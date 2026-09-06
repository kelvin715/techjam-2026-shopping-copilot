"""Official entry point for ARC — the Ask, Rank, Commit shopping agent.

The runtime is deliberately offline and dependency-free. The catalog is loaded
once into in-memory shelf, rarity, structured-attribute, and popularity indexes;
each session then maintains its own incremental constraint state.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src import config
from src.dialog import SessionState, parse
from src.evidence import (
    build_certificate,
    candidate_signals,
    estimate_question_values,
    minimal_counterfactual_explanation,
)
from src.llm import ChatClient, LLMSettings, LLMUsage
from src.policy import available_attributes, choose, emit_count, exact_signature_prefix
from src.rank import diversify_evidence_ties, score_candidates
from src.shelf import Catalog


class Agent:
    """Agent implementation matching the organizer's reset/respond contract."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        llm_settings: LLMSettings | None = None,
        llm_client=None,
    ) -> None:
        self.catalog = Catalog(catalog_path)
        self._sessions: dict[str, SessionState] = {}
        # The optional language layer is off unless ARC_LLM_MODE selects it.
        # With it off, nothing below ever constructs a client or spends a
        # token, so the official scoring path is unchanged.
        self.llm_settings = llm_settings or LLMSettings.from_env()
        if llm_client is not None:
            self.llm = llm_client
        elif self.llm_settings.enabled:
            self.llm = ChatClient(self.llm_settings)
        else:
            self.llm = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            # Defensive fallback: the contract calls reset first, but a malformed
            # request should degrade safely instead of crashing the whole run.
            state = SessionState({})
            self._sessions[session_id] = state

        usage = LLMUsage()
        grounding: dict | None = None
        try:
            # Reaching another turn proves that the previous slate missed. The
            # override parser clears this history before it can affect the new
            # intent.
            state.confirm_previous_misses()
            recognized = parse(user_message, state, self.catalog)
            if self.llm is not None and not recognized:
                # Off-protocol wording: let the model propose a reading and the
                # catalog verify it. Protocol messages never reach the model.
                from src.ground import ground_message

                grounding = ground_message(
                    user_message, state, self.catalog, self.llm, usage
                )
            candidates = self.catalog.candidates(state.shelf)
            pool_widened = False
            if state.shelf is None and state.candidate_pool:
                candidates = state.candidate_pool
                if (
                    config.LLM_POOL_WIDEN_AFTER_MISSES
                    and len(state.proven_misses) >= config.LLM_POOL_WIDEN_AFTER_MISSES
                ):
                    # Failure detection: a grounded shelf guess that has
                    # already refuted a full slate is more likely wrong than
                    # the accumulated evidence. Rank the whole catalog on
                    # that evidence instead of exhausting the guessed pool.
                    candidates = self.catalog.ids
                    pool_widened = True
            rank_source_ids = candidates
            scores = score_candidates(
                self.catalog,
                candidates,
                state.constraints,
                state.profile_tags,
                state.constraint_weights(),
                state.scenario,
                signature_positions_reliable=(
                    state.signature_positions_reliable
                    if config.ADAPT_SIGNATURE_ORDER
                    else True
                ),
            )
            if config.PROVEN_MISS_EXCLUSION and state.proven_misses:
                scores = [
                    (asin, score) for asin, score in scores
                    if asin not in state.proven_misses
                ]

            recovered = False
            if not scores and candidates is not self.catalog.ids:
                # A wrong or exhausted shelf should not terminate the agent.
                # Broaden to the complete read-only catalogue while retaining
                # all grounded constraints and proven misses.
                recovered = True
                rank_source_ids = self.catalog.ids
                scores = score_candidates(
                    self.catalog,
                    self.catalog.ids,
                    state.constraints,
                    state.profile_tags,
                    state.constraint_weights(),
                    state.scenario,
                    signature_positions_reliable=(
                        state.signature_positions_reliable
                        if config.ADAPT_SIGNATURE_ORDER
                        else True
                    ),
                )
                if config.PROVEN_MISS_EXCLUSION and state.proven_misses:
                    scores = [
                        (asin, score) for asin, score in scores
                        if asin not in state.proven_misses
                    ]

            if (
                config.QUESTION_MODE in {
                    "counterfactual", "metric_voi", "answerable_metric_voi"
                }
                and not state.information_complete
                and not state.boundary_signal
            ):
                question_values = estimate_question_values(
                    self.catalog,
                    scores,
                    state.constraints,
                    available_attributes(state),
                )
            else:
                question_values = []
            tail_exploration = False
            if (
                config.TAIL_EXPLORATION_ENABLED
                and turn >= config.TAIL_EXPLORATION_TURN
                and len(state.proven_misses) >= top_k
                and scores
            ):
                evidence_scores = score_candidates(
                    self.catalog,
                    rank_source_ids,
                    state.constraints,
                    state.profile_tags,
                    state.constraint_weights(),
                    state.scenario,
                    popularity_weight=0.0,
                    signature_positions_reliable=(
                        state.signature_positions_reliable
                        if config.ADAPT_SIGNATURE_ORDER
                        else True
                    ),
                )
                if config.PROVEN_MISS_EXCLUSION and state.proven_misses:
                    evidence_scores = [
                        (asin, score) for asin, score in evidence_scores
                        if asin not in state.proven_misses
                    ]
                scores = diversify_evidence_ties(
                    scores,
                    evidence_scores,
                    config.TAIL_EXPLORATION_CORE_WINDOW,
                )
                tail_exploration = True
            refutation_cohort_size = exact_signature_prefix(
                scores, self.catalog.signature
            )
            # The protocol discloses at most four constraints, so four is
            # "complete" for a simulator session. A grounded human session
            # keeps clarifying until the shopper signals exhaustion or the
            # late-turn gate opens.
            disclosed = len(state.constraints)
            if state.grounded:
                disclosed = min(disclosed, config.MAX_DISCLOSED_CONSTRAINTS - 1)
            count = emit_count(
                turn,
                disclosed,
                top_k,
                scores,
                state.information_complete,
                refutation_cohort_size,
            )
            recommendations = [asin for asin, _ in scores[:count]]
            attribute = choose(state, question_values)
            score_lookup = dict(scores)
            details = [
                candidate_signals(
                    self.catalog,
                    asin,
                    score_lookup[asin],
                    state.constraints,
                    state.constraint_weights(),
                    state.scenario,
                    (
                        state.signature_positions_reliable
                        if config.ADAPT_SIGNATURE_ORDER
                        else True
                    ),
                )
                for asin in recommendations[:3]
            ]
            state.last_decision_certificate = build_certificate(
                state=state,
                turn=turn,
                source_candidate_count=len(candidates),
                scores=scores,
                recommendations=recommendations,
                question=attribute,
                question_values=question_values,
                recovered=recovered,
                candidate_details=details,
                tail_exploration=tail_exploration,
                refutation_cohort_size=refutation_cohort_size,
            )
            if grounding is not None or usage.calls:
                state.last_decision_certificate["llm_grounding"] = grounding
                state.last_decision_certificate["llm_usage"] = usage.to_dict()
            if pool_widened:
                state.last_decision_certificate["reason_codes"].append(
                    "grounded_pool_widened_to_catalog"
                )
            excluded = state.proven_misses if config.PROVEN_MISS_EXCLUSION else set()
            state.last_counterfactual_context = {
                "candidate_ids": [
                    asin for asin in rank_source_ids if asin not in excluded
                ],
                "constraints": list(state.constraints),
                "constraint_weights": state.constraint_weights(),
                "profile_tags": list(state.profile_tags),
                "scenario": state.scenario,
                "signature_positions_reliable": (
                    state.signature_positions_reliable
                    if config.ADAPT_SIGNATURE_ORDER
                    else True
                ),
                "original_top": scores[0][0] if scores else None,
            }
            state.last_counterfactual_explanation = None
            state.remember_recommendations(recommendations)
            state.asked.append(attribute)
        except Exception:
            # The official harness treats an exception as an empty turn. Return a
            # valid recovery response so one bad message cannot terminate a run.
            recommendations, attribute = [], "feature"
            state.last_decision_certificate = {
                "turn": turn,
                "action": "RECOVER",
                "reason_codes": ["contract_safe_exception_fallback"],
                "question": attribute,
                "recommendations": [],
            }

        message = self._message(attribute)
        if self.llm is not None and self.llm_settings.says and recommendations:
            try:
                from src.ground import render_message

                rendered = render_message(
                    state.last_decision_certificate,
                    self.catalog.title.get(recommendations[0]),
                    self.llm,
                    usage,
                )
                if rendered:
                    message = rendered
                state.last_decision_certificate["llm_usage"] = usage.to_dict()
            except Exception:
                pass

        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [
                {"parent_asin": asin} for asin in recommendations
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        }

    def explain_last_decision(self, session_id: str) -> dict:
        """Return a copy of the latest non-contract diagnostic certificate."""
        state = self._sessions.get(session_id)
        if state is None:
            return {}
        if state.last_counterfactual_explanation is None:
            context = state.last_counterfactual_context
            state.last_counterfactual_explanation = (
                minimal_counterfactual_explanation(
                    self.catalog,
                    context.get("candidate_ids", []),
                    context.get("constraints", []),
                    context.get("constraint_weights", []),
                    context.get("profile_tags", []),
                    context.get("scenario"),
                    context.get("original_top"),
                    context.get("signature_positions_reliable", True),
                )
                if context
                else {
                    "status": "no_decision_context",
                    "faithful": False,
                    "minimal_removed_count": None,
                }
            )
        certificate = deepcopy(state.last_decision_certificate)
        certificate["minimal_counterfactual_explanation"] = deepcopy(
            state.last_counterfactual_explanation
        )
        return certificate

    @staticmethod
    def _message(attribute: str | None) -> str:
        if attribute is None:
            return "I have enough detail to narrow this down."
        if attribute == "other":
            return "Got it. Anything else that matters for this one?"
        return f"Do you have a preference on {attribute}?"
