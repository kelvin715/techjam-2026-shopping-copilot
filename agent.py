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
from src.telemetry import emit


class Agent:
    """Agent implementation matching the organizer's reset/respond contract."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        llm_settings: LLMSettings | None = None,
        llm_client=None,
        dense=None,
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
        # Research baselines for free-form wording. ``lexical`` reads a message
        # with exact catalog matching only; ``dense`` adds an embedding
        # similarity to the evidence score. Neither runs on protocol wording.
        if self.llm_settings.grounder is not None and config.LLM_WIDEN_RETRIEVE_LIMIT:
            # Free-form sessions may look beyond their shelf pool; build the
            # retrieval index now rather than on a shopper's turn.
            self.catalog.retrieval_index(config.LLM_WIDEN_MAX_DF)
        # ``dense`` (research only): True builds the embedding index for the
        # dense value-expansion / shelf-recall options, which then run only
        # while ``config.DENSE_VALUE_EXPANSION`` / ``config.DENSE_SHELF_RECALL``
        # are switched on and only for off-protocol messages; an object with
        # the ``DenseIndex`` interface is used as is (tests inject one).
        self._dense = None
        if self.llm_settings.mode == "dense" or dense is True:
            from src.dense import DenseIndex

            self._dense = DenseIndex(self.catalog)
        elif dense is not None and dense is not False:
            self._dense = dense

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
            emit("grounding", "running")
            constraints_before = list(state.constraints)
            state.current_turn = turn
            # Reaching another turn proves that the previous slate missed. The
            # override parser clears this history before it can affect the new
            # intent.
            previous_slate = state.previous_slate()
            state.confirm_previous_misses()
            scenario_before = state.scenario
            recognized = parse(user_message, state, self.catalog)
            if not recognized and self.llm_settings.grounder is not None:
                # The template parser may have guessed a scenario from a
                # fragment ("I'm looking for ...") of a sentence it could not
                # otherwise read; the free-text reader decides that itself.
                state.scenario = scenario_before
            dense = self._dense if (
                config.DENSE_VALUE_EXPANSION or config.DENSE_SHELF_RECALL
            ) and self.llm_settings.mode != "dense" else None
            if not recognized and self.llm is None and self.llm_settings.mode == "lexical":
                from src.ground import ground_message_lexical

                grounding = ground_message_lexical(user_message, state, self.catalog, dense)
            elif not recognized and self.llm_settings.mode == "dense":
                grounding = self._dense_note(user_message, state)
            if self.llm is not None and not recognized:
                # Off-protocol wording. The shipped reader lets the catalog
                # read the sentence first and consults the model only when
                # that is not enough; the model-only reader is kept for the
                # attribution baselines. Protocol messages never reach the
                # model on either path.
                if config.LLM_GROUND_CASCADE:
                    from src.ground import ground_message_cascade

                    grounding = ground_message_cascade(
                        user_message, state, self.catalog, self.llm, usage, dense
                    )
                else:
                    from src.ground import ground_message

                    grounding = ground_message(
                        user_message, state, self.catalog, self.llm, usage, dense
                    )
                if (
                    grounding.get("status") != "grounded"
                    and state.shelf is None
                    and not state.candidate_pool
                ):
                    # The model is configured but did not answer (timeout,
                    # outage, unparseable reply). Rather than ranking the
                    # whole catalog by popularity, fall back to a token
                    # overlap between the message and the shelf names so the
                    # shopper still lands in a plausible department.
                    shelves, pool = self.catalog.shelf_pool(user_message)
                    if pool:
                        from src.ground import _apply_deferred_budget

                        state.candidate_pool = pool
                        state.pool_shelves = shelves
                        grounding["fallback_pool"] = {
                            "shelves": shelves[:6],
                            "size": len(pool),
                        }
                        _apply_deferred_budget(state, self.catalog, grounding)
            emit("grounding", "complete", recognized=recognized,
                 grounding=grounding, usage=usage.to_dict())
            emit("evidence", "running")
            if grounding is not None and grounding.get("keeps_slate") and previous_slate:
                # A human building on what was just shown ("the first one
                # looks good, does it come in blue?") has not refuted the
                # slate; only the simulator's continuation rule says so.
                state.unconfirm_misses(previous_slate)
            emit("evidence", "complete", constraints=list(state.constraints),
                 added=[x for x in state.constraints if x not in constraints_before],
                 removed=[x for x in constraints_before if x not in state.constraints],
                 weights=state.constraint_weights(), shelf=state.shelf,
                 shelves=list(state.pool_shelves), proven_misses=len(state.proven_misses))
            emit("ranking", "running")
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
                    # the accumulated evidence. Rank the catalog on that
                    # evidence instead of exhausting the guessed pool.
                    candidates = self._beyond_pool(state, dense)
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
                alternatives=state.constraint_alternatives,
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
                rank_source_ids = (
                    self._beyond_pool(state, dense)
                    if state.grounded and not pool_widened
                    else self.catalog.ids
                )
                scores = score_candidates(
                    self.catalog,
                    rank_source_ids,
                    state.constraints,
                    state.profile_tags,
                    state.constraint_weights(),
                    state.scenario,
                    signature_positions_reliable=(
                        state.signature_positions_reliable
                        if config.ADAPT_SIGNATURE_ORDER
                        else True
                    ),
                    alternatives=state.constraint_alternatives,
                )
                if config.PROVEN_MISS_EXCLUSION and state.proven_misses:
                    scores = [
                        (asin, score) for asin, score in scores
                        if asin not in state.proven_misses
                    ]

            if self.llm_settings.mode == "dense" and self._dense is not None and state.free_text and scores:
                query_vector = self._dense.query(" ".join(state.free_text))
                scores = self._dense.rerank(scores, query_vector)

            emit("ranking", "complete", scores=scores, source_count=len(candidates))
            if (
                config.QUESTION_MODE in {
                    "counterfactual", "metric_voi", "answerable_metric_voi"
                }
                and not state.information_complete
                and not state.boundary_signal
            ):
                emit("mvoi", "running")
                question_values = estimate_question_values(
                    self.catalog,
                    scores,
                    state.constraints,
                    available_attributes(state),
                )
                emit("mvoi", "complete", values=[v.to_dict() for v in question_values],
                     selected=choose(state, question_values), mode=config.QUESTION_MODE,
                     turn_cost=config.QUESTION_TURN_COST)
            else:
                question_values = []
                emit("mvoi", "skipped", reason=(
                    "Preferences exhausted" if state.information_complete else
                    "Boundary response uses the safe question" if state.boundary_signal else
                    "Fixed question policy"))
            # A grounded session that switched department restarts the
            # gate's clock; on the protocol path the offset is always zero.
            effective_turn = max(1, turn - state.turn_offset)
            tail_exploration = False
            if (
                config.TAIL_EXPLORATION_ENABLED
                and effective_turn >= config.TAIL_EXPLORATION_TURN
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
                    alternatives=state.constraint_alternatives,
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
                emit("ranking", "complete", scores=scores,
                     source_count=len(rank_source_ids), refined=True)
            emit("planner", "running")
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
                effective_turn,
                disclosed,
                top_k,
                scores,
                state.information_complete,
                refutation_cohort_size,
            )
            recommendations = [asin for asin, _ in scores[:count]]
            attribute = choose(state, question_values)
            emit("planner", "complete", emitted_count=len(recommendations),
                 recommendations=recommendations, selected_question=attribute)
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
                state.last_decision_certificate["widened_candidates"] = len(candidates)
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
            emit("error", "failed", reason="Agent entered contract-safe recovery")
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

        emit("response", "running")
        message = self._message(attribute)
        if self.llm is not None and self.llm_settings.says and recommendations:
            try:
                from src.ground import render_message

                rendered = render_message(
                    state.last_decision_certificate,
                    self.catalog.title.get(recommendations[0]),
                    self.llm,
                    usage,
                    language=state.language,
                    sample=state.language_sample,
                )
                if rendered:
                    message = rendered
                state.last_decision_certificate["llm_usage"] = usage.to_dict()
            except Exception:
                pass

        response = {
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
        emit("response", "complete", response=response)
        return response

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

    def _beyond_pool(self, state, dense=None) -> list[str]:
        """Candidates for a grounded session that has outgrown its shelf pool.

        A rarity-weighted retrieval over the whole catalog, bounded to
        ``config.LLM_WIDEN_RETRIEVE_LIMIT`` products, so a late turn costs a
        few thousand exact scores rather than fifty thousand. Falls back to
        the full catalog when no constraint carries a discriminative word.
        With the dense recall channel on (research option 2) the products
        closest in embedding space to the shopper's own sentences are
        unioned in, so a paraphrase that shares no rare word with the
        catalog can still reach its product.
        """
        limit = config.LLM_WIDEN_RETRIEVE_LIMIT
        retrieved: list[str] = []
        if limit:
            retrieved = self.catalog.retrieve(
                state.constraints,
                state.constraint_weights(),
                limit,
                config.LLM_WIDEN_MAX_DF,
            )
        if dense is not None and config.DENSE_SHELF_RECALL and config.DENSE_WIDEN_LIMIT and state.grounded_messages:
            query = " ".join(state.grounded_messages[-4:])
            seen = set(retrieved)
            for asin in dense.search_products(query, config.DENSE_WIDEN_LIMIT):
                if asin not in seen:
                    seen.add(asin)
                    retrieved.append(asin)
        if retrieved:
            return retrieved
        return self.catalog.ids

    def _dense_note(self, message: str, state) -> dict:
        """Record a free-form message for the dense baseline; no extraction."""
        from src.ground import _lexical_intent

        intent = _lexical_intent(message)
        trace = {"status": "grounded", "grounder": "dense", "intent": intent,
                 "message": message, "accepted": [], "rejected": [], "removed": [],
                 "tokens": {"prompt": 0, "completion": 0}, "latency_ms": 0.0}
        if intent == "override":
            state.override_seen = True
            state.clear_recommendation_history()
            state.free_text = []
        elif intent == "no_preference":
            asked = state.asked[-1] if state.asked else None
            if asked and asked != "other":
                state.exhausted.add(asked)
                trace["exhausted"] = asked
            elif asked == "other":
                state.information_complete = True
                trace["exhausted"] = "other"
        if intent not in ("no_preference", "reject"):
            state.free_text.append(message)
        if state.shelf is None and not state.candidate_pool:
            shelf = self.catalog.match_shelf(message)
            if shelf is not None:
                state.shelf = shelf
            else:
                shelves, pool = self.catalog.shelf_pool(message)
                if pool:
                    state.pool_shelves = shelves
                    state.candidate_pool = pool
                    trace["pool_size"] = len(pool)
        if state.scenario is None:
            state.scenario = "buying"
        if state.free_text:
            state.grounded = True
            state.signature_positions_reliable = False
        elif intent in ("reject", "no_preference") and state.asked and state.asked[-1] and "exhausted" not in trace:
            state.exhausted.add(state.asked[-1])
            trace["exhausted"] = state.asked[-1]
        state.grounding_trace.append(trace)
        return trace

    @staticmethod
    def _message(attribute: str | None) -> str:
        if attribute is None:
            return "I have enough detail to narrow this down."
        if attribute == "other":
            return "Got it. Anything else that matters for this one?"
        return f"Do you have a preference on {attribute}?"
