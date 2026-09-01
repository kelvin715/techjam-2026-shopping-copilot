"""Counterfactual clarification value and auditable decision certificates.

The official response stays contract-minimal. These helpers instead produce an
internal certificate that can be inspected through
``Agent.explain_last_decision`` without leaking evaluator-only information into
the ranking path.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
import re

from . import config
from .shelf import MATERIALS, norm

_BUDGET = re.compile(r"(?:budget|\$|<=|under\s+\$?\d)", re.I)
_COLOR_WORDS = (
    "color", "black", "white", "blue", "red", "pink", "green",
)


@dataclass(frozen=True)
class QuestionValue:
    """Pool-reduction and score-aligned value of one attribute question."""

    attribute: str
    candidate_count: int
    answer_group_count: int
    expected_remaining: float
    reduction_ratio: float
    answerability: float
    expected_hit_at_10: float
    expected_mrr: float
    ideal_expected_hit_at_10: float
    ideal_expected_mrr: float
    metric_voi: float
    answerable_metric_voi: float

    def to_dict(self) -> dict:
        return asdict(self)


def classify_constraint(value: str) -> str:
    """Mirror the public protocol's visible attribute buckets locally."""
    lowered = norm(value)
    if _BUDGET.search(lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in _COLOR_WORDS):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def plausible_candidates(scores: list[tuple[str, float]]) -> list[str]:
    """Bound planning to the competitive score neighbourhood."""
    if not scores:
        return []
    maximum = max(1, int(config.QUESTION_POOL_MAX))
    best = scores[0][1]
    if best <= 0:
        return [asin for asin, _ in scores[:maximum]]
    cutoff = best * float(config.QUESTION_SCORE_RATIO)
    selected = [asin for asin, score in scores if score >= cutoff][:maximum]
    minimum = min(10, len(scores), maximum)
    if len(selected) < minimum:
        selected = [asin for asin, _ in scores[:minimum]]
    return selected


def _predicted_answer(
    signature: tuple[str, ...],
    known: set[str],
    attribute: str,
) -> tuple[str, ...]:
    remaining = [value for value in signature if value not in known]
    if attribute == "other":
        matches = remaining[:2]
    else:
        matches = [
            value for value in remaining if classify_constraint(value) == attribute
        ][:2]
    return tuple(matches) if matches else ("<none>",)


def estimate_question_values(
    catalog,
    scores: list[tuple[str, float]],
    known_constraints: list[str],
    attributes: list[str],
) -> list[QuestionValue]:
    """Estimate question utility by counterfactual candidate partitioning.

    If candidate ``p`` were the hidden target, its canonical signature predicts
    the protocol-visible answer. Grouping candidates by that answer yields the
    expected surviving pool ``sum(group_size**2) / N``.

    The metric-aligned utility mirrors the rank-sensitive part of the official
    objective. In the ideal estimate every answer group is assumed to rerank
    perfectly. In the answerability-aware estimate, the ``<none>`` branch
    retains the current order because the live protocol only marks that
    attribute exhausted; it does not add a positive ranking constraint.
    """
    candidate_ids = plausible_candidates(scores)
    total = len(candidate_ids)
    if not total:
        return []
    known = {norm(value) for value in known_constraints}
    rank_of = {asin: index for index, asin in enumerate(candidate_ids, start=1)}
    base_hit = min(total, 10) / total
    base_mrr = sum(1.0 / rank for rank in range(1, min(total, 10) + 1)) / total
    base_metric = 0.50 * base_hit + 0.30 * base_mrr
    values: list[QuestionValue] = []
    for attribute in attributes:
        members: dict[tuple[str, ...], list[str]] = {}
        for asin in candidate_ids:
            answer = _predicted_answer(
                catalog.signature.get(asin, ()), known, attribute
            )
            members.setdefault(answer, []).append(asin)
        groups = Counter({answer: len(asins) for answer, asins in members.items()})
        expected = sum(size * size for size in groups.values()) / total
        ideal_hit_numerator = 0.0
        ideal_mrr_numerator = 0.0
        answerable_hit_numerator = 0.0
        answerable_mrr_numerator = 0.0
        answerable_count = 0
        for answer, asins in members.items():
            size = len(asins)
            capped = min(size, 10)
            harmonic = sum(1.0 / rank for rank in range(1, capped + 1))
            ideal_hit_numerator += capped
            ideal_mrr_numerator += harmonic
            if answer == ("<none>",):
                answerable_hit_numerator += sum(
                    rank_of[asin] <= 10 for asin in asins
                )
                answerable_mrr_numerator += sum(
                    1.0 / rank_of[asin]
                    for asin in asins
                    if rank_of[asin] <= 10
                )
            else:
                answerable_count += size
                answerable_hit_numerator += capped
                answerable_mrr_numerator += harmonic
        ideal_hit = ideal_hit_numerator / total
        ideal_mrr = ideal_mrr_numerator / total
        answerable_hit = answerable_hit_numerator / total
        answerable_mrr = answerable_mrr_numerator / total
        ideal_value = 0.50 * ideal_hit + 0.30 * ideal_mrr
        answerable_value = 0.50 * answerable_hit + 0.30 * answerable_mrr
        values.append(QuestionValue(
            attribute=attribute,
            candidate_count=total,
            answer_group_count=len(groups),
            expected_remaining=round(expected, 6),
            reduction_ratio=round(max(0.0, 1.0 - expected / total), 6),
            answerability=round(answerable_count / total, 6),
            expected_hit_at_10=round(answerable_hit, 6),
            expected_mrr=round(answerable_mrr, 6),
            ideal_expected_hit_at_10=round(ideal_hit, 6),
            ideal_expected_mrr=round(ideal_mrr, 6),
            metric_voi=round(
                ideal_value - base_metric - config.QUESTION_TURN_COST, 6
            ),
            answerable_metric_voi=round(
                answerable_value - base_metric - config.QUESTION_TURN_COST, 6
            ),
        ))
    return values


def minimal_counterfactual_explanation(
    catalog,
    candidate_ids: list[str],
    constraints: list[str],
    constraint_weights: list[float],
    profile_tags: list[str],
    scenario: str | None,
    original_top: str | None,
    signature_positions_reliable: bool = True,
) -> dict:
    """Find and verify the smallest evidence removal that changes rank one.

    This is deliberately lazy: it is called by the diagnostic explanation API,
    never on the official response path. With the protocol's four-constraint
    limit there are at most fifteen reranks.
    """
    from .rank import score_candidates

    if original_top is None or not candidate_ids:
        return {
            "status": "no_ranked_candidate",
            "faithful": False,
            "minimal_removed_count": None,
        }
    if not constraints:
        return {
            "status": "no_constraint_evidence_to_remove",
            "faithful": False,
            "minimal_removed_count": None,
            "original_top": original_top,
        }
    weights = list(constraint_weights)
    if len(weights) != len(constraints):
        weights = [1.0] * len(constraints)
    indices = range(len(constraints))
    for removal_count in range(1, len(constraints) + 1):
        for removed in combinations(indices, removal_count):
            removed_set = set(removed)
            kept_constraints = [
                value for index, value in enumerate(constraints)
                if index not in removed_set
            ]
            kept_weights = [
                value for index, value in enumerate(weights)
                if index not in removed_set
            ]
            reranked = score_candidates(
                catalog,
                candidate_ids,
                kept_constraints,
                profile_tags,
                kept_weights,
                scenario,
                signature_positions_reliable=signature_positions_reliable,
            )
            new_top = reranked[0][0] if reranked else None
            if new_top != original_top:
                original_rank = next(
                    (
                        rank
                        for rank, (asin, _) in enumerate(reranked, start=1)
                        if asin == original_top
                    ),
                    None,
                )
                return {
                    "status": "minimal_counterfactual_found",
                    "faithful": True,
                    "minimal_removed_count": removal_count,
                    "removed_constraints": [
                        {
                            "index": index,
                            "constraint": constraints[index],
                            "confidence": round(float(weights[index]), 6),
                        }
                        for index in removed
                    ],
                    "retained_constraints": kept_constraints,
                    "original_top": original_top,
                    "counterfactual_top": new_top,
                    "original_top_counterfactual_rank": original_rank,
                }
    return {
        "status": "top_invariant_to_all_constraint_removals",
        "faithful": False,
        "minimal_removed_count": None,
        "original_top": original_top,
    }


def candidate_signals(
    catalog,
    asin: str,
    final_score: float,
    constraints: list[str],
    constraint_weights: list[float],
    scenario: str | None,
    signature_positions_reliable: bool = True,
) -> dict:
    """Return named, human-auditable evidence for one ranked product."""
    from .rank import _constraint_kind, _satisfies, _signature_score, _typed_score
    from .shelf import tokens

    phrases = [norm(value) for value in constraints if norm(value)]
    weights = constraint_weights
    if len(weights) != len(phrases):
        weights = [1.0] * len(phrases)
    phrase_tokens = [
        [token for token in tokens(phrase) if len(token) > 2]
        for phrase in phrases
    ]
    typed = _typed_score(catalog, asin, phrases, phrase_tokens, weights)
    signature = _signature_score(
        catalog,
        asin,
        phrases,
        weights,
        scenario,
        signature_positions_reliable,
    )
    matches = []
    for phrase, phrase_toks, confidence in zip(phrases, phrase_tokens, weights):
        kind, _ = _constraint_kind(phrase)
        satisfaction = _satisfies(catalog, asin, phrase, phrase_toks)
        matches.append({
            "constraint": phrase,
            "kind": kind,
            "confidence": round(float(confidence), 6),
            "satisfaction": round(satisfaction, 6),
        })
    return {
        "parent_asin": asin,
        "final_rank_score": round(final_score, 6),
        "typed_satisfaction": round(typed, 6),
        "typed_weighted_signal": round(config.TYPED_WEIGHT * typed, 6),
        "signature_likelihood": round(signature, 6),
        "signature_positions_reliable": signature_positions_reliable,
        "signature_weighted_signal": round(config.SIGNATURE_WEIGHT * signature, 6),
        "bounded_popularity_prior": round(catalog.popularity(asin), 6),
        "constraint_evidence": matches,
    }


def build_certificate(
    *,
    state,
    turn: int,
    source_candidate_count: int,
    scores: list[tuple[str, float]],
    recommendations: list[str],
    question: str | None,
    question_values: list[QuestionValue],
    recovered: bool,
    candidate_details: list[dict],
    tail_exploration: bool = False,
) -> dict:
    """Materialise the inspectable evidence behind the turn decision."""
    best = scores[0][1] if scores else None
    second = scores[1][1] if len(scores) > 1 else None
    absolute_margin = None if best is None or second is None else best - second
    relative_margin = (
        None if absolute_margin is None or not best
        else absolute_margin / abs(best)
    )
    selected_value = next(
        (value for value in question_values if value.attribute == question), None
    )
    if recovered:
        action = "RECOVER"
    elif question is None:
        action = "RECOMMEND"
    else:
        action = "ASK"

    reasons: list[str] = []
    if state.information_complete:
        reasons.append("information_complete")
    if len(state.constraints) >= config.MAX_DISCLOSED_CONSTRAINTS:
        reasons.append("maximum_protocol_constraints_observed")
    if state.proven_misses:
        reasons.append("previously_shown_products_excluded")
    if tail_exploration:
        reasons.append("late_evidence_tie_exploration")
    if state.boundary_signal and question == "other":
        reasons.append("boundary_safe_open_question")
    if selected_value is not None and config.QUESTION_MODE == "counterfactual":
        if not state.boundary_signal:
            reasons.append("minimum_expected_surviving_pool")
    if selected_value is not None and config.QUESTION_MODE == "metric_voi":
        reasons.append("maximum_metric_value_of_information")
    if (
        selected_value is not None
        and config.QUESTION_MODE == "answerable_metric_voi"
    ):
        reasons.append("maximum_answerability_aware_metric_value_of_information")
    if question is not None and not reasons:
        reasons.append("more_evidence_has_positive_value")
    if question is None and not reasons:
        reasons.append("no_useful_clarification_remaining")

    return {
        "turn": turn,
        "action": action,
        "reason_codes": reasons,
        "intent_track": state.scenario,
        "shelf": state.shelf,
        "constraints": list(state.constraints),
        "constraint_confidence": state.constraint_weights(),
        "source_candidate_count": source_candidate_count,
        "ranked_candidate_count": len(scores),
        "plausible_candidate_count": (
            question_values[0].candidate_count
            if question_values else min(len(scores), config.QUESTION_POOL_MAX)
        ),
        "proven_miss_count": len(state.proven_misses),
        "excluded_parent_asins": sorted(state.proven_misses),
        "top_score": None if best is None else round(best, 6),
        "top2_absolute_margin": (
            None if absolute_margin is None else round(absolute_margin, 6)
        ),
        "top2_relative_margin": (
            None if relative_margin is None else round(relative_margin, 6)
        ),
        "question": question,
        "selected_question_value": (
            None if selected_value is None else selected_value.to_dict()
        ),
        "question_values": [value.to_dict() for value in question_values],
        "output_gate": {
            "policy": config.GATE_POLICY_LABEL,
            "mode": config.GATE_MODE,
            "proxy_rank_loss_alpha": config.GATE_PROXY_RISK_ALPHA,
            "emitted_count": len(recommendations),
        },
        "tail_exploration": tail_exploration,
        "recommendations": recommendations,
        "candidate_evidence": candidate_details,
    }
