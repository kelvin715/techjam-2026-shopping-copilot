"""Which question to ask next, and how much to commit to."""
from __future__ import annotations
from . import config

ORDER = ["other", "feature", "material", "color", "style", "size", "use_case"]

W_HIT = 0.50
W_MRR = 0.30
W_EFFICIENCY = 0.20

# structurally unreachable: the harness classifier has no bucket for brand or
# category, and price is appended last and almost always truncated away
NEVER = {"brand", "budget", "category"}


def available_attributes(state) -> list[str]:
    return [
        attribute for attribute in ORDER
        if attribute not in NEVER and attribute not in state.exhausted
    ]


def choose(state, question_values=None) -> str | None:
    if state.information_complete:
        return None
    available = available_attributes(state)
    if question_values and not state.boundary_signal:
        by_attribute = {
            value.attribute: value for value in question_values
            if value.attribute in available
        }
        if by_attribute:
            # Stable order breaks exact utility ties in favour of the legacy,
            # empirically strong ``other`` route.
            order = {attribute: index for index, attribute in enumerate(ORDER)}
            if config.QUESTION_MODE == "counterfactual":
                return min(
                    by_attribute.values(),
                    key=lambda value: (
                        value.expected_remaining,
                        order.get(value.attribute, len(ORDER)),
                    ),
                ).attribute
            if config.QUESTION_MODE == "metric_voi":
                return max(
                    by_attribute.values(),
                    key=lambda value: (
                        value.metric_voi,
                        -order.get(value.attribute, len(ORDER)),
                    ),
                ).attribute
            if config.QUESTION_MODE == "answerable_metric_voi":
                return max(
                    by_attribute.values(),
                    key=lambda value: (
                        value.answerable_metric_voi,
                        -order.get(value.attribute, len(ORDER)),
                    ),
                ).attribute
    for attribute in available:
        return attribute
    return "feature"


def _refutation_plan(turn: int, cohort_size: int, top_k: int) -> int:
    """Solve the remaining-session batch size for indistinguishable siblings.

    The state is ``(turn, remaining siblings)``. A failed batch is not wasted:
    continuation proves every emitted sibling wrong, so the next state starts
    after that batch. The recurrence directly prices Hit@10, reciprocal rank,
    and turn efficiency using the published evaluator weights.
    """
    max_turn = config.EVALUATOR_MAX_TURNS
    full_from = config.REFUTATION_FULL_FROM_TURN
    cache: dict[tuple[int, int], tuple[float, int]] = {}

    def solve(current_turn: int, remaining: int) -> tuple[float, int]:
        if remaining <= 0 or current_turn > max_turn:
            return 0.0, 0
        key = (current_turn, remaining)
        if key in cache:
            return cache[key]
        limit = min(top_k, remaining)
        choices = [limit] if current_turn >= full_from else range(1, limit + 1)
        best_value = -1.0
        best_batch = limit
        for batch in choices:
            hit_value = 0.0
            for rank in range(1, batch + 1):
                hit_value += (
                    W_HIT
                    + W_MRR / rank
                    + W_EFFICIENCY * (max_turn + 1 - current_turn) / max_turn
                ) / remaining
            continuation, _ = solve(current_turn + 1, remaining - batch)
            value = hit_value + (remaining - batch) / remaining * continuation
            # On an exact utility tie, prefer the larger batch and earlier hit.
            if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12 and batch > best_batch
            ):
                best_value, best_batch = value, batch
        cache[key] = (best_value, best_batch)
        return cache[key]

    return solve(turn, max(1, cohort_size))[1]


def exact_signature_prefix(scores, signature_lookup) -> int:
    """Count the ranked prefix indistinguishable under the intent protocol."""
    if not scores:
        return 0
    first = signature_lookup.get(scores[0][0], ())
    if not first:
        return 1
    count = 0
    for asin, _ in scores:
        if signature_lookup.get(asin, ()) != first:
            break
        count += 1
    return count


def emit_count(turn: int, n_constraints: int, top_k: int, scores=None,
               information_complete: bool = False,
               refutation_cohort_size: int = 0) -> int:
    """How many recommendations to risk this turn.

    A hit locks in its rank and ends the session, so a premature hit at a poor
    rank costs more than the turns it saves. Commit only as belief concentrates,
    but always commit before the clock runs out.

    count  - expose only the safest prefix until all information is known
    margin - commit to the plausible set: every candidate scoring within
             MARGIN_FRAC of the best. One clear leader means one recommendation.
    """
    complete = information_complete or n_constraints >= config.MAX_DISCLOSED_CONSTRAINTS
    if (
        config.REFUTATION_BATCH_PLANNER
        and complete
        and refutation_cohort_size > 1
        and turn < config.REFUTATION_FULL_FROM_TURN
    ):
        return min(
            top_k,
            _refutation_plan(turn, refutation_cohort_size, top_k),
        )

    if turn >= config.EMIT_LATE_TURN or complete:
        return top_k

    if config.GATE_MODE == "margin" and scores:
        best = scores[0][1]
        if best <= 0:
            return config.EMIT_K0
        cutoff = best * config.MARGIN_FRAC
        k = sum(1 for _, s in scores[:top_k] if s >= cutoff)
        return max(1, min(k, top_k))

    if n_constraints == 0:
        return min(config.EMIT_K0, top_k)
    if n_constraints == 1:
        return min(config.EMIT_K1, top_k)
    return min(config.EMIT_K2, top_k)
