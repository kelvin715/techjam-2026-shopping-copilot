"""Which question to ask next, and how much to commit to."""
from __future__ import annotations
from . import config

ORDER = ["other", "feature", "material", "color", "style", "size", "use_case"]

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


def emit_count(turn: int, n_constraints: int, top_k: int, scores=None,
               information_complete: bool = False) -> int:
    """How many recommendations to risk this turn.

    A hit locks in its rank and ends the session, so a premature hit at a poor
    rank costs more than the turns it saves. Commit only as belief concentrates,
    but always commit before the clock runs out.

    count  - expose only the safest prefix until all information is known
    margin - commit to the plausible set: every candidate scoring within
             MARGIN_FRAC of the best. One clear leader means one recommendation.
    """
    if turn >= config.EMIT_LATE_TURN or information_complete \
            or n_constraints >= config.MAX_DISCLOSED_CONSTRAINTS:
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
