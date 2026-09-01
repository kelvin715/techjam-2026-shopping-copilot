"""Message parsing and per-session memory."""
from __future__ import annotations
import re
from . import config
from .shelf import norm

# The customer only ever speaks in a handful of templates.
_KEY_REQ   = re.compile(r"a key requirement is:\s*(.+?)\s*$", re.I)
_MATTERS   = re.compile(r"what matters is:\s*(.+?)\s*$", re.I)
_NEED_IS   = re.compile(r"what i need is:\s*(.+?)\s*$", re.I)
_LOOKING   = re.compile(r"i'?m looking for\s+(.+?)\s*$", re.I)
_EXHAUSTED = re.compile(r"i don'?t have an additional preference for\s+(\w+)", re.I)
_NO_PREF   = re.compile(r"i don'?t have a preference for\s+(\w+)", re.I)
_NO_INFO   = re.compile(r"not quite right yet", re.I)
_STILL_EXP = re.compile(r",?\s*but i'?m still exploring\.?\s*$", re.I)

# Research robustness lane: additive paraphrases, never replacements for the
# organizer's canonical templates.
_ALT_KEY_REQ = re.compile(
    r"(?:my must-have is|the main thing i need is)\s*:?\s*(.+?)\s*$", re.I
)
_ALT_MATTERS = re.compile(
    r"(?:here is what matters to me|my priorities are)\s*:\s*(.+?)\s*$", re.I
)
_ALT_NEED_IS = re.compile(
    r"(?:my new requirement is|i now need)\s*:\s*(.+?)\s*$", re.I
)
_ALT_LOOKING = re.compile(
    r"(?:i'?m shopping for|i(?:'d| would) like to browse)\s+(.+?)\s*$", re.I
)
_ALT_EXHAUSTED = re.compile(
    r"(?:nothing else to add about|no more preferences? (?:for|on))\s+(\w+)", re.I
)
_ALT_NO_PREF = re.compile(r"no preference (?:for|on)\s+(\w+)", re.I)
_ALT_NO_INFO = re.compile(r"(?:still miss(?:es)? the mark|still aren'?t right)", re.I)
_ALT_STILL_EXP = re.compile(
    r"(?:i haven'?t settled on preferences|i'?m keeping my options open)\.?\s*$",
    re.I,
)


def _clean(value: str) -> str:
    return value.strip().strip(" .;,")


def split_constraints(blob: str, catalog=None, shelf: str | None = None) -> list[str]:
    """Recover at most two protocol constraints without corrupting semicolons.

    The simulator joins two constraints with ``; ``, but a catalog feature may
    itself contain semicolons. If the complete blob is a known signature value,
    preserve it. Otherwise choose a two-part boundary only when both sides are
    independently supported by read-only catalog signatures. The label/target
    is never consulted.
    """
    whole = _clean(blob)
    if not whole:
        return []
    if ";" not in whole or catalog is None or not hasattr(catalog, "signature_support"):
        return [c for c in (_clean(p) for p in whole.split(";")) if c]
    if catalog.signature_support(whole, shelf) > 0:
        return [whole]
    pieces = [piece.strip() for piece in whole.split(";")]
    supported: list[tuple[int, int, int, list[str]]] = []
    for boundary in range(1, len(pieces)):
        left = _clean("; ".join(pieces[:boundary]))
        right = _clean("; ".join(pieces[boundary:]))
        left_support = catalog.signature_support(left, shelf)
        right_support = catalog.signature_support(right, shelf)
        if left_support > 0 and right_support > 0:
            supported.append((
                min(left_support, right_support),
                left_support + right_support,
                -boundary,
                [left, right],
            ))
    if supported:
        return max(supported, key=lambda item: item[:3])[3]
    # The protocol reveals at most two items. Keeping the remainder intact is
    # safer than manufacturing three or more constraints from one feature.
    first = _clean(pieces[0])
    remainder = _clean("; ".join(pieces[1:]))
    return [value for value in (first, remainder) if value]


class SessionState:
    """Everything we have learned in this conversation.

    The starter agent's fatal flaw is having none of this: it rebuilds its
    query from the current message alone and so forgets every constraint.
    """

    def __init__(self, user_profile: dict | None = None) -> None:
        self.profile = user_profile or {}
        tags = (self.profile.get("preference_tags") or [])
        self.profile_tags = [str(t).strip().lower() for t in tags if str(t).strip()]
        self.shelf: str | None = None
        self.constraints: list[str] = []
        self._seen: set[str] = set()
        self._constraint_weight: dict[str, float] = {}
        self._provisional: set[str] = set()
        self.exhausted: set[str] = set()   # attributes with nothing left to give
        self.asked: list[str | None] = []
        self.scenario: str | None = None
        self.boundary_signal = False
        self.override_seen = False
        self.information_complete = False
        self.last_reply_count: int | None = None
        self.signature_positions_reliable = True
        self.proven_misses: set[str] = set()
        self._last_recommendations: list[str] = []
        self.last_decision_certificate: dict = {}
        self.last_counterfactual_context: dict = {}
        self.last_counterfactual_explanation: dict | None = None

    def add(self, value: str, *, provisional: bool = False) -> None:
        value = _clean(value)
        if not value:
            return
        key = norm(value)
        if key in self._seen:
            return
        self._seen.add(key)
        self.constraints.append(value)
        self._constraint_weight[key] = 1.0
        if provisional:
            self._provisional.add(key)

        # The public protocol exposes at most two hard and two soft
        # constraints. Once all four are known, another clarification turn
        # cannot add information.
        if len(self.constraints) >= 4:
            self.information_complete = True

    def decay_provisional(self, decay: float) -> None:
        """Retain a withdrawn opening preference as lower-confidence evidence."""
        keep = max(0.0, min(1.0, float(decay)))
        for key in self._provisional:
            self._constraint_weight[key] = keep

    def constraint_weights(self) -> list[float]:
        return [self._constraint_weight.get(norm(value), 1.0) for value in self.constraints]

    def confirm_previous_misses(self) -> None:
        """Promote the previous slate to negative evidence on continuation.

        ``respond`` is called again only when the prior slate did not end the
        session. Therefore every previously shown ASIN is known not to be the
        current target. Intent override parsing clears this evidence because
        pre-override recommendations are not scored against the new intent.
        """
        self.proven_misses.update(self._last_recommendations)
        self._last_recommendations = []

    def remember_recommendations(self, asins: list[str]) -> None:
        self._last_recommendations = list(dict.fromkeys(asins))

    def clear_recommendation_history(self) -> None:
        self.proven_misses.clear()
        self._last_recommendations = []


def parse(message: str, state: SessionState, catalog) -> None:
    """Update state from one customer message. Mutates state in place."""
    msg = message.strip()
    robust = bool(config.ROBUST_PARSER)
    key_match = _KEY_REQ.search(msg) or (robust and _ALT_KEY_REQ.search(msg))
    matters_match = _MATTERS.search(msg) or (robust and _ALT_MATTERS.search(msg))
    need_match = _NEED_IS.search(msg) or (robust and _ALT_NEED_IS.search(msg))
    looking_match = _LOOKING.search(msg) or (robust and _ALT_LOOKING.search(msg))
    exploring_match = _STILL_EXP.search(msg) or (
        robust and _ALT_STILL_EXP.search(msg)
    )

    # --- shelf (stated once, in turn 1, and never changes) ---
    if state.shelf is None:
        state.shelf = catalog.match_shelf(msg)

    # --- scenario, inferable from the opening template ---
    if state.scenario is None and looking_match:
        if key_match:
            state.scenario = "buying"
        elif exploring_match:
            state.scenario = "browsing_or_boundary"
        else:
            state.scenario = "intent_override"

    # --- messages that carry no information ---
    if _NO_INFO.search(msg) or (robust and _ALT_NO_INFO.search(msg)):
        return
    m = _EXHAUSTED.search(msg) or (robust and _ALT_EXHAUSTED.search(msg))
    if m:
        attribute = m.group(1).lower()
        state.exhausted.add(attribute)
        state.last_reply_count = 0
        if attribute == "other":
            state.information_complete = True
        return
    if _NO_PREF.search(msg) or (robust and _ALT_NO_PREF.search(msg)):
        # boundary session deflecting our first question; not exhausted
        state.boundary_signal = True
        return

    # --- the override turn ---
    if need_match:
        state.override_seen = True
        state.last_reply_count = None
        state.clear_recommendation_history()
        state.decay_provisional(config.OVERRIDE_DECAY)
        for c in split_constraints(need_match.group(1), catalog, state.shelf):
            state.add(c)
        return

    # --- constraints revealed by answering a question ---
    if matters_match:
        revealed = split_constraints(matters_match.group(1), catalog, state.shelf)
        state.last_reply_count = len(revealed)
        if (
            revealed
            and state.asked
            and state.asked[-1] not in (None, "other")
        ):
            state.signature_positions_reliable = False
        for c in revealed:
            state.add(c)
        # "other" is a wildcard in the protocol. It returns up to two of all
        # remaining constraints, so a short batch proves that none remain.
        if state.asked and state.asked[-1] == "other" and len(revealed) < 2:
            state.information_complete = True
        return

    # --- the opening line ---
    if key_match:
        state.add(key_match.group(1))
        return

    m = looking_match
    if m:
        tail = m.group(1)
        if exploring_match:
            return                       # browsing: nothing but the shelf
        if state.shelf:                  # intent_override: shelf then a constraint
            low, sl = tail.lower(), state.shelf.lower()
            if low.startswith(sl):
                rest = _clean(tail[len(sl):])
                if rest:
                    state.add(rest, provisional=True)
