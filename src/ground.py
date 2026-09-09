"""Catalog-verified grounding of free-form shopper language.

The organizer's simulator speaks in a handful of templates that the
deterministic parser in :mod:`src.dialog` understands exactly. Real shoppers
do not. When a message matches no template, this module lets an optional
language model *propose* a structured reading of it, and then lets the
read-only catalog *dispose*: a proposed constraint can only enter session
memory when the catalog contains evidence for it.

    LLM proposes -> catalog verifies -> ranker consumes

Three verification tiers, each with its own confidence weight:

``signature_verbatim``
    the phrase is itself a canonical intent-signature value (the same
    strings the simulator quotes), so it is admitted at full confidence;
``signature_mapped``
    the phrase's content tokens select one canonical signature value
    (``"buckle"`` -> ``"buckle closure"``); admitted at reduced confidence;
``lexical``
    the phrase occurs verbatim in at least one candidate product's text;
    admitted at the lowest confidence so it can order but not dominate.

Anything else is rejected and recorded in the decision certificate. The
model never sees the catalog, never ranks a product, and never sees a label.
"""

from __future__ import annotations

import json
import re

from . import config
from .language import detect, language_name
from .shelf import COLOR_RE, COLORS, MATERIAL_RE, MATERIALS, loose, norm, tokens

INTENTS = ("open", "add", "override", "no_preference", "reject", "other")

TRANSLATE_PROMPT = """Translate the shopper's chat message into natural English for a product search over a clothing, shoes and jewelry catalog.
Keep brand names, product names, sizes, numbers and currency amounts exactly as written; do not convert currencies. Keep the meaning exactly: a cancellation stays a cancellation, "no preference" stays "no preference".
Output the English sentence only."""

SYSTEM_PROMPT = """You convert one shopper chat message into a strict JSON object for a product-search engine over a clothing, shoes and jewelry catalog.
Return exactly these fields:
- "intent": one of
    "open"          a first or new product request;
    "add"           adds preferences to the current search;
    "override"      ONLY when the shopper explicitly cancels or replaces an earlier stated preference ("forget the leather", "instead of black", "changed my mind, I want ...");
    "no_preference" the shopper says they do not care / have no preference / anything works for the attribute just asked (even if they add another wish in the same message);
    "reject"        the shown options are wrong but no new information is given;
    "other".
- "category": a short product-type phrase in the shopper's own words (for example "loafers", "leather belt", "running shoes"), or null.
- "material": one of cotton, polyester, nylon, leather, wool, spandex, silk, rayon, fabric, or null.
- "color": one of black, white, blue, red, pink, green, brown, gray, purple, yellow, orange, or null.
- "budget": a number in US dollars if a price is mentioned, else null.
- "budget_operator": "under" when the shopper states a maximum ("under", "below", "at most", "no more than", "up to", "less than", "within", "max"), "over" for a minimum, "around" for a target price or when unclear; null when budget is null.
- "features": a list of at most 3 short attribute phrases the shopper wants, in catalog-like wording (for example "buckle closure", "zipper closure", "machine wash", "wide fit", "waterproof"). Never repeat the category, material, color or budget here. Do not include vague words such as "nice", "good", "cheap".
- "dropped": earlier preferences the shopper explicitly cancels, else [].
Examples:
"hey i need some comfy leather loafers for work, brown if possible, under 60 bucks" -> {"intent":"open","category":"loafers","material":"leather","color":"brown","budget":60,"budget_operator":"under","features":["comfortable"],"dropped":[]}
"it should have a zipper and be machine washable" -> {"intent":"add","category":null,"material":null,"color":null,"budget":null,"budget_operator":null,"features":["zipper closure","machine wash"],"dropped":[]}
"actually forget the leather thing, i just want it to be black" -> {"intent":"override","category":null,"material":null,"color":"black","budget":null,"features":[],"dropped":["leather"]}
"nah, don't really care about that. just make it sturdy" -> {"intent":"no_preference","category":null,"material":null,"color":null,"budget":null,"features":["sturdy"],"dropped":[]}
"those aren't it, ask me something else" -> {"intent":"reject","category":null,"material":null,"color":null,"budget":null,"features":[],"dropped":[]}
Output only the JSON object."""

RENDER_PROMPT = """You are the voice of a shopping assistant. Write one or two short, warm sentences (at most 40 words) to the shopper.
Rules: mention at most two of the requirements you are matching; describe the top product only with the words given; never invent facts, prices or colours.
If "question_attribute" is set, end by asking about exactly that attribute in plain words (material, color, size, style, feature, use_case; "other" means "anything else that matters to you").
If it is null, do not ask a question.
If "reply_language" is given, write the whole reply in that language, matching the language of "shopper_wrote"; otherwise write in English. Output plain text only."""

_STEM_MIN = 4


def _stem(token: str) -> str:
    if len(token) >= _STEM_MIN and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) >= _STEM_MIN and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    return [_stem(token) for token in tokens(text) if len(token) > 2]


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_list(value: object, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _as_str(item)
        if text and norm(text) not in {norm(existing) for existing in out}:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _budget(value: object) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", value)
        if match:
            try:
                number = float(match.group(0))
                return number if number > 0 else None
            except ValueError:
                return None
    return None


def _format_budget(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"budget around ${text}"


_OPERATOR_WORDS = {
    "under": ("under", "below", "at most", "no more than", "up to", "less than",
              "within", "max", "maximum", "<=", "cheaper than"),
    "over": ("over", "above", "at least", "more than", "minimum", ">="),
}


def _budget_operator(value: object, message: str) -> str:
    """Normalise the model's operator, falling back to the shopper's words."""
    text = str(value or "").strip().lower()
    if text in ("under", "over", "around"):
        return text
    lowered = message.lower()
    for operator, words in _OPERATOR_WORDS.items():
        if any(word in lowered for word in words):
            return operator
    return "around"


def _apply_budget_bound(state, catalog, budget: float, operator: str) -> dict:
    """Treat an explicit price bound as a hard filter on the candidate pool.

    The organizer protocol only ever says ``budget around $n``, so the scored
    path keeps its proximity scoring untouched. A human who says "under $60"
    means a ceiling; products priced above it are removed from the grounded
    pool instead of merely losing a fraction of a score. Products with no
    listed price are kept and flagged as unknown, and an empty result relaxes
    the bound rather than stranding the session.
    """
    if state.pool_before_budget is not None:
        # A new price statement supersedes the previous bound.
        state.candidate_pool = state.pool_before_budget
        state.pool_signature_counts = None
    if state.candidate_pool:
        pool = state.candidate_pool
    elif state.shelf is not None:
        pool = list(catalog.candidates(state.shelf))
    else:
        # "Something under $25" with no department yet: the bound is
        # remembered and applied the moment the session is placed. Filtering
        # the whole catalog here would masquerade as a department.
        state.budget_bound = (operator, budget)
        return {"operator": operator, "bound": budget, "applied": False,
                "reason": "deferred_until_department_placed"}
    kept: list[str] = []
    unknown = 0
    for asin in pool:
        price = catalog.price.get(asin)
        if price is None:
            unknown += 1
            kept.append(asin)
        elif operator == "under" and price <= budget:
            kept.append(asin)
        elif operator == "over" and price >= budget:
            kept.append(asin)
    detail = {
        "operator": operator,
        "bound": budget,
        "pool_before": len(pool),
        "pool_after": len(kept),
        "price_unknown_kept": unknown,
    }
    if kept and len(kept) < len(pool):
        state.pool_before_budget = list(pool)
        state.candidate_pool = kept
        state.pool_signature_counts = None
        detail["applied"] = True
    else:
        detail["applied"] = False
        detail["reason"] = "empty_after_filter" if not kept else "no_product_excluded"
    # Remembered even when nothing was excluded: a later department switch
    # re-applies the bound to the new pool.
    state.budget_bound = (operator, budget)
    return detail


def _apply_deferred_budget(state, catalog, trace: dict) -> None:
    """Apply a price bound stated before the session had a department."""
    if (
        state.budget_bound
        and config.LLM_GROUND_HARD_BUDGET
        and state.pool_before_budget is None
        and (state.candidate_pool or state.shelf is not None)
    ):
        operator, amount = state.budget_bound
        trace["budget_bound"] = _apply_budget_bound(state, catalog, amount, operator)


def _current_pool(catalog, state) -> list[str]:
    """Products the session can currently recommend, before any price bound."""
    if state.pool_before_budget is not None:
        return state.pool_before_budget
    if state.candidate_pool:
        return state.candidate_pool
    if state.shelf is not None:
        return catalog.by_shelf.get(state.shelf, [])
    return []


def _pool_mentions(catalog, pool: list[str], value: str) -> bool:
    """True when at least one product in ``pool`` carries ``value`` in its text."""
    needle = " " + loose(value) + " "
    if len(needle.strip()) < 3:
        return False
    for asin in pool:
        if needle in " " + catalog.ltext[asin] + " ":
            return True
    return False


def _place_category(catalog, state, category: str, intent: str, trace: dict, dense=None) -> None:
    """Route a model-named category to a shelf or a union of shelves.

    On a session's first placement this sets the pool. Later, a category
    that lands in a different department than the current pool switches
    the department (:func:`_switch_department`); one that overlaps the
    current pool is a refinement ("a wide belt" while browsing belts) and
    changes nothing. The token overlap of the category phrase with shelf
    names is what places it; the sentence as a whole is never used here.
    """
    shelf = catalog.match_shelf(category)
    if shelf is not None:
        shelves, pool = [shelf], list(catalog.by_shelf.get(shelf, []))
    else:
        shelves, pool = catalog.shelf_pool(category)
        shelves, pool = _dense_shelves(catalog, category, shelves, pool, dense, trace)
    if not pool:
        trace["category_unplaced"] = category
        return
    current = _current_pool(catalog, state)
    if not current:
        if shelf is not None:
            state.shelf = shelf
            trace["shelf"] = shelf
        else:
            state.pool_shelves = shelves
            state.candidate_pool = pool
            trace["pool_shelves"] = shelves[:12]
            trace["pool_size"] = len(pool)
        _apply_deferred_budget(state, catalog, trace)
        return
    shared = len(set(current) & set(pool))
    overlap = shared / max(1, min(len(current), len(pool)))
    if overlap >= config.LLM_GROUND_SWITCH_OVERLAP:
        trace["category_kept"] = {
            "category": category, "overlap": round(overlap, 3),
            "reason": "refines_current_department",
        }
        return
    if intent not in ("open", "override"):
        # "I'll take the pretty ankle boots one" while browsing slippers is
        # a product description, not a new request: only a new request or a
        # cancellation moves the session.
        trace["category_kept"] = {
            "category": category, "overlap": round(overlap, 3),
            "reason": f"not_a_new_request:{intent}",
        }
        return
    if intent == "open" and _evidence_in_current_department(catalog, state, current, trace):
        # "I'm looking for those hoop earrings with the leopard pattern"
        # while browsing dangle earrings quotes what the current department
        # sells; the shopper is describing a product, not leaving. Only a
        # cancellation switches regardless.
        trace["category_kept"] = {
            "category": category, "overlap": round(overlap, 3),
            "reason": "evidence_in_current_department",
        }
        return
    _switch_department(catalog, state, category, shelf, shelves, pool, overlap, trace)


def _evidence_in_current_department(catalog, state, current: list[str], trace: dict) -> bool:
    """Does this turn's reading quote something the current pool sells?"""
    if trace.get("stage1", {}).get("accepted"):
        return True
    proposal = trace.get("proposal") or {}
    for phrase in proposal.get("features") or []:
        text = norm(phrase)
        if not text:
            continue
        if _support(catalog, state, text) >= 1 or _pool_mentions(catalog, current, text):
            return True
    return False


def _switch_department(catalog, state, category: str, shelf, shelves: list[str],
                       pool: list[str], overlap: float, trace: dict) -> None:
    """"Forget the belt, I want a wallet": move the session to a new department.

    The old pool, the products refuted in it and the questions retired for
    it no longer describe the shopper's request. Feature strings ("buckle
    closure") described the old product type and are dropped. A typed
    material or colour is a preference about the shopper, not the product,
    and is carried at ``OVERRIDE_DECAY`` confidence when the new pool can
    satisfy it; a price bound is the shopper's budget and is re-applied to
    the new pool at full strength.
    """
    previous = list(state.pool_shelves) if state.pool_shelves else ([state.shelf] if state.shelf else [])
    kept: list[dict] = []
    removed: list[str] = []
    for value in list(state.constraints):
        if value.startswith("budget around $"):
            kept.append({"constraint": value, "confidence": 1.0, "reason": "budget_carries"})
            continue
        typed = value in MATERIALS or value.startswith("color:")
        probe = value.split(":", 1)[1].strip() if value.startswith("color:") else value
        if typed and _pool_mentions(catalog, pool, probe):
            state.set_weight(value, config.OVERRIDE_DECAY)
            kept.append({"constraint": value, "confidence": config.OVERRIDE_DECAY,
                         "reason": "typed_preference_carries"})
        else:
            state.remove(value)
            removed.append(value)
    if shelf is not None:
        state.shelf = shelf
        state.candidate_pool = None
        state.pool_shelves = []
    else:
        state.shelf = None
        state.candidate_pool = pool
        state.pool_shelves = shelves
    state.pool_before_budget = None
    state.pool_signature_counts = None
    state.clear_recommendation_history()
    state.exhausted.clear()
    state.information_complete = False
    state.last_reply_count = None
    state.override_seen = True
    state.grounded = True
    # A new request restarts the output gate's clock.
    state.turn_offset = max(0, state.current_turn - 1)
    bound = None
    if state.budget_bound and config.LLM_GROUND_HARD_BUDGET:
        operator, amount = state.budget_bound
        bound = _apply_budget_bound(state, catalog, amount, operator)
    trace["department_switch"] = {
        "category": category,
        "from": previous[:6],
        "to": shelves[:6],
        "pool_size": len(pool),
        "overlap_with_previous": round(overlap, 3),
        "kept": kept,
        "removed": removed,
        "budget_bound": bound,
    }
    if shelf is not None:
        trace["shelf"] = shelf
    else:
        trace["pool_shelves"] = shelves[:12]
        trace["pool_size"] = len(pool)


def _catalog_vocabulary(catalog, state) -> list[str]:
    """Short, common catalog phrases among the products still in play."""
    counts = _pool_support(catalog, state)
    if counts is None:
        if state.shelf is None:
            return []
        counts = catalog.signature_value_count_by_shelf.get(state.shelf, {})
    return catalog.top_signature_values(counts, config.LLM_GROUND_VOCAB_HINTS)


def _user_prompt(message: str, state, vocabulary: list[str]) -> str:
    context = {
        "shelf": state.shelf or (state.pool_shelves[:3] if state.pool_shelves else None),
        "known_constraints": list(state.constraints),
        "last_question": state.asked[-1] if state.asked else None,
    }
    prompt = "Context (JSON): " + json.dumps(context, ensure_ascii=False)
    if vocabulary:
        prompt += (
            "\nCatalog phrases used by products in this category (when the "
            "shopper's meaning matches one of these, use its exact wording in "
            "\"features\"): " + json.dumps(vocabulary, ensure_ascii=False)
        )
    return prompt + "\nShopper message: " + message.strip()


def _demote_dropped(state, dropped: list[str]) -> list[dict]:
    """Apply a cancellation the way the deterministic policy does.

    The submitted agent never deletes a withdrawn preference; it keeps it at
    ``config.OVERRIDE_DECAY`` confidence because the shopper's earlier words
    were still about the same product. ``LLM_GROUND_DROP_MODE`` can switch
    this to a hard removal for deployments where a cancellation is final.
    """
    changed: list[dict] = []
    for item in dropped:
        needle = norm(item)
        needle_tokens = set(content_tokens(needle))
        for existing in list(state.constraints):
            existing_norm = norm(existing)
            if (
                existing_norm == needle
                or needle in existing_norm
                or (needle_tokens and needle_tokens.issubset(set(content_tokens(existing_norm))))
            ):
                if config.LLM_GROUND_DROP_MODE == "remove":
                    if state.remove(existing):
                        changed.append({
                            "dropped": item, "constraint": existing, "action": "removed",
                        })
                else:
                    state.set_weight(existing, config.OVERRIDE_DECAY)
                    changed.append({
                        "dropped": item, "constraint": existing, "action": "decayed",
                        "confidence": config.OVERRIDE_DECAY,
                    })
    return changed


def _pool_support(catalog, state):
    """Signature support restricted to the session's candidate products."""
    if state.shelf is not None or not state.candidate_pool:
        return None
    if state.pool_signature_counts is None:
        counts: dict[str, int] = {}
        for asin in state.candidate_pool:
            for value in catalog.signature.get(asin, ()):
                counts[value] = counts.get(value, 0) + 1
        state.pool_signature_counts = counts
    return state.pool_signature_counts


def _support(catalog, state, value: str) -> int:
    pool_counts = _pool_support(catalog, state)
    if pool_counts is not None:
        return pool_counts.get(value, 0)
    return catalog.signature_support(value, state.shelf)


def _supported_values(catalog, state, min_support: int = 1):
    """Signature values with support among the products still in play."""
    counts = _pool_support(catalog, state)
    if counts is None:
        if state.shelf is not None:
            counts = catalog.signature_value_count_by_shelf.get(state.shelf, {})
        else:
            counts = catalog.signature_value_count
    if min_support <= 1:
        return counts
    return [value for value, count in counts.items() if count >= min_support]


def _expand_value(catalog, state, value: str, phrase: str, dense, detail: dict) -> None:
    """Dense value expansion (research option 1).

    Catalog signature values whose embedding is close to the shopper's phrase
    become alternatives the ranker may credit at reduced weight. Typed values
    (material, colour, budget) are never expanded: their vocabulary is closed
    and a neighbour ("polyester" for "cotton") changes the attribute.
    """
    if dense is None or not config.DENSE_VALUE_EXPANSION:
        return
    if value in MATERIALS or value.startswith("color:") or value.startswith("budget around"):
        return
    generic = {"is discontinued by manufacturer: no"}
    # The nearest neighbours of a catalog string are its spelling variants
    # ("made in the usa or imported" for "made in usa or imported"). Those
    # already match the same products through loose matching; collapse them
    # by content tokens so the slots go to genuinely different strings.
    own = _content_key(value)
    seen = {own}
    alternatives: list[tuple[str, float]] = []
    found = dense.similar_values(
        phrase, _supported_values(catalog, state), 6 * config.DENSE_VALUE_MAX_ALTERNATIVES,
    )
    for alternative, cosine in found:
        if alternative == value or alternative in generic:
            continue
        if alternative.startswith("color:") or alternative in MATERIALS or alternative.startswith("budget around"):
            continue
        key = _content_key(alternative)
        # A string that contains every content word of the admitted value
        # ("silver stainless steel strap") already earns the token credit.
        if key in seen or key >= own:
            continue
        seen.add(key)
        alternatives.append((alternative, cosine))
        if len(alternatives) >= config.DENSE_VALUE_MAX_ALTERNATIVES:
            break
    if alternatives:
        state.constraint_alternatives[norm(value)] = alternatives
        detail["alternatives"] = alternatives


_VARIANT_STOPWORDS = frozenset({"the", "and", "or", "with", "for", "of", "in", "a", "an", "to", "closure"})


def _content_key(value: str) -> frozenset[str]:
    """Content tokens of a catalog string, punctuation and filler removed."""
    return frozenset(
        token for token in tokens(value) if token not in _VARIANT_STOPWORDS
    ) or frozenset({value})


def _dense_shelves(catalog, text: str, shelves: list[str], pool: list[str], dense, trace: dict):
    """Union dense-matched shelves into a token-overlap pool (research option 2)."""
    if dense is None or not config.DENSE_SHELF_RECALL:
        return shelves, pool
    extra = [shelf for shelf, _cosine in dense.similar_shelves(text) if shelf not in shelves]
    if not extra:
        return shelves, pool
    seen = set(pool)
    widened = list(pool)
    for shelf in extra:
        for asin in catalog.by_shelf.get(shelf, ()):
            if asin not in seen:
                seen.add(asin)
                widened.append(asin)
    trace["dense_shelves"] = extra
    return list(shelves) + extra, widened


def _verify_feature(catalog, phrase: str, state, dense=None) -> tuple[str | None, str, float, dict]:
    """Return (constraint, tier, weight, detail) or (None, reason, 0, detail).

    Verification is scoped to the products the session can still recommend:
    the canonical shelf when the wording named one, otherwise the grounded
    candidate pool. A paraphrase must therefore map onto evidence that exists
    among plausible products, not anywhere in the catalog.
    """
    value = norm(phrase)
    if not value:
        return None, "empty", 0.0, {}
    if not config.LLM_GROUND_VERIFY:
        # Attribution baseline: take the model at its word.
        return value, "unverified", config.LLM_GROUND_VERBATIM_WEIGHT, {}
    allowed = set(config.LLM_GROUND_TIERS)
    support = _support(catalog, state, value)
    if support > 0 and "signature_verbatim" in allowed:
        return value, "signature_verbatim", config.LLM_GROUND_VERBATIM_WEIGHT, {
            "support": support,
        }
    phrase_tokens = content_tokens(value)
    if phrase_tokens and "signature_mapped" in allowed:
        limit = len(phrase_tokens) + config.LLM_GROUND_MAPPED_EXTRA_TOKENS
        ranked = []
        for candidate, _global in catalog.signature_values_with_tokens(
            phrase_tokens, state.shelf
        ):
            if len(tokens(candidate)) > limit:
                continue
            local = _support(catalog, state, candidate)
            if local >= config.LLM_GROUND_MAPPED_MIN_SUPPORT:
                ranked.append((candidate, local))
        ranked.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        if ranked:
            best_value, best_support = ranked[0]
            return best_value, "signature_mapped", config.LLM_GROUND_MAPPED_WEIGHT, {
                "from": phrase,
                "support": best_support,
                "alternatives": [item for item, _ in ranked[1:4]],
            }
    if dense is not None and config.DENSE_VALUE_EXPANSION:
        # Dense tier: the phrase names no catalog string, but its embedding
        # sits next to one the candidate products carry ("waterproof" ->
        # "water resistant"). Admitted below the mapped tier, above lexical.
        nearest = dense.similar_values(
            value, _supported_values(catalog, state, config.LLM_GROUND_MAPPED_MIN_SUPPORT), 1,
        )
        if nearest:
            best_value, cosine = nearest[0]
            return best_value, "signature_dense", config.DENSE_VALUE_TIER_WEIGHT, {
                "from": phrase, "cosine": cosine, "support": _support(catalog, state, best_value),
            }
    needle = loose(value)
    if needle and allowed & {"lexical", "lexical_tokens"}:
        pool = state.candidate_pool or catalog.candidates(state.shelf)
        exact_hits = 0
        token_hits = 0
        rare = [
            token for token in tokens(value)
            if len(token) > 2
            and catalog._df.get(token, 0) <= config.LLM_GROUND_RARE_TOKEN_DF
        ]
        for asin in pool:
            text = catalog.ltext[asin]
            if needle in text:
                exact_hits += 1
                if exact_hits >= 3:
                    break
            elif phrase_tokens and all(stem in text for stem in phrase_tokens):
                token_hits += 1
            elif rare and any(f" {token} " in f" {text} " for token in rare):
                # A rare, discriminative word ("thermolite") is evidence even
                # when the rest of the phrase is paraphrased.
                token_hits += 1
        if exact_hits and "lexical" in allowed:
            return value, "lexical", config.LLM_GROUND_LEXICAL_WEIGHT, {
                "hits_sampled": exact_hits,
            }
        if token_hits and "lexical_tokens" in allowed:
            # Every content word occurs on at least one plausible product,
            # just not as one contiguous phrase: weak but real evidence that
            # the ranker's token-fraction scoring can use.
            return value, "lexical_tokens", config.LLM_GROUND_TOKEN_WEIGHT, {
                "hits": token_hits,
            }
    return None, "no_catalog_support", 0.0, {"from": phrase}


def ground_message(message: str, state, catalog, client, usage, dense=None) -> dict:
    """Ground one off-protocol message with the model alone.

    Mutates ``state``; returns a trace. This is the model-only reader kept
    for the attribution baselines; the shipped ``ground`` mode runs
    :func:`ground_message_cascade`, which lets the catalog read the message
    first and consults the model only when that is not enough.
    """
    trace: dict = {
        "status": "skipped",
        "grounder": "llm",
        "message": message,
        "accepted": [],
        "rejected": [],
        "removed": [],
    }
    message = _read_language(message, state, client, usage, trace)
    if trace.get("language", {}).get("english"):
        trace["english"] = message
    if not _llm_extract(message, state, catalog, client, usage, trace, dense):
        state.grounding_trace.append(trace)
        return trace
    return _finish(trace, state, trace["intent"])


def _llm_extract(message: str, state, catalog, client, usage, trace: dict, dense=None) -> bool:
    """Let the model propose a reading and the catalog verify it.

    Writes the proposal, the verification outcome and the model cost into
    ``trace`` and applies every admitted piece of evidence to ``state``.
    Returns False when the model produced nothing usable (endpoint down,
    circuit open, reply not JSON even after repair); the trace then carries
    the failure status and ``state`` is untouched.
    """
    vocabulary = _catalog_vocabulary(catalog, state)
    reply = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(message, state, vocabulary)},
        ],
        max_tokens=config.LLM_GROUND_MAX_TOKENS,
        temperature=0.0,
    )
    trace["vocabulary_hints"] = len(vocabulary)
    usage.absorb(reply)
    trace["tokens"] = {
        "prompt": reply.prompt_tokens,
        "completion": reply.completion_tokens,
    }
    trace["latency_ms"] = round(reply.latency_ms, 1)
    trace["cached"] = reply.cached
    if not reply.ok:
        trace["status"] = "llm_unavailable"
        trace["error"] = reply.error
        return False
    data = reply.json()
    if data is None:
        trace["status"] = "unparseable_reply"
        trace["raw"] = reply.text[:400]
        return False

    intent = _as_str(data.get("intent")) or "other"
    intent = intent.lower()
    if intent not in INTENTS:
        intent = "other"
    trace["intent"] = intent
    trace["proposal"] = {
        "category": _as_str(data.get("category")),
        "material": _as_str(data.get("material")),
        "color": _as_str(data.get("color")),
        "budget": data.get("budget"),
        "budget_operator": data.get("budget_operator"),
        "features": _as_list(data.get("features"), config.LLM_GROUND_MAX_FEATURES),
        "dropped": _as_list(data.get("dropped"), 4),
    }

    # --- dialogue acts that change memory without adding evidence ---------
    if intent == "override":
        state.override_seen = True
        state.last_reply_count = None
        state.clear_recommendation_history()
        state.decay_provisional(config.OVERRIDE_DECAY)
        trace["removed"] = _demote_dropped(state, trace["proposal"]["dropped"])
    elif intent == "no_preference":
        _apply_no_preference(state, trace)

    # --- shelf / candidate pool --------------------------------------------
    category = trace["proposal"]["category"]
    if category:
        _place_category(catalog, state, category, intent, trace, dense)
    if state.scenario is None and intent in ("open", "add"):
        state.scenario = "buying" if (
            trace["proposal"]["material"] or trace["proposal"]["features"]
        ) else "browsing_or_boundary"

    # --- typed evidence: always verifiable against catalog fields ----------
    def accept(value: str, tier: str, weight: float, detail: dict | None = None) -> None:
        before = len(state.constraints)
        state.add(value, weight=weight)
        if len(state.constraints) > before:
            trace["accepted"].append({
                "constraint": value,
                "tier": tier,
                "confidence": weight,
                **(detail or {}),
            })
        elif trace.get("department_switch"):
            # "I want a wallet instead, still leather": a preference carried
            # into the new department at reduced confidence and restated in
            # the same breath is back at full confidence.
            state.set_weight(value, weight)

    features_accepted: list[str] = []
    for phrase in trace["proposal"]["features"]:
        value, tier, weight, detail = _verify_feature(catalog, phrase, state, dense)
        if value is None:
            trace["rejected"].append({"phrase": phrase, "reason": tier, **detail})
            continue
        _expand_value(catalog, state, value, norm(phrase), dense, detail)
        accept(value, tier, weight, detail)
        features_accepted.append(value)

    def as_feature(phrase: str) -> None:
        # A material or colour outside the typed vocabulary ("alloy", "navy")
        # is still a catalog phrase; verify it like any other feature.
        value, tier, weight, detail = _verify_feature(catalog, phrase, state, dense)
        if value is None:
            trace["rejected"].append({"phrase": phrase, "reason": tier, **detail})
        else:
            _expand_value(catalog, state, value, norm(phrase), dense, detail)
            accept(value, tier, weight, detail)
            features_accepted.append(value)

    material = trace["proposal"]["material"]
    if material:
        material_norm = norm(material)
        if material_norm in MATERIALS:
            if any(material_norm in tokens(value) for value in features_accepted):
                trace["rejected"].append({
                    "phrase": material,
                    "reason": "material_already_inside_feature",
                })
            else:
                accept(material_norm, "typed_material", config.LLM_GROUND_VERBATIM_WEIGHT)
        else:
            as_feature(material)

    color = trace["proposal"]["color"]
    if color:
        color_norm = norm(color)
        if color_norm in COLORS:
            accept(f"color: {color_norm}", "typed_color", config.LLM_GROUND_VERBATIM_WEIGHT)
        else:
            as_feature(color)

    budget = _budget(trace["proposal"]["budget"])
    if budget is not None:
        operator = _budget_operator(trace["proposal"]["budget_operator"], message)
        if operator == "around" and state.pool_before_budget is not None:
            # Relaxing to a target price lifts the earlier hard bound.
            state.candidate_pool = state.pool_before_budget
            state.pool_before_budget = None
            state.pool_signature_counts = None
            state.budget_bound = None
            trace["budget_bound"] = {"operator": "around", "bound": budget, "applied": False, "reason": "previous_bound_lifted"}
        elif operator == "around":
            state.budget_bound = None
        # An earlier budget statement is superseded by the new one.
        for existing in list(state.constraints):
            if existing.startswith("budget around $") and existing != _format_budget(budget):
                state.remove(existing)
        if operator in ("under", "over") and config.LLM_GROUND_HARD_BUDGET:
            # An explicit ceiling or floor filters the grounded pool; the
            # proximity term is still recorded so a near-bound product wins
            # a tie among those that satisfy it.
            trace["budget_bound"] = _apply_budget_bound(state, catalog, budget, operator)
            state.grounded = True
        accept(
            _format_budget(budget),
            f"typed_budget_{operator}",
            config.LLM_GROUND_VERBATIM_WEIGHT,
        )

    return True


def _finish(trace: dict, state, intent: str) -> dict:
    """Shared tail of every grounder: retire unanswered questions, flag state."""
    # A reply to a specific question that carries no new evidence means the
    # shopper had nothing to say about that attribute: retire the question,
    # exactly as the protocol's "no additional preference" reply would.
    asked = state.asked[-1] if state.asked else None
    switched = bool(trace.get("department_switch"))
    if (
        asked
        and intent != "open"
        and not trace["accepted"]
        and not trace["removed"]
        and not switched
        and "exhausted" not in trace
    ):
        if asked == "other":
            state.information_complete = True
        else:
            state.exhausted.add(asked)
            state.last_reply_count = 0
        trace["exhausted"] = asked

    if trace["accepted"] or trace["removed"] or switched:
        # A human does not disclose attributes in the simulator's canonical
        # slot order, so positional signature evidence is no longer reliable,
        # and the protocol's four-constraint bound no longer applies.
        state.signature_positions_reliable = False
        state.grounded = True
        if len(state.constraints) < 4 or trace["accepted"] or switched:
            state.information_complete = False
    trace["status"] = "grounded"
    trace["constraints_after"] = list(state.constraints)
    message = trace.get("message")
    if message and intent in ("open", "add", "override", "other") and not trace.get("keeps_slate"):
        if intent == "override":
            state.grounded_messages = []
        state.grounded_messages.append(message)
    state.grounding_trace.append(trace)
    return trace


# ----------------------------------------------------------------------------
# Attribution baseline: template-independent exact matching, no model at all.
# ----------------------------------------------------------------------------

_NO_PREF_WORDS = (
    "don't care", "dont care", "no preference", "doesn't matter", "doesnt matter",
    "any is fine", "anything is fine", "not fussy", "whatever works", "up to you",
    "nothing else", "no more preferences", "not really care", "don't mind", "dont mind",
    "no additional", "nothing in particular",
)
_OVERRIDE_WORDS = (
    "forget", "instead", "scratch that", "ignore", "never mind", "nevermind",
    "changed my mind", "actually i", "actually,", "rather than", "no longer",
)
_REJECT_WORDS = (
    "not those", "not it", "aren't it", "arent it", "none of these", "not what i",
    "not quite", "miss the mark", "not right", "try again", "something else",
    "not these",
)
_BUDGET_RE = re.compile(
    r"(?:\$\s*([0-9]+(?:\.[0-9]+)?)|([0-9]+(?:\.[0-9]+)?)\s*(?:dollars|bucks|usd|\$))", re.I
)


def _lexical_intent(message: str) -> str:
    lowered = " " + message.lower() + " "
    if any(word in lowered for word in _OVERRIDE_WORDS):
        return "override"
    if any(word in lowered for word in _NO_PREF_WORDS):
        return "no_preference"
    if any(word in lowered for word in _REJECT_WORDS):
        return "reject"
    return "add"


def _ngram_matches(message_norm: str, values, minimum_chars: int = 4, loose_of: dict | None = None,
                   first_token_index: dict | None = None) -> list[str]:
    """Catalog strings that occur verbatim in the message, longest first.

    Overlapping matches are resolved toward the longer string, so ``"buckle
    closure"`` wins over ``"buckle"``. Word boundaries are enforced on both
    sides so ``"red"`` does not fire inside ``"covered"``.
    """
    # Punctuation-insensitive on both sides: "buckle closure," in the message
    # still matches the catalog string "buckle closure".
    padded = " " + loose(message_norm) + " "
    found: list[tuple[int, int, str]] = []
    if first_token_index is not None:
        # A value can only occur verbatim if its first word is in the message,
        # so consult the index instead of scanning every catalog string.
        scoped = values if isinstance(values, (set, dict, frozenset)) else set(values)
        seen: set[str] = set()
        narrowed: list[str] = []
        for token in dict.fromkeys(tokens(padded)):
            for value in first_token_index.get(token, ()):
                if value in scoped and value not in seen:
                    seen.add(value)
                    narrowed.append(value)
        values = narrowed
    for value in values:
        needle = loose_of[value] if loose_of is not None and value in loose_of else loose(value)
        if len(needle) < minimum_chars or " " + needle + " " not in padded:
            continue
        start = padded.index(" " + needle + " ")
        found.append((start, start + len(needle) + 2, value))
    found.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    taken: list[tuple[int, int]] = []
    chosen: list[str] = []
    for start, end, value in found:
        if any(start < b and end > a for a, b in taken):
            continue
        taken.append((start, end))
        chosen.append(value)
    return chosen


# Deliberately narrow: an ordinal or demonstrative pointer at a shown item, or
# a question about a shown item's variants. "I love that the strap is
# adjustable" is a preference, not a pointer, and must not match.
_POSITIVE_REFERENCE = re.compile(
    r"\b(?:the|that|this)\s+(?:first|second|third|1st|2nd|3rd|last|top)\s+(?:one|pair|item|product)\b"
    r"|\b(?:like|love|prefer|take|want|keep)\s+(?:that|this)\s+one\b"
    r"|\b(?:these|those)\s+(?:look|seem)\s+(?:good|great|nice|perfect|right|promising)\b"
    r"|\bdo(?:es)?\s+(?:it|they|that|this)\s+come\s+in\b"
    r"|\b(?:is|are)\s+(?:it|they|that|this)\s+available\s+in\b"
    r"|\bcan\s+i\s+(?:get|have)\s+(?:it|that|this|them)\s+in\b"
    r"|\bsame\s+(?:one|thing|style)\s+(?:but|in)\b",
    re.I,
)


def refers_to_shown_products(message: str) -> bool:
    """True when the shopper is building on what was just shown.

    The organizer's simulator continues a session only when the whole slate
    missed, so the controller treats every shown product as a proven miss.
    A human who says "the first one looks good, does it come in blue?" is
    doing the opposite: the slate is the reference point for the next
    question. Such a message keeps the previous slate eligible; an explicit
    rejection in the same message still wins.
    """
    if _lexical_intent(message) in ("reject", "override"):
        return False
    return bool(_POSITIVE_REFERENCE.search(message))


def ground_message_lexical(message: str, state, catalog, dense=None) -> dict:
    """Read a free-form message with exact catalog matching and keyword rules.

    This is the no-model control for the grounding layer: the same state
    updates and the same ranker, but evidence can only enter as a verbatim
    catalog string (signature values, closed material and colour vocabularies,
    a dollar amount) found inside the message. Nothing is paraphrased.
    """
    trace: dict = {
        "status": "skipped",
        "grounder": "lexical",
        "message": message,
        "accepted": [],
        "rejected": [],
        "removed": [],
        "tokens": {"prompt": 0, "completion": 0},
        "latency_ms": 0.0,
    }
    _lexical_extract(message, state, catalog, trace, dense=dense)
    return _finish(trace, state, trace["intent"])


def _pool_from_message(message: str, state, catalog, trace: dict, dense=None) -> None:
    """Union of shelves sharing content tokens with the whole sentence."""
    shelves, pool = catalog.shelf_pool(message)
    shelves, pool = _dense_shelves(catalog, message, shelves, pool, dense, trace)
    if pool:
        state.pool_shelves = shelves
        state.candidate_pool = pool
        trace["pool_shelves"] = shelves[:12]
        trace["pool_size"] = len(pool)
        _apply_deferred_budget(state, catalog, trace)


def _blanket_decay(state, trace: dict) -> None:
    """Without a list of what was cancelled, keep every earlier constraint at
    reduced weight, as the protocol's own override handling does for the
    withdrawn opener."""
    for existing in list(state.constraints):
        state.set_weight(existing, config.OVERRIDE_DECAY)
        trace["removed"].append({"constraint": existing, "action": "decayed",
                                 "confidence": config.OVERRIDE_DECAY})


def _discriminative(catalog, value: str) -> bool:
    """A catalog string that can carry a session on its own.

    Multi-word values ("buckle closure", "stainless steel strap") are; a
    single common word that happens to be a signature value ("shoes",
    "metal", "rubber") is not: it names the department or half of a feature,
    and admitting it as read evidence would hide the rest of the sentence.
    """
    words = [token for token in tokens(value) if len(token) > 2]
    if len(words) >= 2:
        return True
    if not words:
        return False
    return catalog._df.get(words[0], 0) <= config.LLM_GROUND_RARE_TOKEN_DF


def _apply_no_preference(state, trace: dict) -> None:
    """The shopper has nothing to say about the attribute just asked."""
    attribute = state.asked[-1] if state.asked else None
    if attribute and attribute != "other":
        state.exhausted.add(attribute)
        state.last_reply_count = 0
        trace["exhausted"] = attribute
    elif attribute == "other":
        state.information_complete = True
        trace["exhausted"] = "other"


def _apply_dialogue_act(intent: str, state, trace: dict, *, blanket_decay: bool) -> None:
    """Memory changes of a keyword-detected dialogue act, no model involved."""
    if intent == "override":
        state.override_seen = True
        state.last_reply_count = None
        state.clear_recommendation_history()
        state.decay_provisional(config.OVERRIDE_DECAY)
        if blanket_decay:
            _blanket_decay(state, trace)
    elif intent == "no_preference":
        _apply_no_preference(state, trace)


def _lexical_extract(message: str, state, catalog, trace: dict, *,
                     blanket_decay: bool = True, pool_from_message: bool = True,
                     discriminative_only: bool = False,
                     apply_dialogue_acts: bool = True, dense=None) -> None:
    """Stage one of reading a free-form message: exact catalog strings only.

    ``pool_from_message=False`` defers the token-overlap shelf guess so a
    model, if consulted, can name the category more precisely than the whole
    sentence does; the caller falls back to the sentence afterwards.
    ``discriminative_only`` admits a matched catalog string only when it is
    discriminative (see :func:`_discriminative`); weak single-word matches
    are recorded in the trace and left for the model to read in context.
    ``apply_dialogue_acts=False`` only *detects* a cancellation or a "no
    preference" reply and leaves the memory change to the caller, which may
    prefer the model's reading of the sentence.
    """
    message_norm = norm(message)
    intent = _lexical_intent(message)
    trace["intent"] = intent
    if refers_to_shown_products(message):
        trace["keeps_slate"] = True
        # "the first one looks good" refers to the slate; it is not a
        # product attribute even where the catalog happens to say the words.
        message_norm = norm(_POSITIVE_REFERENCE.sub(" ", message))

    if apply_dialogue_acts:
        _apply_dialogue_act(intent, state, trace, blanket_decay=blanket_decay)

    if state.shelf is None and not state.candidate_pool:
        shelf = catalog.match_shelf(message)
        if shelf is not None:
            state.shelf = shelf
            trace["shelf"] = shelf
            _apply_deferred_budget(state, catalog, trace)
        elif pool_from_message:
            _pool_from_message(message, state, catalog, trace, dense)
    if state.scenario is None:
        state.scenario = "buying" if intent == "add" else "browsing_or_boundary"

    def accept(value: str, tier: str, weight: float, detail: dict | None = None) -> None:
        before = len(state.constraints)
        state.add(value, weight=weight)
        if len(state.constraints) > before:
            trace["accepted"].append({"constraint": value, "tier": tier,
                                      "confidence": weight, **(detail or {})})

    # Verbatim signature strings, scoped like the model path: the pool when
    # there is one, otherwise the shelf, otherwise the whole catalog.
    pool_counts = _pool_support(catalog, state)
    if pool_counts is not None:
        values = pool_counts
    elif state.shelf is not None:
        values = catalog.signature_value_count_by_shelf.get(state.shelf, {})
    else:
        values = catalog.signature_value_count
    generic = {"imported", "is discontinued by manufacturer: no"}
    for value in _ngram_matches(
        message_norm, values, loose_of=catalog.signature_values_loose(),
        first_token_index=catalog.signature_values_by_first_token(),
    ):
        if value in generic or value in MATERIALS or value.startswith("color:") or value.startswith("budget around"):
            continue
        if discriminative_only and not _discriminative(catalog, value):
            trace.setdefault("weak_matches", []).append(value)
            continue
        detail = {"support": _support(catalog, state, value)}
        _expand_value(catalog, state, value, value, dense, detail)
        accept(value, "ngram_signature", config.LLM_GROUND_VERBATIM_WEIGHT, detail)

    material = MATERIAL_RE.search(message_norm)
    if material and not any(material.group(1).lower() in tokens(a["constraint"]) for a in trace["accepted"]):
        accept(material.group(1).lower(), "typed_material", config.LLM_GROUND_VERBATIM_WEIGHT)
    color = COLOR_RE.search(message_norm)
    if color:
        accept(f"color: {color.group(1).lower()}", "typed_color", config.LLM_GROUND_VERBATIM_WEIGHT)
    money = _BUDGET_RE.search(message)
    if money:
        amount = float(money.group(1) or money.group(2))
        if amount > 0:
            operator = _budget_operator(None, message)
            for existing in list(state.constraints):
                if existing.startswith("budget around $") and existing != _format_budget(amount):
                    state.remove(existing)
            if operator in ("under", "over") and config.LLM_GROUND_HARD_BUDGET:
                trace["budget_bound"] = _apply_budget_bound(state, catalog, amount, operator)
                state.grounded = True
            accept(_format_budget(amount), f"typed_budget_{operator}", config.LLM_GROUND_VERBATIM_WEIGHT)


# ----------------------------------------------------------------------------
# The shipped reader: the catalog reads first, the model is consulted on demand.
# ----------------------------------------------------------------------------

# Words of ordinary shopper chat that carry no product attribute. Anything
# else the shopper says that is also attribute vocabulary in the catalog is
# a word the catalog stage should have explained.
_CHAT_WORDS = frozenset("""
the and but for with without that this these those they them their there then than
its it's i'm i've i'd i'll we're you're don't doesn't didn't isn't aren't wasn't
can't won't wouldn't couldn't shouldn't not yes yeah nah okay ok sure thanks thank
please just really quite pretty very much more most less some any all both each
either neither one ones something anything nothing everything else other another
need needs needed want wants wanted looking look looks like likes liked love loves
prefer prefers preferred hoping hope wish get got give show find found
have has had having make makes made sure keep keeps let lets
should would could can may might must will shall
about into onto over under around through from
what which who whom whose when where why how
here now then still also too again ever never always often sometimes maybe probably
good great nice fine best better perfect ideal decent proper suitable
pair pairs set sets thing things item items option options stuff kind sort type
buy buying bought purchase shopping shop order
mind care matter bother fussy specific particular preference preferences
right yet again instead rather actually honestly basically definitely
you your yours our ours mine his her hers
was were been being are is am
does did doing done
dollar dollars bucks usd price priced cost costs budget cheap cheaper expensive
max maximum min minimum least most under over around below above between within
only even ever
""".split())


def residual_attribute_words(message: str, trace: dict, catalog) -> list[str]:
    """Attribute vocabulary in the message that stage one did not consume.

    A word counts when it appears in the catalog's own attribute strings
    (the intent-signature vocabulary) and is neither ordinary chat nor part
    of a string stage one matched, a typed value it read, or a weak match it
    noted. Such words are the part of the sentence the catalog could not
    read verbatim and the model is asked to translate.
    """
    consumed: set[str] = set()
    for item in trace.get("accepted", []):
        consumed.update(tokens(str(item.get("constraint", ""))))
    for value in trace.get("weak_matches", []) or []:
        consumed.update(tokens(value))
    if trace.get("shelf"):
        consumed.update(tokens(trace["shelf"]))
    # Department words ("belt", "necklaces") choose the shelf pool; they are
    # not attribute evidence and need no translation.
    consumed.update(catalog.shelf_words())
    vocabulary = catalog.signature_values_by_first_token()
    attribute_words = catalog.signature_vocabulary()
    residual: list[str] = []
    for token in dict.fromkeys(tokens(message)):
        if len(token) <= 2 or token.isdigit() or token in _CHAT_WORDS or token in consumed:
            continue
        stem = _stem(token)
        if token in attribute_words or stem in attribute_words or token in vocabulary:
            residual.append(token)
    return residual


def needs_model(trace: dict, message: str = "", catalog=None, state=None) -> tuple[bool, str]:
    """Decide whether stage one explained the message well enough.

    A cancellation always goes to the model, because only a reader of the
    sentence can say *what* was cancelled. A recognised dialogue act with
    nothing else ("no preference", "not those") is fully explained. Otherwise
    the catalog has read the sentence when it found at least one
    discriminative catalog string in it, the session is placed in a
    department (a shelf name occurred in a message, or a pool exists) and no
    attribute vocabulary is left over; "a zipper closure and hand washing
    only" keeps "hand washing" for the model even though "zipper closure"
    was matched verbatim, and "running shoes for the gym" goes to the model
    to name the department rather than to a token overlap of the sentence
    with shelf names.
    """
    intent = trace.get("intent")
    if intent == "override":
        return True, "cancellation_needs_dropped_list"
    if intent in ("no_preference", "reject"):
        return False, "dialogue_act_recognised"
    if not any(item.get("tier") == "ngram_signature" for item in trace["accepted"]):
        return True, "no_feature_evidence"
    if (
        state is not None
        and catalog is not None
        and config.LLM_GROUND_ESCALATE_UNPLACED
        and state.shelf is None
        and not state.candidate_pool
    ):
        best, pool_size = catalog.department_placement(message)
        if best == 0 or (best == 1 and pool_size > config.LLM_GROUND_CONFIDENT_POOL):
            trace["placement"] = {"shared_words": best, "pool_size": pool_size}
            return True, "department_unplaced"
    if catalog is not None and message:
        residual = residual_attribute_words(message, trace, catalog)
        if residual:
            trace["residual"] = residual
            return True, "attribute_words_left_unread"
    return False, "catalog_string_found"


def _translate(message: str, language: str, client, usage, trace: dict) -> str | None:
    """English for a message the detector read as another language.

    The catalog is English, so the readers are. The translation is one
    short model call recorded in the trace; a reply that is empty, still not
    English, or a failure leaves the original message in place, and the
    cascade then reads it as before (the model stage understands the
    language even when the catalog stage cannot).
    """
    record: dict = {
        "detected": language,
        "name": language_name(language),
        "original": message,
    }
    reply = client.chat(
        [
            {"role": "system", "content": TRANSLATE_PROMPT},
            {"role": "user", "content": message.strip()},
        ],
        max_tokens=config.LLM_TRANSLATE_MAX_TOKENS,
        temperature=0.0,
    )
    usage.absorb(reply)
    record["tokens"] = {"prompt": reply.prompt_tokens, "completion": reply.completion_tokens}
    record["latency_ms"] = round(reply.latency_ms, 1)
    trace["language"] = record
    if not reply.ok:
        record["status"] = "translation_failed"
        record["error"] = reply.error
        return None
    text = " ".join(reply.text.strip().strip('"').strip("\'").split())
    if (
        not text
        or len(text) > 400
        or text.startswith(("{", "[", "```"))
        or not re.search(r"[A-Za-z]", text)
        or detect(text) is not None
    ):
        record["status"] = "translation_rejected"
        record["reply"] = text[:200]
        return None
    record["status"] = "translated"
    record["english"] = text
    return text


def _read_language(message: str, state, client, usage, trace: dict) -> str:
    """Detect the shopper's language; return the text the readers should see."""
    language = detect(message) if config.LLM_TRANSLATE else None
    if language is None:
        return message
    state.language = language
    state.language_sample = message.strip()[:120]
    if client is None:
        trace["language"] = {"detected": language, "name": language_name(language),
                             "original": message, "status": "no_model"}
        return message
    return _translate(message, language, client, usage, trace) or message


def _reclassify_category_phrase(state, trace: dict) -> None:
    """A stage-one string the model read as the category is not an attribute.

    "running shoes" is a feature string of exactly one product and also the
    department the shopper named. Once the model has called it the category,
    keeping it as a verbatim constraint would pin that one product.
    """
    category = (trace.get("proposal") or {}).get("category")
    if not category:
        return
    category_tokens = set(content_tokens(category))
    if not category_tokens:
        return
    stage1 = set(trace.get("stage1", {}).get("accepted", []))
    kept: list[dict] = []
    for item in trace["accepted"]:
        value = item.get("constraint", "")
        value_tokens = set(content_tokens(value))
        if (
            item.get("tier") == "ngram_signature"
            and value in stage1
            and value_tokens
            and value_tokens <= category_tokens
        ):
            state.remove(value)
            trace.setdefault("reclassified", []).append(
                {"constraint": value, "as": "category", "category": category}
            )
            continue
        kept.append(item)
    trace["accepted"] = kept


def ground_message_cascade(message: str, state, catalog, client, usage, dense=None) -> dict:
    """Ground one off-protocol message: catalog first, model only on demand.

    Stage one is :func:`_lexical_extract`, which admits only verbatim catalog
    strings and costs no tokens. Stage two, :func:`_llm_extract`, runs when
    stage one could not explain the message (see :func:`needs_model`). A
    failed model call (outage, open circuit, unparseable reply) leaves stage
    one's reading in place, so the session never loops on the same question.
    """
    trace: dict = {
        "status": "skipped",
        "grounder": "cascade",
        "message": message,
        "accepted": [],
        "rejected": [],
        "removed": [],
        "tokens": {"prompt": 0, "completion": 0},
        "latency_ms": 0.0,
    }
    message = _read_language(message, state, client, usage, trace)
    if trace.get("language", {}).get("english"):
        trace["english"] = message
    _lexical_extract(
        message, state, catalog, trace, blanket_decay=False, pool_from_message=False,
        discriminative_only=True, apply_dialogue_acts=False, dense=dense,
    )
    escalate, reason = needs_model(trace, message, catalog, state)
    stage1_intent = trace["intent"]
    trace["stage1"] = {
        "intent": stage1_intent,
        "accepted": [item["constraint"] for item in trace["accepted"]],
        "escalate": escalate,
        "reason": reason,
    }
    # A keyword-detected cancellation ("instead", "actually, ...") is only a
    # proposal: the words also occur in plain rejections and additions, and
    # clearing the refutation history on a false alarm re-shows products the
    # shopper already turned down. The model's reading decides; without a
    # model the keyword reading stands.
    if escalate and config.LLM_GROUND_CASCADE:
        if _llm_extract(message, state, catalog, client, usage, trace, dense):
            trace["model_status"] = "grounded"
            _reclassify_category_phrase(state, trace)
        else:
            trace["model_status"] = trace["status"]
            trace["intent"] = stage1_intent
            _apply_dialogue_act(stage1_intent, state, trace, blanket_decay=True)
    else:
        if escalate:
            trace["model_status"] = "disabled"
        _apply_dialogue_act(stage1_intent, state, trace, blanket_decay=True)
    if state.shelf is None and not state.candidate_pool:
        # Neither an exact shelf name nor a model category: land the shopper
        # in the shelves that share words with the sentence itself.
        _pool_from_message(message, state, catalog, trace, dense)
    return _finish(trace, state, trace["intent"])


def render_message(certificate: dict, top_title: str | None, client, usage,
                   language: str | None = None, sample: str | None = None) -> str | None:
    """Phrase the customer-facing message from the decision certificate.

    ``language`` (a ``src.language`` code) and ``sample`` (what the shopper
    wrote) make the reply come back in the shopper's language; both are
    absent on English sessions, whose prompt is unchanged.
    """
    payload = {
        "action": certificate.get("action"),
        "requirements_matched": certificate.get("constraints", [])[:4],
        "top_product": (top_title or "")[:120] or None,
        "shown": len(certificate.get("recommendations", []) or []),
        "question_attribute": certificate.get("question"),
    }
    if language:
        payload["reply_language"] = language_name(language)
        payload["shopper_wrote"] = (sample or "")[:120] or None
    reply = client.chat(
        [
            {"role": "system", "content": RENDER_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        max_tokens=config.LLM_RENDER_MAX_TOKENS,
        temperature=0.0,
    )
    usage.absorb(reply)
    if not reply.ok:
        return None
    text = " ".join(reply.text.strip().strip('"').split())
    if not text or len(text) > 320:
        return None
    return text
