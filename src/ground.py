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
import os
import re

from . import config
from .shelf import COLORS, MATERIALS, loose, norm, tokens

INTENTS = ("open", "add", "override", "no_preference", "reject", "other")

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
- "budget": a number in US dollars if a price limit is stated, else null.
- "features": a list of at most 3 short attribute phrases the shopper wants, in catalog-like wording (for example "buckle closure", "zipper closure", "machine wash", "wide fit", "waterproof"). Never repeat the category, material, color or budget here. Do not include vague words such as "nice", "good", "cheap".
- "dropped": earlier preferences the shopper explicitly cancels, else [].
Examples:
"hey i need some comfy leather loafers for work, brown if possible, under 60 bucks" -> {"intent":"open","category":"loafers","material":"leather","color":"brown","budget":60,"features":["comfortable"],"dropped":[]}
"it should have a zipper and be machine washable" -> {"intent":"add","category":null,"material":null,"color":null,"budget":null,"features":["zipper closure","machine wash"],"dropped":[]}
"actually forget the leather thing, i just want it to be black" -> {"intent":"override","category":null,"material":null,"color":"black","budget":null,"features":[],"dropped":["leather"]}
"nah, don't really care about that. just make it sturdy" -> {"intent":"no_preference","category":null,"material":null,"color":null,"budget":null,"features":["sturdy"],"dropped":[]}
"those aren't it, ask me something else" -> {"intent":"reject","category":null,"material":null,"color":null,"budget":null,"features":[],"dropped":[]}
Output only the JSON object."""

RENDER_PROMPT = """You are the voice of a shopping assistant. Write one or two short, warm sentences (at most 40 words) to the shopper.
Rules: mention at most two of the requirements you are matching; describe the top product only with the words given; never invent facts, prices or colours.
If "question_attribute" is set, end by asking about exactly that attribute in plain words (material, color, size, style, feature, use_case; "other" means "anything else that matters to you").
If it is null, do not ask a question. Output plain text only."""

_STEM_MIN = 4


def _stem(token: str) -> str:
    if len(token) >= _STEM_MIN and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) >= _STEM_MIN and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    return [_stem(token) for token in tokens(text) if len(token) > 2]


# A model asked for a nullable field routinely answers with the *string*
# "null" (or "none"/"n/a") instead of JSON null. Passing that through sends a
# phrase called "null" to the verifier, which rejects it and leaves a phantom
# entry in the decision certificate the demo shows.
_NULLISH = {"null", "none", "nil", "n/a", "na", "unknown", "undefined", "-"}


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _NULLISH:
        return None
    return text


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


def _verify_feature(catalog, phrase: str, state) -> tuple[str | None, str, float, dict]:
    """Return (constraint, tier, weight, detail) or (None, reason, 0, detail).

    Verification is scoped to the products the session can still recommend:
    the canonical shelf when the wording named one, otherwise the grounded
    candidate pool. A paraphrase must therefore map onto evidence that exists
    among plausible products, not anywhere in the catalog.
    """
    value = norm(phrase)
    if not value:
        return None, "empty", 0.0, {}
    support = _support(catalog, state, value)
    if support > 0:
        return value, "signature_verbatim", config.LLM_GROUND_VERBATIM_WEIGHT, {
            "support": support,
        }
    phrase_tokens = content_tokens(value)
    if phrase_tokens:
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
    needle = loose(value)
    if needle:
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
        if exact_hits:
            return value, "lexical", config.LLM_GROUND_LEXICAL_WEIGHT, {
                "hits_sampled": exact_hits,
            }
        if token_hits:
            # Every content word occurs on at least one plausible product,
            # just not as one contiguous phrase: weak but real evidence that
            # the ranker's token-fraction scoring can use.
            return value, "lexical_tokens", config.LLM_GROUND_TOKEN_WEIGHT, {
                "hits": token_hits,
            }
    return None, "no_catalog_support", 0.0, {"from": phrase}


def verify_mode() -> str:
    """Which way the reading is produced. ARC_LLM_VERIFY overrides config."""
    mode = str(
        os.environ.get("ARC_LLM_VERIFY", config.LLM_GROUND_VERIFY) or "propose"
    ).strip().lower()
    return mode if mode in ("propose", "iterative") else "propose"


GROUND_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "verify_phrase",
            "description": (
                "Check one candidate phrase against the products the shopper "
                "can still be shown. Returns whether the catalog accepts it, "
                "the exact catalog wording to use, the evidence tier, and the "
                "confidence that tier carries. Never submit a phrase this "
                "rejects."
            ),
            "parameters": {
                "type": "object",
                "properties": {"phrase": {"type": "string"}},
                "required": ["phrase"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_known_values",
            "description": (
                "List common catalog phrases among the products still in "
                "play, to match the shopper's wording to the catalog's own "
                "vocabulary."
            ),
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_reading",
            "description": (
                "Submit the final reading of the shopper's message, with the "
                "same fields the JSON contract describes. Call exactly once "
                "to finish."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "category": {"type": ["string", "null"]},
                    "material": {"type": ["string", "null"]},
                    "color": {"type": ["string", "null"]},
                    "budget": {"type": ["number", "null"]},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "dropped": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["intent"],
            },
        },
    },
]

_ITERATIVE_NOTE = """
You may call verify_phrase before committing to any feature, material or
colour, and list_known_values when the shopper's wording has no obvious
catalog equivalent. verify_phrase runs the very check the catalog applies
afterwards, so use it to repair a phrase now rather than have it rejected
later: submit the exact "constraint" wording it hands back.

Not every accepted phrase is worth submitting. verify_phrase returns a
confidence, and a result marked "weak" will barely influence ranking -- when
you see one, look for the catalog's own word for the same thing through
list_known_values and verify that instead. Prefer one high-confidence phrase
over several weak ones.

Attribution comes first, though. verify_phrase tells you whether a phrase
exists in the catalog, never whether this shopper asked for it, and the
vocabulary list is what other products happen to say -- not a menu to pick
from. Only submit wording the shopper actually expressed in this message. If
they stated no preference, or only rejected what they were shown, the correct
features list is empty. Submitting a plausible catalog phrase they did not ask
for is worse than submitting nothing.

Finish by calling submit_reading exactly once.
"""


def _run_ground_tool(name: str, args: dict, catalog, state) -> dict:
    """Expose the catalog's own verification to the model, unchanged.

    verify_phrase *is* _verify_feature -- the same function that judges the
    proposal afterwards -- so the loop can never admit anything the
    propose-once path would have rejected. The only thing that moves is
    when the verification happens: before the model commits, or after.
    """
    if name == "verify_phrase":
        value, tier, weight, detail = _verify_feature(
            catalog, str(args.get("phrase", "")), state
        )
        if value is None:
            return {"accepted": False, "reason": tier, **detail}
        result = {
            "accepted": True,
            "constraint": value,
            "tier": tier,
            "confidence": weight,
            **detail,
        }
        if weight < config.LLM_GROUND_MAPPED_WEIGHT:
            # Accepting is not the same as being worth submitting. Without
            # this the model stops at the first phrase that returns true and
            # submits the shopper's own wording at 0.5 confidence, when a
            # verbatim catalog value usually exists for the same meaning.
            result["weak"] = True
            result["advice"] = (
                "Low-confidence match: this will carry little weight in "
                "ranking. Call list_known_values and verify a catalog phrase "
                "that means the same thing; submit this one only if nothing "
                "stronger fits."
            )
        return result
    if name == "list_known_values":
        try:
            limit = int(args.get("limit") or config.LLM_GROUND_VOCAB_HINTS)
        except (TypeError, ValueError):
            limit = config.LLM_GROUND_VOCAB_HINTS
        return {"values": _catalog_vocabulary(catalog, state)[: max(1, limit)]}
    return {"error": "unknown tool " + str(name)}


def _absorb(reply, usage, trace) -> None:
    """Accumulate across however many calls the reading took."""
    tokens = trace.setdefault("tokens", {"prompt": 0, "completion": 0})
    usage.absorb(reply)
    tokens["prompt"] += reply.prompt_tokens
    tokens["completion"] += reply.completion_tokens
    trace["latency_ms"] = round(
        float(trace.get("latency_ms") or 0.0) + reply.latency_ms, 1
    )
    # True only if every call behind this reading replayed from cache.
    trace["cached"] = bool(reply.cached) and trace.get("cached", True)


def _propose(message, state, catalog, client, usage, vocabulary, trace):
    """Obtain the model's structured reading. Returns (data, failure_status).

    Both modes return the same dict shape, so everything downstream --
    _verify_feature, the confidence tiers, the shelf pool, the budget, the
    exhaustion rule -- is identical whichever produced it.
    """
    if verify_mode() == "iterative":
        return _propose_iterative(
            message, state, catalog, client, usage, vocabulary, trace
        )
    return _propose_once(message, state, client, usage, vocabulary, trace)


def _propose_once(message, state, client, usage, vocabulary, trace):
    reply = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(message, state, vocabulary)},
        ],
        max_tokens=config.LLM_GROUND_MAX_TOKENS,
        temperature=0.0,
    )
    _absorb(reply, usage, trace)
    if not reply.ok:
        trace["error"] = reply.error
        return None, "llm_unavailable"
    data = reply.json()
    if data is None:
        trace["raw"] = reply.text[:400]
        return None, "unparseable_reply"
    return data, None


def _propose_iterative(message, state, catalog, client, usage, vocabulary, trace):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + _ITERATIVE_NOTE},
        {"role": "user", "content": _user_prompt(message, state, vocabulary)},
    ]
    verifications = 0
    rounds = max(1, int(config.LLM_GROUND_MAX_TOOL_CALLS))
    for index in range(rounds):
        if index == rounds - 1:
            # Last round: the model must commit now. Without this it can
            # spend the whole budget verifying and never submit, which ends
            # the turn with no reading at all.
            messages.append({
                "role": "user",
                "content": (
                    "No more verification. Call submit_reading now with what "
                    "you have, using the catalog wording you confirmed."
                ),
            })
        reply = client.chat(
            messages,
            max_tokens=config.LLM_GROUND_TOOL_MAX_TOKENS,
            temperature=0.0,
            tools=GROUND_TOOLS,
        )
        _absorb(reply, usage, trace)
        if not reply.ok:
            trace["error"] = reply.error
            return None, "llm_unavailable"
        tool_calls = list(reply.tool_calls or [])
        if not tool_calls:
            # The model wanted no tools and answered directly: that is the
            # propose-once contract, so accept it rather than burn the budget.
            data = reply.json()
            trace["tool_verifications"] = verifications
            if data is None:
                trace["raw"] = reply.text[:400]
                return None, "unparseable_reply"
            return data, None
        messages.append({
            "role": "assistant",
            "content": reply.text or None,
            "tool_calls": tool_calls,
        })
        submission = None
        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            if name == "submit_reading":
                submission = args
                result = {"received": True}
            else:
                verifications += 1
                result = _run_ground_tool(name, args, catalog, state)
            messages.append({
                "role": "tool",
                "tool_call_id": str(call.get("id") or ""),
                "content": json.dumps(result, ensure_ascii=False),
            })
        if submission is not None:
            trace["tool_verifications"] = verifications
            return submission, None
    # The loop never committed. Falling back to one plain call is strictly
    # better than returning nothing: iterative must degrade to propose, never
    # to an ungrounded turn.
    trace["tool_verifications"] = verifications
    trace["fell_back_to_propose"] = True
    return _propose_once(message, state, client, usage, vocabulary, trace)


def ground_message(message: str, state, catalog, client, usage) -> dict:
    """Ground one off-protocol message. Mutates ``state``; returns a trace."""
    trace: dict = {
        "status": "skipped",
        "message": message,
        "accepted": [],
        "rejected": [],
        "removed": [],
    }
    vocabulary = _catalog_vocabulary(catalog, state)
    trace["vocabulary_hints"] = len(vocabulary)
    trace["verify_mode"] = verify_mode()
    data, failure = _propose(
        message, state, catalog, client, usage, vocabulary, trace
    )
    if failure is not None:
        trace["status"] = failure
        state.grounding_trace.append(trace)
        return trace

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
        attribute = state.asked[-1] if state.asked else None
        if attribute and attribute != "other":
            state.exhausted.add(attribute)
            state.last_reply_count = 0
            trace["exhausted"] = attribute
        elif attribute == "other":
            state.information_complete = True
            trace["exhausted"] = "other"

    # --- shelf / candidate pool --------------------------------------------
    category = trace["proposal"]["category"]
    if category and state.shelf is None and not state.candidate_pool:
        shelf = catalog.match_shelf(category)
        if shelf is None:
            shelves, pool = catalog.shelf_pool(category)
            if pool:
                state.pool_shelves = shelves
                state.candidate_pool = pool
                trace["pool_shelves"] = shelves[:12]
                trace["pool_size"] = len(pool)
        else:
            state.shelf = shelf
            trace["shelf"] = shelf
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

    features_accepted: list[str] = []
    for phrase in trace["proposal"]["features"]:
        value, tier, weight, detail = _verify_feature(catalog, phrase, state)
        if value is None:
            trace["rejected"].append({"phrase": phrase, "reason": tier, **detail})
            continue
        accept(value, tier, weight, detail)
        features_accepted.append(value)

    def as_feature(phrase: str) -> None:
        # A material or colour outside the typed vocabulary ("alloy", "navy")
        # is still a catalog phrase; verify it like any other feature.
        value, tier, weight, detail = _verify_feature(catalog, phrase, state)
        if value is None:
            trace["rejected"].append({"phrase": phrase, "reason": tier, **detail})
        else:
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
        accept(_format_budget(budget), "typed_budget", config.LLM_GROUND_VERBATIM_WEIGHT)

    # A reply to a specific question that carries no new evidence means the
    # shopper had nothing to say about that attribute: retire the question,
    # exactly as the protocol's "no additional preference" reply would.
    asked = state.asked[-1] if state.asked else None
    if (
        asked
        and intent != "open"
        and not trace["accepted"]
        and not trace["removed"]
        and "exhausted" not in trace
    ):
        if asked == "other":
            state.information_complete = True
        else:
            state.exhausted.add(asked)
            state.last_reply_count = 0
        trace["exhausted"] = asked

    if trace["accepted"] or trace["removed"]:
        # A human does not disclose attributes in the simulator's canonical
        # slot order, so positional signature evidence is no longer reliable,
        # and the protocol's four-constraint bound no longer applies.
        state.signature_positions_reliable = False
        state.grounded = True
        if len(state.constraints) < 4 or trace["accepted"]:
            state.information_complete = False
    trace["status"] = "grounded"
    trace["constraints_after"] = list(state.constraints)
    state.grounding_trace.append(trace)
    return trace


def render_message(certificate: dict, top_title: str | None, client, usage) -> str | None:
    """Phrase the customer-facing message from the decision certificate."""
    payload = {
        "action": certificate.get("action"),
        "requirements_matched": certificate.get("constraints", [])[:4],
        "top_product": (top_title or "")[:120] or None,
        "shown": len(certificate.get("recommendations", []) or []),
        "question_attribute": certificate.get("question"),
    }
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
