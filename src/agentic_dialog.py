"""Second-tier input interpreter: LLM tool-calling fallback for messages the
deterministic template parser (src.dialog.parse) cannot match. Never runs
under config.INPUT_MODE == "template" (the default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import config

_ENV_LOADED = False


def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a local .env file; real env vars take precedence."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def agentic_available() -> bool:
    _load_dotenv()
    return bool(os.environ.get("OPENAI_API_KEY"))


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_catalog_support",
            "description": (
                "Count how many products on the shopper's current shelf "
                "carry this exact attribute phrase in their catalog "
                "signature. Use this to verify a candidate constraint "
                "before submitting it -- a phrase with zero support is not "
                "a real catalog attribute value and must not be submitted "
                "as-is."
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
                "List the most common real attribute values (materials, "
                "colors, features, etc.) seen on the shopper's current "
                "shelf, so you can match the shopper's wording to the "
                "catalog's actual vocabulary (e.g. the shopper says 'jean' "
                "but the catalog only ever says 'denim')."
            ),
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 25}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_extraction",
            "description": (
                "Submit your final, catalog-verified interpretation of the "
                "shopper's message. Always call this exactly once to "
                "finish, even if the message carried no usable information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Zero or more catalog-verified constraint "
                            "phrases the shopper revealed this turn, in the "
                            "catalog's own wording (each verified via "
                            "check_catalog_support)."
                        ),
                    },
                    "is_override": {
                        "type": "boolean",
                        "description": (
                            "True only if the shopper is explicitly "
                            "replacing or retracting an earlier stated "
                            "preference."
                        ),
                    },
                    "no_preference_attribute": {
                        "type": ["string", "null"],
                        "description": (
                            "Set if the shopper declined to state a "
                            "preference for one specific attribute just "
                            "asked about (e.g. 'color'), otherwise null."
                        ),
                    },
                    "exhausted_attribute": {
                        "type": ["string", "null"],
                        "description": (
                            "Set if the shopper said they have nothing "
                            "further to add about one specific attribute "
                            "just asked about, otherwise null."
                        ),
                    },
                    "uninformative": {
                        "type": "boolean",
                        "description": (
                            "True if the message carries no usable "
                            "preference, override, or refusal signal at "
                            "all."
                        ),
                    },
                },
                "required": ["constraints", "is_override", "uninformative"],
            },
        },
    },
]

_SYSTEM_PROMPT = """You are the input-understanding layer of a shopping agent.
The deterministic template parser already tried and failed to recognise this
message, so it is phrased in a way its fixed templates did not anticipate.
Your job is only to classify and extract -- never to rank products or decide
what question to ask next.

Use check_catalog_support and list_known_values to ground any constraint you
extract in real catalog attribute values before submitting it; never submit a
constraint phrase you have not verified has non-zero support. If a phrase has
zero support, try a close catalog synonym via list_known_values, or omit it.

Finish by calling submit_extraction exactly once."""


def _run_tool(name: str, tool_input: dict, catalog, shelf: str | None) -> dict:
    if name == "check_catalog_support":
        phrase = str(tool_input.get("phrase", ""))
        return {"support": catalog.signature_support(phrase, shelf)}
    if name == "list_known_values":
        limit = int(tool_input.get("limit") or 25)
        counts = (
            catalog.signature_value_count_by_shelf.get(shelf, {})
            if shelf else catalog.signature_value_count
        )
        top = sorted(counts.items(), key=lambda item: -item[1])[:limit]
        return {"values": [value for value, _ in top]}
    return {"error": f"unknown tool {name}"}


def _apply_extraction(extraction: dict, state, catalog) -> bool:
    """Mirror dialog.parse's state transitions. Returns True if state changed."""
    if extraction.get("uninformative"):
        return False
    exhausted_attribute = extraction.get("exhausted_attribute")
    if exhausted_attribute:
        attribute = str(exhausted_attribute).strip().lower()
        if attribute:
            state.exhausted.add(attribute)
            state.last_reply_count = 0
            if attribute == "other":
                state.information_complete = True
            return True
    if extraction.get("no_preference_attribute"):
        state.boundary_signal = True
        return True
    is_override = bool(extraction.get("is_override"))
    if is_override:
        state.override_seen = True
        state.last_reply_count = None
        state.clear_recommendation_history()
        state.decay_provisional(config.OVERRIDE_DECAY)
    verified = [
        phrase for phrase in extraction.get("constraints") or []
        if isinstance(phrase, str)
        and phrase.strip()
        and catalog.signature_support(phrase, state.shelf) > 0
    ]
    for phrase in verified:
        state.add(phrase)
    if not is_override:
        state.last_reply_count = len(verified)
    return is_override or bool(verified)


STATUS_UNAVAILABLE = "unavailable"  # no API key/package: model never ran
STATUS_ERROR = "error"              # model ran but failed
STATUS_EXTRACTED = "extracted"      # model ran and applied real signal
STATUS_NO_SIGNAL = "no_signal"      # model ran, correctly found nothing


def agentic_interpret(message: str, state, catalog) -> str:
    """Returns one of the STATUS_* constants above. Never raises."""
    if not agentic_available():
        return STATUS_UNAVAILABLE
    try:
        import openai
    except ImportError:
        return STATUS_UNAVAILABLE

    try:
        client = openai.OpenAI()
        shelf = state.shelf
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        for _ in range(max(1, int(config.AGENTIC_MAX_TOOL_CALLS))):
            response = client.chat.completions.create(
                model=config.AGENTIC_MODEL,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
            )
            reply = response.choices[0].message
            tool_calls = reply.tool_calls or []
            submission = next(
                (call for call in tool_calls if call.function.name == "submit_extraction"),
                None,
            )
            if submission is not None:
                try:
                    extraction = json.loads(submission.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    return STATUS_ERROR
                changed = _apply_extraction(extraction, state, catalog)
                return STATUS_EXTRACTED if changed else STATUS_NO_SIGNAL
            if not tool_calls:
                return STATUS_ERROR
            messages.append(reply.model_dump(exclude_unset=True))
            for call in tool_calls:
                try:
                    args = json.loads(call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result = _run_tool(call.function.name, args, catalog, shelf)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                })
        return STATUS_ERROR
    except Exception:
        return STATUS_ERROR


_REPLY_SYSTEM_PROMPT = (
    "You write one short, natural reply for a shopping assistant. "
    "Under 15 words, one sentence, no filler, no exclamation points. "
    "Never invent product names, prices, or facts."
)


def agentic_reply(attribute: str | None, state) -> str | None:
    """LLM-phrased variant of the outgoing question/acknowledgement.

    Returns None on unavailability or any failure -- caller falls back to
    Agent._message. Capped by config.AGENTIC_REPLY_MAX_TOKENS (generation
    cost) and config.AGENTIC_REPLY_MAX_CHARS (rejects an overlong reply
    instead of trusting the model followed the length instruction).
    """
    if not agentic_available():
        return None
    try:
        import openai
    except ImportError:
        return None
    if attribute is None:
        intent = "Tell the shopper you have enough detail to narrow this down now."
    elif attribute == "other":
        intent = "Ask the shopper if anything else matters for this purchase."
    else:
        intent = f"Ask the shopper if they have a preference on {attribute}."
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=config.AGENTIC_MODEL,
            messages=[
                {"role": "system", "content": _REPLY_SYSTEM_PROMPT},
                {"role": "user", "content": intent},
            ],
            max_tokens=config.AGENTIC_REPLY_MAX_TOKENS,
            temperature=0.7,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text or len(text) > config.AGENTIC_REPLY_MAX_CHARS:
            return None
        return text
    except Exception:
        return None
