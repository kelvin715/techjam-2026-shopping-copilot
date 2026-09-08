"""Attribution baseline: a language model plays the whole agent.

Each turn, a lexical shortlist of candidate products is drawn from the shelf
pool (rarity-weighted token overlap between everything the shopper has said
and the product text, previously shown products excluded). The model reads
the transcript and the shortlist and returns which attribute to ask about and
an ordering of candidate identifiers. Identifiers outside the shortlist are
dropped; an empty or unparseable answer falls back to the shortlist order and
is counted. This is the "let the model rank" alternative that the grounding
layer deliberately avoids. Research only; never on the scored path.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.llm import LLMReply, LLMUsage
from src.shelf import Catalog, tokens

ALLOWED = ("material", "color", "size", "style", "feature", "use_case", "other")

SYSTEM_PROMPT = """You are a shopping assistant. A customer is looking for one specific product from a catalog of clothing, shoes and jewelry. Each turn you see the conversation so far and a numbered list of candidate products. Decide two things:
1. "ask_attribute": one clarifying attribute to ask about next, chosen from material, color, size, style, feature, use_case, other; or null if you do not need to ask.
2. "ranked_ids": the candidate ids ordered from most to least likely to be the customer's product, at most 10 ids, using only ids from the list.
The customer sees the ranked list on every turn, so always return ranked_ids with up to 10 ids, even when you also ask a question. Return only a JSON object with these two keys; ids are plain strings and no explanations."""


class LLMAgent:
    def __init__(self, catalog_path: str | Path, client, shortlist: int = 40) -> None:
        self.catalog = Catalog(catalog_path)
        self.llm = client
        self.shortlist = shortlist
        self._sessions: dict[str, dict] = {}
        self.fallbacks = 0
        # why the shortlist order was shown instead of the model's ranking
        self.fallback_reasons: Counter = Counter()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {"history": [], "shown": set(), "pool": None, "last": {}}

    def _pool(self, session: dict, message: str) -> list[str]:
        if session["pool"] is None:
            shelf = self.catalog.match_shelf(message)
            if shelf is not None:
                session["pool"] = list(self.catalog.by_shelf[shelf])
            else:
                _, pool = self.catalog.shelf_pool(message)
                session["pool"] = pool or list(self.catalog.ids)
        return session["pool"]

    def _shortlist(self, session: dict) -> list[str]:
        # Sorted token order and a full sort key keep the shortlist, and hence
        # the prompt, identical across processes (a set-ordered float sum can
        # differ in the last bit and reorder ties, which breaks cache replay).
        query = sorted({t for role, text in session["history"] if role == "shopper" for t in tokens(text) if len(t) > 2})
        weights = [(t, self.catalog.token_idf(t)) for t in query]
        scored = []
        for asin in session["pool"]:
            if asin in session["shown"]:
                continue
            padded = " " + self.catalog.ltext[asin] + " "
            score = round(sum(w for t, w in weights if f" {t} " in padded), 9)
            scored.append((score, self.catalog.popularity(asin), asin))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [asin for _, _, asin in scored[: self.shortlist]]

    def _prompt(self, session: dict, shortlist: list[str]) -> str:
        lines = ["Conversation so far:"]
        for role, text in session["history"]:
            lines.append(f"{'Customer' if role == 'shopper' else 'Assistant'}: {text}")
        lines.append("\nCandidate products:")
        for index, asin in enumerate(shortlist, start=1):
            title = self.catalog.title.get(asin, "")[:100]
            price = self.catalog.price.get(asin)
            signature = "; ".join(v for v in self.catalog.signature.get(asin, ())[:3])
            lines.append(f"{index}. id={asin} | {title} | {signature[:160]} | price={price if price is not None else 'n/a'}")
        return "\n".join(lines)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        session = self._sessions.setdefault(session_id, {"history": [], "shown": set(), "pool": None, "last": {}})
        session["history"].append(("shopper", user_message))
        self._pool(session, user_message)
        shortlist = self._shortlist(session)
        usage = LLMUsage()
        reply: LLMReply = self.llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": self._prompt(session, shortlist)}],
            max_tokens=300, temperature=0.0,
        )
        usage.absorb(reply)
        data = reply.json() if reply.ok else None
        attribute = None
        ranked: list[str] = []
        reason = None
        if not reply.ok:
            reason = "model_error"
        elif not isinstance(data, dict):
            reason = "unparseable"
        else:
            candidate = data.get("ask_attribute")
            attribute = candidate if candidate in ALLOWED else None
            allowed = set(shortlist)
            items = data.get("ranked_ids") or []
            for item in items:
                # accept plain ids, {"id": ...} objects, and list positions
                if isinstance(item, dict):
                    item = item.get("id") or item.get("asin") or item.get("product_id") or ""
                asin = str(item).strip()
                if asin.isdigit() and 1 <= int(asin) <= len(shortlist):
                    asin = shortlist[int(asin) - 1]
                if asin in allowed and asin not in ranked:
                    ranked.append(asin)
            if not ranked:
                reason = "no_ids" if not items else "ids_off_list"
        fallback = not ranked
        if fallback:
            self.fallbacks += 1
            self.fallback_reasons[reason] += 1
            ranked = list(shortlist)
        recommendations = ranked[:top_k]
        session["shown"].update(recommendations)
        message = f"Do you have a preference on {attribute}?" if attribute else "Here are the closest matches I found."
        session["history"].append(("agent", message))
        session["last"] = {"llm_agent": {"shortlist": len(shortlist), "fallback": fallback, "fallback_reason": reason,
                                         "ask_attribute": attribute, "usage": usage.to_dict()},
                           "llm_usage": usage.to_dict(), "constraints": [], "reason_codes": []}
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
        }

    def explain_last_decision(self, session_id: str) -> dict:
        return dict(self._sessions.get(session_id, {}).get("last", {}))
