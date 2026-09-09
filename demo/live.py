"""Presenter controls and evaluator overlays, outside the Agent boundary."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
import uuid

from src.llm import ChatClient, LLMSettings
from src.telemetry import observe

ROOT = Path(__file__).resolve().parent.parent
MAX_SESSIONS = 64


@dataclass
class LabSession:
    sample: dict | None = None
    turn: int = 0
    done: bool = False
    disclosed: set = field(default_factory=set)
    boundary_used: bool = False
    pending: dict = field(default_factory=dict)
    history: list = field(default_factory=list)


class LiveLab:
    def __init__(self, runtime):
        self.runtime = runtime
        self.sessions: dict[str, LabSession] = {}
        self._samples = None
        self._products = None
        self._rewriters = {}

    def config(self):
        bundle = self.runtime.replay_bundle()
        replay = bundle.get("session_replays", {})
        products = replay.get("products", {})
        cases = []
        for row in replay.get("sessions", []):
            product = products.get(row["target_parent_asin"], {})
            cases.append({
                "id": row["sample_id"], "scenario": row["scenario_type"],
                "title": product.get("title", row["target_parent_asin"]),
                "turns": len(row.get("turns", [])),
            })
        return {"cases": cases, "models": [self._model_info(key) for key in ("gemma", "qwen")],
                "health": self.runtime.health()}

    def _model_settings(self, key):
        if not isinstance(key, str) or key not in {"gemma", "qwen"}:
            raise ValueError("Choose gemma or qwen as the shopper rewrite model")
        prefix = "ARC_DEMO_" + key.upper()
        defaults = {
            "gemma": ("gemma4", "human_language_cache_paraphrase.json"),
            "qwen": ("qwen2.5-7b-instruct", "human_language_cache_qwen_customer.json"),
        }
        model, cache = defaults[key]
        return LLMSettings(
            mode="ground", model=os.environ.get(prefix + "_MODEL", model),
            base_url=os.environ.get(prefix + "_BASE_URL") or os.environ.get("ARC_CUSTOMER_BASE_URL")
            or os.environ.get("ARC_LLM_BASE_URL", ""),
            api_key=os.environ.get(prefix + "_API_KEY") or os.environ.get("ARC_CUSTOMER_API_KEY")
            or os.environ.get("ARC_LLM_API_KEY", "unused"),
            cache_path=os.environ.get(prefix + "_CACHE", str(ROOT / "results" / cache)),
            timeout=LLMSettings.from_env().timeout, max_retries=0,
        )

    def _model_info(self, key):
        settings = self._model_settings(key)
        cached = bool(settings.cache_path and Path(settings.cache_path).is_file())
        return {"id": key, "name": settings.model, "live": bool(settings.base_url),
                "cached": cached, "available": cached or bool(settings.base_url)}

    def rewrite(self, payload):
        from tools.human_language_benchmark import CUSTOMER_PROMPTS

        message = checked_message(payload.get("message"))
        key = payload.get("model", "gemma")
        settings = self._model_settings(key)
        # Serialize the shared client's circuit breaker and cache. No request
        # can change an endpoint; configuration is local environment only.
        with self.runtime._agent_lock:
            client = self._rewriters.get(key)
            if client is None:
                client = ChatClient(settings, replay_only=not bool(settings.base_url))
                self._rewriters[key] = client
            started = time.perf_counter()
            reply = client.chat([
                {"role": "system", "content": CUSTOMER_PROMPTS["paraphrase"]},
                {"role": "user", "content": message},
            ], max_tokens=120, temperature=0.0)
        if not reply.ok or not reply.text.strip():
            raise RuntimeError(
                "No recorded rewrite for this message. Configure ARC_DEMO_"
                + key.upper() + "_BASE_URL for live rewrites."
                if not settings.base_url else "The shopper rewrite model could not complete this request. Try again or send the original."
            )
        return {"original": message, "text": reply.text.strip(), "model": settings.model,
                "source": "cached" if reply.cached else "live",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "tokens": reply.prompt_tokens + reply.completion_tokens}

    def _load_samples(self):
        if self._samples is not None:
            return
        from evaluator.local_evaluator import load_jsonl

        samples = load_jsonl(ROOT / "data/public_set.jsonl")
        self._samples = {s["sample_id"]: s for s in samples}
        needed = {s["ground_truth"]["parent_asin"] for s in samples}
        products = {}
        with self.runtime.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["parent_asin"] in needed:
                    products[row["parent_asin"]] = row
        self._products = products

    def reset(self, payload):
        from evaluator.local_evaluator import coarse_category, initial_message, materialize_hidden_fields

        with self.runtime._agent_lock:
            agent = self.runtime.ensure_agent()
            sample_id = payload.get("sample_id")
            if sample_id is not None and not isinstance(sample_id, str):
                raise ValueError("Scenario ID must be a string")
            session = LabSession()
            profile = {"preference_tags": []}
            target = None
            if sample_id:
                self._load_samples()
                sample = self._samples.get(sample_id)
                if sample is None:
                    raise ValueError("Unknown public scenario")
                target_id = sample["ground_truth"]["parent_asin"]
                if target_id not in self._products:
                    raise ValueError("This scenario's target is not in the loaded catalog")
                card, behavior = materialize_hidden_fields(sample, self._products)
                session.sample = {**sample, "intent_card": card, "behavior": behavior}
                profile = sample["user_profile"]
                disclosed = set()
                product = self._products[target_id]
                opener = initial_message(session.sample, coarse_category(product.get("categories", [])), disclosed)
                session.pending = {"text": opener, "disclosed": disclosed, "boundary_used": False}
                target = self._product(agent, target_id)
            session_id = "lab_" + uuid.uuid4().hex
            # Bound both layers of session state. Eviction occurs under the
            # same lock as respond, so an active turn cannot be evicted.
            while len(self.sessions) >= MAX_SESSIONS:
                oldest = next(iter(self.sessions))
                self.sessions.pop(oldest)
                agent._sessions.pop(oldest, None)
            self.sessions[session_id] = session
            agent.reset(session_id, profile)
            return {"session_id": session_id, "suggestion": session.pending.get("text", ""),
                    "target": target, "sample_id": sample_id,
                    "scenario": session.sample["scenario_type"] if session.sample else "free_chat",
                    "agent_model": agent.llm_settings.model if agent.llm_settings.enabled else None,
                    "reader": agent.llm_settings.mode}

    @staticmethod
    def _product(agent, asin):
        catalog = agent.catalog
        return {"id": asin, "title": catalog.title.get(asin, asin),
                "shelf": catalog.shelf_of.get(asin, ""), "price": catalog.price.get(asin)}

    def _next_message(self, session, response):
        from evaluator.local_evaluator import customer_reply

        if session.sample is None or session.done:
            return ""
        disclosed = set(session.disclosed)
        boundary_used = session.boundary_used
        override = session.sample["behavior"].get("override") or {}
        if session.sample["scenario_type"] == "intent_override" and session.turn + 1 == int(override.get("turn", 3)):
            text = str(override.get("message", "Actually, please ignore my earlier preference."))
            if override.get("new_value"):
                disclosed.add(str(override["new_value"]))
        else:
            text, boundary_used = customer_reply(session.sample, response.get("ask_attribute"), disclosed, boundary_used)
        session.pending = {"text": text, "disclosed": disclosed, "boundary_used": boundary_used}
        return text

    def turn(self, payload, send):
        message = checked_message(payload.get("message"))
        session_id = payload.get("session_id")
        turn = payload.get("turn")
        with self.runtime._agent_lock:
            session = self.sessions.get(session_id) if isinstance(session_id, str) else None
            if session is None:
                raise ValueError("Start a new demo session first")
            if session.done:
                raise ValueError("This session has ended. Start a new session.")
            if type(turn) is not int or turn != session.turn + 1 or not 1 <= turn <= 10:
                raise ValueError("Turn must be the next unprocessed turn, from 1 to 10")
            agent = self.runtime.ensure_agent()
            if payload.get("source_message", message) == session.pending.get("text"):
                session.disclosed = set(session.pending.get("disclosed", set()))
                session.boundary_used = session.pending.get("boundary_used", False)
            events = []
            final_ranking = []
            starts = {}
            started = time.perf_counter()

            def record(event):
                nonlocal final_ranking
                data = dict(event["data"])
                # Keep all ranked IDs on the observer side for the subsequent
                # evaluator join. Stream only display facts and the top ten.
                if "scores" in data:
                    scores = data.pop("scores")
                    final_ranking = [asin for asin, _ in scores]
                    data["count"] = len(scores)
                    data["top"] = [{**self._product(agent, asin), "rank": i + 1,
                                    "score": round(score, 6)} for i, (asin, score) in enumerate(scores[:10])]
                now = time.perf_counter()
                stage = event["stage"]
                if event["status"] == "running":
                    starts[stage] = now
                item = {"session_id": session_id, "turn": turn, "seq": len(events) + 1,
                        "stage": stage, "status": event["status"], "data": deepcopy(data),
                        "elapsed_ms": round((now - started) * 1000, 2)}
                if event["status"] in {"complete", "skipped"}:
                    item["duration_ms"] = round((now - starts.get(stage, now)) * 1000, 2)
                events.append(item)
                send("stage", item)

            # No target, sample, intent card or future reply enters respond.
            with observe(record):
                response = agent.respond(session_id, message, turn, 10)
            certificate = deepcopy(agent._sessions[session_id].last_decision_certificate)
            recommendations = [x["parent_asin"] for x in response["recommendations"]]
            overlay = None
            if session.sample:
                target_id = session.sample["ground_truth"]["parent_asin"]
                override = session.sample["behavior"].get("override") or {}
                eligible = session.sample["scenario_type"] != "intent_override" or turn >= int(override.get("turn", 3))
                candidate_rank = final_ranking.index(target_id) + 1 if target_id in final_ranking else None
                submitted_rank = recommendations.index(target_id) + 1 if target_id in recommendations else None
                overlay = {"candidate_rank": candidate_rank, "recommendation_rank": submitted_rank,
                           "eligible": eligible, "hit": eligible and submitted_rank is not None,
                           "target_id": target_id, "visibility": "post_response_evaluator_only"}
            session.turn = turn
            session.done = turn >= 10 or bool(overlay and overlay["hit"])
            suggestion = self._next_message(session, response)
            result = {"session_id": session_id, "turn": turn, "message": message,
                      "response": response, "certificate": certificate, "overlay": overlay,
                      "products": [self._product(agent, asin) for asin in recommendations],
                      "events": events, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                      "suggestion": suggestion, "done": session.done}
            session.history.append(result)
            send("done", result)
            return result

    def compare(self, payload, send):
        """Run two diagnostic forks without consuming the live conversation."""
        original = checked_message(payload.get("original"))
        rewritten = checked_message(payload.get("rewritten"))
        with self.runtime._agent_lock:
            session_id = payload.get("session_id")
            source = self.sessions.get(session_id) if isinstance(session_id, str) else None
            if source is None or source.done:
                raise ValueError("Choose an active live session to compare")
            agent = self.runtime.ensure_agent()
            results = {}
            for label, message in (("original", original), ("rewritten", rewritten)):
                fork_id = "fork_" + uuid.uuid4().hex
                self.sessions[fork_id] = deepcopy(source)
                agent._sessions[fork_id] = deepcopy(agent._sessions[session_id])
                send("branch", {"branch": label, "status": "running"})
                try:
                    result = self.turn({"session_id": fork_id, "turn": source.turn + 1,
                                        "message": message, "source_message": original}, lambda *_: None)
                    results[label] = result
                    send("branch", {"branch": label, "status": "complete", "result": result})
                finally:
                    self.sessions.pop(fork_id, None)
                    agent._sessions.pop(fork_id, None)
            send("comparison", results)
            return results


def checked_message(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError("Message must contain 1–4000 characters")
    return value.strip()
