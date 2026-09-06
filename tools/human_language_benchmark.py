"""Human-language stress benchmark for the deterministic and hybrid agents.

The organizer's simulator quotes catalog strings inside fixed templates. This
tool keeps the simulator's *policy* byte-identical (it calls the unmodified
``evaluate`` loop, the same intent cards, and the same stopping rule) and
rewrites only the *surface form* of each customer message with a language
model acting as a human shopper:

``natural``     casual wording, every attribute phrase kept verbatim;
``paraphrase``  casual wording and every attribute expressed in the shopper's
                own words, never quoting the listing.

Each level is scored for two arms of the same submitted code base:

``deterministic``  ``ARC_LLM_MODE=off`` (the frozen, token-free submission);
``hybrid``         ``ARC_LLM_MODE=ground`` (LLM proposes, catalog verifies).

The ``canonical`` level (no rewriting) is included as the control: it shows
the hybrid arm making zero model calls on protocol wording.

Rewrites are cached on disk, keyed by level and exact message, so the run is
reproducible and the human shopper says the same thing to both arms. The
customer model and the grounding model may be pointed at different endpoints
so the paraphraser and the reader are not the same network.

Results are research diagnostics, not organizer scores.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src import config
from src.llm import ChatClient, LLMSettings

CUSTOMER_PROMPTS = {
    "natural": (
        "You are a real online shopper typing in a chat box. Rewrite the templated "
        "message below the way you would actually type it: casual, first person, one "
        "or two short sentences. Keep every product attribute phrase EXACTLY as "
        "written (same words, same spelling, same order), keep every number, and keep "
        "the exact meaning: if the message says you have no preference or that the "
        "options were wrong, say that; if it cancels an earlier preference, make the "
        "cancellation clear. Do not add preferences, products or details that are not "
        "in the original. Output only the rewritten message."
    ),
    "paraphrase": (
        "You are a real online shopper typing in a chat box. Rewrite the templated "
        "message below the way you would actually type it: casual, first person, one "
        "or two short sentences. Express every product attribute in your own words: "
        "do NOT quote the original attribute phrases verbatim, use synonyms or plain "
        "descriptions instead, but keep materials, colors, numbers and the product "
        "type recognizable, and keep the exact meaning: if the message says you have "
        "no preference or that the options were wrong, say that; if it cancels an "
        "earlier preference, make the cancellation clear. Do not add preferences, "
        "products or details that are not in the original. Output only the rewritten "
        "message."
    ),
}


class HumanCustomer:
    """Rewrites simulator messages through a cached model call."""

    def __init__(
        self,
        client: ChatClient | None,
        level: str,
        max_failures: int = 0,
        cache_path: str = "",
    ) -> None:
        self.client = client
        self.level = level
        self.rewrites: dict[str, str] = {}
        self.failures = 0
        self.max_failures = max_failures
        self.cache_path = cache_path

    def rewrite(self, message: str) -> str:
        if self.level == "canonical" or self.client is None:
            return message
        cached = self.rewrites.get(message)
        if cached is not None:
            return cached
        reply = self.client.chat(
            [
                {"role": "system", "content": CUSTOMER_PROMPTS[self.level]},
                {"role": "user", "content": message},
            ],
            max_tokens=120,
            temperature=0.0,
        )
        text = " ".join(reply.text.strip().strip('"').split()) if reply.ok else ""
        if not text or len(text) > 600:
            self.failures += 1
            text = message
            if self.failures > self.max_failures:
                # Falling back to the un-rewritten message turns a paraphrase
                # run into a canonical one while still printing plausible
                # scores. Abort on the first one rather than after the arm has
                # been paid for; the usual cause is --customer-model not
                # matching the model --cache was built with.
                raise SystemExit(
                    f"level '{self.level}': rewrite failed for a message and "
                    f"--max-rewrite-failures is {self.max_failures}. The "
                    f"customer model '{self.client.settings.model}' must "
                    f"match the model that built '{self.cache_path}', or the "
                    "endpoint must be reachable. Refusing to report a run "
                    "whose wording silently degraded to canonical."
                )
        self.rewrites[message] = text
        return text


class HumanFacingAgent:
    """Feeds rewritten customer messages to the agent and records the trace."""

    def __init__(self, agent: Agent, customer: HumanCustomer) -> None:
        self.agent = agent
        self.customer = customer
        self.traces: dict[str, list[dict]] = defaultdict(list)
        self.respond_latency_ms: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        rewritten = self.customer.rewrite(user_message)
        started = time.perf_counter()
        response = self.agent.respond(session_id, rewritten, turn, top_k)
        elapsed = (time.perf_counter() - started) * 1000.0
        self.respond_latency_ms.append(elapsed)
        certificate = self.agent.explain_last_decision(session_id)
        self.traces[session_id].append({
            "turn": turn,
            "raw_message": user_message,
            "human_message": rewritten,
            "response": deepcopy(response),
            "grounding": deepcopy(certificate.get("llm_grounding")),
            "llm_usage": deepcopy(certificate.get("llm_usage")),
            "input_interpretation": certificate.get("input_interpretation"),
            "latency_ms": round(elapsed, 2),
        })
        return response


def grounding_summary(wrapper: HumanFacingAgent) -> dict:
    tiers: Counter = Counter()
    rejected: Counter = Counter()
    intents: Counter = Counter()
    statuses: Counter = Counter()
    calls = 0
    latency: list[float] = []
    interpretations: Counter = Counter()
    for trace in wrapper.traces.values():
        for row in trace:
            interpretations[str(row.get("input_interpretation"))] += 1
            grounding = row.get("grounding")
            if not grounding:
                # The agentic arm spends tokens without producing a grounding
                # block, so its usage must be counted before this skip.
                usage = row.get("llm_usage") or {}
                calls += int(usage.get("calls") or 0)
                if usage.get("latency_ms"):
                    latency.append(float(usage["latency_ms"]))
                continue
            statuses[str(grounding.get("status"))] += 1
            if grounding.get("intent"):
                intents[str(grounding["intent"])] += 1
            for item in grounding.get("accepted", []):
                tiers[str(item.get("tier"))] += 1
            for item in grounding.get("rejected", []):
                rejected[str(item.get("reason"))] += 1
            usage = row.get("llm_usage") or {}
            calls += int(usage.get("calls") or 0)
            if usage.get("latency_ms"):
                latency.append(float(usage["latency_ms"]))
    return {
        "grounded_turns": sum(statuses.values()),
        "statuses": dict(statuses),
        "intents": dict(intents),
        "accepted_by_tier": dict(tiers),
        "rejected_by_reason": dict(rejected),
        "model_calls": calls,
        "input_interpretation": dict(interpretations),
        "mean_model_latency_ms_per_turn": round(statistics.fmean(latency), 1) if latency else 0.0,
    }


def sample_rewrites(customer: HumanCustomer, limit: int) -> list[dict]:
    rows = []
    for raw, human in customer.rewrites.items():
        if raw == human:
            continue
        rows.append({"simulator": raw, "human": human})
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--levels", default="canonical,natural,paraphrase")
    parser.add_argument("--arms", default="deterministic,hybrid")
    parser.add_argument("--base-url", default=os.environ.get("ARC_LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("ARC_LLM_MODEL", "gemma4"))
    parser.add_argument("--api-key", default=os.environ.get("ARC_LLM_API_KEY", "unused"))
    parser.add_argument("--customer-base-url", default=os.environ.get("ARC_CUSTOMER_BASE_URL", ""))
    parser.add_argument("--customer-model", default=os.environ.get("ARC_CUSTOMER_MODEL", ""))
    parser.add_argument("--customer-api-key", default=os.environ.get("ARC_CUSTOMER_API_KEY", "unused"))
    parser.add_argument("--customer-insecure", action="store_true",
                        help="skip TLS verification for the customer endpoint (dev proxies only)")
    parser.add_argument("--cache", default="results/human_language_cache.json")
    parser.add_argument("--output", default="results/human_language_benchmark.json")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--max-rewrite-failures", type=int, default=0,
        help=(
            "Abort once this many customer rewrites have fallen back to the "
            "un-rewritten message (default 0). Raise it only for a live "
            "endpoint where an occasional blip is acceptable."
        ),
    )
    args = parser.parse_args()

    levels = [item.strip() for item in args.levels.split(",") if item.strip()]
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    if not args.base_url:
        parser.error("--base-url (or ARC_LLM_BASE_URL) is required")

    ground_settings = LLMSettings(
        mode="ground", base_url=args.base_url.rstrip("/"), model=args.model,
        api_key=args.api_key, timeout=args.timeout, cache_path=args.cache,
    )
    customer_settings = LLMSettings(
        mode="ground",
        base_url=(args.customer_base_url or args.base_url).rstrip("/"),
        model=args.customer_model or args.model,
        api_key=args.customer_api_key if args.customer_base_url else args.api_key,
        timeout=args.timeout,
        cache_path=args.cache,
    )
    customer_client = ChatClient(customer_settings)
    if args.customer_insecure:
        import ssl
        customer_client.ssl_context = ssl._create_unverified_context()  # noqa: SLF001

    samples = load_jsonl(args.dataset)[: args.count]
    ids, categories, products = catalog_index(args.catalog)

    print(f"grounding model: {ground_settings.model} @ {ground_settings.base_url}")
    print(f"customer model:  {customer_settings.model} @ {customer_settings.base_url}")
    agents = {}
    for arm in arms:
        if arm == "deterministic":
            agents[arm] = Agent(args.catalog, llm_settings=LLMSettings(mode="off"))
        elif arm == "hybrid":
            agents[arm] = Agent(args.catalog, llm_settings=ground_settings)
        elif arm == "agentic":
            # The OpenAI tool-calling input fallback, isolated: the ground
            # layer is off so this arm measures INPUT_MODE alone.
            agents[arm] = Agent(args.catalog, llm_settings=LLMSettings(mode="off"))
        else:
            parser.error(f"unknown arm: {arm}")

    experiments: dict[str, dict] = {}
    rewrites_by_level: dict[str, list[dict]] = {}
    for level in levels:
        customer = HumanCustomer(
            customer_client if level != "canonical" else None,
            level,
            max_failures=args.max_rewrite_failures,
            cache_path=args.cache,
        )
        for arm in arms:
            wrapper = HumanFacingAgent(agents[arm], customer)
            previous_input_mode = config.INPUT_MODE
            config.INPUT_MODE = "agentic" if arm == "agentic" else "template"
            started = time.perf_counter()
            try:
                outcome = evaluate(wrapper, samples, ids, categories, products)
            finally:
                config.INPUT_MODE = previous_input_mode
            wall = time.perf_counter() - started
            customer_client.flush()
            client = getattr(agents[arm], "llm", None)
            if client is not None and hasattr(client, "flush"):
                client.flush()
            label = f"{level}__{arm}"
            experiments[label] = {
                "level": level,
                "arm": arm,
                "hit_rate_at_10": outcome["hit_rate_at_10"],
                "mrr": outcome["mrr"],
                "mttc": outcome["mttc"],
                "efficiency": outcome["efficiency"],
                "technical_score": outcome["recommended_technical_score"],
                "reported_token_usage": outcome["reported_token_usage"],
                "scenario_metrics": outcome["scenario_metrics"],
                "agent_turns": sum(len(trace) for trace in wrapper.traces.values()),
                "mean_respond_latency_ms": round(statistics.fmean(wrapper.respond_latency_ms), 1),
                "p95_respond_latency_ms": round(
                    sorted(wrapper.respond_latency_ms)[int(0.95 * (len(wrapper.respond_latency_ms) - 1))], 1
                ),
                "wall_seconds": round(wall, 1),
                "customer_rewrite_failures": customer.failures,
                "grounding": grounding_summary(wrapper),
            }
            print(
                f"{label:<28} HR={outcome['hit_rate_at_10']:.4f} MRR={outcome['mrr']:.4f} "
                f"MTTC={outcome['mttc']:.3f} score={outcome['recommended_technical_score']:.6f} "
                f"tokens={outcome['reported_token_usage']['total_tokens']} "
                f"calls={experiments[label]['grounding']['model_calls']} wall={wall:.0f}s"
            )
        rewrites_by_level[level] = sample_rewrites(customer, 12)

    result = {
        "status": "research_diagnostic_not_official_score",
        "sample_count": len(samples),
        "levels": levels,
        "arms": arms,
        "grounding_model": {"model": ground_settings.model, "base_url": ground_settings.base_url},
        "customer_model": {"model": customer_settings.model, "base_url": customer_settings.base_url},
        "experiments": experiments,
        "sample_rewrites": rewrites_by_level,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
