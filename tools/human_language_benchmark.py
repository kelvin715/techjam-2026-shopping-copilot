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
``cascade``        ``ARC_LLM_MODE=ground`` (the catalog reads first, the model
                   is consulted only when that leaves the sentence unread);
``hybrid``         the model-only reader (every off-protocol message is a
                   model call; LLM proposes, catalog verifies).

The ``canonical`` level (no rewriting) is included as the control: it shows
the hybrid arm making zero model calls on protocol wording.

Rewrites are cached on disk, keyed by level and exact message, so the run is
reproducible and the human shopper says the same thing to both arms. The
customer model and the grounding model may be pointed at different endpoints
so the paraphraser and the reader are not the same network.

External systems that implement the organizer's ``Agent`` contract can be
scored under the same cached rewrites with ``--external NAME=REPO_DIR`` (module
``starter.agent``, class ``Agent`` by default; override with
``NAME=REPO_DIR:module.path:ClassName``). The repository root is put on
``sys.path`` for the import and the process working directory is switched to
it while the agent is constructed, because several entries resolve their own
data files relative to the checkout.

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
from src.shelf import Catalog

# Attribution baselines change one research switch each; the switch is set
# only while that arm is evaluated and restored afterwards.
# ``cascade`` is the shipped ``ground`` mode (catalog reads first, model on
# demand); the ``hybrid*`` arms send every off-protocol message to the model.
ARM_CONFIG = {
    "cascade": {"LLM_GROUND_CASCADE": True},
    "hybrid": {"LLM_GROUND_CASCADE": False},
    "hybrid_unverified": {"LLM_GROUND_CASCADE": False, "LLM_GROUND_VERIFY": False},
    "hybrid_nohints": {"LLM_GROUND_CASCADE": False, "LLM_GROUND_VOCAB_HINTS": 0},
    # Verification on, but every admitted tier at full confidence: separates
    # the admission filter from the confidence down-weighting.
    "hybrid_flatweights": {"LLM_GROUND_CASCADE": False, "LLM_GROUND_MAPPED_WEIGHT": 1.0,
                           "LLM_GROUND_LEXICAL_WEIGHT": 1.0, "LLM_GROUND_TOKEN_WEIGHT": 1.0},
    # Strictest verifier: only exact catalog strings and typed values enter.
    "hybrid_verbatim_only": {"LLM_GROUND_CASCADE": False, "LLM_GROUND_TIERS": ("signature_verbatim",)},
    # Dense research options on top of the shipped cascade (src/dense.py):
    # option 1 expands admitted values into embedding-near catalog strings,
    # option 2 adds an embedding recall channel for shelves and products.
    "cascade_dense_values": {"LLM_GROUND_CASCADE": True, "DENSE_VALUE_EXPANSION": True},
    "cascade_dense_pool": {"LLM_GROUND_CASCADE": True, "DENSE_SHELF_RECALL": True},
    "cascade_dense_both": {"LLM_GROUND_CASCADE": True, "DENSE_VALUE_EXPANSION": True, "DENSE_SHELF_RECALL": True},
    "cascade_dense_values_t75": {"LLM_GROUND_CASCADE": True, "DENSE_VALUE_EXPANSION": True, "DENSE_VALUE_MIN_COSINE": 0.75},
    "cascade_dense_values_t85": {"LLM_GROUND_CASCADE": True, "DENSE_VALUE_EXPANSION": True, "DENSE_VALUE_MIN_COSINE": 0.85},
    # The same options with no model at all (the endpoint-down fallback).
    "lexical_dense_values": {"DENSE_VALUE_EXPANSION": True},
    "lexical_dense_pool": {"DENSE_SHELF_RECALL": True},
    "lexical_dense_both": {"DENSE_VALUE_EXPANSION": True, "DENSE_SHELF_RECALL": True},
}
MODEL_ARMS = tuple(arm for arm in ARM_CONFIG if not arm.startswith("lexical"))
LEXICAL_ARMS = ("lexical",) + tuple(arm for arm in ARM_CONFIG if arm.startswith("lexical"))

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

    def __init__(self, client: ChatClient | None, level: str) -> None:
        self.client = client
        self.level = level
        self.rewrites: dict[str, str] = {}
        self.failures = 0

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
        if not reply.ok and getattr(self.client, "replay_only", False):
            raise RuntimeError(f"strict replay: no cached rewrite for {message!r}")
        text = " ".join(reply.text.strip().strip('"').split()) if reply.ok else ""
        if not text or len(text) > 600:
            self.failures += 1
            text = message
        self.rewrites[message] = text
        return text


def load_external_agent(spec: str, catalog_path: str):
    """Instantiate an external ``Agent`` from ``NAME=REPO_DIR[:module:Class]``."""
    import importlib
    import os

    name, _, target = spec.partition("=")
    if not name or not target:
        raise ValueError(f"--external expects NAME=REPO_DIR[:module:Class], got {spec!r}")
    parts = target.split(":")
    repo = Path(parts[0]).expanduser().resolve()
    module_name = parts[1] if len(parts) > 1 and parts[1] else "starter.agent"
    class_name = parts[2] if len(parts) > 2 and parts[2] else "Agent"
    if not repo.is_dir():
        raise ValueError(f"external repository not found: {repo}")
    catalog_abs = str(Path(catalog_path).resolve())
    previous_cwd = os.getcwd()
    inserted = False
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
        inserted = True
    # Our own ``starter.agent`` and ``agent`` would shadow the external one.
    shadowed = {key: sys.modules.pop(key) for key in list(sys.modules)
                if key in ("agent", "starter", "starter.agent") or key.startswith("src.")}
    try:
        os.chdir(repo)
        module = importlib.import_module(module_name)
        agent_cls = getattr(module, class_name)
        try:
            agent = agent_cls(catalog_abs)
        except TypeError:
            agent = agent_cls()
    finally:
        os.chdir(previous_cwd)
        if inserted:
            sys.path.remove(str(repo))
        # Do not restore the shadowed modules: the external package now owns
        # those names for the rest of the run, and our own agent was already
        # imported before this call.
    return name, agent


class SubprocessAgent:
    """An external ``Agent`` living in its own interpreter and working directory."""

    def __init__(self, spec: str, catalog_path: str) -> None:
        import subprocess

        name, _, target = spec.partition("=")
        if not name or not target:
            raise ValueError(f"--external-isolated expects NAME=REPO_DIR[:module:Class], got {spec!r}")
        parts = target.split(":")
        repo = Path(parts[0]).expanduser().resolve()
        module_name = parts[1] if len(parts) > 1 and parts[1] else "starter.agent"
        class_name = parts[2] if len(parts) > 2 and parts[2] else "Agent"
        if not repo.is_dir():
            raise ValueError(f"external repository not found: {repo}")
        self.name = name
        self.errors = 0
        worker = ROOT / "tools" / "external_agent_worker.py"
        self._proc = subprocess.Popen(
            [sys.executable, str(worker), str(repo), module_name, class_name, str(Path(catalog_path).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            cwd=str(repo), env={**os.environ, "PYTHONPATH": str(repo)},
        )
        ready = json.loads(self._proc.stdout.readline())
        self.module_file = ready.get("module")

    def _call(self, request: dict) -> dict:
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"external agent {self.name} exited")
        return json.loads(line)

    def reset(self, session_id: str, user_profile: dict) -> None:
        reply = self._call({"op": "reset", "session_id": session_id, "user_profile": user_profile})
        if "error" in reply:
            self.errors += 1

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        reply = self._call({"op": "respond", "session_id": session_id, "message": user_message,
                            "turn": turn, "top_k": top_k})
        if "error" in reply:
            self.errors += 1
            return {"message": "", "ask_attribute": None, "recommendations": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        return reply["response"]

    def close(self) -> None:
        try:
            self._call({"op": "quit"})
        except Exception:  # noqa: BLE001
            pass
        self._proc.terminate()


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
        # Read the certificate without triggering the minimal-counterfactual
        # search, which reranks the pool up to fifteen times per turn.
        sessions = getattr(self.agent, "_sessions", None)
        state = sessions.get(session_id) if isinstance(sessions, dict) else None
        certificate = getattr(state, "last_decision_certificate", None)
        if not isinstance(certificate, dict):
            explain = getattr(self.agent, "explain_last_decision", None)
            certificate = explain(session_id) if callable(explain) else {}
        if not isinstance(certificate, dict):
            certificate = {}
        self.traces[session_id].append({
            "turn": turn,
            "raw_message": user_message,
            "human_message": rewritten,
            "response": deepcopy(response),
            "grounding": deepcopy(certificate.get("llm_grounding")),
            "llm_usage": deepcopy(certificate.get("llm_usage")),
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
    for trace in wrapper.traces.values():
        for row in trace:
            usage = row.get("llm_usage") or {}
            grounding = row.get("grounding")
            if not grounding:
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
    parser.add_argument("--arms", default="deterministic,cascade")
    parser.add_argument("--external", action="append", default=[],
                        help="NAME=REPO_DIR[:module:Class]; score an external Agent under the same rewrites (repeatable; in-process)")
    parser.add_argument("--external-isolated", action="append", default=[],
                        help="NAME=REPO_DIR[:module:Class]; same, but in its own interpreter and working directory (preferred)")
    parser.add_argument("--base-url", default=os.environ.get("ARC_LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("ARC_LLM_MODEL", "gemma4"))
    parser.add_argument("--api-key", default=os.environ.get("ARC_LLM_API_KEY", "unused"))
    parser.add_argument("--customer-base-url", default=os.environ.get("ARC_CUSTOMER_BASE_URL", ""))
    parser.add_argument("--customer-model", default=os.environ.get("ARC_CUSTOMER_MODEL", ""))
    parser.add_argument("--customer-api-key", default=os.environ.get("ARC_CUSTOMER_API_KEY", "unused"))
    parser.add_argument("--customer-insecure", action="store_true",
                        help="skip TLS verification for the customer endpoint (dev proxies only)")
    parser.add_argument("--cache", default="results/human_language_cache.json",
                        help="rewrite cache (customer model); the grounding model gets its own file")
    parser.add_argument("--grounding-cache", default=None,
                        help="grounding cache file; default <cache stem>.grounding.json")
    parser.add_argument("--strict-replay", action="store_true",
                        help="serve every rewrite from the cache; abort on a miss")
    parser.add_argument("--strict-grounding", action="store_true",
                        help="also serve every grounding call from its cache; a miss is a grounding failure")
    parser.add_argument("--output", default="results/human_language_benchmark.json")
    parser.add_argument("--traces", action="store_true",
                        help="also write <output stem>.traces.json with every session's turn-level trajectory")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    levels = [item.strip() for item in args.levels.split(",") if item.strip()]
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    if not args.base_url:
        if args.strict_replay and args.strict_grounding:
            # Every rewrite and every model reply comes from the caches; no
            # endpoint is contacted, so none needs to be named.
            args.base_url = "replay://cache"
        else:
            parser.error("--base-url (or ARC_LLM_BASE_URL) is required unless --strict-replay --strict-grounding")

    grounding_cache = args.grounding_cache or str(
        Path(args.cache).with_name(Path(args.cache).stem + ".grounding.json")
    )
    ground_settings = LLMSettings(
        mode="ground", base_url=args.base_url.rstrip("/"), model=args.model,
        api_key=args.api_key, timeout=args.timeout, cache_path=grounding_cache,
    )
    customer_settings = LLMSettings(
        mode="ground",
        base_url=(args.customer_base_url or args.base_url).rstrip("/"),
        model=args.customer_model or args.model,
        api_key=args.customer_api_key if args.customer_base_url else args.api_key,
        timeout=args.timeout,
        cache_path=args.cache,
    )
    customer_client = ChatClient(customer_settings, replay_only=args.strict_replay)
    if args.customer_insecure:
        import ssl
        customer_client.ssl_context = ssl._create_unverified_context()  # noqa: SLF001

    samples = load_jsonl(args.dataset)[: args.count]
    ids, categories, products = catalog_index(args.catalog)

    print(f"grounding model: {ground_settings.model} @ {ground_settings.base_url}")
    print(f"customer model:  {customer_settings.model} @ {customer_settings.base_url}")
    agents = {}
    dense_index = None

    def dense_for(arm: str):
        # One embedding index shared by every dense arm (research dependency).
        nonlocal dense_index
        if "dense" not in arm:
            return None
        if dense_index is None:
            from src.dense import DenseIndex

            dense_index = DenseIndex(Catalog(args.catalog))
        return dense_index

    for arm in arms:
        if arm == "deterministic":
            agents[arm] = Agent(args.catalog, llm_settings=LLMSettings(mode="off"))
        elif arm in MODEL_ARMS:
            agents[arm] = Agent(
                args.catalog,
                llm_settings=ground_settings,
                llm_client=ChatClient(ground_settings, replay_only=args.strict_grounding),
                dense=dense_for(arm),
            )
        elif arm in LEXICAL_ARMS:
            agents[arm] = Agent(args.catalog, llm_settings=LLMSettings(mode="lexical"), dense=dense_for(arm))
        elif arm == "dense":
            agents[arm] = Agent(args.catalog, llm_settings=LLMSettings(mode="dense"))
        elif arm == "llm_agent":
            from tools.llm_agent_baseline import LLMAgent

            agents[arm] = LLMAgent(args.catalog, ChatClient(ground_settings, replay_only=args.strict_grounding))
        else:
            parser.error(f"unknown arm: {arm}")
    for spec in args.external:
        name, external = load_external_agent(spec, args.catalog)
        if name in agents:
            parser.error(f"duplicate arm name: {name}")
        agents[name] = external
        arms.append(name)
        print(f"external arm {name}: {type(external).__module__}.{type(external).__name__}")
    for spec in args.external_isolated:
        external = SubprocessAgent(spec, args.catalog)
        if external.name in agents:
            parser.error(f"duplicate arm name: {external.name}")
        agents[external.name] = external
        arms.append(external.name)
        print(f"isolated external arm {external.name}: {external.module_file}")

    experiments: dict[str, dict] = {}
    rewrites_by_level: dict[str, list[dict]] = {}
    trajectories: dict[str, dict] = {}
    for level in levels:
        customer = HumanCustomer(customer_client if level != "canonical" else None, level)
        for arm in arms:
            wrapper = HumanFacingAgent(agents[arm], customer)
            overrides = ARM_CONFIG.get(arm, {})
            saved = {key: getattr(config, key) for key in overrides}
            for key, value in overrides.items():
                setattr(config, key, value)
            if hasattr(agents[arm], "fallbacks"):
                agents[arm].fallbacks = 0
                agents[arm].fallback_reasons.clear()
            started = time.perf_counter()
            try:
                outcome = evaluate(wrapper, samples, ids, categories, products)
            finally:
                for key, value in saved.items():
                    setattr(config, key, value)
            wall = time.perf_counter() - started
            customer_client.flush()
            client = getattr(agents[arm], "llm", None)
            if client is not None and hasattr(client, "flush"):
                client.flush()
            if args.strict_grounding and getattr(client, "misses", 0):
                raise RuntimeError(
                    f"strict grounding: {client.misses} cache misses for arm {arm!r} at level {level!r}"
                )
            label = f"{level}__{arm}"
            experiments[label] = {
                "level": level,
                "arm": arm,
                "external_module": getattr(agents[arm], "module_file", None),
                "external_errors": getattr(agents[arm], "errors", 0),
                "config_overrides": ARM_CONFIG.get(arm, {}),
                "llm_agent_fallbacks": getattr(agents[arm], "fallbacks", None),
                "llm_agent_fallback_reasons": dict(getattr(agents[arm], "fallback_reasons", {}) or {}),
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
                # Per-session outcomes so arms can be compared with paired
                # bootstrap intervals afterwards.
                "sessions": [
                    {key: row[key] for key in ("sample_id", "scenario_type", "hit", "first_hit_turn", "best_rank", "reciprocal_rank")}
                    for row in outcome["sessions"]
                ],
                "grounding_cache_misses": getattr(getattr(agents[arm], "llm", None), "misses", 0),
            }
            if args.traces:
                # Full trajectories: the simulator's message, the rewrite the
                # agent saw, what it asked and showed, and the grounding trace.
                # The evaluator creates one session per sample in sample order
                # and appends outcome rows in the same order, so the two align.
                rows = outcome["sessions"] if len(outcome["sessions"]) == len(wrapper.traces) else [{}] * len(wrapper.traces)
                trajectories[label] = {
                    session_id: {
                        "sample_id": row.get("sample_id"),
                        "scenario_type": row.get("scenario_type"),
                        "hit": row.get("hit"),
                        "first_hit_turn": row.get("first_hit_turn"),
                        "best_rank": row.get("best_rank"),
                        "turns": [
                            {
                                "turn": step["turn"],
                                "simulator_message": step["raw_message"],
                                "agent_saw": step["human_message"],
                                "ask_attribute": step["response"].get("ask_attribute"),
                                "message": step["response"].get("message"),
                                "recommendations": [rec.get("parent_asin") for rec in step["response"].get("recommendations", [])],
                                "usage": step["response"].get("usage"),
                                "grounding": step.get("grounding"),
                                "latency_ms": step["latency_ms"],
                            }
                            for step in trace
                        ],
                    }
                    for (session_id, trace), row in zip(wrapper.traces.items(), rows)
                }
            print(
                f"{label:<28} HR={outcome['hit_rate_at_10']:.4f} MRR={outcome['mrr']:.4f} "
                f"MTTC={outcome['mttc']:.3f} score={outcome['recommended_technical_score']:.6f} "
                f"tokens={outcome['reported_token_usage']['total_tokens']} "
                f"calls={experiments[label]['grounding']['model_calls']} wall={wall:.0f}s"
            )
        rewrites_by_level[level] = sample_rewrites(customer, 12)

    grounding_client = getattr(agents.get("hybrid"), "llm", None)
    result = {
        "status": "research_diagnostic_not_official_score",
        "caches": {
            "rewrites": args.cache,
            "grounding": grounding_cache,
            "strict_replay": bool(args.strict_replay),
            "grounding_cache_misses": getattr(grounding_client, "misses", 0),
        },
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
    if args.traces:
        traces_path = output.with_name(output.stem + ".traces.json")
        traces_path.write_text(json.dumps(trajectories, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        result["traces"] = str(traces_path)
        print(f"wrote {traces_path}")
    for external in agents.values():
        if isinstance(external, SubprocessAgent):
            external.close()


if __name__ == "__main__":
    main()
