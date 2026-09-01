"""Measure reference startup, evaluation latency, memory, and token cost."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/runtime_profile.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    evaluator_start = time.perf_counter()
    ids, categories, products = catalog_index(args.catalog)
    evaluator_load_seconds = time.perf_counter() - evaluator_start

    agent_start = time.perf_counter()
    agent = Agent(args.catalog)
    agent_startup_seconds = time.perf_counter() - agent_start

    evaluation_start = time.perf_counter()
    outcome = evaluate(agent, samples, ids, categories, products)
    evaluation_seconds = time.perf_counter() - evaluation_start
    response_count = sum(
        int(row["first_hit_turn"] or 10) for row in outcome["sessions"]
    )
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result = {
        "status": "reference_environment_measurement_filesystem_cache_dependent",
        "sample_count": len(samples),
        "response_count": response_count,
        "evaluator_catalog_load_seconds": round(evaluator_load_seconds, 6),
        "agent_startup_seconds": round(agent_startup_seconds, 6),
        "evaluation_wall_seconds": round(evaluation_seconds, 6),
        "mean_evaluation_wall_ms_per_response": round(
            1000.0 * evaluation_seconds / response_count, 6
        ),
        "peak_evaluator_plus_agent_rss_kb": peak_rss_kb,
        "reported_token_usage": outcome["reported_token_usage"],
        "technical_score": outcome["recommended_technical_score"],
        "notes": [
            "Mean time includes simulator and evaluator overhead around Agent.respond.",
            "Peak RSS includes both evaluator and Agent catalog representations.",
            "Startup varies with CPU and filesystem cache; rerun on judging hardware.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
