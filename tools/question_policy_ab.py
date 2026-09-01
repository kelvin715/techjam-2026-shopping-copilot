"""A/B score-aligned clarification policies on an unchanged dataset.

This experiment isolates question selection: proven-miss exclusion and every
ranking/gating parameter stay fixed. ``answerable_metric_voi`` is promoted only
if it survives public, popularity-matched, and uniform-target validation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src import config
from starter.agent import Agent
from tools.research_common import proxy_samples


def compact(result: dict) -> dict:
    return {
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
        "scenario_metrics": result["scenario_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--proxy", choices=("public", "matched", "uniform"), default="public"
    )
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--modes",
        default="fixed,counterfactual,metric_voi,answerable_metric_voi",
    )
    parser.add_argument("--output", default="results/question_policy_ab_metrics.json")
    args = parser.parse_args()

    if args.proxy == "public":
        samples = load_jsonl(args.dataset)
        ids, categories, products = catalog_index(args.catalog)
    else:
        samples, ids, categories, products = proxy_samples(
            args.catalog, args.dataset, args.proxy, args.count, args.seed
        )
    original = config.QUESTION_MODE
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    results: dict[str, dict] = {}
    try:
        for mode in modes:
            config.QUESTION_MODE = mode
            outcome = evaluate(
                Agent(args.catalog), samples, ids, categories, products
            )
            results[mode] = compact(outcome)
            print(
                f"{mode:<24} HR={outcome['hit_rate_at_10']:.6f} "
                f"MRR={outcome['mrr']:.6f} MTTC={outcome['mttc']:.3f} "
                f"score={outcome['recommended_technical_score']:.6f}"
            )
    finally:
        config.QUESTION_MODE = original

    baseline_mode = modes[0]
    baseline = results[baseline_mode]["technical_score"]
    results[f"deltas_vs_{baseline_mode}"] = {
        mode: round(value["technical_score"] - baseline, 6)
        for mode, value in results.items()
        if isinstance(value, dict) and "technical_score" in value
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
