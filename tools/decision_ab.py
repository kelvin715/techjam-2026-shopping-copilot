"""A/B the decision-layer additions on the unchanged public evaluator.

Usage:
    python3 tools/decision_ab.py --catalog /path/to/catalog.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import evaluator.local_evaluator as evaluator
from src import config
from starter.agent import Agent


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/decision_ab_metrics.json")
    args = parser.parse_args()

    samples = evaluator.load_jsonl(args.dataset)
    ids, categories, products = evaluator.catalog_index(args.catalog)
    original = (config.PROVEN_MISS_EXCLUSION, config.QUESTION_MODE)
    variants = (
        ("legacy_decision", False, "fixed"),
        ("proven_miss_only", True, "fixed"),
        ("proven_miss_plus_counterfactual", True, "counterfactual"),
    )
    results: dict[str, dict] = {}
    try:
        for label, exclude, question_mode in variants:
            config.PROVEN_MISS_EXCLUSION = exclude
            config.QUESTION_MODE = question_mode
            result = evaluator.evaluate(
                Agent(args.catalog), samples, ids, categories, products
            )
            results[label] = compact(result)
            print(
                f"{label:<36} HR={result['hit_rate_at_10']:.6f} "
                f"MRR={result['mrr']:.6f} MTTC={result['mttc']:.3f} "
                f"score={result['recommended_technical_score']:.6f}"
            )
    finally:
        config.PROVEN_MISS_EXCLUSION, config.QUESTION_MODE = original

    baseline = results["legacy_decision"]["technical_score"]
    results["deltas_vs_legacy"] = {
        label: round(value["technical_score"] - baseline, 6)
        for label, value in results.items()
        if isinstance(value, dict) and "technical_score" in value
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
