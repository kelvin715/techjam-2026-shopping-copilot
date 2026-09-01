"""Ablate each reranking layer on the unchanged public evaluator.

Usage:
    python3 tools/structured_ab.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import evaluator.local_evaluator as evaluator
from src import config
from starter.agent import Agent


def show(label: str, result: dict) -> None:
    ranks = collections.Counter(
        row["best_rank"] if row["best_rank"] is not None else "miss"
        for row in result["sessions"]
    )
    print(
        f"{label:<22} HR {result['hit_rate_at_10']:.4f}  "
        f"MRR {result['mrr']:.6f}  MTTC {result['mttc']:.3f}  "
        f"SCORE {result['recommended_technical_score']:.6f}  "
        f"rank1={ranks[1]} miss={ranks['miss']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/structured_ab_metrics.json")
    args = parser.parse_args()
    catalog = args.catalog
    samples = evaluator.load_jsonl(args.dataset)
    ids, categories, products = evaluator.catalog_index(catalog)
    agent = Agent(catalog)
    original = (
        config.TYPED_WEIGHT,
        config.SIGNATURE_WEIGHT,
        config.POPULARITY_WEIGHT,
    )
    stages = (
        ("lexical + policy", 0.0, 0.0, 0.0),
        ("+ typed", 0.3, 0.0, 0.0),
        ("+ canonical signature", 0.3, 0.7, 0.0),
        ("+ bounded prior", 0.3, 0.7, 0.55),
    )
    results: dict[str, dict] = {}
    try:
        for label, typed, signature, popularity in stages:
            config.TYPED_WEIGHT = typed
            config.SIGNATURE_WEIGHT = signature
            config.POPULARITY_WEIGHT = popularity
            result = evaluator.evaluate(agent, samples, ids, categories, products)
            show(label, result)
            results[label] = {
                "hit_rate_at_10": result["hit_rate_at_10"],
                "mrr": result["mrr"],
                "mttc": result["mttc"],
                "technical_score": result["recommended_technical_score"],
            }
    finally:
        (
            config.TYPED_WEIGHT,
            config.SIGNATURE_WEIGHT,
            config.POPULARITY_WEIGHT,
        ) = original
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
