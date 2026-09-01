"""Deterministic popularity-matched held-out diagnostic.

This is not an estimate produced from organizer-private data. It synthesizes
sessions for non-public catalog products using the unchanged public evaluator,
the official scenario mix, resampled public profiles, and a target popularity
distribution matched to the public targets.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src import config
from starter.agent import Agent


SCENARIOS = (
    ("buying", 0.40, "easy"),
    ("browsing", 0.40, "medium"),
    ("intent_override", 0.15, "hard"),
    ("boundary", 0.05, "medium"),
)


def rating_number(product: dict) -> int:
    try:
        return max(0, int(product.get("rating_number") or 0))
    except (TypeError, ValueError):
        return 0


def matched_targets(
    products: dict[str, dict],
    pool: list[str],
    reference: list[int],
    rng: random.Random,
    count: int,
) -> list[str]:
    by_count = sorted((rating_number(products[asin]), asin) for asin in pool)
    counts = [value for value, _ in by_count]
    chosen: list[str] = []
    used: set[str] = set()
    while len(chosen) < min(count, len(pool)):
        target_count = reference[rng.randrange(len(reference))]
        start = bisect.bisect_left(counts, target_count)
        for offset in range(len(by_count)):
            for index in (start + offset, start - offset):
                if 0 <= index < len(by_count):
                    asin = by_count[index][1]
                    if asin not in used:
                        used.add(asin)
                        chosen.append(asin)
                        break
            else:
                continue
            break
    rng.shuffle(chosen)
    return chosen


def scenario_sequence(count: int, rng: random.Random) -> list[tuple[str, str]]:
    raw = [(name, difficulty, count * share) for name, share, difficulty in SCENARIOS]
    amounts = {name: int(value) for name, _, value in raw}
    shortfall = count - sum(amounts.values())
    for name, _, value in sorted(raw, key=lambda item: -(item[2] - int(item[2]))):
        if shortfall == 0:
            break
        amounts[name] += 1
        shortfall -= 1
    difficulty_by_name = {name: difficulty for name, difficulty, _ in raw}
    sequence = [
        (name, difficulty_by_name[name])
        for name, _, _ in SCENARIOS
        for _ in range(amounts[name])
    ]
    rng.shuffle(sequence)
    return sequence


def percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank quartile, with the lower middle value for an even-size median."""
    ordered = sorted(values)
    if fraction == 0.5:
        return ordered[(len(ordered) - 1) // 2]
    return ordered[round(fraction * (len(ordered) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--popularity-weight", type=float, default=None)
    parser.add_argument("--output", default="results/matched_proxy_full.json")
    args = parser.parse_args()

    public = load_jsonl(args.dataset)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [row["user_profile"] for row in public]
    ids, categories, products = catalog_index(args.catalog)
    pool = sorted(
        asin
        for asin, product in products.items()
        if asin not in public_targets and product.get("features") and product.get("details")
    )
    reference = sorted(
        rating_number(products[asin])
        for asin in sorted(public_targets)
        if asin in products
    )
    rng = random.Random(args.seed)
    targets = matched_targets(products, pool, reference, rng, args.count)
    scenarios = scenario_sequence(len(targets), rng)
    samples = []
    for position, (asin, (scenario, difficulty)) in enumerate(
        zip(targets, scenarios), start=1
    ):
        samples.append({
            "sample_id": f"proxy_matched_{position:05d}",
            "scenario_type": scenario,
            "category_bucket": "clothing",
            "difficulty_bucket": difficulty,
            "user_profile": rng.choice(profiles),
            "ground_truth": {"parent_asin": asin},
        })

    original_weight = config.POPULARITY_WEIGHT
    if args.popularity_weight is not None:
        config.POPULARITY_WEIGHT = args.popularity_weight
    try:
        outcome = evaluate(
            Agent(args.catalog), samples, ids, categories, products
        )
    finally:
        config.POPULARITY_WEIGHT = original_weight

    counts = [rating_number(products[asin]) for asin in targets]
    summary = {
        "status": "diagnostic_only_not_an_official_private_score",
        "sample_count": outcome["sample_count"],
        "seed": args.seed,
        "popularity_weight": (
            original_weight if args.popularity_weight is None else args.popularity_weight
        ),
        "target_popularity": {
            "median_rating_number": percentile(counts, 0.50),
            "p25_rating_number": percentile(counts, 0.25),
            "p75_rating_number": percentile(counts, 0.75),
        },
        "hit_rate_at_10": outcome["hit_rate_at_10"],
        "mrr": outcome["mrr"],
        "mttc": outcome["mttc"],
        "technical_score": outcome["recommended_technical_score"],
        "scenario_metrics": outcome["scenario_metrics"],
    }
    full = {**summary, "sessions": outcome["sessions"]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
