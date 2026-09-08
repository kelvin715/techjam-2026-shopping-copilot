"""Cross-category generalisation: the same Agent on catalogs it never saw.

Every tunable in ``src/config.py`` was chosen on the organizer's
Clothing_Shoes_and_Jewelry catalog. This diagnostic runs the unchanged
organizer simulator and evaluator over 50,000-product catalogs built from
other Amazon Reviews 2023 categories (``tools/build_category_catalog.py``),
with uniformly sampled targets and the public set's user profiles, and
reports the same Hit@10 / MRR / MTTC / TechnicalScore per category next to
the clothing catalog under the identical protocol.

    python3 tools/cross_category.py --catalogs \\
        clothing=data/catalog.jsonl electronics=data/catalog_electronics.jsonl

Targets are eligible products (non-empty features and details), never a
public-set target, sampled without replacement with a fixed seed; the label
reaches only the evaluator. The per-session dumps go to
``results/cross_category/<name>_full.json`` (ignored) and the committed
summary to ``results/cross_category/summary.json``. A different category is
still the same simulator: this measures whether the design was tuned to
clothing, not whether it survives a different interaction protocol.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from tools.matched_proxy import percentile, rating_number, scenario_sequence


def _departments(catalog_path: str) -> dict[str, str]:
    """``asin -> department`` when the catalog was assembled from several."""
    sidecar = Path(catalog_path).with_suffix(".departments.json")
    if not sidecar.is_file():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8")).get("department_of", {})


def _metrics(rows: list[dict]) -> dict:
    """Evaluator metrics for a subset of per-session outcome rows."""
    if not rows:
        return {}
    hit = sum(1 for row in rows if row["hit"]) / len(rows)
    mrr = sum(float(row["reciprocal_rank"]) for row in rows) / len(rows)
    mttc = sum((row["first_hit_turn"] if row["first_hit_turn"] is not None else 11) for row in rows) / len(rows)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {"sample_count": len(rows), "hit_rate_at_10": round(hit, 6), "mrr": round(mrr, 6), "mttc": round(mttc, 6),
            "technical_score": round(0.5 * hit + 0.3 * mrr + 0.2 * efficiency, 6)}


def run_catalog(name: str, catalog_path: str, profiles: list[dict], public_targets: set[str],
                count: int, seed: int, out_dir: Path, dump_sessions: bool = False) -> dict:
    ids, categories, products = catalog_index(catalog_path)
    department_of = _departments(catalog_path)
    pool = sorted(
        asin for asin, product in products.items()
        if asin not in public_targets and product.get("features") and product.get("details")
    )
    rng = random.Random(seed)
    if department_of:
        # Stratify: an equal number of targets per department, so a large
        # department cannot hide a weak one.
        by_department: dict[str, list[str]] = defaultdict(list)
        for asin in pool:
            by_department[department_of.get(asin, "unknown")].append(asin)
        names = sorted(by_department)
        quota, remainder = divmod(count, len(names))
        targets = []
        for index, department in enumerate(names):
            take = quota + (1 if index < remainder else 0)
            targets.extend(rng.sample(by_department[department], min(take, len(by_department[department]))))
        rng.shuffle(targets)
    else:
        targets = rng.sample(pool, min(count, len(pool)))
    scenarios = scenario_sequence(len(targets), rng)
    samples = [
        {
            "sample_id": f"cross_{name}_{position:05d}",
            "scenario_type": scenario,
            "category_bucket": department_of.get(asin, name),
            "difficulty_bucket": difficulty,
            "user_profile": rng.choice(profiles),
            "ground_truth": {"parent_asin": asin},
        }
        for position, (asin, (scenario, difficulty)) in enumerate(zip(targets, scenarios), start=1)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    if dump_sessions:
        sessions_path = out_dir / f"{name}_sessions.jsonl"
        sessions_path.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        print(f"wrote {sessions_path}", flush=True)
    started = time.perf_counter()
    agent = Agent(catalog_path)
    startup = time.perf_counter() - started
    started = time.perf_counter()
    outcome = evaluate(agent, samples, ids, categories, products)
    wall = time.perf_counter() - started
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    counts = [rating_number(products[asin]) for asin in targets]
    shelves = Counter(agent.catalog.shelf_of.values()) if hasattr(agent, "catalog") else Counter()
    collisions = None
    if department_of and hasattr(agent, "catalog"):
        # Shelves whose products come from more than one department: the
        # simulator's last-two-levels category rule merges them.
        shelf_departments: dict[str, set[str]] = defaultdict(set)
        for asin, shelf in agent.catalog.shelf_of.items():
            shelf_departments[shelf].add(department_of.get(asin, "unknown"))
        mixed_shelves = {shelf for shelf, deps in shelf_departments.items() if len(deps) > 1}
        collisions = {
            "shelves_spanning_departments": len(mixed_shelves),
            "products_on_such_shelves": sum(shelves[shelf] for shelf in mixed_shelves),
            "examples": sorted(mixed_shelves, key=lambda shelf: -shelves[shelf])[:8],
        }
    per_department = None
    if department_of:
        target_of = {sample["sample_id"]: sample["ground_truth"]["parent_asin"] for sample in samples}
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in outcome["sessions"]:
            grouped[department_of.get(target_of.get(row["sample_id"], ""), "unknown")].append(row)
        per_department = {department: _metrics(rows) for department, rows in sorted(grouped.items())}
    signature_lengths = [len(agent.catalog.signature.get(asin, ())) for asin in ids] if hasattr(agent, "catalog") else []
    priced = sum(1 for asin in ids if products[asin].get("price") not in (None, ""))
    summary = {
        "name": name,
        "catalog": catalog_path,
        "catalog_rows": len(ids),
        "eligible_targets": len(pool),
        "sample_count": outcome["sample_count"],
        "seed": seed,
        "catalog_profile": {
            "shelves": len(shelves),
            "largest_shelf": shelves.most_common(1)[0][1] if shelves else None,
            "median_shelf_size": percentile(list(shelves.values()), 0.5) if shelves else None,
            "priced_fraction": round(priced / max(1, len(ids)), 4),
            "mean_signature_slots": round(statistics.fmean(signature_lengths), 3) if signature_lengths else None,
            "target_median_rating_number": percentile(counts, 0.5),
        },
        "hit_rate_at_10": outcome["hit_rate_at_10"],
        "mrr": outcome["mrr"],
        "mttc": outcome["mttc"],
        "technical_score": outcome["recommended_technical_score"],
        "scenario_metrics": outcome["scenario_metrics"],
        "reported_token_usage": outcome["reported_token_usage"],
        "agent_startup_seconds": round(startup, 2),
        "evaluation_wall_seconds": round(wall, 2),
        "mean_wall_ms_per_response": round(1000.0 * wall / max(1, sum(
            (row["first_hit_turn"] if row["first_hit_turn"] is not None else 10) for row in outcome["sessions"]
        )), 2),
        "peak_rss_kb": peak_rss_kb,
        "shelf_collisions": collisions,
        "per_department": per_department,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_full.json").write_text(
        json.dumps({**summary, "sessions": outcome["sessions"]}, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogs", nargs="+", default=["clothing=data/catalog.jsonl"],
                        help="NAME=PATH pairs; the clothing catalog is the reference")
    parser.add_argument("--dataset", default="data/public_set.jsonl", help="public set, for user profiles and excluded targets")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--out-dir", default="results/cross_category")
    parser.add_argument("--dump-sessions", action="store_true",
                        help="also write <name>_sessions.jsonl so the language benchmark can replay the same targets")
    args = parser.parse_args()

    public = load_jsonl(args.dataset)
    public_targets = {str(row["ground_truth"]["parent_asin"]) for row in public}
    profiles = [row["user_profile"] for row in public]
    out_dir = ROOT / args.out_dir
    summary_path = out_dir / "summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"catalogs": {}}
    for spec in args.catalogs:
        name, _, path = spec.partition("=")
        if not name or not path:
            parser.error(f"--catalogs expects NAME=PATH, got {spec!r}")
        print(f"== {name}: {path}", flush=True)
        result = run_catalog(name, path, profiles, public_targets, args.count, args.seed, out_dir, args.dump_sessions)
        existing["catalogs"][name] = result
        print(json.dumps({key: result[key] for key in ("catalog_rows", "hit_rate_at_10", "mrr", "mttc", "technical_score", "catalog_profile")}, indent=2), flush=True)
    existing["status"] = "diagnostic_only_same_simulator_different_catalog"
    existing["sampling"] = "uniform_without_replacement_from_eligible_targets; public-set profiles; public targets excluded"
    existing["count"] = args.count
    existing["seed"] = args.seed
    summary_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    rows = ["| Catalog | Products | Shelves | Priced | Hit@10 | MRR | MTTC | TechnicalScore |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in existing["catalogs"].items():
        profile = result["catalog_profile"]
        rows.append(
            f"| {name} | {result['catalog_rows']:,} | {profile['shelves']} | {profile['priced_fraction']:.0%} | "
            f"{result['hit_rate_at_10']:.4f} | {result['mrr']:.4f} | {result['mttc']:.3f} | {result['technical_score']:.6f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
