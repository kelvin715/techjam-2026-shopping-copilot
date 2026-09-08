"""Paired, scenario-stratified bootstrap intervals between attribution arms.

Reads results/attribution/*.json files that contain per-session outcomes and
writes results/attribution/intervals.json plus a markdown summary. A pair is
(level, shopper, arm A, arm B); the delta is B minus A in per-session score
units: 0.5*hit + 0.3*rr + 0.02*(11 - turn).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "attribution"


def utility(row: dict) -> float:
    turn = row["first_hit_turn"] if row["first_hit_turn"] is not None else 11
    return 0.5 * float(row["hit"]) + 0.3 * float(row["reciprocal_rank"]) + 0.02 * (11 - turn)


def paired(reference: list[dict], alternative: list[dict], seed: int = 20260906, repeats: int = 10000) -> dict:
    a = {r["sample_id"]: r for r in reference}
    b = {r["sample_id"]: r for r in alternative}
    ids = sorted(set(a) & set(b))
    delta = np.array([utility(b[i]) - utility(a[i]) for i in ids])
    scenarios = np.array([a[i]["scenario_type"] for i in ids])
    rng = np.random.default_rng(seed)
    boot = np.zeros(repeats)
    for scenario in sorted(set(scenarios)):
        values = delta[scenarios == scenario]
        boot += rng.choice(values, (repeats, len(values)), replace=True).sum(axis=1) / len(delta)
    return {"n": len(ids), "delta": float(delta.mean()),
            "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "changed_sessions": int(np.count_nonzero(np.abs(delta) > 1e-12))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None,
                        help="directory of benchmark outputs; default results/attribution/replay when present")
    parser.add_argument("--pairs", default="deterministic:lexical,lexical:hybrid,hybrid_unverified:hybrid,hybrid_flatweights:hybrid,hybrid_verbatim_only:hybrid,hybrid_nohints:hybrid,lexical:dense,hybrid:llm_agent,dense:hybrid")
    parser.add_argument("--output", default=None,
                        help="output stem; default <source>/intervals (.json and .md)")
    args = parser.parse_args()
    source = Path(args.source) if args.source else ((OUT / "replay") if (OUT / "replay").is_dir() else OUT)
    output = Path(args.output) if args.output else (source / "intervals" if args.source else OUT / "intervals")
    runs: dict[tuple, dict] = {}
    for path in sorted(source.glob("*_*_*.json")):
        if path.name.startswith(("grounding_", "summary", "intervals", "combined")) or path.name.endswith(".traces.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for exp in data["experiments"].values():
            if exp.get("sessions"):
                runs[(exp["level"], path.stem.split("_", 1)[0], exp["arm"])] = exp
    results = []
    for level, shopper, _ in sorted({(k[0], k[1], None) for k in runs}):
        for pair in args.pairs.split(","):
            ref_arm, alt_arm = pair.split(":")
            ref, alt = runs.get((level, shopper, ref_arm)), runs.get((level, shopper, alt_arm))
            if not ref or not alt:
                continue
            stats = paired(ref["sessions"], alt["sessions"])
            results.append({"level": level, "shopper": shopper, "reference": ref_arm, "alternative": alt_arm,
                            "reference_score": ref["technical_score"], "alternative_score": alt["technical_score"], **stats})
    output.with_suffix(".json").write_text(json.dumps({"rows": results}, indent=2) + "\n")
    lines = ["| level | shopper | A | B | score A | score B | Δ (B−A) | 95% CI | changed |", "|---|---|---|---|---:|---:|---:|---|---:|"]
    for r in results:
        lines.append(f"| {r['level']} | {r['shopper']} | {r['reference']} | {r['alternative']} | {r['reference_score']:.4f} | {r['alternative_score']:.4f} | {r['delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['changed_sessions']} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
