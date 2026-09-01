"""Calibrate the output gate on one proxy split and verify it on another.

The bounded loss is ``1 - reciprocal_rank`` (a miss has loss 1). Candidate
policies are selected with a finite-family Hoeffding correction, then evaluated
once on a target-disjoint held-out split. This is a proxy-distribution risk
control experiment, not a claim about organizer-private data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import evaluate
from src import config
from starter.agent import Agent
from tools.research_common import proxy_samples


@dataclass(frozen=True)
class GatePolicy:
    label: str
    mode: str
    late_turn: int
    margin_frac: float = 0.75
    k0: int = 1
    k1: int = 1
    k2: int = 1


POLICIES = (
    GatePolicy("count_late_4", "count", 4),
    GatePolicy("count_late_3", "count", 3),
    GatePolicy("count_late_5", "count", 5),
    GatePolicy("count_k2_2", "count", 4, k2=2),
    GatePolicy("margin_075", "margin", 4, margin_frac=0.75),
    GatePolicy("margin_090", "margin", 4, margin_frac=0.90),
)


def apply_policy(policy: GatePolicy) -> None:
    config.GATE_MODE = policy.mode
    config.EMIT_LATE_TURN = policy.late_turn
    config.MARGIN_FRAC = policy.margin_frac
    config.EMIT_K0 = policy.k0
    config.EMIT_K1 = policy.k1
    config.EMIT_K2 = policy.k2


def risk_summary(outcome: dict, delta: float) -> dict:
    sessions = outcome["sessions"]
    losses = [1.0 - float(row["reciprocal_rank"]) for row in sessions]
    n = len(losses)
    empirical = sum(losses) / n if n else 1.0
    radius = math.sqrt(math.log(1.0 / delta) / (2.0 * n)) if n else 1.0
    rank_one_coverage = (
        sum(row["best_rank"] == 1 for row in sessions) / n if n else 0.0
    )
    return {
        "sample_count": n,
        "empirical_rank_loss": round(empirical, 6),
        "hoeffding_radius": round(radius, 6),
        "risk_upper_bound": round(min(1.0, empirical + radius), 6),
        "rank_one_coverage": round(rank_one_coverage, 6),
        "hit_rate_at_10": outcome["hit_rate_at_10"],
        "mrr": outcome["mrr"],
        "mttc": outcome["mttc"],
        "technical_score": outcome["recommended_technical_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--calibration-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--output", default="results/risk_gate_calibration.json")
    args = parser.parse_args()

    samples, ids, categories, products = proxy_samples(
        args.catalog, args.dataset, "matched", args.count, args.seed
    )
    split = max(1, min(args.calibration_count, len(samples) - 1))
    calibration, test = samples[:split], samples[split:]
    calibration_targets = {
        row["ground_truth"]["parent_asin"] for row in calibration
    }
    test_targets = {row["ground_truth"]["parent_asin"] for row in test}
    original = (
        config.GATE_MODE,
        config.EMIT_LATE_TURN,
        config.MARGIN_FRAC,
        config.EMIT_K0,
        config.EMIT_K1,
        config.EMIT_K2,
    )
    calibration_delta = args.delta / len(POLICIES)
    candidates: dict[str, dict] = {}
    try:
        for policy in POLICIES:
            apply_policy(policy)
            outcome = evaluate(
                Agent(args.catalog), calibration, ids, categories, products
            )
            candidates[policy.label] = {
                "policy": asdict(policy),
                **risk_summary(outcome, calibration_delta),
            }
            row = candidates[policy.label]
            print(
                f"cal {policy.label:<14} risk={row['empirical_rank_loss']:.4f} "
                f"UCB={row['risk_upper_bound']:.4f} "
                f"score={row['technical_score']:.6f}"
            )
        feasible = [
            row for row in candidates.values()
            if row["risk_upper_bound"] <= args.alpha
        ]
        if feasible:
            selected = max(
                feasible,
                key=lambda row: (row["technical_score"], -row["empirical_rank_loss"]),
            )
            selection_status = "risk_constraint_satisfied"
        else:
            selected = min(
                candidates.values(), key=lambda row: row["risk_upper_bound"]
            )
            selection_status = "no_candidate_satisfied_alpha_selected_lowest_ucb"
        selected_policy = next(
            policy for policy in POLICIES
            if policy.label == selected["policy"]["label"]
        )
        baseline_policy = POLICIES[0]
        held_out: dict[str, dict] = {}
        for label, policy in (
            ("baseline", baseline_policy), ("selected", selected_policy)
        ):
            apply_policy(policy)
            outcome = evaluate(
                Agent(args.catalog), test, ids, categories, products
            )
            held_out[label] = {
                "policy": asdict(policy),
                **risk_summary(outcome, args.delta),
            }
            row = held_out[label]
            print(
                f"test {label:<10} risk={row['empirical_rank_loss']:.4f} "
                f"UCB={row['risk_upper_bound']:.4f} "
                f"score={row['technical_score']:.6f}"
            )
    finally:
        (
            config.GATE_MODE,
            config.EMIT_LATE_TURN,
            config.MARGIN_FRAC,
            config.EMIT_K0,
            config.EMIT_K1,
            config.EMIT_K2,
        ) = original

    result = {
        "status": "proxy_distribution_only_not_a_private_set_guarantee",
        "loss": "1 - reciprocal_rank; miss loss is 1",
        "bound": "Hoeffding upper confidence bound for bounded loss",
        "alpha": args.alpha,
        "delta": args.delta,
        "familywise_calibration_delta": calibration_delta,
        "target_disjoint": calibration_targets.isdisjoint(test_targets),
        "calibration_count": len(calibration),
        "test_count": len(test),
        "selection_status": selection_status,
        "selected_policy": selected["policy"],
        "calibration": candidates,
        "held_out": held_out,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
