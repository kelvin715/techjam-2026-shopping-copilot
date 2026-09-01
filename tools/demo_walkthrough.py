"""Produce a judge-friendly old-versus-new conversational walkthrough.

The comparison uses public labels only for offline evaluation and storytelling;
the Agent never receives the target ASIN. Both arms see the same naturally
paraphrased simulator messages. The old arm disables the additive robust parser,
while the submitted arm enables it. The tool automatically selects a session
that the old arm misses and the submitted arm recovers.
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
from tools.robustness_policy_benchmark import PerturbingAgent


def compact_metrics(outcome: dict) -> dict:
    return {
        "sample_count": outcome["sample_count"],
        "hit_rate_at_10": outcome["hit_rate_at_10"],
        "mrr": outcome["mrr"],
        "mttc": outcome["mttc"],
        "technical_score": outcome["recommended_technical_score"],
    }


def simulator_equivalent_impact(
    old_metrics: dict,
    new_metrics: dict,
    old_response_count: int,
    new_response_count: int,
) -> dict:
    """Translate measured deltas to a transparent 10k-session scale."""
    count = int(new_metrics["sample_count"])
    old_calls = old_response_count / count
    new_calls = new_response_count / count
    return {
        "scope": "simulator_equivalent_not_observed_business_conversion",
        "additional_hit_sessions_per_10k": round(
            10000
            * (new_metrics["hit_rate_at_10"] - old_metrics["hit_rate_at_10"])
        ),
        "average_agent_calls_old": round(old_calls, 4),
        "average_agent_calls_new": round(new_calls, 4),
        "fewer_agent_calls_per_session": round(old_calls - new_calls, 4),
        "mrr_delta": round(new_metrics["mrr"] - old_metrics["mrr"], 6),
        "technical_score_delta": round(
            new_metrics["technical_score"] - old_metrics["technical_score"], 6
        ),
    }


def trace_for_story(trace: list[dict], target: str) -> list[dict]:
    turns: list[dict] = []
    for call in trace:
        response = call["response"]
        asins = [
            str(item.get("parent_asin", ""))
            for item in response.get("recommendations", [])
            if isinstance(item, dict)
        ]
        turns.append({
            "turn": call["turn"],
            "customer": call["transformed_message"],
            "agent_message": response.get("message"),
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": asins,
            "target_rank": asins.index(target) + 1 if target in asins else None,
        })
    return turns


def selected_certificate(wrapper: PerturbingAgent, session_id: str) -> dict:
    certificate = wrapper.explain_last_decision(session_id)
    return {
        "action": certificate.get("action"),
        "reason_codes": certificate.get("reason_codes"),
        "intent_track": certificate.get("intent_track"),
        "constraints": certificate.get("constraints"),
        "proven_miss_count": certificate.get("proven_miss_count"),
        "question": certificate.get("question"),
        "selected_question_value": certificate.get("selected_question_value"),
        "output_gate": certificate.get("output_gate"),
        "recommendations": certificate.get("recommendations"),
        "minimal_counterfactual_explanation": certificate.get(
            "minimal_counterfactual_explanation"
        ),
    }


def run_arm(
    catalog_path: str,
    samples: list[dict],
    ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    robust_parser: bool,
) -> tuple[PerturbingAgent, dict]:
    original = config.ROBUST_PARSER
    try:
        config.ROBUST_PARSER = robust_parser
        wrapper = PerturbingAgent(catalog_path, "natural_paraphrase")
        outcome = evaluate(wrapper, samples, ids, categories, products)
    finally:
        config.ROBUST_PARSER = original
    return wrapper, outcome


def _display_trace(label: str, turns: list[dict]) -> None:
    print(f"\n{label}")
    visible = turns if len(turns) <= 4 else [*turns[:3], turns[-1]]
    for turn in visible:
        suffix = "" if turn["target_rank"] is None else f" TARGET@{turn['target_rank']}"
        print(f"  T{turn['turn']} USER: {turn['customer']}")
        print(
            f"     AGENT ask={turn['ask_attribute']} "
            f"recs={turn['recommendations']}{suffix}"
        )
    if len(turns) > len(visible):
        print(f"     ... {len(turns) - len(visible)} middle turns omitted ...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", default="results/demo_walkthrough.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)[: args.count]
    ids, categories, products = catalog_index(args.catalog)
    old_wrapper, old_outcome = run_arm(
        args.catalog, samples, ids, categories, products, False
    )
    new_wrapper, new_outcome = run_arm(
        args.catalog, samples, ids, categories, products, True
    )
    old_traces = list(old_wrapper.traces.items())
    new_traces = list(new_wrapper.traces.items())
    candidates = [
        index
        for index, (old_session, new_session) in enumerate(
            zip(old_outcome["sessions"], new_outcome["sessions"])
        )
        if not old_session["hit"] and new_session["hit"]
    ]
    if not candidates:
        candidates = [
            index
            for index, (old_session, new_session) in enumerate(
                zip(old_outcome["sessions"], new_outcome["sessions"])
            )
            if new_session["reciprocal_rank"] > old_session["reciprocal_rank"]
        ]
    if not candidates:
        raise SystemExit("no improved session found in the selected sample prefix")
    selected_index = min(
        candidates,
        key=lambda index: (
            new_outcome["sessions"][index]["first_hit_turn"] or 11,
            -(new_outcome["sessions"][index]["reciprocal_rank"]),
            index,
        ),
    )
    sample = samples[selected_index]
    target = str(sample["ground_truth"]["parent_asin"])
    old_session_id, old_trace = old_traces[selected_index]
    new_session_id, new_trace = new_traces[selected_index]
    old_metrics = compact_metrics(old_outcome)
    new_metrics = compact_metrics(new_outcome)
    impact = simulator_equivalent_impact(
        old_metrics,
        new_metrics,
        sum(len(trace) for trace in old_wrapper.traces.values()),
        sum(len(trace) for trace in new_wrapper.traces.values()),
    )
    result = {
        "status": "public_labelled_demo_not_private_evaluation",
        "comparison": (
            "same natural paraphrases; canonical-only parser versus submitted "
            "additive robust parser"
        ),
        "old": {
            "label": "canonical_only_parser",
            "metrics": old_metrics,
        },
        "submitted": {
            "label": "answerability_mvoi_plus_robust_parser",
            "metrics": new_metrics,
        },
        "simulator_equivalent_impact": impact,
        "story": {
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "target_parent_asin": target,
            "target_title": products[target].get("title"),
            "old_outcome": old_outcome["sessions"][selected_index],
            "submitted_outcome": new_outcome["sessions"][selected_index],
            "old_trace": trace_for_story(old_trace, target),
            "submitted_trace": trace_for_story(new_trace, target),
            "submitted_certificate": selected_certificate(
                new_wrapper, new_session_id
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("=== NATURAL-PARAPHRASE SCORECARD ===")
    print(
        f"OLD        HR={old_metrics['hit_rate_at_10']:.4f} "
        f"MRR={old_metrics['mrr']:.4f} MTTC={old_metrics['mttc']:.2f} "
        f"score={old_metrics['technical_score']:.6f}"
    )
    print(
        f"SUBMITTED  HR={new_metrics['hit_rate_at_10']:.4f} "
        f"MRR={new_metrics['mrr']:.4f} MTTC={new_metrics['mttc']:.2f} "
        f"score={new_metrics['technical_score']:.6f}"
    )
    print(
        "10K-SCALE (simulator-equivalent): "
        f"+{impact['additional_hit_sessions_per_10k']} hits, "
        f"-{impact['fewer_agent_calls_per_session']:.2f} calls/session"
    )
    print(
        f"\n=== RECOVERED STORY: {sample['sample_id']} / {sample['scenario_type']} ==="
    )
    print(f"TARGET: {target} | {products[target].get('title')}")
    _display_trace("OLD: constraint wording is lost", trace_for_story(old_trace, target))
    _display_trace(
        "SUBMITTED: evidence is grounded and target is recovered",
        trace_for_story(new_trace, target),
    )
    explanation = result["story"]["submitted_certificate"][
        "minimal_counterfactual_explanation"
    ]
    print(
        "\nAUDIT: "
        f"action={result['story']['submitted_certificate']['action']} "
        f"counterfactual={explanation.get('status')} "
        f"faithful={explanation.get('faithful')}"
    )
    print(f"Full machine-readable story: {output}")


if __name__ == "__main__":
    main()
