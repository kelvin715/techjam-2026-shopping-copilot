"""Trace declared proxy cases without exposing labels to the Agent runtime."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import evaluate, intent_card
from starter.agent import Agent
from tools.research_common import proxy_samples
from src.rank import score_candidates


class TraceAgent:
    def __init__(self, catalog_path: str) -> None:
        self.agent = Agent(catalog_path)
        self.traces: dict[str, list[dict]] = defaultdict(list)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = self.agent.respond(session_id, user_message, turn, top_k)
        state = self.agent._sessions[session_id]
        self.traces[session_id].append({
            "turn": turn,
            "user_message": user_message,
            "response": deepcopy(response),
            "constraints": list(state.constraints),
            "exhausted": sorted(state.exhausted),
            "proven_misses": sorted(state.proven_misses),
            "certificate": deepcopy(state.last_decision_certificate),
        })
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--proxy", choices=("matched", "uniform"), default="uniform")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--output", default="results/failure_trace.json")
    args = parser.parse_args()

    samples, ids, categories, products = proxy_samples(
        args.catalog, args.dataset, args.proxy, args.count, args.seed
    )
    requested = {value.strip() for value in args.sample_ids.split(",") if value.strip()}
    selected = [row for row in samples if row["sample_id"] in requested]
    missing = requested.difference(row["sample_id"] for row in selected)
    if missing:
        raise SystemExit(f"unknown sample ids: {sorted(missing)}")
    wrapper = TraceAgent(args.catalog)
    outcome = evaluate(wrapper, selected, ids, categories, products)
    trace_items = list(wrapper.traces.items())
    cases = []
    for sample, session, (session_id, trace) in zip(
        selected, outcome["sessions"], trace_items
    ):
        target = str(sample["ground_truth"]["parent_asin"])
        state = wrapper.agent._sessions[session_id]
        evidence_scores = score_candidates(
            wrapper.agent.catalog,
            wrapper.agent.catalog.candidates(state.shelf),
            state.constraints,
            state.profile_tags,
            state.constraint_weights(),
            state.scenario,
            popularity_weight=0.0,
            signature_positions_reliable=state.signature_positions_reliable,
        )
        evidence_lookup = dict(evidence_scores)
        target_evidence = evidence_lookup.get(target)
        best_evidence = evidence_scores[0][1] if evidence_scores else None
        target_evidence_rank = next(
            (index for index, (asin, _) in enumerate(evidence_scores, 1)
             if asin == target),
            None,
        )
        evidence_tie_count = (
            sum(abs(score - best_evidence) <= 1e-9 for _, score in evidence_scores)
            if best_evidence is not None
            else 0
        )
        cases.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "target_parent_asin": target,
            "target_title": products[target].get("title"),
            "target_categories": products[target].get("categories"),
            "target_rating_number": products[target].get("rating_number"),
            "derived_intent_card": intent_card(products[target]),
            "offline_target_diagnostic": {
                "evidence_rank": target_evidence_rank,
                "evidence_score": target_evidence,
                "best_evidence_score": best_evidence,
                "best_evidence_tie_count": evidence_tie_count,
                "best_minus_target": (
                    best_evidence - target_evidence
                    if best_evidence is not None and target_evidence is not None
                    else None
                ),
            },
            "outcome": session,
            "trace": trace,
        })
    result = {
        "status": "offline_diagnostic_labels_never_passed_to_agent",
        "proxy": args.proxy,
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print(
            f"\n{case['sample_id']} {case['scenario_type']} "
            f"target={case['target_parent_asin']} hit={case['outcome']['hit']}"
        )
        print(f"  card={case['derived_intent_card']}")
        print(f"  diagnostic={case['offline_target_diagnostic']}")
        for turn in case["trace"]:
            recs = [
                item["parent_asin"]
                for item in turn["response"].get("recommendations", [])
            ]
            selected_value = turn["certificate"].get("selected_question_value") or {}
            print(
                f"  T{turn['turn']} ask={turn['response'].get('ask_attribute')} "
                f"constraints={turn['constraints']} target_shown="
                f"{case['target_parent_asin'] in recs} "
                f"answerability={selected_value.get('answerability')}"
            )


if __name__ == "__main__":
    main()
