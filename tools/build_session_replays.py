"""Record compact evaluator-side traces for every public session.

The submitted Agent receives exactly the inputs used by the unchanged local
evaluator. Ground-truth ASINs are joined only after each response returns, so
the static Session Explorer can show rank and hit correctness without putting
labels on the Agent path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent
from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)


class RecordingAgent:
    """Capture Agent-side inputs and outputs without receiving a target label."""

    def __init__(self, catalog_path: Path) -> None:
        self.agent = Agent(catalog_path)
        self.session_order: list[str] = []
        self.traces: dict[str, list[dict]] = defaultdict(list)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_order.append(session_id)
        self.agent.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        response = self.agent.respond(session_id, user_message, turn, top_k)
        # This certificate is created inside Agent.respond from observed state.
        # Accessing it here avoids invoking the lazy evaluator-side
        # counterfactual and keeps the 200-session build compact.
        state = self.agent._sessions[session_id]  # noqa: SLF001 - build tool
        self.traces[session_id].append({
            "turn": turn,
            "customer": user_message,
            "response": deepcopy(response),
            "certificate": _compact_certificate(
                deepcopy(state.last_decision_certificate)
            ),
        })
        return response


def _compact_certificate(certificate: dict) -> dict:
    selected = certificate.get("selected_question_value") or {}
    gate = certificate.get("output_gate") or {}
    return {
        "action": certificate.get("action"),
        "reason_codes": list(certificate.get("reason_codes") or []),
        "intent_track": certificate.get("intent_track"),
        "shelf": certificate.get("shelf"),
        "constraints": list(certificate.get("constraints") or []),
        "constraint_confidence": list(
            certificate.get("constraint_confidence") or []
        ),
        "source_candidate_count": certificate.get("source_candidate_count"),
        "ranked_candidate_count": certificate.get("ranked_candidate_count"),
        "plausible_candidate_count": certificate.get(
            "plausible_candidate_count"
        ),
        "proven_miss_count": certificate.get("proven_miss_count"),
        "top2_absolute_margin": certificate.get("top2_absolute_margin"),
        "question": certificate.get("question"),
        "selected_question_value": {
            key: selected.get(key)
            for key in (
                "attribute",
                "answerability",
                "expected_hit_at_10",
                "expected_mrr",
                "answerable_metric_voi",
            )
            if key in selected
        } or None,
        "output_gate": {
            "policy": gate.get("policy"),
            "emitted_count": gate.get("emitted_count"),
        },
    }


def _short_title(value: object, limit: int = 92) -> str:
    title = " ".join(str(value or "Product").split())
    if len(title) <= limit:
        return title
    prefix = title[: limit - 1].rsplit(" ", 1)[0]
    return f"{prefix}…"


def _product(product: dict) -> dict:
    raw_price = product.get("price")
    try:
        price = float(raw_price) if raw_price not in (None, "") else None
    except (TypeError, ValueError):
        price = None
    return {
        "parent_asin": str(product.get("parent_asin") or ""),
        "title": _short_title(product.get("title")),
        "store": str(product.get("store") or "Independent seller"),
        "price": price,
    }


def _session_contribution(outcome: dict) -> float:
    hit = float(bool(outcome["hit"]))
    reciprocal_rank = float(outcome["reciprocal_rank"])
    scored_turn = outcome["first_hit_turn"] or 11
    efficiency = max(0.0, min(1.0, (11.0 - float(scored_turn)) / 10.0))
    return round(0.50 * hit + 0.30 * reciprocal_rank + 0.20 * efficiency, 6)


def _assert_matches_expected(result: dict, expected_path: Path | None) -> None:
    if expected_path is None or not expected_path.is_file():
        return
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    for key in (
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "recommended_technical_score",
    ):
        if result.get(key) != expected.get(key):
            raise ValueError(
                f"replay build disagrees with {expected_path}: "
                f"{key}={result.get(key)} expected {expected.get(key)}"
            )
    actual_sessions = {row["sample_id"]: row for row in result["sessions"]}
    for expected_session in expected["sessions"]:
        actual = actual_sessions.get(expected_session["sample_id"])
        if actual != expected_session:
            raise ValueError(
                "replay session disagrees with the recorded public result: "
                f"{expected_session['sample_id']}"
            )


def build_session_replays(
    catalog_path: Path,
    dataset_path: Path,
    expected_path: Path | None = None,
) -> dict:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    wrapper = RecordingAgent(catalog_path)
    result = evaluate(wrapper, samples, catalog_ids, categories, products)
    _assert_matches_expected(result, expected_path)

    sessions: list[dict] = []
    product_ids: set[str] = set()
    for sample, outcome, session_id in zip(
        samples, result["sessions"], wrapper.session_order
    ):
        target = str(sample["ground_truth"]["parent_asin"])
        _, behavior = materialize_hidden_fields(sample, products)
        override = behavior.get("override") or {}
        override_turn = (
            int(override.get("turn", 3))
            if sample["scenario_type"] == "intent_override"
            else None
        )
        turns: list[dict] = []
        for recorded in wrapper.traces[session_id]:
            response = recorded["response"]
            recommendations = normalize_recommendations(
                response.get("recommendations"), catalog_ids
            )
            product_ids.update(recommendations)
            target_rank = (
                recommendations.index(target) + 1
                if target in recommendations else None
            )
            eligible = override_turn is None or recorded["turn"] >= override_turn
            usage = response.get("usage") or {}
            turns.append({
                "turn": recorded["turn"],
                "customer": recorded["customer"],
                "agent_message": response.get("message", ""),
                "ask_attribute": response.get("ask_attribute"),
                "recommendations": recommendations,
                "reported_tokens": int(usage.get("prompt_tokens") or 0)
                + int(usage.get("completion_tokens") or 0),
                "eligible_for_hit": eligible,
                "target_rank": target_rank,
                "hit_this_turn": bool(eligible and target_rank is not None),
                "decision": recorded["certificate"],
            })
        hit_turns = [
            turn for turn in turns if turn["hit_this_turn"]
        ]
        computed_turn = hit_turns[0]["turn"] if hit_turns else None
        computed_rank = hit_turns[0]["target_rank"] if hit_turns else None
        if (
            computed_turn != outcome["first_hit_turn"]
            or computed_rank != outcome["best_rank"]
        ):
            raise ValueError(
                f"post-hoc target join disagrees for {sample['sample_id']}"
            )
        product_ids.add(target)
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "category_bucket": sample.get("category_bucket"),
            "target_parent_asin": target,
            "target_visibility": "evaluator_only_never_sent_to_agent",
            "override_turn": override_turn,
            "outcome": {
                **outcome,
                "technical_score_contribution": _session_contribution(outcome),
            },
            "turns": turns,
        })

    rank_distribution = Counter(
        str(row["outcome"]["best_rank"] or "miss") for row in sessions
    )
    turn_distribution = Counter(
        str(row["outcome"]["first_hit_turn"] or "miss") for row in sessions
    )
    return {
        "status": "verified_public_evaluator_replays",
        "message_source": "unchanged organizer evaluator, canonical reveal",
        "target_join": "post_response_evaluator_only",
        "sample_count": result["sample_count"],
        "metrics": {
            key: result[key]
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "recommended_technical_score",
            )
        },
        "summary": {
            "rank_distribution": dict(sorted(rank_distribution.items())),
            "turn_distribution": dict(sorted(turn_distribution.items())),
        },
        "products": {
            asin: _product(products[asin]) for asin in sorted(product_ids)
        },
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "data/catalog.jsonl"
    )
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/public_set.jsonl"
    )
    parser.add_argument(
        "--expected", type=Path, default=ROOT / "results/public_full.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/public_session_replays.json",
    )
    args = parser.parse_args()
    replay = build_session_replays(args.catalog, args.dataset, args.expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
    ranks = replay["summary"]["rank_distribution"]
    print(
        f"wrote {args.output} | sessions={replay['sample_count']} "
        f"rank1={ranks.get('1', 0)} rank3={ranks.get('3', 0)}"
    )


if __name__ == "__main__":
    main()
