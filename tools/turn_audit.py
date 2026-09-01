"""Research-only audit of where evaluator turns are actually spent.

Replays the organizer's simulator loop while capturing the agent's complete
ranked list each turn, then reports, per session, the earliest turn at which the
target was already rank 1 versus the turn the emit gate actually exposed it.

The split answers the only question that matters once Hit@10 and MRR are
saturated: is MTTC bounded by the ranker or by the output gate? Labels are read
only here, after the agent has produced every response.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")

import agent as agent_module
from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)


def audit(agent, samples, catalog_ids, categories, products) -> list[dict]:
    captured: dict = {}
    original_emit = agent_module.emit_count

    def spy(turn, n_constraints, top_k, scores=None, information_complete=False,
            refutation_cohort_size=0):
        captured["scores"] = scores
        captured["cohort"] = refutation_cohort_size
        captured["complete"] = information_complete
        captured["n_constraints"] = n_constraints
        return original_emit(
            turn, n_constraints, top_k, scores, information_complete,
            refutation_cohort_size,
        )

    agent_module.emit_count = spy
    sessions: list[dict] = []
    try:
        for index, sample in enumerate(samples):
            session_id = f"audit_{index:05d}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            card, behavior = materialize_hidden_fields(sample, products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            override = effective.get("behavior", {}).get("override") or {}
            override_turn = int(override["turn"]) if override else 1

            disclosed: set[str] = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            user_message = initial_message(
                effective, coarse_category(categories.get(target, [])), disclosed
            )
            record = {
                "sample_id": sample.get("sample_id"),
                "scenario": sample["scenario_type"],
                "difficulty": sample.get("difficulty_bucket"),
                "override_turn": override_turn,
                "first_hit_turn": None,
                "rank": None,
                "turns": [],
            }
            for turn in range(1, MAX_TURNS + 1):
                captured.clear()
                response = agent.respond(session_id, user_message, turn, TOP_K)
                full = captured.get("scores") or []
                ranked_full = [asin for asin, _ in full]
                position = (
                    ranked_full.index(target) + 1 if target in ranked_full else None
                )
                lookup = dict(full)
                emitted = normalize_recommendations(
                    response.get("recommendations"), catalog_ids
                )
                record["turns"].append({
                    "turn": turn,
                    "emitted": len(emitted),
                    "target_full_rank": position,
                    "countable": override_applied,
                    "cohort": captured.get("cohort"),
                    "complete": captured.get("complete"),
                    "n_constraints": captured.get("n_constraints"),
                    "ask": response.get("ask_attribute"),
                    "top_asins": ranked_full[:10],
                    "top_score": full[0][1] if full else None,
                    "target_score": lookup.get(target),
                })
                if override_applied and target in emitted:
                    record["first_hit_turn"] = turn
                    record["rank"] = emitted.index(target) + 1
                    break
                if turn == MAX_TURNS:
                    break
                if not override_applied and turn + 1 == override_turn:
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    user_message = str(
                        override.get(
                            "message",
                            "Actually, please ignore my earlier preference.",
                        )
                    )
                else:
                    user_message, boundary_used = customer_reply(
                        effective, response.get("ask_attribute"), disclosed,
                        boundary_used,
                    )
            sessions.append(record)
    finally:
        agent_module.emit_count = original_emit
    return sessions


def summarize(sessions: list[dict]) -> dict:
    """Compare the achieved MTTC against two counterfactual ceilings."""
    achieved, ideal_rank1, ideal_top10 = [], [], []
    blocked_by_gate = 0
    gate_turns_lost = 0
    by_scenario = defaultdict(lambda: {"n": 0, "achieved": 0.0, "ideal": 0.0})
    hist = Counter()

    for record in sessions:
        floor = record["override_turn"] if record["scenario"] == "intent_override" else 1
        hit = record["first_hit_turn"] or MAX_TURNS + 1
        achieved.append(hit)
        hist[hit] += 1

        first_rank1 = None
        first_top10 = None
        for step in record["turns"]:
            if not step["countable"]:
                continue
            position = step["target_full_rank"]
            if position is None:
                continue
            if first_rank1 is None and position == 1:
                first_rank1 = step["turn"]
            if first_top10 is None and position <= TOP_K:
                first_top10 = step["turn"]
        first_rank1 = max(first_rank1 or hit, floor)
        first_top10 = max(first_top10 or hit, floor)
        ideal_rank1.append(first_rank1)
        ideal_top10.append(first_top10)
        if first_rank1 < hit:
            blocked_by_gate += 1
            gate_turns_lost += hit - first_rank1
        bucket = by_scenario[record["scenario"]]
        bucket["n"] += 1
        bucket["achieved"] += hit
        bucket["ideal"] += first_rank1

    def mttc(values):
        return round(sum(values) / len(values), 6) if values else None

    def score(m):
        return round(0.5 + 0.3 + 0.2 * max(0.0, min(1.0, (11 - m) / 10)), 6)

    return {
        "sessions": len(sessions),
        "mttc_achieved": mttc(achieved),
        "mttc_if_gate_perfect_rank1": mttc(ideal_rank1),
        "mttc_if_gate_perfect_top10": mttc(ideal_top10),
        "score_achieved": score(mttc(achieved)),
        "score_if_gate_perfect_rank1": score(mttc(ideal_rank1)),
        "sessions_where_rank1_was_withheld": blocked_by_gate,
        "turns_lost_to_gate": gate_turns_lost,
        "first_hit_turn_histogram": dict(sorted(hist.items())),
        "by_scenario": {
            key: {
                "n": value["n"],
                "mttc_achieved": round(value["achieved"] / value["n"], 6),
                "mttc_ideal_rank1": round(value["ideal"] / value["n"], 6),
                "turns_recoverable": round(value["achieved"] - value["ideal"], 3),
            }
            for key, value in sorted(by_scenario.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/turn_audit.json")
    parser.add_argument("--sessions-output", default="")
    args = parser.parse_args()

    ids, categories, products = catalog_index(args.catalog)
    samples = load_jsonl(args.dataset)
    instance = agent_module.Agent(args.catalog)
    sessions = audit(instance, samples, ids, categories, products)
    summary = summarize(sessions)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if args.sessions_output:
        with open(args.sessions_output, "w", encoding="utf-8") as handle:
            json.dump(sessions, handle, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
