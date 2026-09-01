"""Capture a turn-by-turn decision trace for one public session.

The demo needs to show *why* a product survives: which shelf it came from, which
layer of the reranker separated it, and which candidates the popularity prior
was allowed to touch.  ``Agent.respond`` only materialises detailed signals for
the products it actually emits, so this development tool replays one session and
recomputes the same deterministic ranking with every layer exposed.

Two properties keep the output honest:

* the customer messages come from the unchanged organizer evaluator, not from a
  paraphrase we authored;
* the recomputed top product is asserted against the agent's own certificate, so
  any drift between this tool and the runtime fails the build instead of
  silently producing a prettier story.

Nothing here runs during scoring.  ``agent.py`` never imports this module.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent
from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from src import config
from src.evidence import candidate_signals
from src.rank import score_candidates

MAX_TURNS = 10
TOP_K = 10
DETAIL_COUNT = 12


def _rank_args(state) -> dict:
    """The exact keyword arguments ``Agent.respond`` uses for this session."""
    return {
        "constraints": list(state.constraints),
        "profile_tags": list(state.profile_tags),
        "constraint_weights": state.constraint_weights(),
        "scenario": state.scenario,
        "signature_positions_reliable": (
            state.signature_positions_reliable
            if config.ADAPT_SIGNATURE_ORDER
            else True
        ),
    }


def _drop_proven_misses(scores: list[tuple[str, float]], state) -> list[tuple[str, float]]:
    if not (config.PROVEN_MISS_EXCLUSION and state.proven_misses):
        return scores
    return [
        (asin, value) for asin, value in scores
        if asin not in state.proven_misses
    ]


def _product(catalog, asin: str) -> dict:
    # ``catalog.title`` is normalised for matching; display titles are restored
    # from the raw catalog in ``_restore_display_titles``.
    return {
        "parent_asin": asin,
        "title": catalog.title.get(asin, asin),
        "price": catalog.price.get(asin),
        "average_rating": catalog.rating.get(asin, 0.0),
        "rating_number": catalog.rating_count.get(asin, 0),
    }


def _collect_asins(node: object, found: set[str]) -> None:
    if isinstance(node, dict):
        asin = node.get("parent_asin")
        if isinstance(asin, str):
            found.add(asin)
        for value in node.values():
            _collect_asins(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_asins(value, found)


def _restore_display_titles(trace: dict, catalog_path: Path) -> None:
    """Replace normalised titles with the catalog's original casing, in place."""
    wanted: set[str] = set()
    _collect_asins(trace, wanted)
    originals: dict[str, dict] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            product = json.loads(line)
            asin = str(product.get("parent_asin", ""))
            if asin in wanted:
                originals[asin] = product
                if len(originals) == len(wanted):
                    break

    def apply(node: object) -> None:
        if isinstance(node, dict):
            asin = node.get("parent_asin")
            if isinstance(asin, str) and asin in originals and "title" in node:
                raw = originals[asin]
                node["title"] = " ".join(str(raw.get("title") or asin).split())
                node["store"] = str(raw.get("store") or "Independent seller")
            for value in node.values():
                apply(value)
        elif isinstance(node, list):
            for value in node:
                apply(value)

    apply(trace)


def _score_histogram(scores: list[tuple[str, float]], buckets: int = 24) -> dict:
    """Coarse shape of the whole candidate pool, for a density strip."""
    values = [value for _, value in scores]
    if not values:
        return {"buckets": [], "min": 0.0, "max": 0.0}
    low, high = min(values), max(values)
    if high <= low:
        return {"buckets": [len(values)] + [0] * (buckets - 1), "min": low, "max": high}
    counts = [0] * buckets
    span = high - low
    for value in values:
        index = min(buckets - 1, int((value - low) / span * buckets))
        counts[index] += 1
    return {"buckets": counts, "min": round(low, 6), "max": round(high, 6)}


def capture_turn(agent: Agent, state, turn: int, target: str) -> dict:
    """Recompute the ranking the agent just produced, with every layer visible."""
    catalog = agent.catalog
    shelf_ids = catalog.candidates(state.shelf)
    args = _rank_args(state)

    scores = _drop_proven_misses(score_candidates(catalog, shelf_ids, **args), state)
    core = _drop_proven_misses(
        score_candidates(catalog, shelf_ids, popularity_weight=0.0, **args), state
    )
    core_lookup = dict(core)
    best_core = core[0][1] if core else 0.0

    has_evidence = bool([value for value in state.constraints if value])
    ranked_ids = [asin for asin, _ in scores]
    target_rank = ranked_ids.index(target) + 1 if target in ranked_ids else None

    detail: list[dict] = []
    for position, (asin, final) in enumerate(scores[:DETAIL_COUNT], start=1):
        signals = candidate_signals(
            catalog,
            asin,
            final,
            state.constraints,
            state.constraint_weights(),
            state.scenario,
            args["signature_positions_reliable"],
        )
        candidate_core = core_lookup.get(asin, 0.0)
        detail.append({
            **_product(catalog, asin),
            "rank": position,
            "final_score": round(final, 6),
            "core_score": round(candidate_core, 6),
            "core_gap_to_best": round(best_core - candidate_core, 6),
            "prior_eligible": bool(
                has_evidence and best_core - candidate_core <= config.POPULARITY_WINDOW
            ),
            "typed_satisfaction": signals["typed_satisfaction"],
            "signature_likelihood": signals["signature_likelihood"],
            "popularity_prior": signals["bounded_popularity_prior"],
            "constraint_evidence": signals["constraint_evidence"],
            "is_target": asin == target,
        })

    eligible = sum(
        1 for _, value in core
        if has_evidence and best_core - value <= config.POPULARITY_WINDOW
    )
    certificate = agent.explain_last_decision(state_session_id(agent, state))
    question_value = certificate.get("selected_question_value") or {}

    return {
        "turn": turn,
        "has_evidence": has_evidence,
        "shelf": {"name": state.shelf, "count": len(shelf_ids)},
        "constraints": [
            {"text": text, "confidence": weight}
            for text, weight in zip(state.constraints, state.constraint_weights())
        ],
        "excluded_proven_misses": [
            _product(catalog, asin) for asin in sorted(state.proven_misses)
        ],
        "funnel": {
            "catalog": len(catalog.ids),
            "shelf": len(shelf_ids),
            "ranked": len(scores),
            "prior_eligible": eligible,
            "voi_pool": question_value.get("candidate_count"),
            "emitted": certificate.get("output_gate", {}).get("emitted_count"),
        },
        "top": detail,
        "pool_shape": _score_histogram(scores),
        "target_rank": target_rank,
        "action": certificate.get("action"),
        "reason_codes": certificate.get("reason_codes"),
        "question": certificate.get("question"),
        "question_value": question_value,
        "output_gate": certificate.get("output_gate"),
        "recommendations": certificate.get("recommendations"),
    }


def state_session_id(agent: Agent, state) -> str:
    for session_id, candidate in agent._sessions.items():  # noqa: SLF001 - dev tool
        if candidate is state:
            return session_id
    raise LookupError("session state is not registered on the agent")


def build_trace(catalog_path: Path, sample_id: str) -> dict:
    catalog_ids, categories, products = catalog_index(catalog_path)
    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    sample = next((item for item in samples if item["sample_id"] == sample_id), None)
    if sample is None:
        raise SystemExit(f"sample not found in the public set: {sample_id}")

    agent = Agent(catalog_path)
    session_id = f"trace_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    state = agent._sessions[session_id]  # noqa: SLF001 - dev tool

    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": intent_card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(
        effective, coarse_category(categories.get(target, [])), disclosed
    )

    turns: list[dict] = []
    hit_turn: int | None = None
    best_rank: int | None = None
    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, TOP_K)
        captured = capture_turn(agent, state, turn, target)
        captured["customer"] = user_message
        captured["agent_message"] = response["message"]
        captured["ask_attribute"] = response["ask_attribute"]

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        emitted = captured["recommendations"] or []
        if emitted and captured["top"] and emitted[0] != captured["top"][0]["parent_asin"]:
            raise SystemExit(
                f"turn {turn}: recomputed ranking disagrees with the agent "
                f"({captured['top'][0]['parent_asin']} vs {emitted[0]})"
            )
        turns.append(captured)

        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            break
        if turn == MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    trace = {
        "status": "offline_diagnostic_recomputed_from_the_submitted_runtime",
        "message_source": "unchanged organizer evaluator, canonical reveal",
        "sample_id": sample_id,
        "scenario_type": sample["scenario_type"],
        "target_parent_asin": target,
        "target_visibility": "offline_evaluator_only_never_sent_to_agent",
        "target_product": _product(agent.catalog, target),
        "outcome": {
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
        },
        "weights": {
            "typed": config.TYPED_WEIGHT,
            "signature": config.SIGNATURE_WEIGHT,
            "popularity": config.POPULARITY_WEIGHT,
            "popularity_window": config.POPULARITY_WINDOW,
            "question_turn_cost": config.QUESTION_TURN_COST,
        },
        "turns": turns,
    }
    _restore_display_titles(trace, catalog_path)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--sample-id", default="public_0007")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/decision_trace.json"
    )
    args = parser.parse_args()
    trace = build_trace(args.catalog, args.sample_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    outcome = trace["outcome"]
    print(
        f"wrote {args.output} | {trace['sample_id']} "
        f"turns={len(trace['turns'])} hit_turn={outcome['first_hit_turn']} "
        f"rank={outcome['best_rank']}"
    )


if __name__ == "__main__":
    main()
