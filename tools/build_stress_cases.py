"""Turn the rephrasing stress test into inspectable case studies.

``robustness_policy_benchmark`` reports aggregate pass rates for four ways of
saying the same thing.  Aggregates hide the interesting part: *how* a system
fails when it stops understanding a sentence.  This tool replays the same
sessions, records both arms turn by turn, and classifies each recovered session
by the failure it avoided.

Three failure modes are separated, because they are not equally bad:

``clue_dropped``
    A constraint never reaches the ranker.  The agent is uninformed.
``override_missed``
    The shopper changed their mind and the agent did not notice, so it keeps
    optimising for a preference the shopper has abandoned.  The agent is
    actively wrong, which is worse than being uninformed.
``exhaustion_missed``
    The shopper said they have nothing more to add and the agent did not hear
    it, so it never learns the evidence is complete and keeps holding products
    back.  Ranking is fine; the decision about when to answer is not.

Development tool only.  ``agent.py`` never imports this module.
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
from src import dialog
from tools.robustness_policy_benchmark import PerturbingAgent, _natural

# Every rewrite the stress test can apply, paired with the reader that catches
# it.  ``canonical`` is the organizer's own wording.
REWRITE_RULES = [
    {
        "id": "open_browse",
        "what": "Opening line, still browsing",
        "canonical": "I'm looking for Tees & Blouses Tunics, but I'm still exploring.",
        "rephrased": "I'd like to browse Tees & Blouses Tunics. I haven't settled on preferences.",
        "reader": "_ALT_LOOKING + _ALT_STILL_EXP",
    },
    {
        "id": "open_buying",
        "what": "Opening line with a hard requirement",
        "canonical": "I'm looking for Sweaters. A key requirement is: 100% Cotton.",
        "rephrased": "I'm shopping for Sweaters. The main thing I need is: 100% Cotton.",
        "reader": "_ALT_LOOKING + _ALT_KEY_REQ",
    },
    {
        "id": "reveal",
        "what": "Answering our question with new constraints",
        "canonical": "For that, what matters is: polyester; machine wash.",
        "rephrased": "Here is what matters to me: polyester; machine wash.",
        "reader": "_ALT_MATTERS",
    },
    {
        "id": "override",
        "what": "Changing their mind mid-conversation",
        "canonical": "Actually, ignore my earlier preference. What I need is: long sleeve.",
        "rephrased": "I've changed my mind. My new requirement is: long sleeve.",
        "reader": "_ALT_NEED_IS",
    },
    {
        "id": "exhausted",
        "what": "Nothing more to add on that attribute",
        "canonical": "I don't have an additional preference for material.",
        "rephrased": "Nothing else to add about material.",
        "reader": "_ALT_EXHAUSTED",
    },
    {
        "id": "no_preference",
        "what": "Deflecting the question entirely",
        "canonical": "I don't have a preference for color; please use your judgment.",
        "rephrased": "No preference on color; choose what works.",
        "reader": "_ALT_NO_PREF",
    },
    {
        "id": "rejection",
        "what": "Rejecting the whole result list",
        "canonical": "Those options are not quite right yet. Ask me about one specific attribute.",
        "rephrased": "Those choices still miss the mark; please ask something specific.",
        "reader": "_ALT_NO_INFO",
    },
]

VARIANTS = [
    {
        "id": "canonical",
        "name": "Exactly as the organizers write it",
        "how": "The simulator's own sentences, untouched.",
        "example": "For that, what matters is: polyester.",
    },
    {
        "id": "surface_noise",
        "name": "Shouting and stray formatting",
        "how": "Upper-cased and prefixed, the way a frustrated shopper or a bad client might send it.",
        "example": "  PLEASE NOTE:  FOR THAT, WHAT MATTERS IS: POLYESTER.",
    },
    {
        "id": "natural_paraphrase",
        "name": "Said a different way",
        "how": "The same meaning in a different sentence frame, which is where a template reader breaks.",
        "example": "Here is what matters to me: polyester.",
    },
    {
        "id": "hidden_one",
        "name": "One clue withheld",
        "how": "The shopper simply does not mention one of their constraints. No reader can recover it.",
        "example": "For that, what matters is: polyester.  (the second clue is never said)",
    },
]

# (canonical pattern, alternative pattern, failure mode) for every frame that a
# rephrasing can move out of the canonical reader's reach.
FRAMES = [
    (dialog._KEY_REQ, dialog._ALT_KEY_REQ, "clue_dropped"),
    (dialog._MATTERS, dialog._ALT_MATTERS, "clue_dropped"),
    (dialog._NEED_IS, dialog._ALT_NEED_IS, "override_missed"),
    (dialog._EXHAUSTED, dialog._ALT_EXHAUSTED, "exhaustion_missed"),
    (dialog._NO_PREF, dialog._ALT_NO_PREF, "exhaustion_missed"),
]

MODE_TEXT = {
    "clue_dropped": (
        "The clue never reached the ranker",
        "The agent kept searching with less information than the shopper actually gave it.",
    ),
    "override_missed": (
        "The change of mind went unheard",
        "Worse than missing a clue: the agent kept ranking for a preference the shopper had already abandoned, and never cleared the products it had shown under the old intent.",
    ),
    "exhaustion_missed": (
        "It never learned there was nothing left to ask",
        "The ranking was fine. The agent simply never registered that the evidence was complete, so it kept asking and kept holding products back.",
    ),
}


# Most damaging first: being actively wrong beats being uninformed.
SEVERITY = ["override_missed", "exhaustion_missed", "clue_dropped"]


def classify(messages: list[str]) -> list[str]:
    """Which failure modes a canonical-only reader would hit on these messages."""
    modes: list[str] = []
    for message in messages:
        for canonical, alternative, mode in FRAMES:
            if alternative.search(message) and not canonical.search(message):
                if mode not in modes:
                    modes.append(mode)
    return modes


def repeated_question(trace: list[dict]) -> dict | None:
    """If an arm asked about the same attribute every single turn, say so.

    Computed over the complete trace, not the abridged display turns, so the
    claim survives the middle turns being collapsed for the screen.
    """
    asked = [call["response"].get("ask_attribute") for call in trace]
    if len(asked) >= 3 and asked[0] is not None and len(set(asked)) == 1:
        return {"attribute": asked[0], "turns": len(asked)}
    return None


def pivotal_message(trace: list[dict]) -> dict | None:
    """The first rephrased sentence the canonical reader could not act on."""
    for call in trace:
        said = call["transformed_message"]
        for canonical, alternative, mode in FRAMES:
            if alternative.search(said) and not canonical.search(said):
                return {
                    "turn": call["turn"],
                    "canonical_message": call["raw_message"],
                    "shopper_said": said,
                    "mode": mode,
                }
    return None


def run_arm(catalog_path, samples, ids, categories, products, style, robust):
    original = config.ROBUST_PARSER
    try:
        config.ROBUST_PARSER = robust
        wrapper = PerturbingAgent(catalog_path, style)
        outcome = evaluate(wrapper, samples, ids, categories, products)
    finally:
        config.ROBUST_PARSER = original
    return wrapper, outcome


def compact_turns(trace: list[dict], target: str, limit: int = 4) -> dict:
    """Turn a raw trace into display turns, collapsing an unchanging middle."""
    turns = []
    for call in trace:
        asins = [
            str(item.get("parent_asin", ""))
            for item in call["response"].get("recommendations", [])
            if isinstance(item, dict)
        ]
        turns.append({
            "turn": call["turn"],
            "canonical_message": call["raw_message"],
            "shopper_said": call["transformed_message"],
            "rephrased": call["raw_message"] != call["transformed_message"],
            "agent_asked": call["response"].get("ask_attribute"),
            "shown_count": len(asins),
            "target_rank": asins.index(target) + 1 if target in asins else None,
        })
    if len(turns) <= limit:
        return {"turns": turns, "omitted": 0}
    return {"turns": [*turns[: limit - 1], turns[-1]], "omitted": len(turns) - limit}


def build(catalog_path: str, dataset: str, count: int) -> dict:
    samples = load_jsonl(dataset)[:count]
    ids, categories, products = catalog_index(catalog_path)

    old_wrapper, old_outcome = run_arm(
        catalog_path, samples, ids, categories, products, "natural_paraphrase", False
    )
    new_wrapper, new_outcome = run_arm(
        catalog_path, samples, ids, categories, products, "natural_paraphrase", True
    )
    old_traces = [trace for _, trace in old_wrapper.traces.items()]
    new_traces = [trace for _, trace in new_wrapper.traces.items()]

    cases: list[dict] = []
    for index, sample in enumerate(samples):
        old_session = old_outcome["sessions"][index]
        new_session = new_outcome["sessions"][index]
        old_rr = old_session["reciprocal_rank"]
        new_rr = new_session["reciprocal_rank"]
        if new_rr <= old_rr:
            continue
        target = str(sample["ground_truth"]["parent_asin"])
        # Classify the arm that failed: these are the sentences the canonical
        # reader could not act on.  The two arms diverge after the first
        # misunderstanding, so the new arm's transcript would tell a different
        # story.
        modes = classify(
            [call["transformed_message"] for call in old_traces[index]]
        )
        cases.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "target_parent_asin": target,
            "target_title": products[target].get("title"),
            "failure_modes": modes,
            # The first misunderstanding is the one that set the session off
            # course; later ones are usually downstream of it.
            "primary_mode": modes[0] if modes else "clue_dropped",
            "old_outcome": old_session,
            "new_outcome": new_session,
            "reciprocal_rank_gain": round(new_rr - old_rr, 6),
            "turns_saved": (
                (old_session["first_hit_turn"] or 11)
                - (new_session["first_hit_turn"] or 11)
            ),
            "pivotal": pivotal_message(old_traces[index]),
            "old_repeated_question": repeated_question(old_traces[index]),
            "old_turns": compact_turns(old_traces[index], target),
            "new_turns": compact_turns(new_traces[index], target),
        })

    by_mode: dict[str, list[dict]] = {}
    for case in cases:
        by_mode.setdefault(case["primary_mode"], []).append(case)
    for bucket in by_mode.values():
        bucket.sort(
            key=lambda case: (
                -case["reciprocal_rank_gain"],
                -case["turns_saved"],
                case["sample_id"],
            )
        )

    # One exemplar per failure mode, most instructive mode first.  A session may
    # contain several modes; prefer one where the featured mode came first, so
    # the story we tell is the one that actually caused the miss.
    featured: list[dict] = []
    for mode in SEVERITY:
        candidates = [
            case for case in cases
            if mode in case["failure_modes"]
            and case["sample_id"] not in {item["sample_id"] for item in featured}
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda case: (
                case["primary_mode"] != mode,
                -case["reciprocal_rank_gain"],
                -case["turns_saved"],
                case["sample_id"],
            )
        )
        chosen = dict(candidates[0])
        chosen["featured_for"] = mode
        featured.append(chosen)

    return {
        "status": "self_designed_robustness_probe_not_an_organizer_result",
        "caveat": (
            "We wrote both the rephrasings and the reader that handles them. "
            "Treat this as a probe of our own brittleness, not as an "
            "independent benchmark."
        ),
        "sample_count": len(samples),
        "rewrite_rules": REWRITE_RULES,
        "variants": VARIANTS,
        "mode_text": {
            mode: {"title": title, "why": why}
            for mode, (title, why) in MODE_TEXT.items()
        },
        "recovered_count": len(cases),
        "mode_counts": {mode: len(bucket) for mode, bucket in sorted(by_mode.items())},
        "featured_cases": featured,
        "all_cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stress_cases.json")
    args = parser.parse_args()

    # Sanity check: the documented examples must actually round-trip through the
    # real rewriter, so the table on screen cannot drift from the code.
    for rule in REWRITE_RULES:
        produced = _natural(rule["canonical"])
        if produced != rule["rephrased"]:
            raise SystemExit(
                f"rewrite rule {rule['id']} is out of date:\n"
                f"  documented: {rule['rephrased']}\n"
                f"  produced  : {produced}"
            )

    result = build(args.catalog, args.dataset, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} | recovered={result['recovered_count']} "
        f"modes={result['mode_counts']} featured={len(result['featured_cases'])}"
    )


if __name__ == "__main__":
    main()
