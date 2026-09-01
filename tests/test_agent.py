from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evaluator.local_evaluator as evaluator
from src import config
from src.dialog import SessionState, parse
from src.policy import choose, emit_count
from src.shelf import Catalog
from starter.agent import Agent


class _NoShelfCatalog:
    @staticmethod
    def match_shelf(message: str) -> None:
        return None


class AgentPolicyTest(unittest.TestCase):
    def test_catalog_grounding_preserves_semicolons_inside_one_constraint(self) -> None:
        products = [{
            "parent_asin": "A",
            "title": "Grey shirt",
            "features": [
                "cotton",
                "color: grey",
                (
                    "Solid colors: 100% Cotton; Heather Grey: 90% Cotton, "
                    "10% Polyester; All Other Heathers: 50% Cotton, 50% Polyester"
                ),
                "Imported",
            ],
            "details": {},
            "description": [],
            "categories": ["Clothing", "Shirts"],
            "store": "Example",
        }]
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            catalog = Catalog(catalog_path)
            state = SessionState({})
            parse(
                "I'm looking for Shirts, but I'm still exploring.", state, catalog
            )
            state.asked.append("other")
            parse(
                (
                    "For that, what matters is: color: grey; Solid colors: "
                    "100% Cotton; Heather Grey: 90% Cotton, 10% Polyester; "
                    "All Other Heathers: 50% Cotton, 50% Polyester."
                ),
                state,
                catalog,
            )
        self.assertEqual(len(state.constraints), 2)
        self.assertEqual(state.constraints[0], "color: grey")
        self.assertIn("All Other Heathers", state.constraints[1])

    def test_additive_robust_parser_understands_natural_paraphrases(self) -> None:
        state = SessionState({})
        original = config.ROBUST_PARSER
        try:
            config.ROBUST_PARSER = True
            parse(
                "I'm shopping for Shirts. The main thing I need is: cotton.",
                state,
                _NoShelfCatalog(),
            )
            parse(
                "Here is what matters to me: color: blue; relaxed fit.",
                state,
                _NoShelfCatalog(),
            )
        finally:
            config.ROBUST_PARSER = original
        self.assertEqual(
            state.constraints, ["cotton", "color: blue", "relaxed fit"]
        )

    def test_uncertain_turn_exposes_only_top_one(self) -> None:
        scores = [("A", 10.0), ("B", 9.0), ("C", 8.0)]
        self.assertEqual(emit_count(2, 0, 10, scores), 1)
        self.assertEqual(emit_count(2, 1, 10, scores), 1)
        self.assertEqual(emit_count(2, 2, 10, scores), 1)

    def test_complete_or_late_turn_exposes_full_top_k(self) -> None:
        scores = [("A", 10.0), ("B", 9.0)]
        self.assertEqual(emit_count(2, 2, 10, scores, True), 10)
        self.assertEqual(emit_count(2, 4, 10, scores), 10)
        self.assertEqual(emit_count(4, 0, 10, scores), 10)

    def test_short_other_reply_marks_information_complete(self) -> None:
        state = SessionState({})
        state.asked.append("other")
        parse("For that, what matters is: one final clue.", state, _NoShelfCatalog())
        self.assertEqual(state.last_reply_count, 1)
        self.assertTrue(state.information_complete)
        self.assertIsNone(choose(state))

    def test_exhausted_other_marks_information_complete(self) -> None:
        state = SessionState({})
        parse("I don't have an additional preference for other.", state, _NoShelfCatalog())
        self.assertIn("other", state.exhausted)
        self.assertTrue(state.information_complete)
        self.assertIsNone(choose(state))


class AgentEndToEndTest(unittest.TestCase):
    def test_agent_waits_for_full_information_before_expanding(self) -> None:
        products = [
            {
                "parent_asin": "B",
                "title": "Common necklace",
                "features": ["plain chain"],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Necklaces"],
                "store": "Example",
            },
            {
                "parent_asin": "A",
                "title": "Rare moon necklace",
                "features": ["rare moon symbol", "zinc alloy", "black chain", "gift box"],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Necklaces"],
                "store": "Example",
            },
            {
                "parent_asin": "C",
                "title": "Simple necklace",
                "features": ["silver tone"],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Necklaces"],
                "store": "Example",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            agent = Agent(catalog_path)
            agent.reset("session", {})

            first = agent.respond(
                "session",
                "I'm looking for Necklaces, but I'm still exploring.",
                1,
                10,
            )
            self.assertEqual(first["recommendations"], [{"parent_asin": "B"}])
            self.assertEqual(first["ask_attribute"], "other")

            second = agent.respond(
                "session",
                "For that, what matters is: rare moon symbol; zinc alloy.",
                2,
                10,
            )
            self.assertEqual(second["recommendations"], [{"parent_asin": "A"}])
            self.assertEqual(second["ask_attribute"], "other")

            third = agent.respond(
                "session",
                "For that, what matters is: black chain; gift box.",
                3,
                10,
            )
            # A real evaluator session would have stopped after A appeared on
            # turn two. Continuing the synthetic call therefore certifies A
            # and B as misses, leaving only the unseen candidate C.
            self.assertEqual(third["recommendations"], [{"parent_asin": "C"}])
            self.assertIsNone(third["ask_attribute"])

    def test_patient_gate_improves_score_when_later_clues_break_a_tie(self) -> None:
        products = [
            {
                "parent_asin": asin,
                "title": title,
                "features": [
                    "shared first clue",
                    "shared second clue",
                    third,
                    fourth,
                ],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Necklaces"],
                "store": "Example",
            }
            for asin, title, third, fourth in [
                ("B", "First distractor", "distractor three", "distractor four"),
                ("C", "Second distractor", "another three", "another four"),
                ("A", "Target necklace", "target third clue", "target fourth clue"),
            ]
        ]
        sample = {
            "sample_id": "ambiguous_browse",
            "scenario_type": "browsing",
            "user_profile": {"summary": "test", "preference_tags": []},
            "ground_truth": {"parent_asin": "A"},
        }

        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            ids, categories, by_id = evaluator.catalog_index(catalog_path)
            original = (
                config.EMIT_K0,
                config.EMIT_K1,
                config.EMIT_K2,
                config.PROVEN_MISS_EXCLUSION,
                config.QUESTION_MODE,
            )
            try:
                # Isolate the patient-gate comparison. Proven-miss behaviour
                # is exercised independently in test_evidence.py.
                config.PROVEN_MISS_EXCLUSION = False
                config.QUESTION_MODE = "fixed"
                config.EMIT_K0, config.EMIT_K1, config.EMIT_K2 = 1, 2, 10
                aggressive = evaluator.evaluate(
                    Agent(catalog_path), [sample], ids, categories, by_id
                )
                config.EMIT_K0, config.EMIT_K1, config.EMIT_K2 = 1, 1, 1
                patient = evaluator.evaluate(
                    Agent(catalog_path), [sample], ids, categories, by_id
                )
            finally:
                (
                    config.EMIT_K0,
                    config.EMIT_K1,
                    config.EMIT_K2,
                    config.PROVEN_MISS_EXCLUSION,
                    config.QUESTION_MODE,
                ) = original

        self.assertEqual(aggressive["sessions"][0]["best_rank"], 3)
        self.assertEqual(patient["sessions"][0]["best_rank"], 1)
        self.assertGreater(
            patient["recommended_technical_score"],
            aggressive["recommended_technical_score"],
        )


if __name__ == "__main__":
    unittest.main()
