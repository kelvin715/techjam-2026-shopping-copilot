from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import evaluator.local_evaluator as evaluator
from src import config
from src.dialog import SessionState, parse
from src.policy import choose, emit_count, exact_signature_prefix
from src.shelf import Catalog
from starter.agent import Agent


def setUpModule() -> None:
    # Tests must be hermetic: never depend on a developer's local .env or
    # make real network calls, regardless of the committed config default.
    global _ORIGINAL_INPUT_MODE
    _ORIGINAL_INPUT_MODE = config.INPUT_MODE
    config.INPUT_MODE = "template"


def tearDownModule() -> None:
    config.INPUT_MODE = _ORIGINAL_INPUT_MODE


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

    def test_finite_horizon_plan_enumerates_exact_signature_siblings(self) -> None:
        scores = [("A", 10.0), ("B", 9.0), ("C", 8.0)]
        self.assertEqual(
            emit_count(3, 4, 10, scores, True, refutation_cohort_size=3),
            1,
        )
        self.assertEqual(
            emit_count(8, 4, 10, scores, True, refutation_cohort_size=3),
            10,
        )

    def test_exact_signature_cohort_is_only_a_ranked_prefix(self) -> None:
        signatures = {"A": ("same",), "B": ("same",), "C": ("other",)}
        self.assertEqual(
            exact_signature_prefix(
                [("A", 3.0), ("B", 2.0), ("C", 1.0)], signatures
            ),
            2,
        )

    def test_zero_constraint_cold_start_uses_review_count(self) -> None:
        products = [
            {
                "parent_asin": asin,
                "title": asin,
                "features": ["shared"],
                "details": {},
                "description": [],
                "categories": ["Clothing", "Necklaces"],
                "store": "Example",
                "rating_number": reviews,
            }
            for asin, reviews in (("LOW", 2), ("HIGH", 200))
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            agent = Agent(catalog_path)
            agent.reset("cold", {})
            response = agent.respond(
                "cold",
                "I'm looking for Necklaces, but I'm still exploring.",
                1,
                10,
            )
        self.assertEqual(
            response["recommendations"], [{"parent_asin": "HIGH"}]
        )

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


class AgenticFallbackTest(unittest.TestCase):
    """Hermetic: openai.OpenAI is mocked, no real network calls.

    When the optional package is absent a stub stands in for it, so these
    run in CI too -- an agentic path that only ever ships untested is worse
    than the dependency it avoids.
    """

    def setUp(self) -> None:
        from src import agentic_dialog

        self._stubbed_openai = importlib.util.find_spec("openai") is None
        if self._stubbed_openai:
            stub = types.ModuleType("openai")
            stub.OpenAI = unittest.mock.MagicMock(name="OpenAI")
            sys.modules["openai"] = stub

        self.agentic_dialog = agentic_dialog
        self._saved_circuit = (
            agentic_dialog._consecutive_failures,
            agentic_dialog._circuit_open_until,
        )
        self._temp_dirs: list[str] = []
        agentic_dialog._consecutive_failures = 0
        agentic_dialog._circuit_open_until = 0.0

    def tearDown(self) -> None:
        if self._stubbed_openai:
            sys.modules.pop("openai", None)
        # The breaker is module-global. Leaving it open would silently skip
        # the agentic path in any later test in the same process.
        (
            self.agentic_dialog._consecutive_failures,
            self.agentic_dialog._circuit_open_until,
        ) = self._saved_circuit
        for directory in self._temp_dirs:
            shutil.rmtree(directory, ignore_errors=True)

    def _catalog(self) -> Catalog:
        products = [{
            "parent_asin": "A",
            "title": "Grey shirt",
            "features": ["denim"],
            "details": {},
            "description": [],
            "categories": ["Clothing", "Shirts"],
            "store": "Example",
        }]
        directory = tempfile.mkdtemp()
        self._temp_dirs.append(directory)
        catalog_path = Path(directory) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        return Catalog(catalog_path)

    def _fake_response(self, tool_calls, prompt_tokens=12, completion_tokens=8):
        from types import SimpleNamespace

        message = SimpleNamespace(tool_calls=tool_calls, content=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
        )

    def _fake_tool_call(self, name: str, arguments: dict, call_id="call_1"):
        from types import SimpleNamespace

        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )

    def test_agentic_interpret_applies_extraction_and_tracks_usage(self) -> None:
        from src.llm import LLMUsage

        catalog = self._catalog()
        state = SessionState({})
        state.shelf = "Shirts"
        submission = self._fake_tool_call("submit_extraction", {
            "constraints": ["denim"],
            "is_override": False,
            "uninformative": False,
        })
        usage = LLMUsage()
        with unittest.mock.patch.object(
            self.agentic_dialog, "agentic_available", return_value=True
        ), unittest.mock.patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = (
                self._fake_response([submission])
            )
            status = self.agentic_dialog.agentic_interpret(
                "I need something denim", state, catalog, usage
            )
        self.assertEqual(status, self.agentic_dialog.STATUS_EXTRACTED)
        self.assertIn("denim", state.constraints)
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.prompt_tokens, 12)
        self.assertEqual(usage.completion_tokens, 8)

    def test_agentic_circuit_breaker_opens_after_repeated_failures(self) -> None:
        catalog = self._catalog()
        with unittest.mock.patch.object(
            self.agentic_dialog, "agentic_available", return_value=True
        ), unittest.mock.patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("boom")
            for _ in range(config.AGENTIC_CIRCUIT_FAILURES):
                status = self.agentic_dialog.agentic_interpret(
                    "anything", SessionState({}), catalog
                )
                self.assertEqual(status, self.agentic_dialog.STATUS_ERROR)
            calls_before = mock_openai.return_value.chat.completions.create.call_count
            status = self.agentic_dialog.agentic_interpret(
                "anything", SessionState({}), catalog
            )
            self.assertEqual(status, self.agentic_dialog.STATUS_CIRCUIT_OPEN)
            self.assertEqual(
                mock_openai.return_value.chat.completions.create.call_count,
                calls_before,
            )


    def test_overlong_reply_is_rejected_without_opening_the_circuit(self) -> None:
        """A length-cap rejection is a content failure, not an endpoint one."""
        from types import SimpleNamespace

        overlong = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="x" * (config.AGENTIC_REPLY_MAX_CHARS + 1)
            ))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5),
        )
        with unittest.mock.patch.object(
            self.agentic_dialog, "agentic_available", return_value=True
        ), unittest.mock.patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = overlong
            for _ in range(config.AGENTIC_CIRCUIT_FAILURES + 1):
                self.assertIsNone(self.agentic_dialog.agentic_reply("color"))
            self.assertFalse(self.agentic_dialog._circuit_open())

    def test_extraction_marks_the_session_grounded_and_positions_unreliable(
        self,
    ) -> None:
        """A model reading is provenance: its array order is not evidence."""
        catalog = self._catalog()
        state = SessionState({})
        state.shelf = "Shirts"
        self.assertTrue(state.signature_positions_reliable)
        self.assertFalse(state.grounded)
        submission = self._fake_tool_call("submit_extraction", {
            "constraints": ["denim"],
            "is_override": False,
            "uninformative": False,
        })
        with unittest.mock.patch.object(
            self.agentic_dialog, "agentic_available", return_value=True
        ), unittest.mock.patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = (
                self._fake_response([submission])
            )
            self.agentic_dialog.agentic_interpret(
                "something in that blue jean material", state, catalog
            )
        self.assertIn("denim", state.constraints)
        # Without these, rank._signature_score keeps awarding the positional
        # bonus against the model's array order, and SessionState.add applies
        # the protocol's four-constraint bound to free-form wording.
        self.assertFalse(state.signature_positions_reliable)
        self.assertTrue(state.grounded)

        # An extraction that verified nothing must leave both flags alone.
        unverified = SessionState({})
        unverified.shelf = "Shirts"
        self.agentic_dialog._apply_extraction(
            {
                "constraints": ["not-a-real-catalog-phrase"],
                "is_override": False,
                "uninformative": False,
            },
            unverified,
            catalog,
        )
        self.assertTrue(unverified.signature_positions_reliable)
        self.assertFalse(unverified.grounded)

    def test_category_narrows_the_shelf_instead_of_the_whole_catalog(self) -> None:
        """Without a shelf or pool, agent.py ranks every row in the catalog."""
        catalog = self._catalog()
        state = SessionState({})
        self.assertIsNone(state.shelf)
        submission = self._fake_tool_call("submit_extraction", {
            "category": "shirts",
            "constraints": ["denim"],
            "is_override": False,
            "uninformative": False,
        })
        with unittest.mock.patch.object(
            self.agentic_dialog, "agentic_available", return_value=True
        ), unittest.mock.patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = (
                self._fake_response([submission])
            )
            status = self.agentic_dialog.agentic_interpret(
                "after a shirt in that blue jean material", state, catalog
            )
        self.assertEqual(status, self.agentic_dialog.STATUS_EXTRACTED)
        self.assertEqual(state.shelf, "Shirts")
        self.assertIn("denim", state.constraints)

        # A category on its own is still something learned: the turn must not
        # be reported as no_signal, or the caller re-reads an answered turn.
        only_category = SessionState({})
        self.assertTrue(self.agentic_dialog._apply_extraction(
            {
                "category": "shirts",
                "constraints": [],
                "is_override": False,
                "uninformative": False,
            },
            only_category,
            catalog,
        ))
        self.assertEqual(only_category.shelf, "Shirts")

    def test_no_preference_retires_the_question_that_was_asked(self) -> None:
        """boundary_signal is a scenario hint; exhausted is what retires it."""
        catalog = self._catalog()
        state = SessionState({})
        state.asked.append("color")
        self.agentic_dialog._apply_extraction(
            {"no_preference_attribute": "colour", "constraints": [],
             "is_override": False, "uninformative": False},
            state,
            catalog,
        )
        self.assertIn("color", state.exhausted)   # what we asked, not "colour"
        self.assertEqual(state.last_reply_count, 0)
        self.assertFalse(state.boundary_signal)

        # "other" is the wildcard question: nothing left to disclose.
        wildcard = SessionState({})
        wildcard.asked.append("other")
        self.agentic_dialog._apply_extraction(
            {"no_preference_attribute": "other", "constraints": [],
             "is_override": False, "uninformative": False},
            wildcard,
            catalog,
        )
        self.assertTrue(wildcard.information_complete)

        # Deflecting before anything was asked is the boundary case.
        deflect = SessionState({})
        self.agentic_dialog._apply_extraction(
            {"no_preference_attribute": "color", "constraints": [],
             "is_override": False, "uninformative": False},
            deflect,
            catalog,
        )
        self.assertTrue(deflect.boundary_signal)
        self.assertFalse(deflect.exhausted)

    def test_client_is_bounded_by_timeout_and_no_retries(self) -> None:
        """Without these the breaker cannot be reached in bounded time."""
        submission = self._fake_tool_call("submit_extraction", {
            "constraints": [], "is_override": False, "uninformative": True,
        })
        with unittest.mock.patch.object(
            self.agentic_dialog, "agentic_available", return_value=True
        ), unittest.mock.patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = (
                self._fake_response([submission])
            )
            self.agentic_dialog.agentic_interpret(
                "anything", SessionState({}), self._catalog()
            )
        mock_openai.assert_called_with(
            timeout=config.AGENTIC_TIMEOUT_SECONDS,
            max_retries=config.AGENTIC_MAX_RETRIES,
        )


if __name__ == "__main__":
    unittest.main()
