"""Dense research options: a translator for values, a recall channel for shelves.

Both are off by default and never touch a protocol message. The tests use a
deterministic hashing embedder (no model download) or a scripted stub.
"""
from __future__ import annotations

import tempfile
import unittest

from agent import Agent
from src import config
from src.dialog import SessionState
from src.llm import LLMSettings
from src.rank import score_candidates
from src.shelf import Catalog
from tests.test_llm_grounding import ScriptedClient, _config_override, _write_catalog

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - research dependency
    HAVE_NUMPY = False


class StubDense:
    """Scripted stand-in for ``DenseIndex``: records every query it is asked."""

    def __init__(self, values=None, shelves=None, products=None) -> None:
        self.values = values or {}
        self.shelves = shelves or {}
        self.products = products or []
        self.calls: list[tuple[str, str]] = []

    def similar_values(self, phrase, allowed=None, limit=3, min_cosine=None):
        self.calls.append(("values", phrase))
        allowed = None if allowed is None else set(allowed)
        out = [
            (value, cosine) for value, cosine in self.values.get(phrase, [])
            if value != phrase and (allowed is None or value in allowed)
        ]
        return out[:limit]

    def similar_shelves(self, text, limit=None, min_cosine=None):
        self.calls.append(("shelves", text))
        return list(self.shelves.get(text, []))

    def search_products(self, text, limit):
        self.calls.append(("products", text))
        return list(self.products)[:limit]


class DenseIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.catalog = Catalog(_write_catalog(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @unittest.skipUnless(HAVE_NUMPY, "numpy is a research dependency")
    def test_hash_embedder_index_maps_a_synonym_onto_a_catalog_value(self) -> None:
        from src.dense import DenseIndex, HashEmbedder

        index = DenseIndex(
            self.catalog, cache_dir=self._tmp.name,
            embedder=HashEmbedder(synonyms={"fastening": "closure"}),
        )
        found = index.similar_values("buckle fastening", None, 3, min_cosine=0.8)
        self.assertEqual(found[0][0], "buckle closure")
        self.assertGreaterEqual(found[0][1], 0.99)
        # Scoped to the values a session can use, and never itself.
        self.assertEqual(index.similar_values("buckle closure", ["snap closure"], 3, min_cosine=0.8), [])
        self.assertEqual(index.similar_values("buckle closure", ["buckle closure"], 3, min_cosine=0.8), [])
        # Shelves by name embedding; products by text embedding.
        self.assertEqual(index.similar_shelves("karate belts", limit=1, min_cosine=0.5)[0][0], "Martial Arts Karate Belts")
        self.assertEqual(index.search_products("penny loafer", 1), ["LOAFER_C"])


class RankerAlternativesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.catalog = Catalog(_write_catalog(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_alternative_earns_reduced_credit_and_nothing_without_it(self) -> None:
        ids = ["BELT_A", "BELT_B", "BELT_E"]
        # "snap closure" is BELT_B's own string; "buckle closure" is an alternative.
        plain = dict(score_candidates(self.catalog, ids, ["snap closure"], scenario="buying"))
        expanded = dict(score_candidates(
            self.catalog, ids, ["snap closure"], scenario="buying",
            alternatives={"snap closure": [("buckle closure", 0.85)]},
        ))
        self.assertEqual(plain["BELT_B"], expanded["BELT_B"])
        self.assertGreater(expanded["BELT_A"], plain["BELT_A"])
        self.assertGreater(expanded["BELT_B"], expanded["BELT_A"])
        # An empty mapping is the protocol path: identical scores.
        self.assertEqual(plain, dict(score_candidates(self.catalog, ids, ["snap closure"], scenario="buying", alternatives={})))


class DenseOptionsInTheAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.catalog_path = _write_catalog(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _agent(self, script, dense):
        client = ScriptedClient(script)
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client, dense=dense,
        )
        agent.reset("s", {})
        return agent, client

    def test_switched_off_the_index_is_never_consulted(self) -> None:
        dense = StubDense(values={"buckle fastening": [("buckle closure", 0.9)]})
        agent, _ = self._agent([{"intent": "open", "category": "belt", "features": ["buckle fastening"]}], dense)
        agent.respond("s", "I'm looking for Accessories Belts. A key requirement is: leather.", 1, 10)
        agent.respond("s", "need a belt with a buckle fastening", 2, 10)
        self.assertEqual(dense.calls, [])
        self.assertEqual(agent._sessions["s"].constraint_alternatives, {})

    def test_dense_tier_admits_the_nearest_supported_value(self) -> None:
        dense = StubDense(values={"buckle fastening": [("buckle closure", 0.9)]})
        agent, client = self._agent([{"intent": "open", "category": "belts", "features": ["buckle fastening"]}], dense)
        with _config_override(DENSE_VALUE_EXPANSION=True):
            response = agent.respond("s", "need a belt with a buckle fastening", 1, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        accepted = {item["constraint"]: item for item in grounding["accepted"]}
        self.assertEqual(accepted["buckle closure"]["tier"], "signature_dense")
        self.assertEqual(accepted["buckle closure"]["confidence"], config.DENSE_VALUE_TIER_WEIGHT)
        self.assertEqual(accepted["buckle closure"]["from"], "buckle fastening")
        self.assertIn(response["recommendations"][0]["parent_asin"], {"BELT_A", "BELT_E"})
        # Protocol wording on the next turn never reaches the index.
        calls_before = len(dense.calls)
        with _config_override(DENSE_VALUE_EXPANSION=True):
            agent.respond("s", "For that, what matters is: color: black.", 2, 10)
        self.assertEqual(len(dense.calls), calls_before)
        self.assertEqual(len(client.calls), 1)

    def test_expansion_records_alternatives_the_ranker_credits(self) -> None:
        dense = StubDense(values={"snap closure": [("buckle closure", 0.85), ("snap closure", 1.0)]})
        agent, _ = self._agent([{"intent": "open", "category": "belts", "features": ["snap closure"]}], dense)
        with _config_override(DENSE_VALUE_EXPANSION=True):
            response = agent.respond("s", "need a belt that snaps shut", 1, 10)
        state = agent._sessions["s"]
        self.assertEqual(state.constraint_alternatives, {"snap closure": [("buckle closure", 0.85)]})
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertEqual(grounding["accepted"][0]["alternatives"], [("buckle closure", 0.85)])
        # The verbatim product still leads; the alternatives follow the karate belt.
        shown = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(shown[0], "BELT_B")
        # Cancelling the constraint drops its alternatives with it.
        state.remove("snap closure")
        self.assertEqual(state.constraint_alternatives, {})

    def test_typed_values_are_never_expanded(self) -> None:
        dense = StubDense(values={"leather": [("genuine leather", 0.95)], "color: black": [("color: brown", 0.9)]})
        agent, _ = self._agent([{"intent": "open", "category": "belts", "material": "leather", "color": "black"}], dense)
        with _config_override(DENSE_VALUE_EXPANSION=True):
            agent.respond("s", "a black leather belt please", 1, 10)
        self.assertEqual(agent._sessions["s"].constraint_alternatives, {})

    def test_shelf_recall_unions_dense_shelves_into_the_pool(self) -> None:
        dense = StubDense(shelves={"waist strap": [("Accessories Belts", 0.83)]})
        agent, _ = self._agent([{"intent": "open", "category": "waist strap", "features": []}], dense)
        with _config_override(DENSE_SHELF_RECALL=True):
            agent.respond("s", "hunting for a waist strap", 1, 10)
        state = agent._sessions["s"]
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertEqual(grounding["dense_shelves"], ["Accessories Belts"])
        self.assertIn("Accessories Belts", state.pool_shelves)
        self.assertIn("BELT_A", state.candidate_pool)
        self.assertEqual(state.grounded_messages, ["hunting for a waist strap"])

    def test_widening_unions_dense_products_after_misses(self) -> None:
        dense = StubDense(shelves={"waist strap": [("Martial Arts Karate Belts", 0.83)]}, products=["BELT_A"])
        agent, _ = self._agent([{"intent": "open", "category": "waist strap", "features": []}], dense)
        with _config_override(DENSE_SHELF_RECALL=True, LLM_POOL_WIDEN_AFTER_MISSES=1):
            agent.respond("s", "hunting for a waist strap", 1, 10)
            state = agent._sessions["s"]
            self.assertEqual(state.candidate_pool, ["KARATE_D"])
            state.proven_misses.add("KARATE_D")
            candidates = agent._beyond_pool(state, dense)
        self.assertIn("BELT_A", candidates)
        self.assertIn(("products", "hunting for a waist strap"), dense.calls)


if __name__ == "__main__":
    unittest.main()
