"""The optional language layer: off by default, catalog-verified when on."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent
from src import config
from src.dialog import SessionState, parse
from src.llm import FakeChatClient, LLMReply, LLMSettings
from src.shelf import Catalog

PRODUCTS = [
    {
        "parent_asin": "BELT_A",
        "title": "Hide & Drink rustic full grain leather belt",
        "features": ["100% Leather", "Buckle closure", "Imported"],
        "details": {"Department": "Mens", "Color": "Brown"},
        "description": ["Handmade brown leather belt with a heavy buckle."],
        "categories": ["Clothing", "Men", "Accessories", "Belts"],
        "price": 28.0,
        "average_rating": 4.7,
        "rating_number": 900,
        "store": "Hide & Drink",
    },
    {
        "parent_asin": "BELT_B",
        "title": "Sporty polyester web belt",
        "features": ["Polyester", "Snap closure"],
        "details": {"Department": "Mens"},
        "description": ["Black polyester belt."],
        "categories": ["Clothing", "Men", "Accessories", "Belts"],
        "price": 11.0,
        "average_rating": 4.1,
        "rating_number": 40,
        "store": "Sporty",
    },
    {
        "parent_asin": "LOAFER_C",
        "title": "Classic penny loafer",
        "features": ["Leather sole", "Slip-on"],
        "details": {"Department": "Mens"},
        "description": ["Black leather loafer."],
        "categories": ["Clothing", "Men", "Shoes", "Loafers & Slip-Ons"],
        "price": 60.0,
        "average_rating": 4.4,
        "rating_number": 300,
        "store": "Classic",
    },
    {
        "parent_asin": "BELT_E",
        "title": "Plain dress belt",
        "features": ["Genuine Leather", "Buckle closure"],
        "details": {"Department": "Mens"},
        "description": ["Black dress belt."],
        "categories": ["Clothing", "Men", "Accessories", "Belts"],
        "price": 20.0,
        "average_rating": 4.2,
        "rating_number": 120,
        "store": "Plain",
    },
    {
        "parent_asin": "KARATE_D",
        "title": "Karate rank belt",
        "features": ["Cotton", "Double stitched"],
        "details": {},
        "description": ["White cotton karate belt."],
        "categories": ["Sports", "Martial Arts", "Karate Belts"],
        "price": 8.0,
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Dojo",
    },
]


class ScriptedClient(FakeChatClient):
    """Returns canned JSON replies in order, regardless of the prompt."""

    def __init__(self, script: list[dict], *, fail: bool = False) -> None:
        super().__init__(fail=fail)
        self.script = [json.dumps(item) for item in script]

    def chat(self, messages, *, max_tokens=200, temperature=0.0) -> LLMReply:
        self.calls.append(messages)
        if self.fail:
            return LLMReply(text="", error="fake failure")
        text = self.script.pop(0) if self.script else "{}"
        return LLMReply(text=text, prompt_tokens=100, completion_tokens=20, latency_ms=1.0)


def _write_catalog(directory: str) -> Path:
    path = Path(directory) / "catalog.jsonl"
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in PRODUCTS),
        encoding="utf-8",
    )
    return path


class LLMSettingsTest(unittest.TestCase):
    def test_off_unless_mode_and_endpoint_are_both_set(self) -> None:
        self.assertFalse(LLMSettings.from_env({}).enabled)
        self.assertFalse(LLMSettings.from_env({"ARC_LLM_MODE": "ground"}).enabled)
        self.assertFalse(
            LLMSettings.from_env({"ARC_LLM_BASE_URL": "http://x/v1"}).enabled
        )
        settings = LLMSettings.from_env(
            {"ARC_LLM_MODE": "assist", "ARC_LLM_BASE_URL": "http://x/v1/"}
        )
        self.assertTrue(settings.enabled)
        self.assertTrue(settings.says)
        self.assertEqual(settings.base_url, "http://x/v1")
        self.assertEqual(LLMSettings.from_env({"ARC_LLM_MODE": "banana"}).mode, "off")

    def test_reply_json_tolerates_code_fences(self) -> None:
        reply = LLMReply(text='```json\n{"intent": "open", "features": ["a"]}\n```')
        self.assertEqual(reply.json(), {"intent": "open", "features": ["a"]})
        self.assertIsNone(LLMReply(text="not json").json())


class GroundingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.catalog_path = _write_catalog(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parse_reports_whether_a_template_matched(self) -> None:
        catalog = Catalog(self.catalog_path)
        state = SessionState({})
        self.assertTrue(parse(
            "I'm looking for Accessories Belts. A key requirement is: 100% Leather.",
            state, catalog,
        ))
        self.assertFalse(parse("need a belt for my husband, rustic, buckle", state, catalog))

    def test_unknown_category_reports_nothing_learned(self) -> None:
        catalog = Catalog(self.catalog_path)
        state = SessionState({})
        self.assertFalse(parse("I'm looking for some alloy necklaces.", state, catalog))
        self.assertIsNone(state.shelf)
        self.assertTrue(parse(
            "Those options are not quite right yet. Ask me about one specific attribute.",
            state, catalog,
        ))

    def test_shelf_pool_widens_a_tiny_best_match(self) -> None:
        catalog = Catalog(self.catalog_path)
        shelves, pool = catalog.shelf_pool("leather belt")
        self.assertIn("Accessories Belts", shelves)
        self.assertEqual(set(pool), {"BELT_A", "BELT_B", "BELT_E", "KARATE_D"})
        # "karate belt" matches one two-product shelf best; widening keeps
        # every other belt shelf reachable.
        narrow_shelves, narrow_pool = catalog.shelf_pool("karate belt", min_products=0)
        self.assertEqual(narrow_pool, ["KARATE_D"])
        wide_shelves, wide_pool = catalog.shelf_pool("karate belt", min_products=3)
        self.assertGreater(len(wide_pool), len(narrow_pool))
        self.assertEqual(wide_shelves[0], narrow_shelves[0])
        self.assertEqual(catalog.shelf_pool("zzz"), ([], []))

    def test_vocabulary_hints_prefer_short_common_phrases(self) -> None:
        catalog = Catalog(self.catalog_path)
        counts = {
            "imported": 50, "buckle closure": 20, "color: black": 30, "leather": 40,
            "12345": 9, "a": 1, "made to last by artisans in a faraway workshop": 15,
        }
        hints = catalog.top_signature_values(counts, limit=10)
        self.assertEqual(hints[0], "buckle closure")
        for banned in ("imported", "color: black", "leather", "12345", "a"):
            self.assertNotIn(banned, hints)
        self.assertNotIn("made to last by artisans in a faraway workshop", hints)

    def test_dead_endpoint_opens_the_circuit_and_stops_waiting(self) -> None:
        from src.llm import ChatClient, LLMSettings

        # Port 1 on loopback refuses instantly: no network, no timeout.
        client = ChatClient(LLMSettings(
            mode="ground", base_url="http://127.0.0.1:1/v1", model="none",
            timeout=1.0,
        ))
        message = [{"role": "user", "content": "hi"}]
        for _ in range(config.LLM_CIRCUIT_FAILURES):
            self.assertFalse(client.chat(message).ok)
        blocked = client.chat(message)
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.error, "circuit_open")
        self.assertEqual(blocked.prompt_tokens, 0)

    def test_off_mode_spends_no_tokens_on_any_wording(self) -> None:
        agent = Agent(self.catalog_path, llm_settings=LLMSettings(mode="off"))
        self.assertIsNone(agent.llm)
        agent.reset("s", {})
        free = agent.respond("s", "need a belt, rustic, real leather", 1, 10)
        agent.reset("t", {})
        protocol = agent.respond(
            "t", "I'm looking for Accessories Belts. A key requirement is: 100% Leather.", 1, 10
        )
        for response in (free, protocol):
            self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_protocol_messages_never_reach_the_model(self) -> None:
        client = ScriptedClient([{"intent": "open"}])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        agent.respond(
            "s", "I'm looking for Accessories Belts. A key requirement is: 100% Leather.", 1, 10
        )
        agent.respond("s", "For that, what matters is: Buckle closure; Imported.", 2, 10)
        self.assertEqual(client.calls, [])
        self.assertEqual(
            agent.explain_last_decision("s").get("llm_usage"), None
        )

    def test_free_text_is_grounded_to_catalog_evidence(self) -> None:
        client = ScriptedClient([{
            "intent": "open",
            "category": "leather belt",
            "material": "leather",
            "color": "brown",
            "budget": 30,
            "features": ["buckle", "unicorn glitter"],
            "dropped": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        response = agent.respond(
            "s", "looking for a belt for my husband, rustic, brown, real leather, under 30", 1, 10
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(response["usage"], {"prompt_tokens": 100, "completion_tokens": 20})
        self.assertEqual(response["recommendations"][0]["parent_asin"], "BELT_A")
        certificate = agent.explain_last_decision("s")
        grounding = certificate["llm_grounding"]
        self.assertEqual(grounding["status"], "grounded")
        accepted = {item["constraint"]: item["tier"] for item in grounding["accepted"]}
        self.assertEqual(accepted["buckle closure"], "signature_mapped")
        self.assertEqual(accepted["leather"], "typed_material")
        self.assertEqual(accepted["color: brown"], "typed_color")
        self.assertEqual(accepted["budget around $30"], "typed_budget")
        self.assertEqual(
            [item["phrase"] for item in grounding["rejected"]], ["unicorn glitter"]
        )
        self.assertIn("Accessories Belts", grounding["pool_shelves"])
        self.assertTrue(any("Karate Belts" in shelf for shelf in grounding["pool_shelves"]))
        self.assertEqual(grounding["pool_size"], 4)
        self.assertEqual(certificate["pool_shelves"], grounding["pool_shelves"])
        confidence = dict(zip(certificate["constraints"], certificate["constraint_confidence"]))
        self.assertLess(confidence["buckle closure"], confidence["leather"])

    def test_override_forgets_the_dropped_constraint_and_the_old_misses(self) -> None:
        client = ScriptedClient([
            {
                "intent": "open", "category": "belt", "material": "leather",
                "features": ["buckle"], "dropped": [],
            },
            {
                "intent": "override", "color": "black", "features": [],
                "dropped": ["leather"],
            },
        ])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        first = agent.respond("s", "belt, leather, with a buckle", 1, 10)
        shown = [item["parent_asin"] for item in first["recommendations"]]
        self.assertTrue(shown)
        agent.respond("s", "actually forget the leather, make it black", 2, 10)
        certificate = agent.explain_last_decision("s")
        confidence = dict(zip(certificate["constraints"], certificate["constraint_confidence"]))
        self.assertEqual(confidence["leather"], config.OVERRIDE_DECAY)
        self.assertIn("color: black", certificate["constraints"])
        self.assertEqual(certificate["proven_miss_count"], 0)
        self.assertEqual(
            certificate["llm_grounding"]["removed"][0]["action"], "decayed"
        )
        original = config.LLM_GROUND_DROP_MODE
        try:
            config.LLM_GROUND_DROP_MODE = "remove"
            state = agent._sessions["s"]
            from src.ground import _demote_dropped
            self.assertEqual(_demote_dropped(state, ["leather"])[0]["action"], "removed")
            self.assertNotIn("leather", state.constraints)
        finally:
            config.LLM_GROUND_DROP_MODE = original

    def test_no_preference_exhausts_the_attribute_just_asked(self) -> None:
        client = ScriptedClient([{"intent": "no_preference"}])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        first = agent.respond(
            "s", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10
        )
        asked = first["ask_attribute"]
        self.assertIsNotNone(asked)
        agent.respond("s", "honestly i don't care about that", 2, 10)
        certificate = agent.explain_last_decision("s")
        state = agent._sessions["s"]
        if asked == "other":
            self.assertTrue(state.information_complete)
        else:
            self.assertIn(asked, state.exhausted)
        self.assertEqual(certificate["llm_grounding"]["intent"], "no_preference")
        self.assertEqual(client.calls[0][-1]["role"], "user")

    def test_unknown_material_is_verified_as_a_feature_phrase(self) -> None:
        client = ScriptedClient([{
            "intent": "open", "category": "belt", "material": "full grain",
            "color": "chestnut", "features": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        agent.respond("s", "a full grain belt in chestnut", 1, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        accepted = {item["constraint"]: item["tier"] for item in grounding["accepted"]}
        self.assertEqual(accepted.get("full grain"), "lexical")
        self.assertEqual(
            [item["phrase"] for item in grounding["rejected"]], ["chestnut"]
        )

    def test_reply_without_new_evidence_retires_the_question_asked(self) -> None:
        client = ScriptedClient([{"intent": "reject"}])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        first = agent.respond(
            "s", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10
        )
        asked = first["ask_attribute"]
        second = agent.respond("s", "hmm not those", 2, 10)
        state = agent._sessions["s"]
        if asked == "other":
            self.assertTrue(state.information_complete)
        else:
            self.assertIn(asked, state.exhausted)
            self.assertNotEqual(second["ask_attribute"], asked)
        self.assertEqual(agent.explain_last_decision("s")["llm_grounding"]["exhausted"], asked)

    def test_model_failure_degrades_to_the_deterministic_path(self) -> None:
        client = ScriptedClient([], fail=True)
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="assist", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        response = agent.respond("s", "belt, leather, with a buckle", 1, 10)
        self.assertIsInstance(response["message"], str)
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertEqual(
            agent.explain_last_decision("s")["llm_grounding"]["status"], "llm_unavailable"
        )

    def test_assist_mode_renders_the_message_but_not_the_ranking(self) -> None:
        client = ScriptedClient([
            {"intent": "open", "category": "belt", "material": "leather", "features": ["buckle"]},
        ])
        client.script.append("Great, a leather belt with a buckle it is. Any colour in mind?")
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="assist", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        response = agent.respond("s", "belt, leather, with a buckle", 1, 10)
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(response["message"].startswith("Great, a leather belt"))
        self.assertEqual(response["recommendations"][0]["parent_asin"], "BELT_A")
        self.assertEqual(response["usage"], {"prompt_tokens": 200, "completion_tokens": 40})


if __name__ == "__main__":
    unittest.main()
