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
    {
        "parent_asin": "SNEAKER_G",
        "title": "Trail running sneaker",
        "features": ["Running shoes", "Breathable mesh", "Rubber sole"],
        "details": {"Department": "Mens"},
        "description": ["Lightweight black running shoe for the gym."],
        "categories": ["Clothing", "Men", "Shoes", "Running Shoes"],
        "price": 55.0,
        "average_rating": 4.5,
        "rating_number": 500,
        "store": "Trail",
    },
    {
        "parent_asin": "GIFT_F",
        "title": "Ugly holiday sweater for the family",
        "features": ["Polyester", "Pullover"],
        "details": {},
        "description": ["Festive sweater for the whole family and the dog."],
        "categories": ["Clothing", "Gifts", "Gift Guide", "for the Family"],
        "price": 30.0,
        "average_rating": 4.3,
        "rating_number": 70,
        "store": "Festive",
    },
]


class _config_override:
    """Set research switches on ``src.config`` for one block, then restore."""

    def __init__(self, **values) -> None:
        self.values = values
        self.saved: dict = {}

    def __enter__(self) -> None:
        for key, value in self.values.items():
            self.saved[key] = getattr(config, key)
            setattr(config, key, value)

    def __exit__(self, *exc) -> None:
        for key, value in self.saved.items():
            setattr(config, key, value)


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

    def test_explicit_price_ceiling_filters_the_grounded_pool(self) -> None:
        reply = json.dumps({
            "intent": "open", "category": "belt", "material": None, "color": None,
            "budget": 22, "budget_operator": "under", "features": [], "dropped": [],
        })
        agent = Agent(self.catalog_path, llm_settings=LLMSettings(mode="ground", base_url="fake://"), llm_client=FakeChatClient({"*": reply}))
        agent.reset("s", {"preference_tags": []})
        response = agent.respond("s", "need a belt under 22 bucks", 1, 10)
        certificate = agent.explain_last_decision("s")
        bound = certificate["llm_grounding"]["budget_bound"]
        self.assertTrue(bound["applied"])
        self.assertEqual(bound["operator"], "under")
        state = agent._sessions["s"]
        self.assertNotIn("BELT_A", state.candidate_pool)   # $28 is above the ceiling
        self.assertIn("BELT_B", state.candidate_pool)      # $12
        self.assertIn("BELT_E", state.candidate_pool)      # $20
        self.assertNotIn("BELT_A", [r["parent_asin"] for r in response["recommendations"]])
        # A ceiling nobody satisfies is relaxed rather than stranding the session.
        reply_low = json.dumps({
            "intent": "open", "category": "belt", "material": None, "color": None,
            "budget": 1, "budget_operator": "under", "features": [], "dropped": [],
        })
        agent2 = Agent(self.catalog_path, llm_settings=LLMSettings(mode="ground", base_url="fake://"), llm_client=FakeChatClient({"*": reply_low}))
        agent2.reset("t", {"preference_tags": []})
        agent2.respond("t", "belt under a dollar", 1, 10)
        bound2 = agent2.explain_last_decision("t")["llm_grounding"]["budget_bound"]
        self.assertFalse(bound2["applied"])
        self.assertEqual(bound2["reason"], "empty_after_filter")

    def test_a_later_budget_statement_supersedes_the_ceiling(self) -> None:
        first = json.dumps({"intent": "open", "category": "belt", "material": None, "color": None,
                            "budget": 22, "budget_operator": "under", "features": [], "dropped": []})
        second = json.dumps({"intent": "add", "category": None, "material": None, "color": None,
                             "budget": 30, "budget_operator": "around", "features": [], "dropped": []})
        agent = Agent(self.catalog_path, llm_settings=LLMSettings(mode="ground", base_url="fake://"),
                      llm_client=ScriptedClient([json.loads(first), json.loads(second)]))
        agent.reset("s", {"preference_tags": []})
        agent.respond("s", "belt under 22 bucks", 1, 10)
        state = agent._sessions["s"]
        self.assertNotIn("BELT_A", state.candidate_pool)
        agent.respond("s", "actually around 30 is fine", 2, 10)
        self.assertIn("BELT_A", state.candidate_pool)
        self.assertIn("budget around $30", state.constraints)
        self.assertNotIn("budget around $22", state.constraints)
        self.assertEqual(agent.explain_last_decision("s")["llm_grounding"]["budget_bound"]["reason"], "previous_bound_lifted")

    def test_budget_operator_falls_back_to_the_shoppers_words(self) -> None:
        from src.ground import _budget_operator
        self.assertEqual(_budget_operator(None, "no more than 40 dollars please"), "under")
        self.assertEqual(_budget_operator("around", "under 40"), "around")
        self.assertEqual(_budget_operator(None, "at least 40"), "over")
        self.assertEqual(_budget_operator(None, "roughly 40"), "around")

    def test_model_outage_falls_back_to_a_token_overlap_pool(self) -> None:
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=FakeChatClient(fail=True),
        )
        agent.reset("s", {"preference_tags": []})
        response = agent.respond("s", "need a rustic leather belt with a buckle", 1, 10)
        certificate = agent.explain_last_decision("s")
        grounding = certificate["llm_grounding"]
        # The catalog stage still read the sentence: the model outage is
        # recorded, the session is not stranded.
        self.assertEqual(grounding["status"], "grounded")
        self.assertEqual(grounding["model_status"], "llm_unavailable")
        self.assertIn("Accessories Belts", grounding["pool_shelves"])
        self.assertEqual(
            [item["constraint"] for item in grounding["accepted"]], ["leather"]
        )
        self.assertTrue(all(
            agent.catalog.shelf_of[r["parent_asin"]].endswith("Belts")
            for r in response["recommendations"]
        ))
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_flush_merges_with_entries_another_client_wrote(self) -> None:
        import tempfile
        from src.llm import ChatClient, LLMSettings
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "cache.json")
            a = ChatClient(LLMSettings(mode="ground", base_url="fake://", cache_path=path))
            b = ChatClient(LLMSettings(mode="ground", base_url="fake://", cache_path=path))
            a._cache["ka"] = {"text": "A", "prompt_tokens": 1, "completion_tokens": 1}
            a._cache_dirty = True
            a.flush()
            b._cache["kb"] = {"text": "B", "prompt_tokens": 1, "completion_tokens": 1}
            b._cache_dirty = True
            b.flush()
            on_disk = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(set(on_disk), {"ka", "kb"})
            replay = ChatClient(LLMSettings(mode="ground", base_url="fake://", cache_path=path), replay_only=True)
            miss = replay.chat([{"role": "user", "content": "never cached"}])
            self.assertFalse(miss.ok)
            self.assertEqual(replay.misses, 1)

    def test_lexical_grounder_admits_only_verbatim_catalog_strings(self) -> None:
        agent = Agent(self.catalog_path, llm_settings=LLMSettings(mode="lexical"))
        agent.reset("s", {"preference_tags": []})
        response = agent.respond("s", "need a leather belt with buckle closure, brown, under 25 dollars", 1, 10)
        certificate = agent.explain_last_decision("s")
        grounding = certificate["llm_grounding"]
        self.assertEqual(grounding["grounder"], "lexical")
        tiers = {item["constraint"]: item["tier"] for item in grounding["accepted"]}
        self.assertEqual(tiers.get("buckle closure"), "ngram_signature")
        self.assertEqual(tiers.get("leather"), "typed_material")
        self.assertEqual(tiers.get("color: brown"), "typed_color")
        self.assertEqual(tiers.get("budget around $25"), "typed_budget_under")
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertTrue(grounding["budget_bound"]["applied"])
        # A paraphrase that is not a catalog string admits nothing.
        agent.reset("t", {"preference_tags": []})
        agent.respond("t", "something that fastens with a metal clasp", 1, 10)
        self.assertEqual(agent.explain_last_decision("t")["llm_grounding"]["accepted"], [])
        # Dialogue acts still work without a model.
        from src.ground import _lexical_intent
        self.assertEqual(_lexical_intent("honestly i don't care about that"), "no_preference")
        self.assertEqual(_lexical_intent("actually, forget the leather"), "override")
        self.assertEqual(_lexical_intent("those aren't it, try again"), "reject")

    def test_cascade_reads_verbatim_catalog_strings_without_the_model(self) -> None:
        client = ScriptedClient([])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        # The department was named on the protocol turn, so the session is
        # placed; every attribute in the free-form reply is a verbatim
        # catalog string. No call.
        agent.respond("s", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10)
        response = agent.respond(
            "s", "need a brown leather one with buckle closure, under 25 dollars", 2, 10
        )
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertEqual(grounding["grounder"], "cascade")
        self.assertFalse(grounding["stage1"]["escalate"])
        self.assertEqual(grounding["stage1"]["reason"], "catalog_string_found")
        self.assertEqual(client.calls, [])
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        tiers = {item["constraint"]: item["tier"] for item in grounding["accepted"]}
        self.assertEqual(tiers["buckle closure"], "ngram_signature")
        self.assertEqual(tiers["leather"], "typed_material")
        self.assertEqual(tiers["color: brown"], "typed_color")
        # "under 25 dollars" is a hard ceiling: the $28 belt is out, the $20
        # leather buckle belt wins.
        self.assertTrue(grounding["budget_bound"]["applied"])
        shown = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(shown[0], "BELT_E")
        self.assertNotIn("BELT_A", shown)

    def test_cascade_asks_the_model_to_name_an_unplaced_department(self) -> None:
        # "belt" is not a shelf name. The catalog read every attribute, but
        # placing the shopper in a department by the token overlap of the
        # whole sentence with shelf names is a last resort, not a reading:
        # the model names the category, and the price ceiling stated before
        # any department existed is applied to the pool it chooses.
        client = ScriptedClient([{
            "intent": "open", "category": "belt", "material": None, "color": None,
            "budget": None, "features": [], "dropped": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        # The fixture's belt shelves hold four products, which the real
        # catalog's threshold would call a firm one-word placement; lower it
        # so "belt" alone counts as weak here.
        with _config_override(LLM_GROUND_CONFIDENT_POOL=0):
            response = agent.respond(
                "s", "need a brown leather belt with buckle closure, under 25 dollars", 1, 10
            )
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertTrue(grounding["stage1"]["escalate"])
        self.assertEqual(grounding["stage1"]["reason"], "department_unplaced")
        self.assertEqual(grounding["placement"], {"shared_words": 1, "pool_size": 4})
        self.assertEqual(len(client.calls), 1)
        self.assertIn("Accessories Belts", grounding["pool_shelves"])
        self.assertNotIn("Karate Belts", grounding["pool_shelves"])
        self.assertTrue(grounding["budget_bound"]["applied"])
        shown = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(shown[0], "BELT_E")
        self.assertNotIn("BELT_A", shown)

    def test_a_shelf_named_in_free_text_is_placed_without_the_model(self) -> None:
        client = ScriptedClient([])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        agent.respond("s", "hmm, Accessories Belts I guess, with buckle closure", 1, 10)
        certificate = agent.explain_last_decision("s")
        # The template parser reads any exact shelf name; the grounder is
        # not needed for the department and the model is not called.
        self.assertNotIn("llm_grounding", certificate)
        self.assertEqual(client.calls, [])
        self.assertEqual(agent._sessions["s"].shelf, "Accessories Belts")

    def test_cascade_consults_the_model_for_a_paraphrase(self) -> None:
        client = ScriptedClient([{
            "intent": "open", "category": "belt", "material": "leather", "color": None,
            "budget": None, "features": ["buckle"], "dropped": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        response = agent.respond("s", "something in real leather that fastens with a buckle", 1, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertTrue(grounding["stage1"]["escalate"])
        self.assertEqual(grounding["stage1"]["reason"], "no_feature_evidence")
        # Stage one already admitted the typed material; the model added the
        # feature the catalog could not read verbatim, and it was verified.
        self.assertEqual(grounding["stage1"]["accepted"], ["leather"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(grounding["model_status"], "grounded")
        tiers = {item["constraint"]: item["tier"] for item in grounding["accepted"]}
        self.assertEqual(tiers["buckle closure"], "signature_mapped")
        self.assertEqual(response["usage"], {"prompt_tokens": 100, "completion_tokens": 20})
        # The model's category, not the whole sentence, chose the pool.
        self.assertIn("Accessories Belts", grounding["pool_shelves"])

    def test_cascade_repairs_a_near_miss_json_reply_and_retires_the_question(self) -> None:
        broken = ('```json\n{"intent":"no_preference","category":null,"material":null,'
                  '"color":null,"budget":null,"budget_operator":null,"features":[],"dropped:[]}\n```')
        client = FakeChatClient({"*": broken})
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        first = agent.respond("s", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10)
        asked = first["ask_attribute"]
        # Wording the catalog stage does not recognise as a dialogue act, so
        # the model is consulted and its reply needs repair.
        agent.respond("s", "meh, either way is fine by me", 2, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertEqual(grounding["model_status"], "grounded")
        self.assertEqual(grounding["intent"], "no_preference")
        self.assertEqual(grounding["exhausted"], asked)

    def test_cascade_survives_a_failed_model_reply_without_looping(self) -> None:
        client = FakeChatClient({"*": "not json at all"})
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        first = agent.respond("s", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10)
        asked = first["ask_attribute"]
        second = agent.respond("s", "meh, either way is fine by me", 2, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertEqual(grounding["model_status"], "unparseable_reply")
        self.assertEqual(grounding["status"], "grounded")
        # A reply that taught the reader nothing retires the question asked,
        # exactly as the protocol's "no additional preference" would, so the
        # agent does not ask the same thing again.
        self.assertEqual(grounding["exhausted"], asked)
        self.assertNotEqual(second["ask_attribute"], asked)

    def test_cascade_sends_a_cancellation_to_the_model(self) -> None:
        client = ScriptedClient([{
            "intent": "override", "category": None, "material": None, "color": "black",
            "budget": None, "features": [], "dropped": ["leather"],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10)
        agent.respond("s", "need a leather one with buckle closure", 2, 10)
        self.assertEqual(client.calls, [])
        agent.respond("s", "actually, forget the leather, black is what matters", 3, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertEqual(grounding["stage1"]["reason"], "cancellation_needs_dropped_list")
        self.assertEqual(len(client.calls), 1)
        removed = {item["constraint"]: item["action"] for item in grounding["removed"]}
        self.assertEqual(removed, {"leather": "decayed"})
        state = agent._sessions["s"]
        weights = dict(zip(state.constraints, state.constraint_weights()))
        self.assertEqual(weights["leather"], 0.5)
        self.assertEqual(weights["buckle closure"], 1.0)
        self.assertEqual(weights["color: black"], 1.0)

    def test_keyword_cancellation_is_only_a_proposal_until_the_model_agrees(self) -> None:
        client = ScriptedClient([{
            "intent": "reject", "category": None, "material": None, "color": None,
            "budget": None, "features": [], "dropped": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10)
        first = agent.respond("s", "need a leather one with buckle closure", 2, 10)
        shown = [item["parent_asin"] for item in first["recommendations"]]
        # "instead" reads as a cancellation to the keyword stage, but the
        # model reads the sentence as a plain rejection: the refuted slate
        # must stay refuted and nothing may be decayed.
        agent.respond("s", "not those, show me something else instead", 3, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertEqual(grounding["stage1"]["intent"], "override")
        self.assertEqual(grounding["intent"], "reject")
        self.assertEqual(grounding["removed"], [])
        state = agent._sessions["s"]
        self.assertTrue(set(shown) <= state.proven_misses)
        self.assertFalse(state.override_seen)
        self.assertEqual(set(state.constraint_weights()), {1.0})

    def test_positive_follow_up_keeps_the_previous_slate(self) -> None:
        client = ScriptedClient([{
            "intent": "add", "category": None, "material": None, "color": "brown",
            "budget": None, "features": [], "dropped": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        first = agent.respond("s", "need a leather belt with buckle closure", 1, 10)
        shown = [item["parent_asin"] for item in first["recommendations"]]
        self.assertTrue(shown)
        second = agent.respond("s", "the first one looks good, does it come in brown?", 2, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertTrue(grounding["keeps_slate"])
        state = agent._sessions["s"]
        self.assertFalse(set(shown) & state.proven_misses)
        self.assertIn(shown[0], [item["parent_asin"] for item in second["recommendations"]])
        # An explicit rejection is still a refutation.
        agent.respond("s", "no, those aren't it", 3, 10)
        state = agent._sessions["s"]
        self.assertTrue(set(shown) <= state.proven_misses)
        from src.ground import refers_to_shown_products
        self.assertFalse(refers_to_shown_products("not those, the first one is wrong"))
        self.assertTrue(refers_to_shown_products("I like the second one, can I get it in blue?"))
        # A stated preference is not a pointer at the slate.
        self.assertFalse(refers_to_shown_products("I love that the strap is adjustable"))
        self.assertFalse(refers_to_shown_products("I really like the screw-on tunnels"))

    def test_unverified_switch_admits_any_proposal(self) -> None:
        reply = json.dumps({"intent": "open", "category": "belt", "material": None, "color": None,
                            "budget": None, "features": ["unicorn glitter"], "dropped": []})
        original = config.LLM_GROUND_VERIFY
        try:
            config.LLM_GROUND_VERIFY = False
            agent = Agent(self.catalog_path, llm_settings=LLMSettings(mode="ground", base_url="fake://"),
                          llm_client=FakeChatClient({"*": reply}))
            agent.reset("s", {"preference_tags": []})
            agent.respond("s", "a belt with unicorn glitter", 1, 10)
            accepted = {a["constraint"]: a["tier"] for a in agent.explain_last_decision("s")["llm_grounding"]["accepted"]}
            self.assertEqual(accepted.get("unicorn glitter"), "unverified")
        finally:
            config.LLM_GROUND_VERIFY = original

    def test_llm_agent_baseline_only_returns_shortlisted_ids(self) -> None:
        from tools.llm_agent_baseline import LLMAgent
        client = FakeChatClient({"*": json.dumps({"ask_attribute": "color", "ranked_ids": ["BELT_B", "NOT_A_PRODUCT", "BELT_A"]})})
        agent = LLMAgent(self.catalog_path, client, shortlist=10)
        agent.reset("s", {"preference_tags": []})
        response = agent.respond("s", "looking for a leather belt", 1, 10)
        ids = [r["parent_asin"] for r in response["recommendations"]]
        self.assertEqual(ids, ["BELT_B", "BELT_A"])
        self.assertEqual(response["ask_attribute"], "color")
        self.assertEqual(response["usage"], {"prompt_tokens": 100, "completion_tokens": 20})
        broken = LLMAgent(self.catalog_path, FakeChatClient({"*": "not json"}), shortlist=10)
        broken.reset("s", {"preference_tags": []})
        fallback = broken.respond("s", "looking for a leather belt", 1, 10)
        self.assertTrue(fallback["recommendations"])
        self.assertEqual(broken.fallbacks, 1)

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
        self.assertIn(accepted["budget around $30"], ("typed_budget_around", "typed_budget_under"))
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
        # A recognised dialogue act is fully explained by the catalog stage;
        # the model is not consulted for it.
        self.assertEqual(client.calls, [])
        self.assertEqual(
            certificate["llm_grounding"]["stage1"]["reason"], "dialogue_act_recognised"
        )
        # The model-only reader still handles the same message through the model.
        with _config_override(LLM_GROUND_CASCADE=False):
            agent.reset("m", {})
            agent.respond("m", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10)
            agent.respond("m", "honestly i don't care about that", 2, 10)
        self.assertEqual(client.calls[0][-1]["role"], "user")
        self.assertEqual(agent.explain_last_decision("m")["llm_grounding"]["grounder"], "llm")

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
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertEqual(grounding["status"], "grounded")
        self.assertEqual(grounding["model_status"], "llm_unavailable")

    def test_shelf_pool_ignores_function_words(self) -> None:
        catalog = Catalog(self.catalog_path)
        # "for the" must not carry a sentence to "Gift Guide for the Family".
        shelves, pool = catalog.shelf_pool("running shoes for the gym, breathable and lightweight")
        self.assertEqual(shelves[0], "Shoes Running Shoes")
        self.assertEqual(pool[0], "SNEAKER_G")
        self.assertNotIn("Gift Guide for the Family", shelves)
        self.assertEqual(catalog.shelf_pool("for the and with"), ([], []))

    def test_a_sentence_no_shelf_word_can_place_goes_to_the_model(self) -> None:
        client = ScriptedClient([{
            "intent": "open", "category": "running shoes", "material": None, "color": None,
            "budget": None, "features": [], "dropped": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        response = agent.respond("s", "I need something with a rubber sole for the gym", 1, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        # Stage one matched "rubber sole" verbatim (a feature string of one
        # product) but no word of the sentence names a shelf: the model is
        # asked to name the department instead of a token overlap.
        self.assertEqual(grounding["stage1"]["accepted"], ["rubber sole"])
        self.assertEqual(grounding["stage1"]["reason"], "department_unplaced")
        self.assertEqual(grounding["placement"], {"shared_words": 0, "pool_size": 0})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(grounding["pool_shelves"][0], "Shoes Running Shoes")
        self.assertNotIn("Gift Guide for the Family", grounding["pool_shelves"])
        self.assertEqual(response["recommendations"][0]["parent_asin"], "SNEAKER_G")

    def test_a_category_phrase_the_model_names_is_not_kept_as_a_constraint(self) -> None:
        client = ScriptedClient([{
            "intent": "open", "category": "running shoes", "material": None, "color": None,
            "budget": None, "features": ["breathable mesh"], "dropped": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        response = agent.respond("s", "I need running shoes for the gym, breathable please", 1, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        # "running shoes" is both a feature string of one product and the
        # department; "breathable" is attribute vocabulary stage one could
        # not read verbatim, so the model is consulted.
        self.assertEqual(grounding["stage1"]["accepted"], ["running shoes"])
        self.assertEqual(grounding["stage1"]["reason"], "attribute_words_left_unread")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(grounding["pool_shelves"][0], "Shoes Running Shoes")
        # The model called that phrase the category, so it is no longer a
        # constraint pinning that single product; the feature it read stays.
        state = agent._sessions["s"]
        self.assertNotIn("running shoes", state.constraints)
        self.assertEqual(grounding["reclassified"][0]["constraint"], "running shoes")
        self.assertIn("breathable mesh", state.constraints)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "SNEAKER_G")

    def test_a_category_named_while_adding_a_preference_does_not_switch(self) -> None:
        client = ScriptedClient([
            {"intent": "open", "category": "belt", "features": ["buckle closure"], "dropped": []},
            # The shopper quotes a product description that happens to name
            # another department; the intent is "add", so the belts stay.
            {"intent": "add", "category": "running shoes", "features": ["rubber sole"], "dropped": []},
        ])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        agent.respond("s", "a belt with a buckle", 1, 10)
        agent.respond("s", "the one described as running shoes style with a rubber sole", 2, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertNotIn("department_switch", grounding)
        self.assertEqual(grounding["category_kept"]["reason"], "not_a_new_request:add")
        self.assertIn("Accessories Belts", agent._sessions["s"].pool_shelves)

    def test_category_switch_moves_the_session_to_the_new_department(self) -> None:
        client = ScriptedClient([
            {
                "intent": "open", "category": "belt", "material": "leather", "color": None,
                "budget": 40, "budget_operator": "under", "features": ["buckle closure"], "dropped": [],
            },
            {
                "intent": "override", "category": "loafers", "material": None, "color": None,
                "budget": None, "features": [], "dropped": ["belt"],
            },
        ])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        first = agent.respond("s", "a leather belt with a buckle, under 40 dollars", 1, 10)
        self.assertTrue(all(
            agent.catalog.shelf_of[r["parent_asin"]] == "Accessories Belts"
            for r in first["recommendations"]
        ))
        second = agent.respond("s", "actually forget the belt, I want loafers instead", 2, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        switch = grounding["department_switch"]
        self.assertEqual(switch["category"], "loafers")
        self.assertIn("Accessories Belts", switch["from"])
        self.assertEqual(switch["to"], ["Shoes Loafers & Slip-Ons"])
        self.assertEqual(agent._sessions["s"].candidate_pool, ["LOAFER_C"])
        # Belt evidence that no loafer can carry is gone; the budget stays.
        self.assertIn("buckle closure", switch["removed"])
        kept = {item["constraint"]: item["reason"] for item in switch["kept"]}
        self.assertEqual(kept["budget around $40"], "budget_carries")
        state = agent._sessions["s"]
        self.assertNotIn("buckle closure", state.constraints)
        self.assertIn("budget around $40", state.constraints)
        # The refuted belts and the retired questions belonged to the old
        # department; the new one starts clean and shows loafers.
        self.assertEqual(state.proven_misses, set())
        self.assertEqual(state.exhausted, set())
        shown = [r["parent_asin"] for r in second["recommendations"]]
        self.assertEqual(shown, ["LOAFER_C"])
        # The output gate's clock restarted with the new request: the switch
        # turn counts as turn one, so the late-turn opening is not reached
        # two turns later.
        self.assertEqual(state.turn_offset, 1)
        self.assertFalse(state.information_complete)

    def test_a_category_that_refines_the_current_department_does_not_switch(self) -> None:
        client = ScriptedClient([
            {"intent": "open", "category": "belt", "features": ["buckle closure"], "dropped": []},
            {"intent": "add", "category": "dress belt", "features": [], "dropped": []},
        ])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        first = agent.respond("s", "a belt with a buckle", 1, 10)
        shown = [r["parent_asin"] for r in first["recommendations"]]
        agent.respond("s", "more of a dress belt really", 2, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertNotIn("department_switch", grounding)
        self.assertEqual(grounding["category_kept"]["reason"], "refines_current_department")
        state = agent._sessions["s"]
        self.assertIn("buckle closure", state.constraints)
        # Continuation still refuted the first slate.
        self.assertTrue(set(shown) <= state.proven_misses)

    def test_a_foreign_language_message_is_translated_before_the_catalog_reads_it(self) -> None:
        from src.language import detect

        self.assertEqual(detect("我想买一条皮带，要有搭扣"), "zh")
        self.assertEqual(detect("saya mau beli ikat pinggang kulit"), "id")
        self.assertEqual(detect("je cherche une ceinture en cuir"), "fr")
        self.assertEqual(detect("tôi muốn mua một chiếc thắt lưng da"), "vi")
        self.assertIsNone(detect("I want a leather belt with a buckle"))
        self.assertIsNone(detect("I just need 进口 and Pull On closure."))
        self.assertIsNone(detect("I'm looking for Accessories Belts, but I'm still exploring."))
        self.assertIsNone(detect("need sepatu for the gym"))

        class TranslatingClient(ScriptedClient):
            def chat(self, messages, *, max_tokens=200, temperature=0.0):
                system = messages[0]["content"]
                if system.startswith("Translate"):
                    self.calls.append(messages)
                    return LLMReply(text="I want a leather belt with a buckle closure",
                                    prompt_tokens=30, completion_tokens=12, latency_ms=1.0)
                return super().chat(messages, max_tokens=max_tokens, temperature=temperature)

        client = TranslatingClient([{
            "intent": "open", "category": "belt", "material": "leather", "color": None,
            "budget": None, "features": ["buckle closure"], "dropped": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="assist", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        response = agent.respond("s", "我想买一条皮带，皮的，要有搭扣", 1, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        language = grounding["language"]
        self.assertEqual(language["detected"], "zh")
        self.assertEqual(language["status"], "translated")
        self.assertEqual(grounding["english"], "I want a leather belt with a buckle closure")
        # The catalog read the English: leather and the closure are verbatim.
        state = agent._sessions["s"]
        self.assertIn("leather", state.constraints)
        self.assertIn("buckle closure", state.constraints)
        self.assertIn("Accessories Belts", grounding["pool_shelves"])
        self.assertEqual(state.language, "zh")
        # translate + render: the English placed itself ("belt" over four
        # belts) and every attribute was a verbatim catalog string, so the
        # grounding model was not needed. Both calls are counted.
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(response["usage"]["prompt_tokens"], 30 + 100)
        render_payload = json.loads(client.calls[-1][-1]["content"])
        self.assertEqual(render_payload["reply_language"], "Chinese")
        self.assertTrue(render_payload["shopper_wrote"].startswith("我想买"))
        self.assertIn("reply_language", client.calls[-1][0]["content"])

    def test_a_failed_translation_leaves_the_original_message_for_the_model(self) -> None:
        reading = {
            "intent": "open", "category": "belt", "material": "leather", "color": None,
            "budget": None, "features": [], "dropped": [],
        }
        client = ScriptedClient([reading, reading])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        # The scripted client answers the translation request with the
        # grounding JSON: not English prose, so it is rejected and the model
        # stage reads the original sentence itself.
        agent.respond("s", "我想买一条皮带", 1, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertEqual(grounding["language"]["status"], "translation_rejected")
        self.assertNotIn("english", grounding)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(grounding["model_status"], "grounded")
        self.assertIn("Accessories Belts", grounding["pool_shelves"])
        self.assertEqual(agent._sessions["s"].language, "zh")

    def test_english_never_triggers_a_translation_call(self) -> None:
        client = ScriptedClient([{
            "intent": "open", "category": "belt", "material": "leather", "features": [], "dropped": [],
        }])
        agent = Agent(
            self.catalog_path,
            llm_settings=LLMSettings(mode="ground", base_url="fake://"),
            llm_client=client,
        )
        agent.reset("s", {})
        agent.respond("s", "something in leather, a belt maybe", 1, 10)
        grounding = agent.explain_last_decision("s")["llm_grounding"]
        self.assertNotIn("language", grounding)
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(agent._sessions["s"].language)

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
