from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request
from http.server import ThreadingHTTPServer

from agent import Agent
from demo.server import DemoRuntime, make_handler
from src.llm import LLMSettings, LLMReply
from src.policy import _refutation_plan
from src.telemetry import emit, observe


class LiveLabTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.catalog = root / "catalog.jsonl"
        self.catalog.write_text("".join(json.dumps({
            "parent_asin": asin, "title": "Leather blue walking shoes " + asin,
            "features": ["leather", "color: blue", "leather sole", "slip-on comfort"],
            "categories": ["Clothing", "Shoes", "Loafers"],
            "rating_number": 100 - i, "average_rating": 4.5,
        }) + "\n" for i, asin in enumerate(["A", "B", "C", "D"])))
        bundle = root / "bundle.json"
        bundle.write_text("{}")
        self.runtime = DemoRuntime(bundle, self.catalog, live_requested=True)
        self.runtime._agent = Agent(self.catalog, llm_settings=LLMSettings(mode="off"))
        self.lab = self.runtime.lab

    def new_session(self):
        return self.lab.reset({})["session_id"]

    def turn(self, session_id, turn=1, message="I'm looking for Shoes Loafers, but I'm still exploring."):
        events = []
        result = self.lab.turn({"session_id": session_id, "turn": turn, "message": message},
                               lambda event, data: events.append((event, data)))
        return result, events

    def test_observation_preserves_every_response_and_certificate(self):
        agent = self.runtime.ensure_agent()
        agent.reset("plain", {})
        agent.reset("observed", {})
        messages = ["I'm looking for Shoes Loafers, but I'm still exploring.",
                    "For that, what matters is: leather; color: blue.",
                    "For that, what matters is: leather sole; slip-on comfort."]
        for turn, message in enumerate(messages, 1):
            plain = agent.respond("plain", message, turn, 10)
            events = []
            with observe(events.append):
                observed = agent.respond("observed", message, turn, 10)
            self.assertEqual(plain, observed)
            self.assertEqual(agent._sessions["plain"].last_decision_certificate,
                             agent._sessions["observed"].last_decision_certificate)
            self.assertEqual(events[0]["stage"], "grounding")
            self.assertEqual(events[-1]["stage"], "response")
            self.assertEqual(events[-1]["data"]["response"], observed)

    def test_broken_observer_cannot_change_output(self):
        agent = self.runtime.ensure_agent()
        agent.reset("a", {}); agent.reset("b", {})
        message = "I'm looking for Shoes Loafers, but I'm still exploring."
        expected = agent.respond("a", message, 1, 10)
        def fail(_):
            raise RuntimeError("viewer disconnected")
        with observe(fail):
            result = agent.respond("b", message, 1, 10)
        self.assertEqual(result, expected)

    def test_dp_trace_is_the_utility_actually_optimized(self):
        events = []
        with observe(events.append):
            batch = _refutation_plan(3, 4, 10)
        result = events[0]["data"]
        self.assertEqual(result["selected_batch"], batch)
        self.assertEqual([c["batch"] for c in result["choices"]], [1, 2, 3, 4])
        for choice in result["choices"]:
            self.assertAlmostEqual(choice["hit_probability"] + choice["miss_probability"], 1)
            self.assertAlmostEqual(choice["expected_value"], choice["hit_value"] + choice["miss_probability"] * choice["continuation_value"])
        self.assertAlmostEqual(result["expected_value"], max(c["expected_value"] for c in result["choices"]))

    def test_internal_rank_and_submitted_rank_are_distinct_and_joined_after_return(self):
        sid = self.new_session()
        sample = {"ground_truth": {"parent_asin": "D"}, "scenario_type": "buying",
                  "behavior": {}, "intent_card": {"hard_constraints": ["leather"], "soft_preferences": []}}
        self.lab.sessions[sid].sample = sample
        agent = self.runtime.ensure_agent()
        original = agent.respond
        returned = False
        def respond(*args):
            nonlocal returned
            self.assertNotIn(sample, args)
            result = original(*args)
            returned = True
            return result
        def receive(event, data):
            if event == "stage":
                self.assertFalse(returned)
                self.assertNotIn("overlay", data)
                self.assertNotIn("target_id", data["data"])
                self.assertNotIn("scores", data["data"])
            if event == "done":
                self.assertTrue(returned)
        with patch.object(agent, "respond", side_effect=respond):
            result = self.lab.turn({"session_id": sid, "turn": 1,
                                    "message": "I'm looking for Shoes Loafers, but I'm still exploring."}, receive)
        self.assertEqual(result["overlay"]["candidate_rank"], 4)
        self.assertIsNone(result["overlay"]["recommendation_rank"])
        self.assertEqual(result["overlay"]["visibility"], "post_response_evaluator_only")

    def test_ordering_rejects_duplicate_and_out_of_order_turns(self):
        sid = self.new_session()
        result, events = self.turn(sid)
        seq = [data["seq"] for kind, data in events if kind == "stage"]
        self.assertEqual(seq, list(range(1, len(seq) + 1)))
        self.assertEqual(events[-1][0], "done")
        self.assertIsNone(result["overlay"])
        with self.assertRaises(ValueError):
            self.turn(sid, 1)
        with self.assertRaises(ValueError):
            self.turn(sid, 3)
        self.assertEqual(self.lab.sessions[sid].turn, 1)

    def test_comparison_forks_do_not_mutate_the_conversation(self):
        sid = self.new_session()
        self.turn(sid)
        agent = self.runtime.ensure_agent()
        before = deepcopy(agent._sessions[sid].__dict__)
        history = deepcopy(self.lab.sessions[sid].history)
        result = self.lab.compare({"session_id": sid, "original": "For that, what matters is: leather.",
                                   "rewritten": "For that, what matters is: color: blue."}, lambda *_: None)
        self.assertEqual(before, agent._sessions[sid].__dict__)
        self.assertEqual(history, self.lab.sessions[sid].history)
        self.assertEqual(self.lab.sessions[sid].turn, 1)
        self.assertEqual(set(self.lab.sessions), {sid})
        self.assertNotEqual(result["original"]["certificate"]["constraints"], result["rewritten"]["certificate"]["constraints"])

    def test_rewriter_only_receives_the_message_and_labels_cache_provenance(self):
        from src.llm import FakeChatClient

        client = FakeChatClient()
        self.lab._rewriters["qwen"] = client
        with patch.object(client, "chat", return_value=LLMReply(text="Blue shoes, please.", cached=True)) as chat:
            output = self.lab.rewrite({"message": "I need blue shoes.", "model": "qwen", "target_id": "D"})
        self.assertEqual(output["source"], "cached")
        self.assertEqual(output["model"], "qwen2.5-7b-instruct")
        self.assertEqual(chat.call_args.args[0][1], {"role": "user", "content": "I need blue shoes."})
        self.assertNotIn("target_id", json.dumps(chat.call_args.args))

    def test_observers_are_thread_local(self):
        seen = []
        with observe(seen.append):
            thread = threading.Thread(target=lambda: emit("foreign", "running"))
            thread.start(); thread.join()
            emit("own", "running")
        self.assertEqual([e["stage"] for e in seen], ["own"])

    def test_stream_arrives_before_agent_finishes(self):
        sid = self.new_session()
        release = threading.Event()
        agent = self.runtime.ensure_agent()
        original = agent.respond
        def slow_response(*args):
            emit("grounding", "running")
            if not release.wait(5):
                raise RuntimeError("test client did not receive the live event")
            return original(*args)
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.runtime))
        except PermissionError:
            self.skipTest("local sockets are disabled in this sandbox")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(agent, "respond", side_effect=slow_response):
                request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/lab/turn",
                    data=json.dumps({"session_id": sid, "turn": 1, "message": "I'm looking for Shoes Loafers, but I'm still exploring."}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=8) as response:
                    self.assertEqual(response.headers.get_content_type(), "text/event-stream")
                    self.assertEqual(response.readline(), b"event: stage\n")
                    first = json.loads(response.readline().decode()[6:])
                    self.assertEqual(first["status"], "running")
                    self.assertNotIn("response", first["data"])
                    self.assertEqual(self.lab.sessions[sid].turn, 0)
                    release.set()
                    rest = response.read().decode()
                    self.assertIn("event: done\n", rest)
        finally:
            release.set(); server.shutdown(); server.server_close(); thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
