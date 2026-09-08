"""Run an external ``Agent`` in its own process and serve it over JSON lines.

Usage (spawned by ``tools/human_language_benchmark.py --external-isolated``):

    python3 tools/external_agent_worker.py REPO_DIR MODULE CLASS CATALOG_PATH

The worker changes into ``REPO_DIR`` before importing, so the entry's relative
data paths resolve, and nothing from this repository is on its import path.
Requests arrive on stdin, one JSON object per line: ``{"op": "reset", ...}``
or ``{"op": "respond", ...}``; each reply is one JSON line. An exception in
the external agent is reported as ``{"error": ...}`` and the caller turns it
into an empty, contract-valid turn, which is what the official evaluator does.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback


def main() -> None:
    repo, module_name, class_name, catalog = sys.argv[1:5]
    os.chdir(repo)
    sys.path.insert(0, repo)
    module = importlib.import_module(module_name)
    agent_cls = getattr(module, class_name)
    try:
        agent = agent_cls(catalog)
    except TypeError:
        agent = agent_cls()
    sys.stdout.write(json.dumps({"ready": True, "module": module.__file__}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        try:
            if request["op"] == "reset":
                agent.reset(request["session_id"], request["user_profile"])
                reply = {"ok": True}
            elif request["op"] == "respond":
                reply = {"ok": True, "response": agent.respond(
                    request["session_id"], request["message"], request["turn"], request["top_k"]
                )}
            elif request["op"] == "quit":
                break
            else:
                reply = {"error": f"unknown op {request['op']!r}"}
        except Exception:  # noqa: BLE001 - mirror the evaluator: report, do not crash
            reply = {"error": traceback.format_exc()}
        sys.stdout.write(json.dumps(reply, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
