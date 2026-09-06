"""Interactive REPL for testing input-understanding generalizability.

Shows, per turn, whether the template parser matched, what the agentic
interpreter would extract if not, and the real Agent.respond() output.

Usage:
    python tools/interactive_agent.py --catalog data/catalog.jsonl
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent
from src import config
from src.agentic_dialog import agentic_available, agentic_interpret
from src.dialog import parse


def _probe(message: str, state, catalog) -> tuple[bool, str | None, list[str]]:
    """Non-mutating probe: what would each path do with this message?

    Returns (template_matched, agentic_status, agentic_constraints).
    """
    template_probe = copy.deepcopy(state)
    template_matched = parse(message, template_probe, catalog)
    if template_matched:
        return True, None, []
    agentic_probe = copy.deepcopy(state)
    status = agentic_interpret(message, agentic_probe, catalog)
    return False, status, list(agentic_probe.constraints)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--input-mode", choices=["template", "agentic"], default="agentic",
        help=(
            "This REPL is the interactive entry point, so it opts into "
            "'agentic' for itself rather than relying on the global default, "
            "which stays 'template' so the scored path is deterministic and "
            "token-free."
        ),
    )
    args = parser.parse_args()

    config.INPUT_MODE = args.input_mode
    agent = Agent(args.catalog)
    session_id = "interactive"
    agent.reset(session_id, {})
    turn = 0

    print(f"INPUT_MODE (live): {config.INPUT_MODE}")
    print(f"Agentic API key loaded: {agentic_available()}")
    print("Type shopper messages. Commands: /reset  /quit")
    print("-" * 70)

    while True:
        try:
            message = input(f"[turn {turn + 1}] you> ").strip()
        except EOFError:
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/reset":
            agent.reset(session_id, {})
            turn = 0
            print("(session reset)")
            continue

        turn += 1
        state = agent._sessions[session_id]

        skip_probe = config.INPUT_MODE == "agentic"
        if not skip_probe:
            t_matched, a_status, a_constraints = _probe(message, state, agent.catalog)
            print(f"  [probe] template parser matched: {t_matched}")
            if not t_matched:
                print(f"  [probe] agentic status: {a_status}")
                if a_constraints:
                    print(f"  [probe] agentic constraints: {a_constraints}")

        response = agent.respond(session_id, message, turn, args.top_k)
        certificate = agent.explain_last_decision(session_id)
        recs = [item["parent_asin"] for item in response["recommendations"][:5]]

        print(f"  agent> {response['message']}")
        print(f"  ask_attribute: {response['ask_attribute']}")
        print(f"  recommendations (top 5): {recs}")
        print(f"  input_interpretation (actual): {certificate.get('input_interpretation')}")
        print(f"  session constraints so far: {state.constraints}")
        print(
            f"  boundary_signal={state.boundary_signal}  "
            f"exhausted={sorted(state.exhausted)}  "
            f"override_seen={state.override_seen}  "
            f"information_complete={state.information_complete}"
        )
        print("-" * 70)


if __name__ == "__main__":
    main()
