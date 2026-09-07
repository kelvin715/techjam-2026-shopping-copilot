"""Interactive REPL for trying the agent on wording you type yourself.

Shows, per turn, whether the deterministic parser matched and -- when the
optional language layer is configured -- what it read from the message.

    python tools/interactive_agent.py --catalog data/catalog.jsonl

The layer is off unless ARC_LLM_* is set, in which case any
OpenAI-compatible endpoint works without installing anything:

    export ARC_LLM_MODE=ground ARC_LLM_BASE_URL=https://api.openai.com/v1
    export ARC_LLM_MODEL=gpt-4o-mini ARC_LLM_API_KEY=sk-...
    export ARC_LLM_VERIFY=iterative      # optional: verify-before-commit
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
from src import ground
from src.dialog import parse
from src.llm import LLMSettings


def _probe(message: str, state, catalog) -> bool:
    """Non-mutating probe of the deterministic parser. Never calls out."""
    return parse(message, copy.deepcopy(state), catalog)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    settings = LLMSettings.from_env()
    agent = Agent(args.catalog, llm_settings=settings)
    session_id = "interactive"
    agent.reset(session_id, {})
    turn = 0

    print(f"language layer: {settings.mode}"
          f"{' @ ' + settings.base_url if settings.base_url else ' (off)'}")
    print(f"verification:   {ground.verify_mode()}")
    print("Type shopper messages. Commands: /reset  /quit")
    print("-" * 70)

    while True:
        try:
            message = input(f"[turn {turn + 1}] you> ").strip()
        except EOFError:
            break
        if not message or message == "/quit":
            if message == "/quit":
                break
            continue
        if message == "/reset":
            agent.reset(session_id, {})
            turn = 0
            print("(session reset)")
            continue

        turn += 1
        state = agent._sessions[session_id]
        print(f"  [probe] template parser matched: "
              f"{_probe(message, state, agent.catalog)}")

        response = agent.respond(session_id, message, turn, args.top_k)
        certificate = agent.explain_last_decision(session_id)
        recs = [item["parent_asin"] for item in response["recommendations"][:5]]

        print(f"  agent> {response['message']}")
        print(f"  ask_attribute: {response['ask_attribute']}")
        print(f"  recommendations (top 5): {recs}")
        print(f"  input_interpretation: {certificate.get('input_interpretation')}")
        grounding = certificate.get("llm_grounding")
        if grounding:
            print(f"  grounding: status={grounding.get('status')} "
                  f"mode={grounding.get('verify_mode')} "
                  f"verifications={grounding.get('tool_verifications', 0)}")
            for item in grounding.get("accepted", []):
                print(f"    + {item.get('constraint')} "
                      f"({item.get('tier')}, confidence {item.get('confidence')})")
            for item in grounding.get("rejected", []):
                print(f"    - {item.get('phrase')} ({item.get('reason')})")
        print(f"  constraints so far: {state.constraints}")
        print("-" * 70)


if __name__ == "__main__":
    main()
