"""τ-Rec-inspired language robustness and protocol-policy benchmark.

The benchmark keeps catalog products and ground-truth ASINs unchanged. It
varies only how already-authorised user evidence is revealed: canonical,
surface noise, natural paraphrase, or one hidden clue. Results are research
diagnostics and must not be presented as official τ-Rec numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent
from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    catalog_index,
    evaluate,
    load_jsonl,
)
from src import config
from src.shelf import norm


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _natural(message: str) -> str:
    replacements = (
        (r"^I'm looking for (.+?)\. A key requirement is:\s*(.+?)\.$",
         r"I'm shopping for \1. The main thing I need is: \2."),
        (r"^I'm looking for (.+?), but I'm still exploring\.$",
         r"I'd like to browse \1. I haven't settled on preferences."),
        (r"^I'm looking for (.+?)\. (.+)$", r"I'm shopping for \1. \2"),
        (r"^For that, what matters is:\s*(.+?)\.$",
         r"Here is what matters to me: \1."),
        (r"^I don't have an additional preference for (\w+)\.$",
         r"Nothing else to add about \1."),
        (r"^I don't have a preference for (\w+); please use your judgment\.$",
         r"No preference on \1; choose what works."),
        (r"^Those options are not quite right yet\..*$",
         r"Those choices still miss the mark; please ask something specific."),
        (r"^Actually, ignore my earlier preference\. What I need is:\s*(.+?)\.$",
         r"I've changed my mind. My new requirement is: \1."),
    )
    for pattern, replacement in replacements:
        if re.search(pattern, message, re.I):
            return re.sub(pattern, replacement, message, flags=re.I)
    return message


class PerturbingAgent:
    def __init__(self, catalog_path: str, style: str) -> None:
        self.agent = Agent(catalog_path)
        self.style = style
        self.traces: dict[str, list[dict]] = defaultdict(list)
        self.hidden_once: set[str] = set()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def transform(self, session_id: str, message: str) -> str:
        if self.style == "canonical":
            return message
        if self.style == "surface_noise":
            return "  PLEASE NOTE:  " + message.upper()
        if self.style == "natural_paraphrase":
            return _natural(message)
        if self.style == "hidden_one" and session_id not in self.hidden_once:
            match = re.search(r"what matters is:\s*(.+?)\.$", message, re.I)
            if match:
                pieces = [part.strip() for part in match.group(1).split(";")]
                if len(pieces) >= 2:
                    self.hidden_once.add(session_id)
                    return "For that, what matters is: " + "; ".join(pieces[:-1]) + "."
        return message

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        transformed = self.transform(session_id, user_message)
        response = self.agent.respond(session_id, transformed, turn, top_k)
        self.traces[session_id].append({
            "turn": turn,
            "raw_message": user_message,
            "transformed_message": transformed,
            "response": deepcopy(response),
        })
        return response

    def explain_last_decision(self, session_id: str) -> dict:
        return self.agent.explain_last_decision(session_id)


def policy_audit(wrapper: PerturbingAgent, catalog_ids: set[str]) -> dict:
    violations: list[dict] = []
    calls = 0
    reported_tokens = 0
    for session_id, trace in wrapper.traces.items():
        seen_since_override: set[str] = set()
        if len(trace) > 10:
            violations.append({"session_id": session_id, "type": "turn_limit"})
        for call in trace:
            calls += 1
            message = call["transformed_message"].lower()
            if "changed my mind" in message or "what i need is" in message:
                seen_since_override.clear()
            response = call["response"]
            required = {"message", "ask_attribute", "recommendations", "usage"}
            if not isinstance(response, dict) or not required.issubset(response):
                violations.append({"session_id": session_id, "type": "contract"})
                continue
            attribute = response["ask_attribute"]
            if attribute is not None and attribute not in ALLOWED_ATTRIBUTES:
                violations.append({"session_id": session_id, "type": "attribute"})
            recommendations = response["recommendations"]
            asins = [
                str(item.get("parent_asin", ""))
                for item in recommendations
                if isinstance(item, dict)
            ] if isinstance(recommendations, list) else []
            if (
                len(asins) != len(recommendations)
                or len(asins) > 10
                or len(asins) != len(set(asins))
                or any(asin not in catalog_ids for asin in asins)
            ):
                violations.append({"session_id": session_id, "type": "slate"})
            repeated = seen_since_override.intersection(asins)
            if repeated:
                violations.append({
                    "session_id": session_id,
                    "type": "proven_miss_repeated",
                    "asins": sorted(repeated),
                })
            seen_since_override.update(asins)
            usage = response.get("usage") or {}
            reported_tokens += int(usage.get("prompt_tokens") or 0)
            reported_tokens += int(usage.get("completion_tokens") or 0)
    return {
        "session_count": len(wrapper.traces),
        "response_count": calls,
        "reported_token_usage": reported_tokens,
        "violation_count": len(violations),
        "violations_by_type": {
            kind: sum(row["type"] == kind for row in violations)
            for kind in sorted({row["type"] for row in violations})
        },
        "pass": not violations and reported_tokens == 0,
    }


def explanation_audit(wrapper: PerturbingAgent, limit: int) -> dict:
    audited = 0
    found = 0
    faithful = 0
    grounded = 0
    statuses: dict[str, int] = defaultdict(int)
    for session_id, trace in list(wrapper.traces.items())[:limit]:
        certificate = wrapper.explain_last_decision(session_id)
        explanation = certificate.get("minimal_counterfactual_explanation", {})
        status = str(explanation.get("status", "missing"))
        statuses[status] += 1
        audited += 1
        if status == "minimal_counterfactual_found":
            found += 1
            faithful += int(explanation.get("faithful") is True)
        transcript = norm(" ".join(row["transformed_message"] for row in trace))
        constraints = [norm(value) for value in certificate.get("constraints", [])]
        grounded += int(all(value in transcript for value in constraints))
    return {
        "audited_sessions": audited,
        "grounded_certificate_rate": round(grounded / audited, 6) if audited else None,
        "counterfactual_found": found,
        "faithful_when_found": round(faithful / found, 6) if found else None,
        "statuses": dict(statuses),
    }


def pass_at_k(successes: list[bool], k: int) -> float:
    n = len(successes)
    c = sum(successes)
    if n < k:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--explanation-sample", type=int, default=8)
    parser.add_argument("--output", default="results/robustness_policy_benchmark.json")
    args = parser.parse_args()

    all_samples = load_jsonl(args.dataset)
    samples = all_samples[: min(args.count, len(all_samples))]
    ids, categories, products = catalog_index(args.catalog)
    before_hash = file_sha256(args.catalog)
    styles = ("canonical", "surface_noise", "natural_paraphrase", "hidden_one")
    original_robust = config.ROBUST_PARSER
    experiments: dict[str, dict] = {}
    success_by_parser: dict[str, dict[str, list[bool]]] = {}
    try:
        for parser_label, enabled in (("canonical_parser", False), ("robust_parser", True)):
            config.ROBUST_PARSER = enabled
            success_by_parser[parser_label] = {}
            for style in styles:
                wrapper = PerturbingAgent(args.catalog, style)
                outcome = evaluate(wrapper, samples, ids, categories, products)
                label = f"{parser_label}__{style}"
                experiments[label] = {
                    "parser_enabled": enabled,
                    "reveal_style": style,
                    "hit_rate_at_10": outcome["hit_rate_at_10"],
                    "mrr": outcome["mrr"],
                    "mttc": outcome["mttc"],
                    "technical_score": outcome["recommended_technical_score"],
                    "policy_audit": policy_audit(wrapper, ids),
                    "explanation_audit": explanation_audit(
                        wrapper, args.explanation_sample
                    ),
                }
                success_by_parser[parser_label][style] = [
                    bool(row["hit"]) for row in outcome["sessions"]
                ]
                print(
                    f"{label:<42} HR={outcome['hit_rate_at_10']:.4f} "
                    f"MRR={outcome['mrr']:.4f} "
                    f"score={outcome['recommended_technical_score']:.6f}"
                )
    finally:
        config.ROBUST_PARSER = original_robust
    after_hash = file_sha256(args.catalog)

    aggregate: dict[str, dict] = {}
    for parser_label, by_style in success_by_parser.items():
        per_sample = [
            [by_style[style][index] for style in styles]
            for index in range(len(samples))
        ]
        aggregate[parser_label] = {
            "variant_pass_at_1": round(
                sum(pass_at_k(row, 1) for row in per_sample) / len(per_sample), 6
            ),
            "variant_pass_at_2": round(
                sum(pass_at_k(row, 2) for row in per_sample) / len(per_sample), 6
            ),
            "variant_pass_at_4": round(
                sum(pass_at_k(row, 4) for row in per_sample) / len(per_sample), 6
            ),
            "pass_all_variants": round(
                sum(all(row) for row in per_sample) / len(per_sample), 6
            ),
        }

    result = {
        "status": "tau_rec_inspired_local_diagnostic_not_official_tau_rec",
        "catalog_mutation": {
            "sha256_before": before_hash,
            "sha256_after": after_hash,
            "unchanged": before_hash == after_hash,
        },
        "sample_count": len(samples),
        "variants": list(styles),
        "pass_at_k_note": (
            "Computed across deterministic reveal variants, not stochastic model trials."
        ),
        "experiments": experiments,
        "aggregate_robustness": aggregate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
