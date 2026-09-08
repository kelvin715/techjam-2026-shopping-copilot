"""Rarity-weighted exact matching with bounded structured reranking."""

from __future__ import annotations

import hashlib
import math
import re

from . import config
from .shelf import COLORS, MATERIALS, loose, norm, tokens

_PRICE_RE = re.compile(
    r"(?:\$|<=|under|around)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.I
)


def _global_weight(catalog, phrase_tokens: list[str]) -> float:
    if not phrase_tokens:
        return 1.0
    idfs = sorted((catalog.token_idf(token) for token in phrase_tokens), reverse=True)
    return sum(idfs[:3]) / min(len(idfs), 3)


def _constraint_kind(phrase: str) -> tuple[str, str | float | None]:
    value = norm(phrase)
    if value in MATERIALS:
        return "material", value
    if value.startswith("color:"):
        color = value.split(":", 1)[1].strip()
        if color in COLORS:
            return "color", "gray" if color == "grey" else color
    if value in COLORS:
        return "color", "gray" if value == "grey" else value
    if "budget" in value or "$" in value or "under" in value:
        match = _PRICE_RE.search(value)
        if match:
            try:
                return "budget", float(match.group(1))
            except ValueError:
                pass
    return "phrase", None


def _satisfies(
    catalog,
    asin: str,
    phrase: str,
    phrase_tokens: list[str],
    kind_value: tuple[str, str | float | None] | None = None,
    needle: str | None = None,
) -> float:
    """Typed satisfaction of one constraint by one product.

    ``kind_value`` and ``needle`` are pure functions of the phrase; callers
    scoring a whole candidate pool pass them precomputed so the per-product
    loop does no string parsing.
    """
    kind, value = kind_value if kind_value is not None else _constraint_kind(phrase)
    # ``ltext`` is the product text with punctuation runs collapsed to single
    # spaces, so a padded substring test is exactly token membership without
    # materialising a token set per product.
    if kind == "material" and isinstance(value, str):
        if catalog.first_material.get(asin) == value:
            return 1.0
        return 0.6 if f" {value} " in f" {catalog.ltext[asin]} " else 0.0
    if kind == "color" and isinstance(value, str):
        if catalog.first_color.get(asin) == value:
            return 1.0
        padded = f" {catalog.ltext[asin]} "
        if value == "gray":
            return 0.6 if " gray " in padded or " grey " in padded else 0.0
        return 0.6 if f" {value} " in padded else 0.0
    if kind == "budget" and isinstance(value, float):
        price = catalog.price.get(asin)
        if price is None or value <= 0:
            return 0.25
        return max(0.0, 1.0 - abs(price - value) / value)

    if needle is None:
        needle = loose(phrase)
    blob = catalog.ltext[asin]
    if needle and needle in blob:
        return 1.0
    if not phrase_tokens:
        return 0.0
    padded = f" {blob} "
    hits = sum(f" {token} " in padded for token in phrase_tokens)
    return 0.7 * hits / len(phrase_tokens)


def _phrase_plan(
    phrases: list[str],
    phrase_tokens: list[list[str]],
    constraint_weights: list[float],
) -> list[tuple[str, list[str], float, tuple, str, float]]:
    """Everything about a constraint that does not depend on the product."""
    plan = []
    for phrase, phrase_toks, confidence in zip(phrases, phrase_tokens, constraint_weights):
        specificity = (1.0 + min(2.0, len(phrase_toks) / 6.0)) * confidence
        plan.append((
            phrase, phrase_toks, confidence, _constraint_kind(phrase), loose(phrase), specificity,
        ))
    return plan


def _typed_score(
    catalog,
    asin: str,
    phrases: list[str],
    phrase_tokens: list[list[str]],
    constraint_weights: list[float],
    plan: list | None = None,
    alternatives: dict | None = None,
) -> float:
    if not phrases:
        return 0.0
    if plan is None:
        plan = _phrase_plan(phrases, phrase_tokens, constraint_weights)
    total = 0.0
    weight_sum = 0.0
    for phrase, phrase_toks, _confidence, kind_value, needle, specificity in plan:
        satisfaction = _satisfies(
            catalog, asin, phrase, phrase_toks, kind_value, needle
        )
        if alternatives and satisfaction < 1.0:
            # Dense value expansion: a catalog string the shopper's phrase
            # may have meant counts at reduced credit (research option 1).
            for alternative, _cosine in alternatives.get(phrase, ()):
                alt = config.DENSE_VALUE_ALT_WEIGHT * _satisfies(
                    catalog, asin, alternative,
                    [token for token in tokens(alternative) if len(token) > 2],
                )
                if alt > satisfaction:
                    satisfaction = alt
        total += specificity * satisfaction
        weight_sum += specificity
    return total / weight_sum if weight_sum else 0.0


def _signature_score(
    catalog,
    asin: str,
    phrases: list[str],
    constraint_weights: list[float],
    scenario: str | None,
    positions_reliable: bool = True,
    alternatives: dict | None = None,
) -> float:
    """Likelihood that the candidate's canonical card produced these clues."""
    signature = catalog.signature.get(asin, ())
    if not signature or not phrases:
        return 0.0
    use_positions = positions_reliable and scenario != "intent_override"
    total = 0.0
    weight_sum = 0.0
    for index, (phrase, confidence) in enumerate(
        zip(phrases, constraint_weights)
    ):
        if use_positions and index < len(signature) and phrase == signature[index]:
            satisfaction = 1.0
        elif phrase in signature:
            satisfaction = 0.7
        elif alternatives and any(
            alternative in signature for alternative, _cosine in alternatives.get(phrase, ())
        ):
            satisfaction = 0.7 * config.DENSE_VALUE_ALT_WEIGHT
        else:
            satisfaction = 0.0
        total += confidence * satisfaction
        weight_sum += confidence
    return total / weight_sum if weight_sum else 0.0


def _popularity_enabled(scenario: str | None) -> bool:
    scope = config.POPULARITY_SCOPE
    if scope == "buying":
        return scenario == "buying"
    if scope == "buying_override":
        return scenario in ("buying", "intent_override")
    return scope == "all"


def score_candidates(
    catalog,
    cand_ids: list[str],
    constraints: list[str],
    profile_tags: list[str] | None = None,
    constraint_weights: list[float] | None = None,
    scenario: str | None = None,
    popularity_weight: float | None = None,
    signature_positions_reliable: bool = True,
    alternatives: dict | None = None,
):
    """Score ``cand_ids`` against the shopper's constraints, best first.

    ``alternatives`` (research option 1, dense value expansion) maps a
    normalised constraint to catalog strings it may stand for; a product
    carrying only an alternative earns ``config.DENSE_VALUE_ALT_WEIGHT`` of
    the credit. It is ``None`` on the protocol path, where every branch below
    is unchanged.
    """
    if not cand_ids:
        return []
    if not alternatives:
        alternatives = None
    phrases = [phrase for phrase in (norm(value) for value in constraints) if phrase]
    if not phrases:
        if config.COLD_START_PRIOR == "review_count":
            # ``rating_number`` is part of the frozen, disclosed catalog. Log
            # compression keeps the diagnostic score bounded while preserving
            # the review-count ordering; no target or session label is used.
            denominator = catalog._log_max_ratings or 1.0
            cold = [
                (
                    asin,
                    math.log1p(catalog.rating_count.get(asin, 0)) / denominator,
                )
                for asin in cand_ids
            ]
            # Python's sort is stable, so products with equal evidence retain
            # the frozen catalog order instead of gaining an arbitrary ASIN
            # lexicographic preference.
            return sorted(cold, key=lambda item: -item[1])
        return [(asin, 0.0) for asin in cand_ids]
    if constraint_weights is None or len(constraint_weights) != len(phrases):
        constraint_weights = [1.0] * len(phrases)

    texts = catalog.text
    phrase_tokens = [
        [token for token in tokens(phrase) if len(token) > 2]
        for phrase in phrases
    ]
    plan = _phrase_plan(phrases, phrase_tokens, constraint_weights)
    candidate_count = len(cand_ids)
    weights = []
    for phrase, phrase_toks in zip(phrases, phrase_tokens):
        if config.WEIGHT_MODE == "global":
            weights.append(_global_weight(catalog, phrase_toks))
            continue
        local = math.log(
            1.0
            + candidate_count
            / (1.0 + sum(phrase in texts[asin] for asin in cand_ids))
        )
        if config.WEIGHT_MODE == "local":
            weights.append(local)
        else:
            weights.append(local * _global_weight(catalog, phrase_toks))

    typed_kinds = {"material", "color", "budget"}
    scoring_plan = [
        (phrase, loose_phrase, weight, phrase_toks, confidence, kind_value,
         kind_value[0] in typed_kinds,
         [(alternative, loose(alternative)) for alternative, _cosine in alternatives.get(phrase, ())]
         if alternatives else ())
        for (phrase, phrase_toks, confidence, kind_value, loose_phrase, _specificity), weight
        in zip(plan, weights)
    ]
    raw: list[float] = []
    mode = config.MATCH_MODE
    for asin in cand_ids:
        text = texts[asin]
        loose_text = catalog.ltext[asin]
        padded = " " + text + " "
        total = 0.0
        for phrase, loose_phrase, weight, phrase_toks, confidence, kind_value, typed, alts in scoring_plan:
            # The protocol synthesises typed clues such as ``color: grey`` and
            # ``budget around $25`` from catalog fields. Those literal strings
            # need not occur in searchable_text, so treating them as ordinary
            # phrases unfairly rewards products that happen to serialise a
            # matching field label. Score them through the same structured
            # interpretation that generated the clue instead.
            if typed:
                satisfaction = _satisfies(
                    catalog, asin, phrase, phrase_toks, kind_value, loose_phrase
                )
                total += (
                    confidence
                    * weight
                    * config.PHRASE_HIT
                    * satisfaction
                )
                continue
            hit_raw = mode in ("raw", "both") and phrase in text
            hit_loose = mode in ("loose", "both") and loose_phrase in loose_text
            if hit_raw:
                total += confidence * weight * config.PHRASE_HIT
            elif hit_loose:
                hit_value = (
                    config.PHRASE_HIT if mode == "loose" else config.LOOSE_HIT
                )
                total += confidence * weight * hit_value
            else:
                fraction = 0.0
                if phrase_toks:
                    fraction = sum(
                        f" {token} " in padded for token in phrase_toks
                    ) / len(phrase_toks)
                alt_credit = 0.0
                if alts:
                    # An alternative catalog string present in the product
                    # text counts at reduced credit (dense value expansion).
                    for alternative, loose_alternative in alts:
                        if alternative in text or loose_alternative in loose_text:
                            alt_credit = config.PHRASE_HIT * config.DENSE_VALUE_ALT_WEIGHT
                            break
                if alt_credit > config.TOKEN_MAX * fraction:
                    total += confidence * weight * alt_credit
                elif fraction:
                    total += confidence * weight * config.TOKEN_MAX * fraction
            if (
                config.TITLE_BONUS != 1.0
                and (hit_raw or hit_loose)
                and phrase in catalog.title[asin]
            ):
                total += (
                    confidence
                    * weight
                    * config.PHRASE_HIT
                    * (config.TITLE_BONUS - 1.0)
                )

        if config.PROFILE_BONUS and profile_tags:
            hits = sum(f" {tag} " in padded for tag in profile_tags)
            if hits:
                total += config.PROFILE_BONUS * hits / len(profile_tags)
        raw.append(total)

    top_raw = max(raw, default=0.0)
    core: list[float] = []
    for asin, base in zip(cand_ids, raw):
        normalised = base / top_raw if top_raw > 0 else 0.0
        typed = _typed_score(
            catalog, asin, phrases, phrase_tokens, constraint_weights, plan, alternatives
        )
        signature = _signature_score(
            catalog,
            asin,
            phrases,
            constraint_weights,
            scenario,
            signature_positions_reliable,
            alternatives,
        )
        core.append(
            normalised
            + config.TYPED_WEIGHT * typed
            + config.SIGNATURE_WEIGHT * signature
        )

    best_core = max(core, default=0.0)
    applied_popularity_weight = (
        config.POPULARITY_WEIGHT
        if popularity_weight is None
        else max(0.0, float(popularity_weight))
    )
    use_popularity = (
        applied_popularity_weight > 0
        and len(phrases) >= config.POPULARITY_MIN_CONSTRAINTS
        and _popularity_enabled(scenario)
    )
    output: list[tuple[float, int, str]] = []
    for index, (asin, score) in enumerate(zip(cand_ids, core)):
        if use_popularity and best_core - score <= config.POPULARITY_WINDOW:
            score += applied_popularity_weight * catalog.popularity(asin)
        output.append((-score, index, asin))
    output.sort()
    return [(asin, -negative) for negative, _, asin in output]


def diversify_evidence_ties(
    final_scores: list[tuple[str, float]],
    evidence_scores: list[tuple[str, float]],
    core_window: float,
) -> list[tuple[str, float]]:
    """Deterministically explore only the best evidence-equivalent group.

    The stable ASIN hash removes popularity/catalog-order bias after repeated
    failed slates. Candidates below the declared core window never move ahead
    of the evidence-tied group.
    """
    if not final_scores or not evidence_scores:
        return final_scores
    evidence = dict(evidence_scores)
    best_core = max(evidence.get(asin, float("-inf")) for asin, _ in final_scores)
    tolerance = max(0.0, float(core_window))
    tied = [
        item for item in final_scores
        if best_core - evidence.get(item[0], float("-inf")) <= tolerance
    ]
    if len(tied) < 2:
        return final_scores
    tied_ids = {asin for asin, _ in tied}
    tied.sort(key=lambda item: hashlib.blake2b(
        item[0].encode("utf-8"), digest_size=8
    ).digest())
    return [*tied, *[item for item in final_scores if item[0] not in tied_ids]]


def rank(
    catalog, cand_ids: list[str], constraints: list[str], top_k: int
) -> list[str]:
    return [
        asin
        for asin, _ in score_candidates(catalog, cand_ids, constraints)[:top_k]
    ]
