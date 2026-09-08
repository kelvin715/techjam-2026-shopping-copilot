"""All tunables in one place so the sweeps can vary them."""

PHRASE_HIT = 2.0
TOKEN_MAX = 0.5
WEIGHT_MODE = "global"

EMIT_LATE_TURN = 4
EMIT_K0 = 1
EMIT_K1 = 1
EMIT_K2 = 1
MAX_DISCLOSED_CONSTRAINTS = 4

# With no shopper evidence at all, catalog order is not a meaningful prior.
# Rank only that cold-start state by the disclosed aggregate review count; as
# soon as one constraint arrives, the evidence ranker below takes over.
COLD_START_PRIOR = "review_count"

# Once disclosure is exhausted, byte-identical intent signatures cannot be
# separated by another question. Plan how many of those siblings to expose per
# turn, using continuation as grounded refutation feedback. Turns 8--10 always
# expose the full allowance so shortening cannot lose a baseline Hit@10.
REFUTATION_BATCH_PLANNER = True
REFUTATION_FULL_FROM_TURN = 8
EVALUATOR_MAX_TURNS = 10

COVERAGE_FIRST = False
LENGTH_NORM = 0.0
PARTIAL_MIN = 0.5

MATCH_MODE = "raw"
LOOSE_HIT = 1.5

GATE_MODE = "count"
MARGIN_FRAC = 0.75
GATE_POLICY_LABEL = "finite_horizon_refutation_plan"
GATE_PROXY_RISK_ALPHA = 0.18

TITLE_BONUS = 1.0
PROFILE_BONUS = 0.0

TYPED_WEIGHT = 0.3
# Canonical signature likelihood is the only ranking term that improved all
# three panels at once (public, popularity-matched, uniform long-tail), so it
# carries more weight than the raw lexical and typed evidence it arbitrates.
SIGNATURE_WEIGHT = 0.9

POPULARITY_WEIGHT = 0.55
POPULARITY_WINDOW = 0.15
POPULARITY_MIN_CONSTRAINTS = 1
POPULARITY_SCOPE = "all"

OVERRIDE_DECAY = 0.5

# A response that did not end the session is evaluator-grounded negative
# evidence: every ASIN shown on that response is not the current target.
PROVEN_MISS_EXCLUSION = True

# After several complete slates have failed, popularity ordering has exhausted
# its value. Explore only candidates tied on evidence; never relax relevance.
TAIL_EXPLORATION_ENABLED = True
TAIL_EXPLORATION_TURN = 7
TAIL_EXPLORATION_CORE_WINDOW = 1e-9

# Attribute-specific questions can reveal canonical signature fields out of
# order. Positional likelihood is used only while the observed sequence is
# still protocol-guaranteed to preserve catalog order.
ADAPT_SIGNATURE_ORDER = True

# Question planning. ``fixed`` preserves the measured legacy order;
# ``counterfactual`` minimises the expected surviving pool; ``metric_voi``
# optimises an idealised Hit@10/MRR value-of-information objective; and
# ``answerable_metric_voi`` discounts branches that the protocol answers with
# "no additional preference" and therefore cannot rerank.
QUESTION_MODE = "answerable_metric_voi"
QUESTION_POOL_MAX = 512
QUESTION_SCORE_RATIO = 0.75
QUESTION_TURN_COST = 0.02

# Additive parser coverage for semantically equivalent user wording. The
# canonical protocol remains the first parse path and is unchanged.
ROBUST_PARSER = True

# Optional hybrid language layer. ``off`` keeps the scored path byte-identical
# and token-free. ``ground`` lets an OpenAI-compatible model propose a
# structured reading of off-protocol shopper wording; every proposal is
# verified against read-only catalog evidence before it can touch ranking.
# ``assist`` additionally phrases the customer-facing message from the
# decision certificate. ``ARC_LLM_MODE``, ``ARC_LLM_BASE_URL``,
# ``ARC_LLM_MODEL``, ``ARC_LLM_API_KEY``, ``ARC_LLM_TIMEOUT`` and
# ``ARC_LLM_CACHE`` override these at Agent construction.
LLM_MODE = "off"
LLM_BASE_URL = ""
LLM_MODEL = "gemma4"
LLM_TIMEOUT_SECONDS = 30.0
LLM_GROUND_MAX_TOKENS = 220
LLM_RENDER_MAX_TOKENS = 90
LLM_GROUND_MAX_FEATURES = 3
# Confidence attached to grounded evidence by verification tier.
LLM_GROUND_VERBATIM_WEIGHT = 1.0
LLM_GROUND_MAPPED_WEIGHT = 0.8
LLM_GROUND_LEXICAL_WEIGHT = 0.6
LLM_GROUND_TOKEN_WEIGHT = 0.5
# A paraphrase may map onto a canonical value only when that value is short
# (phrase tokens + this allowance) and appears on at least this many products
# the session can still recommend.
LLM_GROUND_MAPPED_EXTRA_TOKENS = 3
LLM_GROUND_MAPPED_MIN_SUPPORT = 2
# Common catalog phrases from the candidate pool shown to the model so a
# paraphrase can be expressed in the catalog's own words before verification.
LLM_GROUND_VOCAB_HINTS = 40
# A token occurring on at most this many catalog products counts as rare and
# can ground a paraphrased phrase on its own.
LLM_GROUND_RARE_TOKEN_DF = 250
# ``decay`` keeps a cancelled preference at OVERRIDE_DECAY confidence, as the
# deterministic policy does; ``remove`` deletes it.
LLM_GROUND_DROP_MODE = "decay"
# In a grounded (free-form) session, once this many products have been shown
# and refuted, stop trusting the guessed shelf pool and look beyond it.
LLM_POOL_WIDEN_AFTER_MISSES = 10
# Looking beyond the pool goes through a rarity-weighted inverted index
# (tokens on at most ``LLM_WIDEN_MAX_DF`` products) and keeps the best
# ``LLM_WIDEN_RETRIEVE_LIMIT`` products for exact scoring, instead of
# re-scoring all 50,000 products on every later turn. ``0`` restores the
# full-catalog scan.
LLM_WIDEN_RETRIEVE_LIMIT = 2000
LLM_WIDEN_MAX_DF = 12500
# The shipped free-text reader is a cascade: the catalog reads the message
# first (verbatim catalog strings, closed material / colour vocabularies, a
# dollar amount; zero tokens), and the model is consulted only when that
# leaves the sentence unexplained or the shopper cancels something. ``False``
# sends every off-protocol message to the model (the model-only baseline).
LLM_GROUND_CASCADE = True
# A grounded "under $n" / "over $n" is a hard bound on the candidate pool;
# the protocol's "budget around $n" keeps its proximity scoring.
LLM_GROUND_HARD_BUDGET = True
# Attribution baselines. ``LLM_GROUND_VERIFY=False`` admits every model
# proposal as evidence without a catalog check; ``LLM_GROUND_VOCAB_HINTS=0``
# removes the catalog vocabulary from the prompt. Both are research switches
# for the ablations in the paper; the shipped defaults are verify=True, 40.
LLM_GROUND_VERIFY = True
# ``LLM_GROUND_TIERS`` restricts which verification tiers may admit evidence
# (research switch); the shipped default admits all four.
LLM_GROUND_TIERS = ("signature_verbatim", "signature_mapped", "lexical", "lexical_tokens")
# Dense-retrieval baseline: cosine similarity between the shopper's free
# text and a product-text embedding, min-max normalised within the pool and
# added to the evidence score with this weight.
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_WEIGHT = 1.0
# Dense as a translator (research option 1, off by default; never runs on
# protocol wording). An admitted feature phrase is expanded into the pool's
# signature values within ``DENSE_VALUE_MIN_COSINE`` of its embedding, and
# the ranker credits such an alternative at ``DENSE_VALUE_ALT_WEIGHT`` of a
# verbatim hit. A phrase with no catalog support at all is admitted as its
# nearest supported signature value at ``DENSE_VALUE_TIER_WEIGHT``
# confidence (tier ``signature_dense``, between mapped 0.8 and lexical 0.6).
DENSE_VALUE_EXPANSION = False
DENSE_VALUE_MIN_COSINE = 0.80
DENSE_VALUE_MAX_ALTERNATIVES = 5
DENSE_VALUE_ALT_WEIGHT = 0.5
DENSE_VALUE_TIER_WEIGHT = 0.7
# Dense as a recall channel (research option 2, off by default). A category
# phrase is mapped onto the shelves whose name embedding is within
# ``DENSE_SHELF_MIN_COSINE`` (at most ``DENSE_SHELF_LIMIT``), unioned with
# the token-overlap shelves; a session that has outgrown its pool unions
# ``DENSE_WIDEN_LIMIT`` products retrieved by embedding similarity of the
# shopper's own sentences with the rarity-weighted lexical retrieval.
DENSE_SHELF_RECALL = False
DENSE_SHELF_MIN_COSINE = 0.80
DENSE_SHELF_LIMIT = 5
DENSE_WIDEN_LIMIT = 500
# Endpoint circuit breaker. A shared or hosted model server can go down
# mid-session; without this every later turn pays the full timeout twice
# before falling back. After this many consecutive failures the client stops
# calling out for the cooldown and the agent runs its deterministic path at
# full speed.
LLM_CIRCUIT_FAILURES = 2
LLM_CIRCUIT_COOLDOWN_SECONDS = 60.0
