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

# Second-tier LLM fallback for template-parser misses. "template" or "agentic".
INPUT_MODE = "template"
AGENTIC_MODEL = "gpt-4o-mini"
AGENTIC_MAX_TOOL_CALLS = 3

# Caps the outgoing reply generator: keeps replies short and token spend low.
AGENTIC_REPLY_MAX_TOKENS = 20
AGENTIC_REPLY_MAX_CHARS = 120

# Circuit breaker, mirroring LLM_CIRCUIT_* below.
AGENTIC_CIRCUIT_FAILURES = 2
AGENTIC_CIRCUIT_COOLDOWN_SECONDS = 60.0

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
LLM_TIMEOUT_SECONDS = 20.0
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
# and refuted, stop trusting the guessed shelf pool and rank the full catalog.
LLM_POOL_WIDEN_AFTER_MISSES = 10
# Endpoint circuit breaker. A shared or hosted model server can go down
# mid-session; without this every later turn pays the full timeout twice
# before falling back. After this many consecutive failures the client stops
# calling out for the cooldown and the agent runs its deterministic path at
# full speed.
LLM_CIRCUIT_FAILURES = 2
LLM_CIRCUIT_COOLDOWN_SECONDS = 60.0
