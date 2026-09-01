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
SIGNATURE_WEIGHT = 0.7

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
