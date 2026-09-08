# Dense retrieval as a translator and as a recall channel (research options 1 and 2)

Run 2026-09-08 on the cached shopper rewrites (`--strict-replay`), grounding model `gemma4`
(gemma-4-26b-a4b-nvfp4) live for prompts not already in the per-arm cache, embeddings
`BAAI/bge-small-en-v1.5`. 200 public sessions per cell. Reference arms (`cascade`, `lexical`)
are the strict replays from `results/attribution/replay_v2/strict/`; every dense arm ran on the
identical rewrites, so the intervals in `intervals.md` are paired and scenario-stratified.

## What each arm does

| Arm | Switch | Mechanism |
|---|---|---|
| `*_dense_values` (option 1) | `DENSE_VALUE_EXPANSION` | an admitted feature phrase is expanded into up to 5 catalog signature values within cosine 0.80 of it (spelling variants and supersets collapsed); the ranker credits an alternative at 0.5 of a verbatim hit; a phrase with no catalog support is admitted as its nearest supported value (`signature_dense`, confidence 0.7) |
| `*_dense_pool` (option 2) | `DENSE_SHELF_RECALL` | shelves whose name embedding is within 0.80 of the category phrase (at most 5) are unioned into the token-overlap pool; after a refuted slate, 500 products nearest to the shopper's own sentences are unioned into the widening retrieval |
| `*_dense_both` | both | |
| `cascade_dense_values_t75/t85` | threshold sweep | cosine 0.75 / 0.85 |

The scored path is untouched: with both switches on, the canonical (template) control reproduces
`0.980400` with zero model calls and zero embedding queries, and the organizer evaluator on the
200 public sessions is byte-identical (`tests/test_dense_options.py`, evaluator rerun).

## Scores

| Shopper / wording | cascade | +values | +pool | +both | lexical | lexical+values | lexical+pool | lexical+both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma, verbatim | 0.9718 | 0.9710 | 0.9718 | 0.9708 | 0.9710 | 0.9710 | 0.9710 | 0.9710 |
| gemma, paraphrase | 0.7818 | 0.7771 | 0.7639 | 0.7762 | 0.6386 | 0.6391 | 0.6385 | 0.6389 |
| qwen, verbatim | 0.9041 | 0.9038 | 0.9017 | 0.9022 | 0.8842 | 0.8839 | 0.8843 | 0.8838 |
| qwen, paraphrase | 0.7490 | 0.7381 | 0.7341 | 0.7421 | 0.7123 | 0.7106 | 0.7137 | 0.7120 |

Threshold sweep, gemma paraphrase, cascade+values: 0.75 → 0.7768, 0.80 → 0.7771, 0.85 → 0.7869
(the only positive cell, +0.005 [−0.018, +0.028]).

## Paired deltas (B − A, per-session utility 0.5·hit + 0.3·rr + 0.02·(11 − turn))

| Wording | Shopper | +values | +pool | +both |
|---|---|---:|---:|---:|
| verbatim | gemma | −0.001 [−0.002, +0.000] | 0.000 | −0.001 [−0.003, +0.000] |
| verbatim | qwen | −0.000 [−0.001, +0.001] | −0.002 [−0.005, −0.001] | −0.002 [−0.004, −0.000] |
| paraphrase | gemma | −0.005 [−0.032, +0.023] | −0.018 [−0.045, +0.008] | −0.006 [−0.040, +0.029] |
| paraphrase | qwen | −0.011 [−0.033, +0.009] | −0.015 [−0.033, +0.001] | −0.007 [−0.033, +0.018] |

No cell is significantly positive; two verbatim cells for the pool channel are significantly
negative. The no-model (`lexical`) arms change fewer than 20 sessions per cell and move by less
than 0.002.

## Why (gemma paraphrase, per-session)

**Option 1 fixes 6 misses and creates 7.** Fixed: `waterproof → water resistant`,
`stainless steel strap → stainless steel band`, `zipper closure → closure: zip front closure`,
`slip on closure → pull on closure`. Broken: `100% cotton → 100% polyester / 100% nylon`,
`pull on closure → wrap closure / toggle closure`, `rubber sole → ethylene vinyl acetate sole /
fabric sole`, `hand wash → machine wash`. The gains (+5.67 utility over 21 sessions) and losses
(−6.60 over 19) cancel. A general-purpose sentence embedding measures topical closeness, not
attribute equivalence: on a calibration set the paraphrase pairs (`waterproof ~ water resistant`
0.839) and the attribute-changing siblings (`rubber sole ~ leather sole` 0.841,
`100% cotton ~ 100% polyester` 0.855, `machine wash ~ hand wash only` 0.741) overlap completely,
so no threshold separates them; 0.85 merely admits fewer of both.

**Option 2 fixes 4 misses and creates 7.** Dense shelves fired in 92 of 200 sessions (median
pool 646 products). They rescued `women's denim → Women Jeans`, `scrub pants → Medical Scrub
Bottoms`, `walking shoes → Athletic Walking`, and lost sessions where the wider pool admitted
decoys carrying the same verbatim values as the target (`casual dresses` pool grown to 2,175
products: rank 1 → 8).

## Reading

In this benchmark the shopper's paraphrase is checked against a catalog whose products differ by
exactly the attribute words a sentence embedding blurs. A dense channel earns its place in
production because it is trained on click and purchase logs for the vocabulary-mismatch
queries that an inverted index cannot reach; an off-the-shelf embedding over signature strings
is the weak version of that and, here, a wash. The language-layer ranking stays what it was:
the catalog reads the sentence first, the model translates on demand, and every admitted piece
of evidence is a catalog string.
