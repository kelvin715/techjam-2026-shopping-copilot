# ARC Decision Lab

## Final presentation: Live Decision Lab

```bash
python3 demo/server.py --catalog data/catalog.jsonl --live --prewarm
```

Open **<http://127.0.0.1:8765/lab.html>**. The existing presentation also links
to the new page. Catalog preparation can take a while on a cold start; prewarm
before the presentation.

The new page ties chat, candidate rankings, and the flow diagram to real stage
events streamed from `Agent.respond`. Sending a message starts the flow;
the response appears when the agent finishes. Click a node to inspect its
measured result, or a historical chat turn to restore that turn's snapshot.
The target is joined on the evaluator side only after the response returns.
Internal candidate rank and submitted recommendation rank are separate.
Free conversation has no ground-truth target.

Suggested presentation scenarios:

- **public_0099 (default):** five turns, target ranks 36 → 4 → 3 → 2 → 1;
  actual MVOI and Batch Planner calculations, including batch-size alternatives.
- **public_0144:** intent override plus a planner that chooses batches of two.
- **public_0187:** a no-preference response followed by progressively stronger evidence.

The planner panel shows DP only when it actually runs; otherwise it explains
the output gate. Question values are protocol estimates, and a negative MVOI
does not itself stop the current policy from asking.

### Shopper rewrites and agent language understanding

The **Paraphrase with** selector controls the shopper rewriter, separately
from the agent model shown in the toolbar. Both `gemma4` and
`qwen2.5-7b-instruct` can use the existing recorded rewrite caches. A cache miss
without a configured endpoint is reported explicitly. Preview a rewrite,
check its meaning, then choose **Use rewrite** and send. **Compare from this
turn** runs original and rewritten messages from identical session snapshots
without advancing the live conversation. Model caches are shared, so timing
in a comparison can include warm-cache effects.

Without model configuration, the demo uses the catalog-only free-text reader.
To enable the agent's optional language layer:

```bash
ARC_LLM_MODE=assist \
ARC_LLM_BASE_URL=http://your-agent-host:8000/v1 \
ARC_LLM_MODEL=gemma4 \
python3 demo/server.py --catalog data/catalog.jsonl --live --prewarm
```

Set `ARC_DEMO_GEMMA_BASE_URL` and `ARC_DEMO_QWEN_BASE_URL` to enable live
shopper rewrites. Each also accepts `_MODEL`, `_API_KEY`, and `_CACHE` in place
of `_BASE_URL`. The fallback endpoint is `ARC_CUSTOMER_BASE_URL`, then
`ARC_LLM_BASE_URL`. The rewriter receives only the shopper message and a
meaning-preserving rewrite instruction. Runtime configuration never comes
from the browser, and new rewrite results stay in memory rather than modifying
the research caches. `ARC_LLM_TIMEOUT` bounds model requests.

### Replay, export, and validation

**Replay** can play the last live run or a recorded public scenario, with chat
and flow advancing together. **Replay this calculation** slows an individual
turn's recorded events for explanation. Live mode adds no artificial delay.
Measured stage durations stay visible during replay. **Export run** downloads
the conversation, events, certificates, and evaluator observations as JSON.

Six featured scenarios include full recorded stage traces. Other historical
public sessions show only the data captured in their original recordings;
unrecorded internal ranks and planner values are marked unavailable.
The static site exports this replay interface, with live controls disabled.

After changing the agent, regenerate the six full traces, then rebuild the
main bundle:

```bash
python3 tools/build_live_demo_replays.py
python3 -m unittest discover -s tests -v
node tests/live_lab_frontend.mjs
python3 tools/build_demo_bundle.py --catalog data/catalog.jsonl --tests-verified
```

The optional observer is request-local, carries no hidden target, and cannot
alter the response if a viewer disconnects. Tests cover response parity,
planner values, target isolation, turn ordering, comparison state isolation,
and streaming a running-stage event before the response completes.

## Existing presentation and session explorer

The demo is an offline, judge-facing presentation of ARC — the submitted
Ask · Rank · Commit agent. It does not change or
import into the official `Agent.reset/respond` path.

## Run the verified replay

```bash
python3 demo/server.py
```

Open <http://127.0.0.1:8765>. The eight-page Presentation and the ninth-page
Session Explorer work without the 50,000-row catalog. Every displayed state is
copied from a measured run of the unchanged public evaluator. Page 9 can select
all 200 public sessions and replay their conversations, rankings, compact
decision certificates, and evaluator-only target results one turn at a time.

Useful recording URLs:

```text
http://127.0.0.1:8765/?scene=0
http://127.0.0.1:8765/?scene=0&autoplay=1
http://127.0.0.1:8765/?scene=7
http://127.0.0.1:8765/?scene=8
```

`scene` is zero-based and clamps to the last page, so the four-turn case study
is pages 4–7 (`?scene=3` … `?scene=6`), the evaluation evidence is page 8
(`?scene=7`), and the Session Explorer is page 9 (`?scene=8`). The recorded
three-minute story can still end on page 8; page 9 is a useful judge-led proof
surface after the video.

Keyboard controls: left/right arrows, space to play or pause, and `F` for
fullscreen.

## Rebuild the replay bundle

After changing the Agent, regenerate the all-session traces and rebuild:

```bash
python3 tools/build_session_replays.py
python3 tools/build_demo_bundle.py \
  --catalog data/catalog.jsonl \
  --output demo/data/demo_bundle.json
```

`build_session_replays.py` executes all 200 public cases through the submitted
Agent and the unchanged local evaluator, then checks every outcome against
`results/public_full.json`. The wrapper records the Agent response first and
joins the hidden target only afterwards. The frontend never reruns or estimates
the evaluator.

The server checks every recorded source hash. A mismatch changes the green
`VERIFIED REPLAY` badge to `STALE BUNDLE`; do not record while it is stale.

After the complete test suite passes, build with `--tests-verified` so the
evaluation page can accurately state that all currently discovered tests passed:

```bash
python3 -m unittest discover -v
python3 tools/build_demo_bundle.py --catalog data/catalog.jsonl --tests-verified
```

If the catalog is unavailable, omit `--catalog`. The replay still works, but
catalog-enriched product details and exact shelf count will be unavailable.

## Publish the replay for reviewers

The replay is a static page. Export it, and the whole demo is a directory a
reviewer can open from any URL:

```bash
python3 tools/build_static_site.py
python3 -m http.server 8080 --directory site
```

The export marks `index.html` with `data-deploy="static"`, so the frontend reads
`data/demo_bundle.json` and `data/health.json` instead of the server API. Asset
paths are relative, so the site works at a domain root and under a project-page
subpath such as `https://<user>.github.io/<repo>/`.

The freshness check runs at export time and a stale replay fails the build, so a
published verified badge always reflects sources that matched. Rebuild the
bundle first if the export refuses.

`.github/workflows/pages.yml` runs the same export on every push to `main`. In
the repository settings, set Pages → Source to **GitHub Actions** once.

The Live Agent is not exported. It needs the catalog and a persistent Python
process, neither of which GitHub Pages provides. The complete 200-session
explorer is exported, so online reviewers can still select a case, simulate the
conversation turn by turn, and verify the result.

## Optional live engine

```bash
python3 demo/server.py \
  --catalog data/catalog.jsonl \
  --live \
  --prewarm
```

Live mode is visibly separated from the verified replay. The HTTP service binds
to `127.0.0.1`, accepts only bounded JSON, and does not upload files or make
network calls.

### Page 9 modes

Page 9 defaults to **Verified sessions**:

- filter by Buying, Browsing, Intent Override, or Boundary;
- choose any of the 200 public session IDs or sample one at random;
- step through only the messages and responses available at each turn;
- inspect the ordered IDs returned to the evaluator and the target's observed
  rank; and
- audit the compact action, evidence, miss, margin, answerability, and reason
  fields captured after that response.

The target card and hit/rank overlay are labeled evaluator-only. They are joined
after `Agent.respond` and were never sent to the Agent. This mode is available
on the static site.

With live mode on, switch Page 9 to **Live Agent**:

- type any shopper message, or load one of the recorded openers as a starting
  point; the presets come from the replayed trace, so they always match a real
  catalog shelf;
- each turn shows the response the evaluator would receive — the clarification,
  `ask_attribute`, the ordered `parent_asin` list, and the reported token count;
- **Explain this decision** calls `Agent.explain_last_decision` and prints the
  certificate: action, reason codes, clues held, pool sizes, top-2 margin, and
  the verified minimal counterfactual;
- the session stops at the protocol's tenth turn; **New session** starts a fresh
  `Agent.reset`.

Product titles come from a separate `catalog_facts` field so the response object
stays byte-identical to what the Agent returned. They are the normalised catalog
strings the index holds, which is why they are lower-case. Without `--live`,
only that secondary tab explains how to enable the engine; Verified sessions
remains fully interactive.
