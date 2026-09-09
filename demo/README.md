# ARC Decision Lab

## Final presentation: Live Decision Lab

```bash
python3 demo/server.py --catalog data/catalog.jsonl --live --prewarm
```

Open **<http://127.0.0.1:8765/lab.html>**. Catalog preparation can take a while
on a cold start; prewarm before the presentation.

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

## GitHub Pages deployment

**<https://kelvin715.github.io/techjam-2026-shopping-copilot/>** opens the
Decision Lab. The previous presentation is retired from the published site.
The `/lab.html` address also opens the same new demo.

```bash
python3 tools/build_static_site.py
python3 -m http.server 8080 --directory site
```

The exporter publishes `lab.html` as `index.html`, copies only the Decision
Lab assets, and marks the document with `data-deploy="static"`. The page reads
`data/demo_bundle.json`, `data/health.json`, and `lab-replays.json` from relative
paths, so the project subpath works without a backend. A stale source manifest
fails the export; rebuild the bundle after changing its recorded sources.

`.github/workflows/pages.yml` publishes this export on every push to `main`.
GitHub Pages serves the synchronized replay. Live shopper input, model calls,
and paraphrase comparisons use the local Python server described above.
