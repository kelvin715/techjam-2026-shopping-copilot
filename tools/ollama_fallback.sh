#!/usr/bin/env bash
# Set up the offline fallback for ARC's optional language layer.
#
# Installs Ollama into ~/.local (no root), pulls a small instruct model,
# starts the server, and verifies that ARC's own client can reach it. Every
# step is skipped when it is already done, so re-running is cheap.
#
#   bash tools/ollama_fallback.sh                 # install + pull + serve + verify
#   bash tools/ollama_fallback.sh qwen2.5:3b      # a smaller model
#   bash tools/ollama_fallback.sh qwen2.5:7b 11435 # pin the port
#
# macOS: install Ollama from https://ollama.com/download instead (the app or
# `brew install ollama`), then run this script — it detects an ollama already
# on PATH and only does the pull, serve and verify steps.

set -euo pipefail

MODEL="${1:-qwen2.5:7b}"
PORT="${2:-11434}"
PREFIX="${OLLAMA_PREFIX:-$HOME/.local}"
HOST="127.0.0.1:${PORT}"
BASE_URL="http://${HOST}/v1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- install
if command -v ollama >/dev/null 2>&1; then
  OLLAMA="$(command -v ollama)"
  say "ollama already on PATH: $OLLAMA"
elif [ -x "$PREFIX/bin/ollama" ]; then
  OLLAMA="$PREFIX/bin/ollama"
  say "ollama already installed: $OLLAMA"
else
  case "$(uname -s)" in
    Linux) : ;;
    *) echo "Install Ollama from https://ollama.com/download first, then re-run." >&2; exit 1 ;;
  esac
  say "installing ollama into $PREFIX (no root required)"
  TAG="$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
          | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)"
  [ -n "$TAG" ] || { echo "could not resolve the latest ollama release" >&2; exit 1; }
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  # Releases ship .tar.zst; older ones shipped .tgz. Try both.
  if curl -fL --retry 3 -o "$TMP/ollama.tar.zst" \
       "https://github.com/ollama/ollama/releases/download/${TAG}/ollama-linux-amd64.tar.zst"; then
    command -v zstd >/dev/null 2>&1 || { echo "zstd is required to extract this release" >&2; exit 1; }
    mkdir -p "$PREFIX"
    tar -C "$PREFIX" --zstd -xf "$TMP/ollama.tar.zst"
  else
    curl -fL --retry 3 -o "$TMP/ollama.tgz" \
      "https://github.com/ollama/ollama/releases/download/${TAG}/ollama-linux-amd64.tgz"
    mkdir -p "$PREFIX"
    tar -C "$PREFIX" -xzf "$TMP/ollama.tgz"
  fi
  OLLAMA="$PREFIX/bin/ollama"
  say "installed $($OLLAMA --version 2>&1 | tail -1)"
fi

# ------------------------------------------------------------------ serve
if curl -sf --max-time 2 "http://${HOST}/api/version" >/dev/null 2>&1; then
  say "server already listening on ${HOST}"
else
  say "starting the server on ${HOST}"
  LOG="${TMPDIR:-/tmp}/ollama-${PORT}.log"
  OLLAMA_HOST="$HOST" nohup "$OLLAMA" serve >"$LOG" 2>&1 &
  for _ in $(seq 1 40); do
    curl -sf --max-time 2 "http://${HOST}/api/version" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf --max-time 2 "http://${HOST}/api/version" >/dev/null 2>&1 \
    || { echo "server did not come up; see $LOG" >&2; exit 1; }
  say "server up (log: $LOG)"
fi

# ------------------------------------------------------------------- pull
if OLLAMA_HOST="$HOST" "$OLLAMA" list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$MODEL"; then
  say "model $MODEL already present"
else
  say "pulling $MODEL (a few GB, one time)"
  OLLAMA_HOST="$HOST" "$OLLAMA" pull "$MODEL"
fi

# ----------------------------------------------------------------- verify
say "verifying ARC's client against $BASE_URL"
cd "$ROOT"
ARC_LLM_BASE_URL="$BASE_URL" ARC_LLM_MODEL="$MODEL" python3 tools/llm_keepalive.py --once

say "checking that the grounding prompt returns usable JSON"
ARC_LLM_BASE_URL="$BASE_URL" ARC_LLM_MODEL="$MODEL" python3 - "$MODEL" "$BASE_URL" <<'PY'
import json, sys, time
from src.llm import ChatClient, LLMSettings
from src.ground import SYSTEM_PROMPT

model, base_url = sys.argv[1], sys.argv[2]
client = ChatClient(LLMSettings(mode="ground", base_url=base_url, model=model, timeout=600))
probes = [
    ("open", "looking for a belt for my husband, something rustic with a buckle, real leather"),
    ("override", "actually forget the leather, i just want it black"),
    ("no_preference", "honestly i don't care about that"),
]
ok = True
for expected, message in probes:
    started = time.perf_counter()
    reply = client.chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Shopper message: " + message}],
        max_tokens=200,
    )
    elapsed = time.perf_counter() - started
    parsed = reply.json()
    good = bool(parsed) and parsed.get("intent") == expected
    ok = ok and good
    print(f"  [{elapsed:5.1f}s] intent={None if not parsed else parsed.get('intent'):<14} "
          f"expected={expected:<14} {'ok' if good else 'MISMATCH'}")
    if parsed:
        print("           " + json.dumps(
            {k: parsed.get(k) for k in ("category", "material", "color", "features", "dropped")},
            ensure_ascii=False))
print("\nverdict:", "usable for the live demo" if ok else
      "this model misreads the dialogue acts — try a larger one")
sys.exit(0 if ok else 1)
PY

say "ready — start the demo with:"
cat <<EOF

  ARC_LLM_MODE=assist ARC_LLM_BASE_URL=${BASE_URL} ARC_LLM_MODEL=${MODEL} \\
    python3 demo/server.py --catalog data/catalog.jsonl --live --prewarm

Then open http://127.0.0.1:8765/ and switch to the Live Agent tab.
Measure this laptop before trusting it on stage:

  ARC_LLM_BASE_URL=${BASE_URL} ARC_LLM_MODEL=${MODEL} \\
    python3 tools/human_language_benchmark.py --count 20 --levels paraphrase \\
      --arms hybrid --customer-model qwen2.5-7b-instruct \\
      --cache results/human_language_cache_qwen_customer.json \\
      --output results/ollama_fallback_check.json

EOF
