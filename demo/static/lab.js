import { STAGES, emptyView, applyStage, readEventStream } from "./lab-state.js";

const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const n = (value, digits = 0) => value == null ? "—" : Number(value).toLocaleString("en", { maximumFractionDigits: digits });
const rank = value => value == null ? "—" : `#${n(value)}`;
const signed = value => `${value > 0 ? "+" : ""}${n(value, 4)}`;
const productIcon = '<svg viewBox="0 0 32 32" fill="none" aria-hidden="true"><path d="m11 5-7 5 4 7 3-2v12h10V15l3 2 4-7-7-5c0 5-10 5-10 0Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M13 12h6" stroke="currentColor" stroke-width="1.5"/></svg>';
let config, bundle, recorded = { sessions: [] };
let mode = "live", session = null, history = [], view = emptyView(), selectedStage = null;
let busy = false, suggestion = "", preview = null, viewedTurn = null, playbackToken = 0;
let replayRows = [], replayIndex = 0, lastLive = null, activeController = null;

function error(message = "") { $("error").textContent = message; $("error").hidden = !message; }
function notice(message = "") { $("notice").textContent = message; $("notice").hidden = !message; }
async function jsonRequest(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const value = await response.json();
  if (!response.ok || value.error) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

function updateControls() {
  const ended = mode === "live" && Boolean(history.at(-1)?.done);
  if (config) {
    $("engine-status").className = `pill ${mode === "live" ? "live" : "replay"}`;
    $("engine-status").textContent = mode === "live" ? "LIVE ENGINE" : "RECORDED REPLAY";
  }
  $("live-mode").classList.toggle("selected", mode === "live");
  $("replay-mode").classList.toggle("selected", mode === "replay");
  $("live-mode").setAttribute("aria-pressed", String(mode === "live"));
  $("replay-mode").setAttribute("aria-pressed", String(mode === "replay"));
  $("live-mode").disabled = busy || !config?.health.live_available;
  $("replay-mode").disabled = busy;
  $("scenario").disabled = busy;
  $("reset").disabled = busy;
  $("message").disabled = busy || ended || mode !== "live" || !session;
  $("send").disabled = busy || ended || mode !== "live" || !session || !$("message").value.trim();
  $("suggest").disabled = busy || ended || !suggestion;
  $("rewrite").disabled = busy || ended || !$("message").value.trim() || !session;
  $("rewrite-model").disabled = busy;
  $("export").disabled = busy || !history.length;
  $("replay-trace").disabled = busy || !history.length;
  $("replay-next").disabled = busy || replayIndex >= replayRows.length;
  $("replay-all").disabled = busy || replayIndex >= replayRows.length;
  $("replay-controls").hidden = mode !== "replay";
  document.querySelector(".composer").hidden = mode !== "live";
  $("turn-badge").textContent = `Turn ${viewedTurn ?? history.length} / 10`;
  $("history-note").hidden = viewedTurn == null || viewedTurn === history.length;
}

function populateCases() {
  const previous = $("scenario").value;
  const entries = config.cases;
  const featured = { public_0099: "Featured · MVOI + Batch Planner", public_0187: "No-preference response", public_0144: "Intent change + Batch Planner" };
  const ordered = [...entries].sort((a, b) => Number(b.id in featured) - Number(a.id in featured));
  $("scenario").innerHTML = (mode === "live" ? '<option value="">Free conversation · no hidden target</option>' : (lastLive ? '<option value="last-live">Your last live conversation</option>' : "")) + ordered.map(s => `<option value="${esc(s.id)}">${esc(featured[s.id] || s.scenario.replaceAll("_", " "))} / ${esc(s.id)} · ${esc(s.title.slice(0, 90))}</option>`).join("");
  $("scenario").value = entries.some(s => s.id === previous) ? previous : (entries.some(s => s.id === "public_0099") ? "public_0099" : entries[0]?.id || "");
}

function blankConversation() {
  $("transcript").innerHTML = `<div class="welcome"><div class="avatar">a↗</div><div><h3>A little detail goes a long way.</h3><p>${mode === "live" ? "Start with a need. Watch ARC turn each message into evidence, a better shortlist, and its next decision." : "Replay a recorded conversation. The messages and decision flow advance together, one turn at a time."}</p><span class="little-label">YOUR WORDS → BETTER MATCHES</span></div></div>`;
}

async function resetSession() {
  if (busy) return;
  error(); preview = null; $("rewrite-preview").hidden = true; $("comparison").hidden = true;
  history = []; viewedTurn = null; replayIndex = 0; view = emptyView(); selectedStage = null;
  session = null; suggestion = ""; $("message").value = ""; $("latency").textContent = "";
  blankConversation(); renderAll();
  if (mode === "replay") {
    const id = $("scenario").value;
    if (id === "last-live" && lastLive) {
      session = structuredClone(lastLive.session); replayRows = structuredClone(lastLive.history);
    } else {
      const rich = recorded.sessions.find(s => s.sample_id === id);
      if (rich) { session = rich.session; replayRows = rich.turns; }
      else {
        const row = bundle.session_replays.sessions.find(s => s.sample_id === id);
        if (!row) { error("No recorded conversation is available for this scenario."); return; }
        const p = bundle.session_replays.products[row.target_parent_asin] || {};
        session = { target: { ...p, id: row.target_parent_asin }, scenario: row.scenario_type, reader: "recorded", sample_id: id };
        replayRows = row.turns.map(t => legacyRecord(t, row));
      }
    }
    notice(replayRows[0]?.legacy ? "Historical replay: internal ranks and full planner calculations were not captured in this recording. Available values are shown as recorded." : "Recorded computation · playback is slowed for explanation. Stage timings retain the measured values.");
  } else {
    busy = true; updateControls();
    notice("Starting the live agent and preparing the scenario…");
    try {
      session = await jsonRequest("/api/lab/reset", { sample_id: $("scenario").value || null });
      suggestion = session.suggestion; $("message").value = suggestion;
      notice();
    } catch (e) { error(e.message); notice(); }
    finally { busy = false; }
  }
  $("reader").textContent = session ? `Agent: ${session.agent_model || (mode === "replay" ? "recorded run" : "catalog reader · model off")}` : "";
  renderAll(); updateControls();
}

function renderTarget() {
  const target = session?.target;
  const current = history.find(r => r.turn === (viewedTurn ?? history.length));
  const overlay = current?.overlay;
  const caption = current?.legacy ? `Internal rank not recorded${overlay?.hit ? ` · hit at submitted #${overlay.recommendation_rank}` : ""}` : overlay?.eligible === false ? "Before intent override · not scored" : overlay?.hit ? `✓ Hit at submitted #${overlay.recommendation_rank}` : overlay && overlay.candidate_rank == null ? "Outside candidate pool" : "Candidate rank";
  $("target-card").innerHTML = target ? `<div class="product-icon">${productIcon}</div><div class="target-info"><span class="eyebrow">HIDDEN TARGET · EVALUATOR VIEW ONLY</span><strong>${esc(target.title || target.id)}</strong><small>${esc(target.id)}${target.price != null ? ` · $${esc(target.price)}` : ""}</small></div><div class="target-rank"><strong>${rank(overlay?.candidate_rank)}</strong><small>${esc(caption)}</small></div>` : `<div class="product-icon">${productIcon}</div><div class="target-info"><span class="eyebrow">FREE CONVERSATION</span><strong>Your preferences lead the search.</strong><small>No hidden target is assigned. Candidate results still update live.</small></div>`;
}

function renderChart() {
  const rows = history.filter(r => r.overlay && r.turn <= (viewedTurn ?? history.length));
  if (!rows.length) { $("rank-chart").innerHTML = `<div class="chart-empty">${session?.target ? "Send a message to start the rank trail" : "Choose a scenario to follow a known target"}</div>`; return; }
  const values = rows.flatMap(r => [r.overlay.candidate_rank, r.overlay.recommendation_rank]).filter(x => x != null);
  const max = Math.max(2, ...values);
  const x = turn => 28 + (turn - 1) * (490 / Math.max(3, rows.length - 1));
  const y = value => 16 + Math.log(value) / Math.log(max) * 38;
  let svg = `<svg viewBox="0 0 550 83" role="img" aria-label="Target ranks by turn, rank one at the top"><line x1="20" x2="535" y1="16" y2="16" stroke="#e7ece2" stroke-dasharray="3 4"/><text x="0" y="19" font-size="8" fill="#91a089">#1</text>`;
  for (const [key, color] of [["candidate_rank", "#176b50"], ["recommendation_rank", "#df886c"]]) {
    let previous = null;
    for (const row of rows) {
      const value = row.overlay[key];
      if (value == null) { previous = null; continue; }
      if (previous) svg += `<line x1="${x(previous.turn)}" y1="${y(previous.overlay[key])}" x2="${x(row.turn)}" y2="${y(value)}" stroke="${color}" stroke-width="1.8"/>`;
      svg += `<g data-turn="${row.turn}" style="cursor:pointer"><circle cx="${x(row.turn)}" cy="${y(value)}" r="${key === "candidate_rank" ? 4 : 2.8}" fill="${color}" stroke="white" stroke-width="1.2"/><title>Turn ${row.turn}: ${key === "candidate_rank" ? "candidate" : "submitted"} rank ${value}</title><text x="${x(row.turn)}" y="${y(value) - (key === "candidate_rank" ? 8 : -14)}" text-anchor="middle" font-size="8" fill="${color}">#${value}</text></g>`;
      previous = row;
    }
  }
  for (const row of rows) svg += `<text data-turn="${row.turn}" x="${x(row.turn)}" y="80" text-anchor="middle" font-size="8" fill="#8f9b88" style="cursor:pointer">T${row.turn}${row.overlay.candidate_rank == null && !row.legacy ? " ∅" : ""}</text>`;
  $("rank-chart").innerHTML = svg + "</svg>";
}

function renderCandidates() {
  const data = view.nodes.ranking.data;
  const top = data.top || [];
  $("candidate-count").textContent = data.count == null ? "Waiting for a message" : `${n(data.count)} candidates`;
  $("ranking-note").textContent = data.reported ? "Submitted list · internal ordering unavailable" : "Top 5 shown · ranking score";
  $("candidates").innerHTML = top.length ? top.slice(0, 5).map(p => `<div class="candidate-row ${p.id === session?.target?.id ? "target" : ""}"><span class="position">${p.rank}</span><span class="title" title="${esc(p.title)}">${esc(p.title || p.id)}</span><span class="score">${p.id === session?.target?.id ? '<span class="target-label">TARGET</span>' : p.score != null ? n(p.score, 3) : ""}</span></div>`).join("") : '<div class="empty-candidates">Matching products will appear here as the agent ranks.</div>';
  const evidence = view.nodes.evidence.data;
  $("evidence").innerHTML = (evidence.constraints || []).map(c => `<span class="evidence-chip ${(evidence.added || []).includes(c) ? "added" : ""}" title="${esc(c)}">${(evidence.added || []).includes(c) ? "+ " : ""}${esc(c)}</span>`).join("") + (evidence.removed || []).map(c => `<span class="evidence-chip removed" title="${esc(c)}">${esc(c)}</span>`).join("");
}

function nodeSummary(stage, node) {
  if (node.status === "running") return "Computing…";
  if (node.status === "waiting") return stage.hint;
  if (node.status === "skipped") return "Skipped this turn";
  const d = node.data;
  if (stage.id === "grounding") return d.recognized ? "Protocol understood · 0 model calls" : `${n(d.usage?.calls || 0)} model calls`;
  if (stage.id === "evidence") return `${n(d.constraints?.length || 0)} preferences held`;
  if (stage.id === "ranking") return `${n(d.count)} ranked candidates`;
  if (stage.id === "mvoi") return d.selected ? `Next question: ${d.selected}` : "No question selected";
  if (stage.id === "planner") return `${d.active ? "Batch DP" : "Output gate"} · ${n(d.emitted_count)} shown`;
  return d.response?.ask_attribute ? "Question + recommendations" : "Recommendations ready";
}

function renderFlow() {
  $("flow").innerHTML = STAGES.map((s, i) => {
    const node = view.nodes[s.id];
    const mark = ({ waiting: "○", running: "●", complete: "✓", skipped: "↷", failed: "!" })[node.status] || "○";
    return `${i ? `<div class="flow-edge ${node.status === "running" ? "active" : ""}" aria-hidden="true"><span>→</span></div>` : ""}<button class="flow-node ${esc(node.status)} ${selectedStage === s.id ? "selected" : ""}" data-stage="${s.id}" aria-pressed="${selectedStage === s.id}" aria-label="${s.name}: ${node.status}"><span class="node-top"><span class="node-number">0${i + 1}</span><span class="node-mark">${mark}</span></span><strong>${s.name}</strong><small>${esc(nodeSummary(s, node))}</small></button>`;
  }).join("");
  $("flow-status").textContent = busy ? (mode === "replay" ? `Replaying turn ${view.turn}` : `Processing turn ${view.turn}`) : view.failed ? `Turn ${view.turn} · recovered` : view.turn ? `Turn ${view.turn} · complete` : "Ready for turn 1";
}

function inspector(title, explanation, body) {
  return `<div class="inspector-summary"><div class="inspector-intro"><h3>${title}</h3><p>${explanation}</p></div><div class="inspector-body">${body}</div></div>`;
}
function facts(items) { return `<div class="fact-grid">${items.map(([label, value]) => `<div><small>${label}</small><strong>${esc(value)}</strong></div>`).join("")}</div>`; }

function renderInspector() {
  const id = selectedStage;
  if (!id) { $("inspector").innerHTML = '<div class="inspector-empty">↗ <span>The flow follows the agent’s real processing stages. Select a node to inspect its inputs, calculations, and results.</span></div>'; return; }
  const node = view.nodes[id], d = node.data;
  const stage = STAGES.find(s => s.id === id);
  if (node.status === "waiting" || node.status === "running") { $("inspector").innerHTML = inspector(stage.name, node.status === "running" ? stage.working : "This stage has not run in the selected turn.", '<div class="inspector-empty">Results appear when this stage completes.</div>'); return; }
  if (node.status === "skipped") { $("inspector").innerHTML = inspector(stage.name, "This stage did not run, or its calculation was not recorded.", `<div class="inspector-empty">${esc(d.reason || "Not captured in this historical replay.")}</div>`); return; }
  let html;
  if (id === "mvoi") {
    const values = d.values || [], max = Math.max(.001, ...values.map(v => Math.abs(v.answerable_metric_voi || 0)));
    html = inspector("Why ask this question?", "Candidate partitions estimate ranking gain. The no-preference branch keeps the existing order.", `<table class="value-table"><thead><tr><th>Question</th><th>Answerable</th><th>Expected pool</th><th>Expected MRR</th><th>Net MVOI</th></tr></thead><tbody>${values.map(v => `<tr class="${v.attribute === d.selected ? "chosen" : ""}"><td>${esc(v.attribute)}${v.attribute === d.selected ? " ✓" : ""}</td><td>${n(v.answerability * 100, 1)}%</td><td>${n(v.expected_remaining, 1)}</td><td>${n(v.expected_mrr, 3)}</td><td class="bar-cell"><span class="value-bar ${v.answerable_metric_voi < 0 ? "negative" : ""}" style="width:${Math.max(2, Math.abs(v.answerable_metric_voi) / max * 80)}px"></span>${signed(v.answerable_metric_voi)}</td></tr>`).join("")}</tbody></table><p class="calculation-note">0.50 × E[Hit@10] + 0.30 × E[MRR] − current rank utility − ${n(d.turn_cost ?? .02, 2)}. Answerability is a protocol estimate. A negative value is not an automatic stop rule.</p>`);
  } else if (id === "planner") {
    html = d.active ? inspector("How many should we recommend?", `${n(d.cohort_size)} indistinguishable siblings · ${n(d.remaining_turns)} turns remain. The planner evaluates hit utility and the continuation after a miss.`, `<table class="value-table"><thead><tr><th>Batch size</th><th>Hit chance</th><th>Hit contribution</th><th>Miss × continuation</th><th>Total value</th></tr></thead><tbody>${(d.choices || []).map(c => `<tr class="${c.batch === d.selected_batch ? "chosen" : ""}"><td>${c.batch}${c.batch === d.selected_batch ? " ✓" : ""}</td><td>${n(c.hit_probability * 100, 1)}%</td><td>${n(c.hit_value, 4)}</td><td>${n(c.miss_probability * c.continuation_value, 4)}</td><td>${n(c.expected_value, 4)}</td></tr>`).join("")}</tbody></table><p class="calculation-note">Each remaining sibling is equally likely in this estimate. A failed batch is refuted by the simulator’s continuation rule. This rule does not describe every human follow-up.</p>`) : inspector("Output gate active", "Batch DP runs when disclosure is complete and multiple signature siblings remain before the forced full-output turns.", `${facts([["Recommended this turn", n(d.emitted_count)], ["Signature siblings", n(d.cohort_size)], ["Next question", d.selected_question || "None"]])}<p class="calculation-note">${esc(d.reason || "The output gate selected this prefix.")}</p>`);
  } else if (id === "grounding") {
    html = inspector("From language to evidence", d.recognized ? "The protocol parser understood this message directly." : "The catalog reader runs first; the configured language layer can resolve wording it cannot read.", `${facts([["Reader", d.recognized ? "Protocol" : d.grounding?.status || "Catalog"], ["Model calls", n(d.usage?.calls || 0)], ["Stage time", `${n(node.duration, 1)} ms`]])}${d.grounding ? `<details><summary>Inspect grounding evidence</summary><pre>${esc(JSON.stringify(d.grounding, null, 2))}</pre></details>` : ""}`);
  } else if (id === "evidence") {
    html = inspector("What changed this turn?", "Preferences persist across turns. Changes and withdrawn preferences are recorded separately.", `${facts([["Preferences held", n(d.constraints?.length || 0)], ["Added", n(d.added?.length || 0)], ["Removed", n(d.removed?.length || 0)], ["Refuted products", n(d.proven_misses)]])}<p class="calculation-note">${esc(d.shelf || (d.shelves || []).slice(0, 3).join(" / ") || "Department not placed yet")}</p>`);
  } else if (id === "ranking") {
    html = inspector("Evidence determines the order", "The target is joined by the evaluator after respond returns. Ranking receives only shopper evidence and catalog data.", `${facts([["Ranked candidates", n(d.count)], ["Source pool", n(d.source_count)], ["Stage time", `${n(node.duration, 1)} ms`]])}<p class="calculation-note">${d.refined ? "The final order includes late evidence-tie exploration; MVOI was evaluated before that refinement." : "The candidate shortlist updates at this stage. Target ranks and submitted ranks are finalized after the response."}</p>`);
  } else {
    const response = d.response || {};
    html = inspector("The agent’s actual response", "A turn can contain both a clarification and recommendations.", `${facts([["Ask about", response.ask_attribute || "No further question"], ["Recommendations", n(response.recommendations?.length || 0)], ["Reported tokens", n((response.usage?.prompt_tokens || 0) + (response.usage?.completion_tokens || 0))]])}<details><summary>Inspect response contract</summary><pre>${esc(JSON.stringify(response, null, 2))}</pre></details>`);
  }
  $("inspector").innerHTML = html;
}

function renderAll() { renderTarget(); renderChart(); renderCandidates(); renderFlow(); renderInspector(); }

function appendUser(message, turn, rewrite) {
  $("transcript").querySelector(".welcome")?.remove();
  const div = document.createElement("div"); div.className = "message user";
  div.innerHTML = `<div class="message-meta"><button class="turn-link" data-turn="${turn}">Turn ${turn}</button><span>SHOPPER</span></div><div class="bubble">${esc(message)}</div>${rewrite ? `<div class="rewrite-attribution">${esc(rewrite.model)} · ${rewrite.source === "cached" ? "recorded" : "live"} paraphrase</div><details class="rewrite-attribution"><summary>Original message</summary>${esc(rewrite.original)}</details>` : ""}`;
  $("transcript").append(div);
  const pending = document.createElement("div"); pending.id = "pending"; pending.className = "pending";
  pending.innerHTML = '<span class="pending-dot"></span><span id="pending-text">Receiving your message…</span>';
  $("transcript").append(pending); scrollChat();
}

function appendAgent(response, turn, products = []) {
  $("pending")?.remove();
  const div = document.createElement("div"); div.className = "message agent"; div.dataset.replyTurn = turn;
  div.innerHTML = `<div class="message-meta"><strong>ARC</strong><span>SHOPPING COPILOT</span><button class="turn-link" data-turn="${turn}">View decision ↗</button></div><div class="bubble">${esc(response.message)}</div><div class="mini-products">${products.slice(0, 3).map((p, i) => `<div class="mini-product" title="${esc(p.title)}"><b>${i + 1}</b><span>${esc(p.title || p.id)}</span></div>`).join("")}${products.length > 3 ? `<span class="muted">+${products.length - 3} more</span>` : ""}</div>`;
  $("transcript").append(div); scrollChat();
}
function scrollChat() { $("transcript").scrollTo({ top: $("transcript").scrollHeight, behavior: "instant" }); }

function receiveStage(event, record, autoFocus = true) {
  if (!applyStage(view, event)) return;
  if (record) record.events.push(event);
  const stage = STAGES.find(s => s.id === event.stage);
  if (event.status === "running" && stage) {
    if (autoFocus) selectedStage = event.stage;
    if ($("pending-text")) $("pending-text").textContent = stage.working;
  }
  renderCandidates(); renderFlow(); renderInspector();
}

async function sendMessage() {
  if (busy || !session || history.at(-1)?.done) return;
  const message = $("message").value.trim(); if (!message) return;
  const rewrite = preview?.text === message ? preview : null;
  const source = rewrite?.original || message;
  const sessionId = session.session_id;
  const turn = history.length + 1;
  error(); viewedTurn = null; busy = true; selectedStage = null; view = emptyView(turn);
  const record = { turn, message, events: [], rewrite };
  history.push(record); appendUser(message, turn, rewrite); updateControls(); renderAll();
  preview = null; $("rewrite-preview").hidden = true;
  let completed = false;
  activeController = new AbortController();
  try {
    const response = await fetch("/api/lab/turn", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId, turn, message, source_message: source }), signal: activeController.signal });
    await readEventStream(response, (type, data) => {
      if (type === "error") throw new Error(data.error);
      if (data.session_id !== sessionId || data.turn !== turn) return;
      if (type === "stage") receiveStage(data, record);
      if (type === "done") {
        completed = true; Object.assign(record, data, { rewrite });
        appendAgent(data.response, turn, data.products);
        suggestion = data.suggestion || ""; $("message").value = suggestion;
        if (data.done) {
          const p = document.createElement("div"); p.className = "hit-banner";
          p.textContent = data.overlay?.hit ? `Target found · turn ${turn} · submitted rank #${data.overlay.recommendation_rank}` : "Ten-turn horizon reached. Start a new session to continue.";
          $("transcript").append(p); scrollChat();
        }
        $("latency").textContent = `${n(data.latency_ms, 1)} ms / turn`;
        selectedStage = data.events.some(e => e.stage === "planner" && e.data.active) ? "planner" : "mvoi";
        lastLive = { session: structuredClone(session), history: structuredClone(history) };
      }
    });
    if (!completed) throw new Error("The connection ended before the response arrived. Start a new session before retrying.");
  } catch (e) {
    error(e.message); $("pending")?.remove(); record.failed = true; record.done = true; view.failed = true;
    for (const node of Object.values(view.nodes)) if (node.status === "running") node.status = "failed";
  } finally {
    busy = false; activeController = null; renderAll(); updateControls();
  }
}

function showTurn(turn) {
  if (busy) return;
  const record = history.find(r => r.turn === turn); if (!record) return;
  viewedTurn = turn; view = emptyView(turn);
  for (const event of record.events || []) applyStage(view, event);
  selectedStage = record.events?.some(e => e.stage === "planner" && e.data.active) ? "planner" : "mvoi";
  $("latency").textContent = record.latency_ms != null ? `${n(record.latency_ms, 1)} ms / turn` : "Timing not recorded";
  renderAll(); updateControls();
}

async function rewriteMessage() {
  if (busy || !$("message").value.trim()) return;
  const original = $("message").value.trim(); error(); busy = true; updateControls();
  $("rewrite").textContent = "Rewriting…";
  try {
    preview = await jsonRequest("/api/lab/rewrite", { message: original, model: $("rewrite-model").value });
    $("rewrite-preview").innerHTML = `<small>${esc(preview.model)} · ${preview.source === "cached" ? "Recorded model output" : "Live model output"} · ${n(preview.latency_ms, 1)} ms</small><p>${esc(preview.text)}</p><small>Check that the rewrite preserves the shopper’s requirements.</small><div class="preview-actions"><button id="use-rewrite" class="primary-button">Use rewrite</button><button id="compare" class="text-button">Compare from this turn</button><button id="discard-rewrite" class="text-button">Dismiss</button></div>`;
    $("rewrite-preview").hidden = false;
    $("use-rewrite").onclick = () => { $("message").value = preview.text; $("rewrite-preview").hidden = true; updateControls(); };
    $("discard-rewrite").onclick = () => { preview = null; $("rewrite-preview").hidden = true; };
    $("compare").onclick = comparePreview;
  } catch (e) { error(e.message); }
  finally { busy = false; $("rewrite").textContent = "✧ Preview rewrite"; updateControls(); }
}

async function comparePreview() {
  if (busy || !preview || !session) return;
  busy = true; updateControls(); error(); $("comparison").hidden = false;
  $("comparison").innerHTML = '<h2>Same conversation. Two ways to say it.</h2><p>Running original and rewritten messages from identical session snapshots. Your live conversation stays at this turn.</p><div class="comparison-grid"><div id="branch-original">Original · waiting…</div><div id="branch-rewritten">Paraphrase · waiting…</div></div>';
  let complete = false;
  try {
    const response = await fetch("/api/lab/compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: session.session_id, original: preview.original, rewritten: preview.text }) });
    await readEventStream(response, (type, data) => {
      if (type === "error") throw new Error(data.error);
      if (type === "comparison") complete = true;
      if (type !== "branch" || !["original", "rewritten"].includes(data.branch)) return;
      const el = $(`branch-${data.branch}`);
      if (data.status === "running") { el.textContent = `${data.branch === "original" ? "Original" : "Paraphrase"} · processing…`; return; }
      const r = data.result;
      el.innerHTML = `<h3>${data.branch === "original" ? "Original message" : "Paraphrased message"}</h3><p>${esc(r.message)}</p>${facts([["Candidate rank", rank(r.overlay?.candidate_rank)], ["Next question", r.response.ask_attribute || "None"], ["Shown", n(r.products.length)]])}<p>${esc(r.certificate.constraints?.join(" · ") || "No preferences extracted")}</p><span class="muted">${n(r.certificate.llm_usage?.calls || 0)} model calls · ${n(r.latency_ms, 1)} ms · latency may include warm-cache effects</span>`;
    });
    if (!complete) throw new Error("Comparison stream ended early. Retry the comparison.");
  } catch (e) { error(e.message); }
  finally { busy = false; updateControls(); }
}

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function animateRecord(record, appendMessages) {
  const token = ++playbackToken;
  view = emptyView(record.turn); selectedStage = null; viewedTurn = record.turn;
  if (appendMessages) appendUser(record.message, record.turn, record.rewrite);
  $("flow-status").classList.add("replay");
  for (const event of record.events || []) {
    if (token !== playbackToken) return;
    receiveStage(event, null);
    $("flow-status").textContent = `Calculation replay · turn ${record.turn}`;
    await delay(event.status === "running" ? 450 : 110);
  }
  if (appendMessages) appendAgent(record.response, record.turn, record.products);
  $("flow-status").classList.remove("replay");
  $("latency").textContent = record.latency_ms != null ? `${n(record.latency_ms, 1)} ms measured / turn` : "Timing not recorded";
}

async function replayNext(all = false) {
  if (busy || replayIndex >= replayRows.length) return;
  busy = true; updateControls();
  do {
    const record = structuredClone(replayRows[replayIndex++]);
    // Reveal only this turn's observation after its recorded response.
    history.push({ turn: record.turn, events: [], message: record.message });
    await animateRecord(record, true);
    history[history.length - 1] = record;
    renderAll();
    if (all && replayIndex < replayRows.length) await delay(650);
  } while (all && replayIndex < replayRows.length);
  busy = false; viewedTurn = null; renderAll(); updateControls();
}

function legacyRecord(turn, row) {
  const cert = turn.decision || {}, products = bundle.session_replays.products;
  const response = { message: turn.agent_message, ask_attribute: turn.ask_attribute, recommendations: turn.recommendations.map(id => ({ parent_asin: id })), usage: { prompt_tokens: turn.reported_tokens || 0, completion_tokens: 0 } };
  const facts = turn.recommendations.map(id => ({ ...(products[id] || {}), id }));
  let seq = 0;
  const event = (stage, status, data) => ({ stage, status, data, seq: ++seq, turn: turn.turn });
  const events = [event("grounding", "skipped", { reason: "Grounding trace not captured in this recording." }), event("evidence", "complete", { constraints: cert.constraints || [], shelf: cert.shelf, proven_misses: cert.proven_miss_count }), event("ranking", "complete", { top: facts.map((p, i) => ({ ...p, rank: i + 1 })), count: cert.ranked_candidate_count, reported: true }), event("mvoi", cert.selected_question_value?.attribute ? "complete" : "skipped", { values: cert.selected_question_value?.attribute ? [cert.selected_question_value] : [], selected: cert.question, reason: "No question values recorded." }), event("planner", "skipped", { reason: `Full planner calculation not captured. Actual submitted count: ${facts.length}.` }), event("response", "complete", { response })];
  return { turn: turn.turn, message: turn.customer, response, products: facts, events, legacy: true, certificate: cert, overlay: { candidate_rank: null, recommendation_rank: turn.target_rank, eligible: turn.eligible_for_hit, hit: turn.hit_this_turn }, done: turn.turn === row.turns.length };
}

$("chat-form").addEventListener("submit", e => { e.preventDefault(); sendMessage(); });
$("message").addEventListener("input", updateControls);
$("message").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendMessage(); } });
$("suggest").onclick = () => { $("message").value = suggestion; preview = null; $("rewrite-preview").hidden = true; updateControls(); $("message").focus(); };
$("reset").onclick = resetSession;
$("scenario").onchange = resetSession;
$("rewrite").onclick = rewriteMessage;
$("latest").onclick = () => { showTurn(history.length); viewedTurn = null; updateControls(); };
$("flow").onclick = e => { const node = e.target.closest("[data-stage]"); if (node) { selectedStage = node.dataset.stage; renderFlow(); renderInspector(); } };
for (const id of ["transcript", "rank-chart"]) $(id).onclick = e => { const el = e.target.closest("[data-turn]"); if (el) showTurn(Number(el.dataset.turn)); };
$("fullscreen").onclick = async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch (e) { error(e.message); } };
$("live-mode").onclick = () => { if (!busy && mode !== "live") { mode = "live"; populateCases(); resetSession(); } };
$("replay-mode").onclick = () => { if (!busy && mode !== "replay") { mode = "replay"; populateCases(); if (lastLive) $("scenario").value = "last-live"; resetSession(); } };
$("replay-next").onclick = () => replayNext(false);
$("replay-all").onclick = () => replayNext(true);
$("replay-trace").onclick = async () => {
  if (busy || !history.length) return;
  busy = true; updateControls();
  await animateRecord(history[(viewedTurn ?? history.length) - 1], false);
  busy = false; renderAll(); updateControls();
};
$("export").onclick = () => {
  const body = JSON.stringify({ format: "arc-live-lab-v1", mode, session, turns: history }, null, 2);
  const url = URL.createObjectURL(new Blob([body], { type: "application/json" }));
  const a = document.createElement("a"); a.href = url; a.download = `arc-${session?.sample_id || "conversation"}-${Date.now()}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};

async function init() {
  try {
    const isStatic = document.documentElement.dataset.deploy === "static";
    const results = await Promise.allSettled([
      jsonRequest(isStatic ? "./data/health.json" : "/api/lab/config"),
      jsonRequest(isStatic ? "./data/demo_bundle.json" : "/api/demo"),
      jsonRequest("./lab-replays.json"),
    ]);
    if (results[0].status !== "fulfilled") throw results[0].reason;
    if (results[1].status !== "fulfilled") throw results[1].reason;
    bundle = results[1].value;
    config = isStatic ? { health: results[0].value, models: [], cases: bundle.session_replays.sessions.map(s => ({ id: s.sample_id, scenario: s.scenario_type, title: bundle.session_replays.products[s.target_parent_asin]?.title || s.target_parent_asin })) } : results[0].value;
    if (results[2].status === "fulfilled") recorded = results[2].value;
    mode = config.health.live_available ? "live" : "replay";
    $("engine-status").className = `pill ${mode === "live" ? "live" : "replay"}`;
    $("engine-status").textContent = mode === "live" ? "LIVE ENGINE" : "RECORDED REPLAY";
    $("rewrite-model").innerHTML = (config.models || []).map(m => `<option value="${esc(m.id)}" ${!m.available ? "disabled" : ""}>${esc(m.name)} · ${m.live ? "live + cache" : "recorded"}</option>`).join("");
    populateCases(); await resetSession();
    if (!config.health.live_available && !isStatic) notice("Recorded replay · for live conversations start: python3 demo/server.py --catalog data/catalog.jsonl --live --prewarm");
  } catch (e) { error(`Could not load the Decision Lab: ${e.message}`); $("engine-status").textContent = "UNAVAILABLE"; }
}
init();
