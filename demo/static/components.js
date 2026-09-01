import { SCENES } from "./state.js";

const escapeMap = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#039;",
};

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => escapeMap[character]);
}

const score = (value) => Number(value).toFixed(6);
const decimal = (value, places = 2) => Number(value).toFixed(places);
const percent = (value, places = 0) => `${(Number(value) * 100).toFixed(places)}%`;
const compact = (value) => new Intl.NumberFormat("en-US", { notation: "compact" }).format(value);
const count = (value) => Number(value).toLocaleString("en-US");

// The six stages of the pipeline, in the order a message travels through them.
// `where` points at the file a judge can open to check the claim.
const PIPELINE = [
  {
    id: "read",
    name: "Read",
    method: "Template parser with alternative phrasings, plus catalog-grounded clue splitting",
    purpose: "Turn a sentence into clues the ranker can actually use",
    where: "src/dialog.py",
  },
  {
    id: "remember",
    name: "Remember",
    method: "Per-session state: clues with confidence, products already ruled out, clue-order reliability",
    purpose: "Build up across turns instead of re-querying from the latest message",
    where: "src/dialog.py",
  },
  {
    id: "narrow",
    name: "Narrow",
    method: "Reconstruct the category shelf from the opening line and search only there",
    purpose: "50,000 down to a few hundred, with no vector database",
    where: "src/shelf.py",
  },
  {
    id: "rank",
    name: "Rank",
    method: "Three evidence layers, then a popularity prior that is only allowed near the top",
    purpose: "Order by evidence, and let popularity break ties but never win them",
    where: "src/rank.py",
  },
  {
    id: "decide",
    name: "Decide",
    method: "Value-of-information over possible answers, then a risk-calibrated output gate",
    purpose: "Choose what to ask, and how many products are safe to show now",
    where: "src/evidence.py · src/policy.py",
  },
  {
    id: "explain",
    name: "Explain",
    method: "Evidence record plus the smallest clue removal that changes the winner",
    purpose: "Make every answer re-runnable instead of narrated after the fact",
    where: "src/evidence.py",
  },
];

const REASON_TEXT = {
  previously_shown_products_excluded:
    "Dropped the product IDs submitted on earlier turns. The evaluator continued, so none matched the hidden target.",
  maximum_answerability_aware_metric_value_of_information:
    "Asked the question with the best expected payoff, after discounting answers the shopper can't give.",
  information_complete: "Every clue this shopper can give has now been given.",
  maximum_protocol_constraints_observed: "We already hold every clue the conversation can reveal.",
  boundary_safe_open_question: "The shopper deflected the last attribute, so we asked an open question instead.",
  late_evidence_tie_exploration: "Several products are tied on evidence, so we rotate through them.",
  more_evidence_has_positive_value: "One more question is still worth the extra turn.",
  no_useful_clarification_remaining: "Nothing left worth asking. Time to show the list.",
};

const ASK_TEXT = {
  other: "anything else that matters",
  material: "the fabric",
  color: "the colour",
  size: "the size",
  style: "the style",
  brand: "a brand",
  budget: "the budget",
  feature: "a particular feature",
  use_case: "what it's for",
  category: "the category",
};

const askedAbout = (attribute) =>
  attribute ? (ASK_TEXT[attribute] || String(attribute).replaceAll("_", " ")) : "nothing further";

function icon(name) {
  const paths = {
    check: '<path d="m5 12 4 4L19 6"/>',
    shield: '<path d="M12 3 20 6v5c0 5-3.4 8.3-8 10-4.6-1.7-8-5-8-10V6l8-3Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/>',
    spark: '<path d="m12 2 1.7 5.3L19 9l-5.3 1.7L12 16l-1.7-5.3L5 9l5.3-1.7L12 2Z"/><path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z"/>',
    arrow: '<path d="M5 12h14M14 7l5 5-5 5"/>',
    branch: '<path d="M6 3v6a3 3 0 0 0 3 3h9"/><path d="m14 8 4 4-4 4"/><circle cx="6" cy="3" r="2"/>',
    lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    play: '<path d="m8 5 11 7-11 7V5Z"/>',
    pause: '<path d="M8 5v14M16 5v14"/>',
    back: '<path d="m15 18-6-6 6-6"/>',
    reset: '<path d="M4 7v5h5"/><path d="M5.5 16a8 8 0 1 0 .7-9L4 9"/>',
    expand: '<path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5"/>',
    terminal: '<path d="m5 7 4 4-4 4M11 17h8"/>',
    close: '<path d="m6 6 12 12M18 6 6 18"/>',
    question: '<path d="M9.2 9a3 3 0 1 1 4 2.8c-.8.3-1.2 1-1.2 1.9v.3"/><circle cx="12" cy="18" r=".6" fill="currentColor"/>',
  };
  return `<svg class="icon" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${paths[name] || ""}</svg>`;
}

function badge(label, tone = "neutral", glyph = "") {
  return `<span class="badge badge-${tone}">${glyph ? icon(glyph) : ""}${esc(label)}</span>`;
}

function header(bundle, health, state) {
  const fresh = Boolean(health.replay_fresh);
  const modeBadge = fresh
    ? badge("Verified replay", "cyan", "shield")
    : badge("Out of date", "warning", "shield");
  const rows = bundle.catalog?.row_count || 50000;
  const sceneId = SCENES[state.scene]?.id;
  const walkMatch = sceneId?.match(/^walk-(\d)$/);
  const turnLabel = walkMatch ? walkMatch[1] : "—";
  const contextStat = turnLabel === "—"
    ? '<span class="turn-stat">Horizon <strong>H = 10</strong></span>'
    : `<span class="turn-stat">Case turn <strong>${turnLabel}</strong> of 4</span>`;
  return `
    <header class="status-bar">
      <button class="brand" data-action="home" aria-label="Back to the start">
        <span class="brand-mark" aria-hidden="true">π</span>
        <span class="brand-copy"><strong>ARC · ASK · RANK · COMMIT</strong><small>TechJam 2026 · evidence-grounded decisions</small></span>
      </button>
      <div class="system-status">
        ${modeBadge}
        ${badge("No internet", "quiet")}
        ${badge("0 AI tokens", "quiet")}
        <span class="catalog-stat"><strong>${compact(rows)}</strong> products</span>
        ${contextStat}
      </div>
    </header>
    ${fresh ? "" : `
      <div class="stale-banner" role="alert">
        ${icon("shield")} Some source files changed after this run was recorded. Rebuild before you record video.
        <code>${esc((health.changed_sources || []).slice(0, 2).join(", "))}</code>
      </div>
    `}
  `;
}

function sceneHeader(kicker, title, copy = "") {
  return `
    <div class="scene-heading">
      <p class="eyebrow">${esc(kicker)}</p>
      <h1>${title}</h1>
      ${copy ? `<p class="scene-copy">${copy}</p>` : ""}
    </div>
  `;
}

/* ---------------------------------------------------------------- pipeline */

function pipelineStrip(activeIds = [], { compactStrip = true } = {}) {
  const active = new Set(activeIds);
  return `
    <div class="pipeline ${compactStrip ? "is-compact" : ""}" aria-label="Pipeline stages">
      ${PIPELINE.map((stage, index) => `
        ${index ? `<i class="pipeline-link" aria-hidden="true"></i>` : ""}
        <div class="pipeline-stage ${active.has(stage.id) ? "is-active" : ""}">
          <span class="pipeline-index">${index + 1}</span>
          <strong>${esc(stage.name)}</strong>
          ${compactStrip ? "" : `
            <p class="pipeline-method">${esc(stage.method)}</p>
            <p class="pipeline-purpose">${icon("arrow")}${esc(stage.purpose)}</p>
            <code>${esc(stage.where)}</code>
          `}
        </div>
      `).join("")}
    </div>
  `;
}

/* ------------------------------------------------------------------ scenes */

function problemScene(bundle) {
  const trace = bundle.explore_trace || bundle.trace;
  const first = trace.turns[0];
  return `
    <section class="scene scene-problem" aria-labelledby="problem-title">
      ${sceneHeader(
        "Problem formulation · finite horizon H = 10",
        '<span id="problem-title">Conversational shopping is a partially observable sequential decision problem.</span>',
        "The target product and the shopper’s complete preference set are hidden. Each turn reveals only partial evidence, so the system must acquire information before it commits to an answer."
      )}
      <div class="formal-model" aria-label="Sequential decision problem formulation">
        <article class="model-node is-hidden">
          <span class="model-symbol">s</span>
          <div><small>Hidden state</small><code>s = (p*, C*)</code><p>target product + complete constraints</p></div>
        </article>
        <i>${icon("arrow")}</i>
        <article class="model-node">
          <span class="model-symbol">o<sub>t</sub></span>
          <div><small>Observation</small><code>o<sub>t</sub> = utterance<sub>t</sub></code><p>one partial clue at a time</p></div>
        </article>
        <i>${icon("arrow")}</i>
        <article class="model-node is-state">
          <span class="model-symbol">b<sub>t</sub></span>
          <div><small>Information state</small><code>b<sub>t</sub> ≈ Update(b<sub>t−1</sub>, o<sub>t</sub>)</code><p>shelf · constraints · confidence · misses</p></div>
        </article>
        <i>${icon("arrow")}</i>
        <article class="model-node is-action">
          <span class="model-symbol">a<sub>t</sub></span>
          <div><small>Policy action</small><code>a<sub>t</sub> = (q<sub>t</sub>, k<sub>t</sub>)</code><p>next question + result count</p></div>
        </article>
      </div>
      <div class="problem-thesis">
        <blockquote><span>The first observation</span>“${esc(first.customer)}”</blockquote>
        <div class="three-decisions">
          <article><span>01</span><div><strong>ASK</strong><small>Which missing constraint is worth one more turn?</small></div></article>
          <article><span>02</span><div><strong>RANK</strong><small>Which products are supported by the evidence so far?</small></div></article>
          <article><span>03</span><div><strong>COMMIT</strong><small>How many results are safe to expose now?</small></div></article>
        </div>
      </div>
      <div class="scored-on">
        <span class="micro-label">Policy objective</span>
        <code>J(π) = 0.50 · Hit@10  +  0.30 · MRR  +  0.20 · Efficiency</code>
        <span>Find the exact product, rank it high, and spend fewer turns.</span>
      </div>
    </section>
  `;
}

function headlineScene(bundle) {
  const base = bundle.proof.baseline;
  const ours = bundle.proof.public;
  const rows = [
    ["Found it in the top 10", base.hit_rate_at_10, ours.hit_rate_at_10, 3],
    ["Rank quality (MRR)", base.mrr, ours.mrr, 4],
    ["Turns to find it", base.mttc, ours.mttc, 2],
  ];
  return `
    <section class="scene scene-headline" aria-labelledby="headline-title">
      ${sceneHeader(
        "Empirical result · organizer evaluator",
        '<span id="headline-title">The same task. A better decision policy.</span>',
        "All 200 public sessions, the organizer’s unchanged scoring code, and no network or model tokens on our side."
      )}
      <div class="headline-compare">
        <article class="headline-card is-base">
          <span class="micro-label">Organizer baseline · BM25 retrieval</span>
          <strong>${score(base.technical_score)}</strong>
          <p>Burns ${decimal(base.mttc, 2)} turns on average and finds the product for only 1 shopper in 8.</p>
        </article>
        <div class="headline-arrow">${icon("arrow")}<span>same data<br>same evaluator</span></div>
        <article class="headline-card is-ours">
          <span class="micro-label">Submitted constraint-grounded policy</span>
          <strong>${score(ours.technical_score)}</strong>
          <p>Finds every shopper's product, usually by turn two.</p>
        </article>
      </div>
      <table class="compare-table">
        <thead><tr><th>Measure</th><th>Baseline</th><th>Our policy</th><th>Change</th></tr></thead>
        <tbody>
          ${rows.map(([label, a, b, places]) => `
            <tr>
              <td>${esc(label)}</td>
              <td class="num">${decimal(a, places)}</td>
              <td class="num is-ours">${decimal(b, places)}</td>
              <td class="num delta">${b > a ? "+" : ""}${decimal(b - a, places)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
      <p class="scene-footnote">Baseline figures come from <code>docs/baseline_results.json</code> in the participant kit. Ours come from running the organizers' <code>local_evaluator</code> unmodified.</p>
    </section>
  `;
}

function methodScene() {
  return `
    <section class="scene scene-method" aria-labelledby="method-title">
      ${sceneHeader(
        "Method · one deterministic turn loop",
        '<span id="method-title">One partial observation enters. One grounded action leaves.</span>',
        "Every turn executes the same inspectable loop. No trained RL policy and no hidden model call—just catalog evidence, state, and a metric-aligned decision."
      )}
      <div class="method-flow" aria-label="One turn of the submitted method">
        <article><span>01</span><small>OBSERVE</small><code>o<sub>t</sub></code><p>shopper utterance</p></article>
        <article><span>02</span><small>GROUND</small><code>clues<sub>t</sub></code><p>catalog-checkable evidence</p></article>
        <article><span>03</span><small>UPDATE</small><code>b<sub>t</sub></code><p>constraints · confidence · misses</p></article>
        <article><span>04</span><small>NARROW</small><code>C<sub>t</sub> ⊂ P</code><p>50K → category shelf</p></article>
        <article><span>05</span><small>RANK</small><code>R<sub>t</sub></code><p>evidence first · prior bounded</p></article>
        <article class="is-decision"><span>06</span><small>DECIDE</small><code>(q<sub>t</sub>, k<sub>t</sub>)</code><p><b>ASK</b> safe prefix &nbsp;·&nbsp; <b>COMMIT</b> full Top 10</p></article>
      </div>
      <div class="method-loop-note"><span>${icon("reset")} If ASK: receive o<sub>t+1</sub> and repeat with memory</span><strong>If complete or late: COMMIT the full Top 10</strong></div>
      <div class="method-detail-grid">
        <article class="algorithm-card">
          <div class="algorithm-head"><span>Algorithm 1</span><strong>ASK · RANK · COMMIT</strong><small>executed on every turn</small></div>
          <ol class="pseudo-code">
            <li><code><b>clues</b><sub>t</sub> ← GROUND(o<sub>t</sub>, catalog)</code></li>
            <li><code>b<sub>t</sub> ← UPDATE(b<sub>t−1</sub>, clues<sub>t</sub>, misses)</code></li>
            <li><code>C<sub>t</sub> ← SHELF(b<sub>t</sub>); &nbsp; R<sub>t</sub> ← RANK(C<sub>t</sub> | b<sub>t</sub>)</code></li>
            <li><code>q<sub>t</sub> ← argmax<sub>q</sub> ANSWERABLE_MVOI(q | b<sub>t</sub>)</code></li>
            <li><code>k<sub>t</sub> ← GATE(b<sub>t</sub>, R<sub>t</sub>, horizon) &nbsp; ∈ [1, 10]</code></li>
            <li><code><b>return</b> reply(q<sub>t</sub>), &nbsp; TOP-k<sub>t</sub>(R<sub>t</sub>)</code></li>
          </ol>
        </article>
        <article class="mvoi-card">
          <div class="mvoi-head"><div><span>Core innovation · ASK policy</span><h2>Answerability-aware Metric Value of Information</h2></div><code>ANSWERABLE_MVOI</code></div>
          <p class="mvoi-thesis"><strong>Candidate reduction asks “how much smaller?”</strong><span>We ask: “Can the shopper's answer actually improve the official score enough to justify another turn?”</span></p>
          <div class="mvoi-formula"><span>A-MVOI(q)</span><code>[ .50 E(Hit@10) + .30 E(MRR) ]<sub>answerable</sub> − U<sub>now</sub> − .02</code></div>
          <ol class="mvoi-steps">
            <li><span>1</span><p><strong>Imagine each plausible target</strong><small>Use every competitive product as a counterfactual hidden target.</small></p></li>
            <li><span>2</span><p><strong>Predict visible answers</strong><small>Its catalog signature tells us which answer branch q would reveal.</small></p></li>
            <li class="is-answerability"><span>3</span><p><strong>Keep &lt;none&gt; honest</strong><small>No preference means no new constraint—the current ranking stays.</small></p></li>
            <li><span>4</span><p><strong>Score, then charge .02</strong><small>Expected Hit@10 + MRR gain, minus one more conversation turn.</small></p></li>
          </ol>
          <div class="mvoi-choice"><span>Choose</span><code>q<sub>t</sub>* = argmax<sub>q</sub> A-MVOI(q | b<sub>t</sub>)</code><strong>High discrimination ≠ high value if the shopper cannot answer.</strong></div>
        </article>
      </div>
      <div class="method-support">
        <article><span>RANK · evidence before popularity</span><code>S = S<sub>rare</sub> + .30S<sub>typed</sub> + .70S<sub>clue-combination</sub></code></article>
        <article><span>COMMIT · result-count gate</span><code>k<sub>t</sub> = π<sub>gate</sub>(b<sub>t</sub>) ∈ [1, 10]</code></article>
      </div>
      <p class="scene-footnote">This is the submitted runtime path: <code>src/dialog.py → src/shelf.py → src/rank.py → src/evidence.py → src/policy.py</code>.</p>
    </section>
  `;
}

/* -------------------------------------------------------------- walkthrough */

function funnelRow(turn) {
  const f = turn.funnel;
  const planningStep = turn.action !== "ASK"
    ? { label: "Evidence window", value: count(f.prior_eligible), note: "eligible for tie-break", state: "active" }
    : Number(f.voi_pool) > 0
      ? { label: "Planned against", value: count(f.voi_pool), note: turn.has_evidence ? "top of the list" : "planning cap, nothing to rank yet", state: "active" }
      : { label: "Question mode", value: "OPEN", note: turn.constraints.length ? "open clarification" : "boundary-safe pivot", state: "active" };
  const steps = [
    { label: "Catalog", value: compact(f.catalog), note: "read only", state: "done" },
    { label: "Right shelf", value: count(f.shelf), note: esc(turn.shelf.name), state: "done" },
    { label: "Still standing", value: count(f.ranked), note: f.ranked < f.shelf ? `${f.shelf - f.ranked} ruled out` : "nothing ruled out yet", state: "done" },
    planningStep,
    { label: "Submitted", value: count(f.emitted), note: "ranked IDs this turn", state: "output" },
  ];
  return `
    <div class="funnel" aria-label="How the candidate set shrinks">
      ${steps.map((step, index) => `
        ${index ? `<i>${icon("arrow")}</i>` : ""}
        <div class="funnel-step is-${step.state}">
          <span>${esc(step.label)}</span>
          <strong>${step.value}</strong>
          <small>${step.note}</small>
        </div>
      `).join("")}
    </div>
  `;
}

function scoreBar(candidate, weights, scale) {
  const typed = weights.typed * candidate.typed_satisfaction;
  const signature = weights.signature * candidate.signature_likelihood;
  const lexical = Math.max(0, candidate.core_score - typed - signature);
  const popularity = Math.max(0, candidate.final_score - candidate.core_score);
  const segments = [
    ["lexical", lexical, "Rare-word match"],
    ["typed", typed, "Typed constraint match"],
    ["signature", signature, "Clue combination match"],
    ["popularity", popularity, "Popularity tie-break"],
  ];
  return `
    <div class="score-bar" role="img" aria-label="Score made of ${segments.filter(([, v]) => v > 0).map(([, v, l]) => `${l} ${decimal(v, 2)}`).join(", ")}">
      ${segments.map(([key, value, label]) => value <= 0 ? "" : `
        <span class="seg seg-${key}" style="width:${(value / scale) * 100}%" title="${esc(label)}: ${decimal(value, 4)}"></span>
      `).join("")}
    </div>
  `;
}

function candidateRow(candidate, weights, scale, submittedIds = []) {
  const isSubmitted = submittedIds.includes(candidate.parent_asin);
  return `
    <li class="candidate ${candidate.is_target ? "is-target" : ""} ${candidate.prior_eligible ? "is-eligible" : ""} ${isSubmitted ? "is-submitted" : ""}">
      <span class="candidate-rank">${candidate.rank}</span>
      <span class="candidate-body">
        <span class="candidate-heading">
          <strong title="${esc(candidate.title)}">${esc(candidate.store)} · ${esc(candidate.title)}</strong>
          ${isSubmitted ? "<em>SUBMITTED</em>" : ""}
        </span>
        ${scoreBar(candidate, weights, scale)}
      </span>
      <span class="candidate-score">${decimal(candidate.final_score, 3)}</span>
    </li>
  `;
}

function submittedList(turn) {
  const submittedIds = turn.recommendations || [];
  if (!submittedIds.length) return "";
  const leadId = submittedIds[0];
  const lead = turn.top.find((candidate) => candidate.parent_asin === leadId);
  const leadLabel = lead
    ? `${lead.store ? `${lead.store} · ` : ""}${lead.title}`
    : leadId;
  const remainder = submittedIds.length - 1;
  return `
    <div class="submitted-list" aria-label="Structured recommendations submitted to the evaluator">
      <div class="submitted-list-head">
        <span>Structured output · sent with this reply</span>
        <strong>${count(submittedIds.length)} product${submittedIds.length === 1 ? "" : "s"}</strong>
      </div>
      <div class="submitted-product">
        <span class="submitted-rank">1</span>
        <span class="submitted-product-copy">
          <strong title="${esc(leadLabel)}">${esc(leadLabel)}</strong>
          <code>${esc(leadId)}</code>
        </span>
        ${remainder ? `<span class="submitted-more">+${count(remainder)} more</span>` : ""}
      </div>
      <small>Scored by the evaluator separately from the natural-language message.</small>
    </div>
  `;
}

function provenMissBlock(turn) {
  const misses = turn.excluded_proven_misses || [];
  if (!misses.length) return "";
  return `
    <div class="ruled-out explore-ruled-out">
      <span class="micro-label">Confirmed misses before this turn · ${count(misses.length)}</span>
      <p class="protocol-inference">The evaluator reached turn ${turn.turn}, so these earlier submitted IDs did not match the hidden target.</p>
      ${misses.map((product) => `
        <div class="ruled-item">
          <span>${icon("close")}</span>
          <span>${esc(product.store || "")}${product.store ? " · " : ""}${esc(product.title)}<code>${esc(product.parent_asin)}</code></span>
        </div>
      `).join("")}
    </div>
  `;
}

function scoreLegend() {
  return `
    <div class="score-guide">
      <span class="micro-label">What each score colour means</span>
      <ul class="score-legend">
        <li><i class="seg-lexical"></i><span><strong>Rare-word match</strong><small>Uncommon shopper words found in the catalog text.</small></span></li>
        <li><i class="seg-typed"></i><span><strong>Typed constraint match</strong><small>Material, colour, or price requirements checked as fields.</small></span></li>
        <li><i class="seg-signature"></i><span><strong>Clue combination match</strong><small>Whether the observed clues jointly fit the same product.</small></span></li>
        <li><i class="seg-popularity"></i><span><strong>Popularity tie-break</strong><small>Review volume breaks only near-ties; it cannot override clues.</small></span></li>
      </ul>
    </div>
  `;
}

function conversationSoFar(trace, upTo) {
  return `
    <div class="chat-stream">
      ${trace.turns.slice(0, upTo).map((turn) => `
        <div class="chat-message shopper ${turn.turn === upTo ? "is-key" : ""}">
          <small>Shopper, turn ${turn.turn}</small><p>${esc(turn.customer)}</p>
        </div>
        <div class="chat-message agent">
          <small>${turn.action === "ASK" ? `Us · ASK about ${esc(askedAbout(turn.ask_attribute))}` : "Us · COMMIT the ranked list"}</small><p>${esc(turn.agent_message)}</p>
          ${turn.turn === upTo ? submittedList(turn) : ""}
        </div>
      `).join("")}
    </div>
  `;
}

function walkthroughTrajectory(trace) {
  return `
    <div class="walk-trajectory" aria-label="Observation history">
      ${trace.turns.map((item, index) => `
        <article class="walk-trajectory-step ${index === trace.turns.length - 1 ? "is-current" : ""}">
          <span class="walk-turn">T${item.turn}</span>
          <div><small>${exploreStage(item, index, trace.turns)} · evaluator target ${item.target_rank ? `#${item.target_rank}` : "outside rank"}</small><p>${esc(item.customer)}</p></div>
        </article>
      `).join("")}
    </div>
  `;
}

function walkthroughScene(bundle, index) {
  const trace = bundle.explore_trace || bundle.trace;
  const turn = index === 0 ? trace.turns[0] : trace.turns[trace.turns.length - 1];
  const weights = trace.weights;
  const scale = Math.max(...turn.top.map((c) => c.final_score), 1);
  const q = turn.question_value || {};
  const heading = index === 0
    ? {
        kicker: "Policy step · ASK",
        title: "No evidence, no confident ranking. Acquire information.",
        copy: `At turn one the information state contains only a shelf. All ${count(turn.funnel.shelf)} candidates score zero, so the policy spends one turn on the question with the highest expected value.`,
      }
    : {
        kicker: "Policy step · RANK + COMMIT",
        title: "Four turns turn uncertainty into a rank-one decision.",
        copy: `The shopper declines one question, the policy pivots, and two later observations move the evaluator target from #${trace.turns[0].target_rank} to #${turn.target_rank}. Only then does the output gate commit.`,
      };
  return `
    <section class="scene scene-walk" aria-label="Walkthrough turn ${turn.turn}">
      <div class="walk-head">
        <div>${sceneHeader(heading.kicker, `<span>${esc(heading.title)}</span>`, esc(heading.copy))}</div>
        ${pipelineStrip(index === 0 ? ["read", "narrow", "decide"] : ["read", "remember", "narrow", "rank", "decide"])}
      </div>
      <div class="walk-grid">
        <section class="walk-panel">
          <div class="panel-heading"><span class="panel-index">${icon("question")}</span><div><small>Observation o<sub>t</sub></small><h2>What the shopper revealed</h2></div></div>
          ${index === 0 ? conversationSoFar(trace, turn.turn) : walkthroughTrajectory(trace)}
          <div class="clue-block">
            <span class="micro-label">Information state b<sub>t</sub> · grounded constraints</span>
            ${turn.constraints.length ? turn.constraints.map((clue) => `
              <div class="clue"><span class="chip-check">${icon("check")}</span><span><strong>${esc(clue.text)}</strong><small>matched in the catalog · confidence ${decimal(clue.confidence, 1)}</small></span></div>
            `).join("") : '<p class="clue-empty">None yet. That is the point of this turn.</p>'}
          </div>
          ${turn.excluded_proven_misses.length && index === 0 ? `
            <div class="ruled-out">
              <span class="micro-label">Ruled out by the conversation continuing</span>
              ${turn.excluded_proven_misses.map((p) => `<div class="ruled-item"><span>${icon("close")}</span><span>${esc(p.store || "")} · ${esc(p.title)}<small>we showed it, the shopper kept talking, so it wasn't the one</small></span></div>`).join("")}
            </div>
          ` : turn.excluded_proven_misses.length ? `
            <div class="ruled-summary">${icon("close")}<span><strong>${count(turn.excluded_proven_misses.length)} earlier submitted lists excluded</strong><small>The shopper continued, so those provisional products cannot be the target.</small></span></div>
          ` : ""}
        </section>

        <section class="walk-panel is-wide">
          <div class="panel-heading"><span class="panel-index">${icon("branch")}</span><div><small>Policy component π<sub>rank</sub></small><h2>${index === 0 ? "Everything is tied at zero" : "How the top of the list is built"}</h2></div>${turn.target_rank ? badge(`Their product is #${turn.target_rank}`, turn.target_rank === 1 ? "green" : "quiet", turn.target_rank === 1 ? "check" : "") : ""}</div>
          ${funnelRow(turn)}
          ${index === 0 ? `
            <div class="tied-note">
              ${icon("shield")}
              <div>
                <strong>All ${count(turn.funnel.ranked)} candidates score 0.000000.</strong>
                <p>The list below is just shelf order, not a ranking, and we say so rather than dressing it up. Their product happens to sit at position ${count(turn.target_rank)}. We show a single exploratory product and spend the turn asking instead.</p>
              </div>
            </div>
          ` : `
            <div class="window-note">
              ${icon("shield")}
              <div>
                <strong>${turn.funnel.prior_eligible === 1 ? "Exactly one" : count(turn.funnel.prior_eligible)} of ${count(turn.funnel.ranked)} candidates ${turn.funnel.prior_eligible === 1 ? "was" : "were"} close enough for popularity to matter at all.</strong>
                <p>Popularity is capped at ${weights.popularity} and only applies within ${weights.popularity_window} of the best evidence score. The runner-up is ${decimal(turn.top[1].core_gap_to_best, 4)} behind, so no amount of reviews could have moved it. This ranking is evidence, not sales.</p>
              </div>
            </div>
          `}
          <ol class="candidate-list">
            ${turn.top.slice(0, 6).map((c) => candidateRow(c, weights, scale, turn.recommendations)).join("")}
          </ol>
          ${index === 0 ? "" : scoreLegend()}
        </section>

        <section class="walk-panel">
          <div class="panel-heading"><span class="panel-index">${icon("spark")}</span><div><small>Policy components π<sub>ask</sub> + π<sub>commit</sub></small><h2>Ask, or commit?</h2></div></div>
          ${turn.action === "ASK" ? `
            <div class="decision-metrics">
              <div title="Products we plan against"><span>Planning against</span><strong>${count(q.candidate_count)}</strong></div>
              <div title="Distinct answers the shopper could give"><span>Possible answers</span><strong>${count(q.answer_group_count)}</strong></div>
              <div title="Expected products still in play once they answer"><span>Left after they answer</span><strong>${decimal(q.expected_remaining, 1)}</strong></div>
              <div class="accent" title="Expected score gained by asking, minus the cost of one more turn. Internally: answerability-aware metric value of information (MVOI).">
                <span>Worth asking</span><strong>+${decimal(q.metric_voi, 3)}</strong>
              </div>
            </div>
            <p class="decision-equation">q* = argmax<sub>q</sub> E[ΔScore | b<sub>t</sub>, q] − ${weights.question_turn_cost}</p>
            <p class="decision-explain">We treat every plausible product as if it were the hidden one, predict what the shopper would answer, and score the resulting trajectory. Asking about <strong>${esc(askedAbout(turn.question))}</strong> has the highest value after charging for one more turn.</p>
          ` : `
            <div class="decision-metrics commit-metrics">
              <div><span>Start rank</span><strong>#${count(trace.turns[0].target_rank)}</strong></div>
              <div><span>After broad evidence</span><strong>#${count(trace.turns[trace.turns.length - 2].target_rank)}</strong></div>
              <div><span>Grounded clues</span><strong>${count(turn.constraints.length)}</strong></div>
              <div class="accent"><span>Final rank</span><strong>#${count(turn.target_rank)}</strong></div>
            </div>
            <p class="decision-equation">k<sub>t</sub> = π<sub>gate</sub>(b<sub>t</sub>) = ${count(turn.output_gate.emitted_count)}</p>
            <p class="decision-explain">The information state is complete, the target has a decisive evidence lead, and no useful clarification remains. The policy stops asking and opens the full Top 10.</p>
          `}
          <div class="gate-summary">
            ${icon("shield")}
            <div><span>So how many IDs do we submit?</span><strong>${count(turn.output_gate.emitted_count)} of 10</strong><small>${turn.action === "ASK" ? "the rest stay gated until the clues are complete" : "the evidence is complete; submit the full Top 10"}</small></div>
          </div>
          <div class="reason-block">
            <span class="micro-label">Recorded reasons</span>
            ${(turn.reason_codes || []).map((r) => `<p class="reason-code">${icon("check")}<span>${esc(REASON_TEXT[r] || r.replaceAll("_", " "))}</span></p>`).join("")}
          </div>
        </section>
      </div>
    </section>
  `;
}

function turnConversation(turn) {
  return `
    <div class="chat-stream">
      <div class="chat-message shopper is-key">
        <small>Shopper, turn ${turn.turn}</small><p>${esc(turn.customer)}</p>
      </div>
      <div class="chat-message agent">
        <small>${turn.action === "ASK" ? `Us · ASK about ${esc(askedAbout(turn.ask_attribute))}` : "Us · COMMIT the ranked list"}</small><p>${esc(turn.agent_message)}</p>
        ${submittedList(turn)}
      </div>
    </div>
  `;
}

function turnCandidates(turn) {
  const visible = turn.top.slice(0, 6);
  const target = turn.top.find((candidate) => candidate.is_target);
  if (target && target.rank <= 10 && !visible.some((candidate) => candidate.parent_asin === target.parent_asin)) {
    return [...visible.slice(0, 5), target];
  }
  return visible;
}

function turnWalkthroughScene(bundle, index) {
  const trace = bundle.explore_trace || bundle.trace;
  const turn = trace.turns[index];
  const weights = trace.weights;
  const scale = Math.max(...turn.top.map((candidate) => candidate.final_score), 1);
  const q = turn.question_value || {};
  const headings = [
    {
      kicker: "Turn 1 of 4 · ASK",
      title: "No evidence, no confident ranking. Acquire information.",
      copy: `The opening line identifies a ${count(turn.funnel.shelf)}-product shelf but reveals no distinguishing clue. The policy submits one exploratory ID and asks the highest-value question.`,
      pipeline: ["read", "narrow", "decide"],
    },
    {
      kicker: "Turn 2 of 4 · PIVOT",
      title: "A declined question is not a product constraint.",
      copy: "The shopper cannot answer feature. We preserve an empty constraint state, infer one prior submitted ID is a miss from the evaluator continuing, and pivot to an open question.",
      pipeline: ["read", "remember", "decide"],
    },
    {
      kicker: "Turn 3 of 4 · UPDATE + ASK",
      title: "Broad evidence helps, but it is not enough to commit.",
      copy: `Two leather clues enter the information state and move the evaluator-only target diagnostic from #${trace.turns[1].target_rank} to #${turn.target_rank}. The list is better, but still ambiguous.`,
      pipeline: ["read", "remember", "rank", "decide"],
    },
    {
      kicker: "Turn 4 of 4 · RANK + COMMIT",
      title: "Specific structure resolves the uncertainty.",
      copy: `Two more clues create a decisive evidence gap, move the evaluator-only target to #${turn.target_rank}, and open the output gate from one submitted ID to the full top ten.`,
      pipeline: ["read", "remember", "narrow", "rank", "decide"],
    },
  ];
  const heading = headings[index];
  const rankNote = !turn.has_evidence
    ? `
      <div class="tied-note">
        ${icon("shield")}
        <div>
          <strong>All ${count(turn.funnel.ranked)} remaining candidates score 0.000000.</strong>
          <p>${turn.turn === 1
            ? `This is shelf order, not a confident ranking. The target's #${count(turn.target_rank)} position is evaluator-only. We submit only the first exploratory ID while asking.`
            : `One earlier ID is now a confirmed miss, but the shopper's refusal added no positive evidence. The target moves mechanically to #${count(turn.target_rank)}; the policy still does not know which product it is.`}</p>
        </div>
      </div>
    `
    : turn.action === "ASK"
      ? `
        <div class="window-note">
          ${icon("shield")}
          <div>
            <strong>Useful evidence, but ${count(turn.funnel.prior_eligible)} candidates remain inside the popularity window.</strong>
            <p>The two leather clues lift the target diagnostic to #${count(turn.target_rank)}, but many loafers share that material. Only rank one is submitted, and the policy keeps asking rather than treating an internal top-ten position as a hit.</p>
          </div>
        </div>
      `
      : `
        <div class="window-note">
          ${icon("shield")}
          <div>
            <strong>Exactly one of ${count(turn.funnel.ranked)} candidates is close enough for popularity to matter.</strong>
            <p>Popularity is capped at ${weights.popularity} and applies only within ${weights.popularity_window} of the best core score. The runner-up is ${decimal(turn.top[1].core_gap_to_best, 4)} behind, so evidence—not review volume—determines rank one.</p>
          </div>
        </div>
      `;
  return `
    <section class="scene scene-walk" aria-label="Walkthrough turn ${turn.turn}">
      <div class="walk-head">
        <div>${sceneHeader(heading.kicker, `<span>${esc(heading.title)}</span>`, esc(heading.copy))}</div>
        ${pipelineStrip(heading.pipeline)}
      </div>
      <div class="walk-grid">
        <section class="walk-panel">
          <div class="panel-heading"><span class="panel-index">${icon("question")}</span><div><small>Observation o<sub>t</sub></small><h2>What happened this turn</h2></div></div>
          ${turnConversation(turn)}
          <div class="clue-block">
            <span class="micro-label">Information state b<sub>t</sub> · grounded constraints</span>
            ${turn.constraints.length ? turn.constraints.map((clue) => `
              <div class="clue"><span class="chip-check">${icon("check")}</span><span><strong>${esc(clue.text)}</strong><small>matched in the catalog · confidence ${decimal(clue.confidence, 1)}</small></span></div>
            `).join("") : `<p class="clue-empty">None yet. ${turn.turn === 2 ? "The refusal is remembered as an exhausted question, not invented as a preference." : "That is the point of this turn."}</p>`}
          </div>
          ${turn.excluded_proven_misses.length === 1 ? provenMissBlock(turn) : turn.excluded_proven_misses.length > 1 ? `
            <div class="ruled-summary">${icon("close")}<span><strong>${count(turn.excluded_proven_misses.length)} earlier submitted IDs excluded</strong><small>The evaluator reached turn ${turn.turn}, so those products cannot be the hidden target.</small></span></div>
          ` : ""}
        </section>

        <section class="walk-panel is-wide">
          <div class="panel-heading"><span class="panel-index">${icon("branch")}</span><div><small>Policy component π<sub>rank</sub></small><h2>${turn.has_evidence ? "How the ranking changed" : "Why there is no confident ranking yet"}</h2></div>${turn.target_rank ? badge(`Evaluator target #${turn.target_rank}`, turn.target_rank === 1 ? "green" : "quiet", turn.target_rank === 1 ? "check" : "") : ""}</div>
          ${funnelRow(turn)}
          ${rankNote}
          <ol class="candidate-list">
            ${turnCandidates(turn).map((candidate) => candidateRow(candidate, weights, scale, turn.recommendations)).join("")}
          </ol>
          ${turn.has_evidence ? scoreLegend() : ""}
        </section>

        <section class="walk-panel">
          <div class="panel-heading"><span class="panel-index">${icon("spark")}</span><div><small>Policy components π<sub>ask</sub> + π<sub>commit</sub></small><h2>Ask, or commit?</h2></div></div>
          ${q.candidate_count ? `
            <div class="decision-metrics">
              <div title="Products we plan against"><span>Planning against</span><strong>${count(q.candidate_count)}</strong></div>
              <div title="Distinct answers the shopper could give"><span>Possible answers</span><strong>${count(q.answer_group_count)}</strong></div>
              <div title="Expected products still in play once they answer"><span>Left after answer</span><strong>${decimal(q.expected_remaining, 1)}</strong></div>
              <div class="accent"><span>Worth asking</span><strong>+${decimal(q.metric_voi, 3)}</strong></div>
            </div>
            <p class="decision-equation">q* = argmax<sub>q</sub> E[ΔScore | b<sub>t</sub>, q] − ${weights.question_turn_cost}</p>
            <p class="decision-explain">The planner predicts possible answers for each plausible product. Asking about <strong>${esc(askedAbout(turn.question))}</strong> has the highest expected payoff after charging for one more turn.</p>
          ` : turn.action === "ASK" ? `
            <div class="decision-metrics commit-metrics">
              <div><span>Grounded clues</span><strong>${count(turn.constraints.length)}</strong></div>
              <div><span>Confirmed misses</span><strong>${count(turn.excluded_proven_misses.length)}</strong></div>
              <div><span>Still standing</span><strong>${count(turn.funnel.ranked)}</strong></div>
              <div class="accent"><span>Evaluator-only rank</span><strong>#${count(turn.target_rank)}</strong></div>
            </div>
            <p class="decision-equation">q<sub>${turn.turn}</sub> = other &nbsp; · &nbsp; k<sub>${turn.turn}</sub> = 1</p>
            <p class="decision-explain">${turn.turn === 2
              ? "The shopper declined feature, so the policy marks that attribute exhausted and switches to a boundary-safe open question. No unsupported constraint is added."
              : "Leather improves the ranking but does not isolate one product. The policy keeps clarification open and submits only the safest rank-one ID."}</p>
          ` : `
            <div class="decision-metrics commit-metrics">
              <div><span>Start rank</span><strong>#${count(trace.turns[0].target_rank)}</strong></div>
              <div><span>After broad evidence</span><strong>#${count(trace.turns[2].target_rank)}</strong></div>
              <div><span>Grounded clues</span><strong>${count(turn.constraints.length)}</strong></div>
              <div class="accent"><span>Final rank</span><strong>#${count(turn.target_rank)}</strong></div>
            </div>
            <p class="decision-equation">k<sub>t</sub> = π<sub>gate</sub>(b<sub>t</sub>) = ${count(turn.output_gate.emitted_count)}</p>
            <p class="decision-explain">The information state is complete and rank one has a decisive evidence lead. The policy stops asking and submits the full Top 10.</p>
          `}
          <div class="gate-summary">
            ${icon("shield")}
            <div><span>So how many IDs do we submit?</span><strong>${count(turn.output_gate.emitted_count)} of 10</strong><small>${turn.action === "ASK" ? "the rest stay gated until the clues are complete" : "the evidence is complete; submit the full Top 10"}</small></div>
          </div>
          <div class="reason-block">
            <span class="micro-label">Recorded reasons</span>
            ${(turn.reason_codes || []).map((reason) => `<p class="reason-code">${icon("check")}<span>${esc(REASON_TEXT[reason] || reason.replaceAll("_", " "))}</span></p>`).join("")}
          </div>
        </section>
      </div>
    </section>
  `;
}

/* ---------------------------------------------------------------- evidence */

function metricCard(label, data, tone, note) {
  return `
    <article class="metric-card metric-${tone}">
      <span>${esc(label)}</span>
      <strong>${score(data.technical_score)}</strong>
      <p>${esc(note)}</p>
      <dl><div><dt>Found in top 10</dt><dd>${decimal(data.hit_rate_at_10, 3)}</dd></div><div><dt>Rank quality</dt><dd>${decimal(data.mrr, 3)}</dd></div><div><dt>Turns to find</dt><dd>${decimal(data.mttc, 2)}</dd></div></dl>
    </article>
  `;
}

function proofContent(bundle, compactView = false) {
  const proof = bundle.proof;
  const para = proof.paraphrase;
  const tests = proof.tests;
  return `
    <div class="proof-grid">
      ${metricCard("The organizers' public set", proof.public, "public", `All ${count(proof.public.sample_count)} sessions, scored by their evaluator, unchanged.`)}
      ${metricCard("Products we'd never seen", proof.matched, "matched", `${count(proof.matched.sample_count)} targets we held out ourselves, at everyday popularity.`)}
      ${metricCard("The rare, hard-to-find tail", proof.uniform, "uniform", `${count(proof.uniform.sample_count)} targets we held out ourselves, mostly barely-reviewed products.`)}
    </div>
    <div class="paraphrase-card">
      <div class="paraphrase-head"><div><span class="micro-label">Stress test we wrote ourselves · when the shopper rephrases · same ${esc(para.sample_count)} targets</span><h2>Twelve shoppers we would otherwise lose.</h2></div>${badge("Our own test, not the organizers'", "warning", "shield")}</div>
      <div class="ab-columns">
        <div class="ab-system old"><span>Without the alternative phrasings</span><strong>${Math.round(para.old.hit_rate_at_10 * para.sample_count)} / ${esc(para.sample_count)}</strong><small>shoppers found</small><dl><div><dt>Turns per shopper</dt><dd>${decimal(para.impact.average_agent_calls_old, 2)}</dd></div><div><dt>Score</dt><dd>${score(para.old.technical_score)}</dd></div></dl></div>
        <div class="ab-change"><span>+12</span>${icon("arrow")}<small>more shoppers<br>per hundred</small></div>
        <div class="ab-system new"><span>Constraint-grounded policy</span><strong>${Math.round(para.submitted.hit_rate_at_10 * para.sample_count)} / ${esc(para.sample_count)}</strong><small>shoppers found</small><dl><div><dt>Turns per shopper</dt><dd>${decimal(para.impact.average_agent_calls_new, 2)}</dd></div><div><dt>Score</dt><dd>${score(para.submitted.technical_score)}</dd></div></dl></div>
      </div>
      <p class="honest-note">${icon("shield")} Being straight about this one: we wrote both the rephrasings and the code that catches them, so treat it as a self-designed robustness probe rather than an independent result. On the organizers' own wording the two columns are identical at ${score(para.submitted.technical_score)}.</p>
    </div>
    <div class="engineering-row ${compactView ? "is-compact" : ""}">
      <div><strong>0</strong><span>calls to any API</span></div>
      <div><strong>0</strong><span>AI tokens, so $0 to run</span></div>
      <div><strong>${decimal(proof.runtime.mean_evaluation_wall_ms_per_response, 1)}ms</strong><span>typical reply, on a laptop</span></div>
      <div><strong>${esc(tests.count)}</strong><span>${tests.verified_pass ? "tests in verified suite" : "tests in the suite"}</span></div>
      <div><strong>${proof.robustness.catalog_mutation.unchanged ? "Untouched" : "Changed"}</strong><span>catalog, byte for byte</span></div>
    </div>
    <p class="proof-disclaimer">The two held-out sets are ours, built from catalog products the public sessions never use. They are diagnostics, not a claim about the organizers' private set, and nothing here is measured sales.</p>
  `;
}

function proofScene(bundle) {
  return `
    <section class="scene scene-proof" aria-labelledby="proof-title">
      ${sceneHeader(
        "Evaluation suite · beyond one trajectory",
        '<span id="proof-title">One successful trace is a demonstration. This is the evidence.</span>'
      )}
      ${proofContent(bundle, true)}
    </section>
  `;
}

/* ------------------------------------------------------------- playground */

const SCENARIO_LABEL = {
  buying: "Buying",
  browsing: "Browsing",
  intent_override: "Intent Override",
  boundary: "Boundary",
};

function sessionModeTabs(mode, health) {
  return `
    <div class="sx-tabs" aria-label="Session Explorer mode">
      <button class="${mode === "replay" ? "is-active" : ""}" data-action="session-mode-replay">
        ${icon("shield")}Verified sessions
      </button>
      <button class="${mode === "live" ? "is-active" : ""}" data-action="session-mode-live">
        ${icon("terminal")}Live Agent
        <span>${health.live_available ? "ready" : "optional"}</span>
      </button>
    </div>
  `;
}

function replayProduct(replays, asin) {
  return replays.products?.[asin] || {
    parent_asin: asin,
    title: "Catalog product",
    store: "",
    price: null,
  };
}

function replaySessionOption(session) {
  const outcome = session.outcome;
  return `${session.sample_id} · ${SCENARIO_LABEL[session.scenario_type] || session.scenario_type} · Rank ${outcome.best_rank ?? "miss"} · T${outcome.first_hit_turn ?? "—"}`;
}

function replayTurnStatus(turn) {
  if (!turn.eligible_for_hit) return badge("Waiting for override", "warning", "shield");
  if (turn.target_rank !== null && turn.target_rank !== undefined) {
    return badge(`HIT @ #${turn.target_rank}`, "green", "check");
  }
  return badge("Target not submitted", "quiet");
}

function replayConversation(session, selectedTurn) {
  return session.turns.slice(0, selectedTurn).map((turn) => `
    <div class="sx-turn-pair ${turn.turn === selectedTurn ? "is-current" : ""}">
      <div class="live-message"><small>Shopper · turn ${turn.turn}</small><p>${esc(turn.customer)}</p></div>
      <div class="live-message is-agent">
        <small>ARC · turn ${turn.turn}</small>
        <p>${esc(turn.agent_message)}</p>
        <code>ask_attribute: ${esc(String(turn.ask_attribute))} · ${turn.recommendations.length} submitted · ${turn.reported_tokens} tokens</code>
      </div>
    </div>
  `).join("");
}

function replayRecommendations(replays, session, turn) {
  if (!turn.recommendations.length) {
    return `<p class="live-placeholder">No product IDs were submitted on this turn.</p>`;
  }
  return `
    <ol class="pg-list sx-results-list">
      ${turn.recommendations.map((asin, index) => {
        const product = replayProduct(replays, asin);
        const isTarget = asin === session.target_parent_asin;
        return `
          <li class="${isTarget ? "is-target" : ""}">
            <span class="pg-rank">${index + 1}</span>
            <span>
              <span class="pg-title">${esc(product.title)}</span>
              <span class="pg-asin">${esc(asin)}${product.store ? ` · ${esc(product.store)}` : ""}${product.price !== null && product.price !== undefined ? ` · $${decimal(product.price)}` : ""}</span>
            </span>
            ${isTarget ? `<strong class="sx-target-flag">TARGET</strong>` : ""}
          </li>
        `;
      }).join("")}
    </ol>
  `;
}

function replayDecision(turn) {
  const decision = turn.decision || {};
  const questionValue = decision.selected_question_value || {};
  return `
    <div class="sx-decision">
      <div class="pg-chips">
        ${badge(decision.action || "—", decision.action === "RECOMMEND" ? "green" : "neutral")}
        ${badge(`${count(decision.ranked_candidate_count || 0)} ranked`, "quiet")}
        ${badge(`${count(decision.proven_miss_count || 0)} ruled out`, "quiet")}
      </div>
      <dl>
        <div><dt>Grounded clues</dt><dd>${esc((decision.constraints || []).join(" · ") || "none yet")}</dd></div>
        <div><dt>Top-2 margin</dt><dd>${decision.top2_absolute_margin === null || decision.top2_absolute_margin === undefined ? "—" : decimal(decision.top2_absolute_margin, 4)}</dd></div>
        <div><dt>Question value</dt><dd>${questionValue.answerable_metric_voi === null || questionValue.answerable_metric_voi === undefined ? "—" : decimal(questionValue.answerable_metric_voi, 4)}</dd></div>
        <div><dt>Answerability</dt><dd>${questionValue.answerability === null || questionValue.answerability === undefined ? "—" : percent(questionValue.answerability)}</dd></div>
      </dl>
      <p class="pg-note">${esc((decision.reason_codes || []).map((code) => REASON_TEXT[code] || code).join(" "))}</p>
    </div>
  `;
}

function sessionReplayScene(bundle, health, state) {
  const replays = bundle.session_replays || { sessions: [], products: {}, summary: {} };
  const explorer = state.sessionExplorer || {};
  const filtered = explorer.scenario === "all"
    ? replays.sessions
    : replays.sessions.filter((session) => session.scenario_type === explorer.scenario);
  const session = filtered.find((row) => row.sample_id === explorer.sessionId)
    || replays.sessions.find((row) => row.sample_id === explorer.sessionId)
    || filtered[0]
    || replays.sessions[0];
  if (!session) {
    return `
      <section class="scene scene-sessions">
        ${sceneHeader(
          "Session Explorer · bundle update required",
          "The loaded replay bundle predates the all-session explorer.",
          "The frontend is ready, but the server is still serving an older in-memory demo_bundle.json. Rebuild the bundle, then restart an already-running server once."
        )}
        ${sessionModeTabs("replay", health)}
        <div class="pg-offline">
          <p class="micro-label">Expected field missing · session_replays.sessions</p>
          <code>python3 tools/build_session_replays.py<br>python3 tools/build_demo_bundle.py --catalog data/catalog.jsonl --tests-verified</code>
          <p class="pg-note">Then stop the old process with Ctrl+C and run <code>python3 demo/server.py</code> again.</p>
        </div>
      </section>
    `;
  }
  const selectedTurn = Math.max(1, Math.min(session.turns.length, Number(explorer.turn || 1)));
  const turn = session.turns[selectedTurn - 1];
  const target = replayProduct(replays, session.target_parent_asin);
  const outcome = session.outcome;
  const rankDistribution = replays.summary?.rank_distribution || {};
  return `
    <section class="scene scene-sessions" aria-labelledby="sessions-title">
      ${sceneHeader(
        `Session Explorer · ${count(replays.sample_count || replays.sessions.length)} public evaluator runs`,
        '<span id="sessions-title">Pick any session. Replay every turn. Verify the target rank.</span>',
        "These are compact traces from the unchanged evaluator, not a browser simulation. Ground truth is joined only after each Agent response and never enters ARC."
      )}
      ${sessionModeTabs("replay", health)}
      <div class="sx-toolbar">
        <label><span>Scenario</span><select id="session-filter">
          <option value="all" ${explorer.scenario === "all" ? "selected" : ""}>All 200 sessions</option>
          ${Object.entries(SCENARIO_LABEL).map(([value, label]) => `<option value="${value}" ${explorer.scenario === value ? "selected" : ""}>${esc(label)}</option>`).join("")}
        </select></label>
        <label class="sx-session-select"><span>Session</span><select id="session-select">
          ${filtered.map((row) => `<option value="${esc(row.sample_id)}" ${row.sample_id === session.sample_id ? "selected" : ""}>${esc(replaySessionOption(row))}</option>`).join("")}
        </select></label>
        <button class="secondary-button" data-action="session-random">${icon("reset")}Random session</button>
        <div class="sx-corpus-proof"><strong>${count(rankDistribution["1"] || 0)}</strong><span>Rank 1</span><strong>${count(rankDistribution["3"] || 0)}</strong><span>Rank 3</span></div>
      </div>
      <div class="sx-case-head">
        <div><span class="micro-label">Evaluator-only ground truth · never sent to Agent</span><h2>${esc(target.title)}</h2><code>${esc(session.target_parent_asin)}</code></div>
        <div class="sx-case-badges">
          ${badge(SCENARIO_LABEL[session.scenario_type] || session.scenario_type, "neutral")}
          ${badge(session.difficulty_bucket || "unlabelled", "quiet")}
          ${badge(`HIT @ #${outcome.best_rank} · T${outcome.first_hit_turn}`, "green", "check")}
          ${badge(`session score ${decimal(outcome.technical_score_contribution, 3)}`, "cyan")}
        </div>
      </div>
      <div class="sx-turn-strip">
        ${session.turns.map((item) => `<button class="${item.turn === selectedTurn ? "is-active" : ""} ${item.turn < selectedTurn ? "is-past" : ""}" data-action="session-set-turn" data-turn="${item.turn}"><span>T${item.turn}</span>${item.hit_this_turn ? `HIT #${item.target_rank}` : item.eligible_for_hit ? "searching" : "override pending"}</button>`).join("")}
      </div>
      <div class="sx-layout">
        <div class="sx-panel sx-conversation">
          <div class="section-label"><span>${icon("terminal")}</span><div><small>Messages actually seen by ARC</small><h2>Conversation through turn ${selectedTurn}</h2></div></div>
          <div class="sx-conversation-scroll">${replayConversation(session, selectedTurn)}</div>
          <div class="turn-selector"><button data-action="session-prev-turn" ${selectedTurn <= 1 ? "disabled" : ""}>${icon("back")}Previous</button><span>Turn <strong>${selectedTurn}</strong> of ${session.turns.length}</span><button data-action="session-next-turn" ${selectedTurn >= session.turns.length ? "disabled" : ""}>Next${icon("arrow")}</button></div>
        </div>
        <div class="sx-panel sx-results">
          <div class="section-label"><span>${icon("shield")}</span><div><small>Post-response evaluator overlay</small><h2>Submitted list and correctness</h2></div>${replayTurnStatus(turn)}</div>
          <div class="sx-verdict ${turn.hit_this_turn ? "is-hit" : ""}"><span>${turn.eligible_for_hit ? "Scored on this turn" : `Not scorable until override on T${session.override_turn}`}</span><strong>${turn.target_rank ? `Target ranked #${turn.target_rank}` : "Target not in submitted list"}</strong></div>
          ${replayRecommendations(replays, session, turn)}
          ${replayDecision(turn)}
          <p class="sx-truth-note">${icon("lock")} Target rank is computed after <code>Agent.respond</code>. ARC receives only profile, message, turn, and top_k.</p>
        </div>
      </div>
    </section>
  `;
}

// Openers come from the recorded trace, so a preset is always a message the
// real catalog can answer. Nothing here is scripted: the reply is produced by
// the submitted Agent at request time.
const playgroundPresets = (bundle) =>
  (bundle.explore_trace?.turns || [])
    .map((turn) => turn.customer)
    .filter(Boolean);

function playgroundOffline(health) {
  const reason = health.catalog_present === false
    ? "The 50,000-product catalog is not in this checkout: the data terms keep it out of the repository."
    : "The server was started without the live engine.";
  return `
    <div class="pg-offline">
      <p class="micro-label">Live engine unavailable · verified sessions ready</p>
      <p>${esc(reason)} Only this Live Agent tab needs the engine; switch back to Verified sessions to explore all 200 runs without a catalog or network.</p>
      <code>python3 demo/server.py --catalog data/catalog.jsonl --live --prewarm</code>
      <p class="pg-note">Download the catalog first, as described in the README setup section, then reload this page.</p>
    </div>
  `;
}

function playgroundLog(playground) {
  if (!playground.log.length) {
    return `<p class="live-placeholder">Send a message, or start from one of the recorded openers below.</p>`;
  }
  return playground.log.map((entry) => entry.role === "shopper"
    ? `<div class="live-message"><small>Shopper · turn ${entry.turn}</small><p>${esc(entry.text)}</p></div>`
    : `<div class="live-message is-agent"><small>Agent · turn ${entry.turn}</small><p>${esc(entry.message)}</p><code>ask_attribute: ${esc(String(entry.ask))} · ${entry.shown} shown · ${entry.tokens} tokens</code></div>`
  ).join("");
}

function playgroundResults(playground) {
  const last = playground.last;
  if (!last) {
    return `<p class="live-placeholder">The submitted list and its evidence appear here.</p>`;
  }
  const recommendations = last.response.recommendations || [];
  const usage = last.response.usage || {};
  const facts = last.facts || {};
  return `
    <div class="pg-chips">
      ${badge(last.response.ask_attribute ? `ASK · ${last.response.ask_attribute}` : "COMMIT", last.response.ask_attribute ? "neutral" : "green", last.response.ask_attribute ? "question" : "check")}
      ${badge(`${recommendations.length} submitted`, "quiet")}
      ${badge(`${Number(usage.prompt_tokens || 0) + Number(usage.completion_tokens || 0)} tokens`, "quiet", "lock")}
    </div>
    ${recommendations.length ? `
      <ol class="pg-list">
        ${recommendations.map((item, index) => {
          const fact = facts[item.parent_asin] || {};
          return `
            <li>
              <span class="pg-rank">${index + 1}</span>
              <span>
                <span class="pg-title">${esc(fact.title || "product")}</span>
                <span class="pg-asin">${esc(item.parent_asin)}${fact.price ? ` · $${decimal(fact.price)}` : ""}${fact.rating ? ` · ${decimal(fact.rating, 1)}★ (${count(fact.rating_count || 0)})` : ""}</span>
              </span>
            </li>
          `;
        }).join("")}
      </ol>
      <p class="pg-note">Titles are the normalised catalog text the Agent indexes. The response itself carries only ordered <code>parent_asin</code> values, exactly as the evaluator receives them.</p>
    ` : `<p class="live-placeholder">This turn submitted no products: the Agent asked first.</p>`}
    <button class="secondary-button" data-action="pg-explain" ${playground.busy ? "disabled" : ""}>${icon("branch")}Explain this decision</button>
    ${playground.certificate ? playgroundCertificate(playground.certificate) : ""}
  `;
}

function playgroundCertificate(certificate) {
  const counterfactual = certificate.minimal_counterfactual_explanation || {};
  const removed = (counterfactual.removed_constraints || [])
    .map((item) => item.constraint)
    .join(", ");
  return `
    <div class="pg-cert">
      <p class="micro-label">Decision certificate · ${esc(certificate.action || "")}</p>
      <ul>
        ${(certificate.reason_codes || []).map((code) => `<li>${esc(REASON_TEXT[code] || code)}</li>`).join("")}
      </ul>
      <dl>
        <div><dt>Clues held</dt><dd>${esc((certificate.constraints || []).join(" · ") || "none yet")}</dd></div>
        <div><dt>Shelf pool</dt><dd>${count(certificate.source_candidate_count || 0)} → ${count(certificate.ranked_candidate_count || 0)} ranked</dd></div>
        <div><dt>Ruled out</dt><dd>${count(certificate.proven_miss_count || 0)} previously submitted</dd></div>
        <div><dt>Top-2 margin</dt><dd>${certificate.top2_absolute_margin === null || certificate.top2_absolute_margin === undefined ? "—" : decimal(certificate.top2_absolute_margin, 4)}</dd></div>
      </dl>
      <p class="pg-note">${counterfactual.faithful
        ? `Smallest evidence removal that changes rank one: <strong>${esc(removed)}</strong>. Re-ranked without it, the current winner falls to #${esc(String(counterfactual.original_top_counterfactual_rank ?? "—"))}.`
        : `Minimal counterfactual: ${esc(counterfactual.status || "not available for this turn")}.`}</p>
    </div>
  `;
}

function livePlaygroundScene(bundle, health, state) {
  const playground = state.playground || {};
  const live = Boolean(health.live_available);
  const presets = playgroundPresets(bundle);
  const exhausted = playground.turn >= 10;
  return `
    <section class="scene scene-playground" aria-labelledby="playground-title">
      ${sceneHeader(
        "Live Agent · optional local backend",
        '<span id="playground-title">Ask it something we did not rehearse.</span>',
        "Every reply below is produced by the same <code>Agent.respond</code> the evaluator calls, over the same read-only catalog, with no network and no model tokens."
      )}
      ${sessionModeTabs("live", health)}
      <div class="explore-layout">
        <div class="explore-conversation">
          <div class="section-label">
            <span>${icon("terminal")}</span>
            <div><small>Live session</small><h2>Shopper</h2></div>
            ${live ? badge("Engine ready", "green", "check") : badge("Replay only", "warning", "shield")}
          </div>
          <div class="live-log">${playgroundLog(playground)}</div>
          ${live ? `
            <form id="live-form" autocomplete="off">
              <label for="live-input">Shopper message</label>
              <input id="live-input" name="message" type="text" maxlength="4000" placeholder="I'm looking for Loafers &amp; Slip-Ons. A key requirement is: leather." ${exhausted || playground.busy ? "disabled" : ""}>
              <button class="primary-button" type="submit" ${exhausted || playground.busy ? "disabled" : ""}>${icon(playground.busy ? "reset" : "arrow")}${playground.busy ? "Thinking" : "Send"}</button>
            </form>
            <div class="pg-presets">
              ${presets.map((message, index) => `<button class="pg-preset" data-preset="${index}" title="${esc(message)}">Recorded turn ${index + 1}</button>`).join("")}
            </div>
          ` : playgroundOffline(health)}
          ${playground.error ? `<p class="live-error">${esc(playground.error)}</p>` : ""}
          <div class="turn-selector">
            <span>Turn <strong>${playground.turn}</strong> of 10${exhausted ? " · horizon reached" : ""}</span>
            <button data-action="pg-reset" ${playground.busy ? "disabled" : ""}>${icon("reset")}New session</button>
          </div>
        </div>
        <div class="explore-results">
          <div class="section-label">
            <span>${icon("spark")}</span>
            <div><small>Structured output · sent with this reply</small><h2>Agent</h2></div>
          </div>
          ${playgroundResults(playground)}
        </div>
      </div>
    </section>
  `;
}

function sessionExplorerScene(bundle, health, state) {
  return state.sessionExplorer?.mode === "live"
    ? livePlaygroundScene(bundle, health, state)
    : sessionReplayScene(bundle, health, state);
}

/* ------------------------------------------------------------------- views */

function storyView(bundle, health, state) {
  const scenes = {
    problem: () => problemScene(bundle),
    method: () => methodScene(),
    headline: () => headlineScene(bundle),
    "walk-1": () => turnWalkthroughScene(bundle, 0),
    "walk-2": () => turnWalkthroughScene(bundle, 1),
    "walk-3": () => turnWalkthroughScene(bundle, 2),
    "walk-4": () => turnWalkthroughScene(bundle, 3),
    proof: () => proofScene(bundle),
    sessions: () => sessionExplorerScene(bundle, health, state),
  };
  const id = SCENES[state.scene].id;
  return `<main class="main-stage" id="main-content" data-scene="${esc(id)}">${scenes[id]()}</main>`;
}

function replayControls(state) {
  return `
    <footer class="replay-controls">
      <div class="scene-progress" aria-label="Chapters">
        ${SCENES.map((scene, index) => `<button class="progress-dot ${index === state.scene ? "is-active" : ""} ${index < state.scene ? "is-past" : ""}" data-scene="${index}" aria-label="Go to ${esc(scene.label)}" aria-current="${index === state.scene ? "step" : "false"}"><span>${esc(scene.label)}</span></button>`).join("")}
      </div>
      <div class="transport">
        <button data-action="restart" aria-label="Start over">${icon("reset")}</button>
        <button data-action="back" aria-label="Back" ${state.scene === 0 ? "disabled" : ""}>${icon("back")}</button>
        <button class="play-button" data-action="play" aria-label="${state.playing ? "Pause" : "Play"}">${icon(state.playing ? "pause" : "play")}</button>
        <button data-action="next" aria-label="Next" ${state.scene === SCENES.length - 1 ? "disabled" : ""}>${icon("arrow")}</button>
        <button data-action="fullscreen" aria-label="Fullscreen">${icon("expand")}</button>
      </div>
      <div class="scene-counter"><strong>${state.scene + 1}</strong> of ${SCENES.length}</div>
    </footer>
  `;
}

export function renderApp(bundle, health, state) {
  return `
    ${header(bundle, health, state)}
    ${storyView(bundle, health, state)}
    ${replayControls(state)}
  `;
}

export function renderFatal(error) {
  return `
    <main class="fatal-screen" id="main-content">
      <div class="brand-mark brand-mark-large">π</div>
      <p class="eyebrow">Nothing to show yet</p>
      <h1>We couldn't load the recorded run.</h1>
      <p>${esc(error?.message || error)}</p>
      <code>python3 demo/server.py</code>
    </main>
  `;
}
