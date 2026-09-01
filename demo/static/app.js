import { renderApp, renderFatal } from "./components.js";
import { createStore, emptyPlayground, initialState, SCENES } from "./state.js";

const root = document.querySelector("#app");
const store = createStore(initialState);

// A static export (tools/build_static_site.py) marks the document so the replay
// reads plain files instead of the local server API.  The live engine is never
// part of a static export, so its endpoints stay server-only.
const STATIC_DEPLOY = document.documentElement.dataset.deploy === "static";
const ENDPOINTS = STATIC_DEPLOY
  ? { demo: "./data/demo_bundle.json", health: "./data/health.json" }
  : { demo: "/api/demo", health: "/api/health" };

let bundle;
let health;
let autoplayTimer = null;
// Set after a playground round trip so the re-rendered input keeps the caret.
let refocusLiveInput = false;

async function getJSON(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(value.error || `${response.status} ${response.statusText}`);
  }
  return value;
}

const postJSON = (path, payload) => getJSON(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});

function updateLocation(state) {
  const url = new URL(window.location.href);
  // The presentation is the only view, so an older ``?mode=`` link degrades to
  // it instead of pointing at a section that no longer exists.
  url.searchParams.delete("mode");
  url.searchParams.set("scene", state.scene);
  if (SCENES[state.scene]?.id === "sessions") {
    const explorer = state.sessionExplorer;
    url.searchParams.set("session", explorer.sessionId);
    url.searchParams.set("turn", String(explorer.turn));
    url.searchParams.set("session_mode", explorer.mode);
    if (explorer.scenario === "all") url.searchParams.delete("scenario");
    else url.searchParams.set("scenario", explorer.scenario);
  } else {
    for (const key of ["session", "turn", "session_mode", "scenario"]) {
      url.searchParams.delete(key);
    }
  }
  if (state.playing) url.searchParams.set("autoplay", "1");
  else url.searchParams.delete("autoplay");
  history.replaceState(null, "", url);
}

function syncAutoplay(state) {
  window.clearTimeout(autoplayTimer);
  autoplayTimer = null;
  if (!state.playing) return;
  if (state.scene >= SCENES.length - 1) {
    store.update({ playing: false });
    return;
  }
  autoplayTimer = window.setTimeout(() => {
    const current = store.getState();
    if (current.playing) {
      const next = Math.min(SCENES.length - 1, current.scene + 1);
      store.update({ scene: next });
    }
  }, SCENES[state.scene].duration);
}

function render(state) {
  if (!bundle || !health) return;
  root.innerHTML = renderApp(bundle, health, state);
  document.body.dataset.scene = SCENES[state.scene].id;
  const log = root.querySelector(".live-log");
  if (log) log.scrollTop = log.scrollHeight;
  if (refocusLiveInput) {
    refocusLiveInput = false;
    root.querySelector("#live-input")?.focus();
  }
  updateLocation(state);
  syncAutoplay(state);
}

store.subscribe(render);

/* --------------------------------------------------------------- playground */

function updateSessionExplorer(patch) {
  const sessionExplorer = {
    ...store.getState().sessionExplorer,
    ...patch,
  };
  store.update({ sessionExplorer, playing: false });
}

function filteredReplaySessions(scenario = store.getState().sessionExplorer.scenario) {
  const sessions = bundle?.session_replays?.sessions || [];
  return scenario === "all"
    ? sessions
    : sessions.filter((session) => session.scenario_type === scenario);
}

// The playground talks to the local server's live engine, which holds the same
// Agent the evaluator drives. A static export has no engine, so every entry
// point below is a no-op there and the page renders its offline card instead.
function updatePlayground(patch) {
  const playground = { ...store.getState().playground, ...patch };
  store.update({ playground });
}

function liveEnabled() {
  return !STATIC_DEPLOY && Boolean(health?.live_available);
}

async function playgroundSend(message) {
  const text = String(message || "").trim();
  const playground = store.getState().playground;
  if (!liveEnabled() || playground.busy || !text || playground.turn >= 10) return;
  const turn = playground.turn + 1;
  updatePlayground({ busy: true, error: null });
  try {
    let sessionId = playground.sessionId;
    if (!sessionId) {
      sessionId = `playground-${Date.now().toString(36)}`;
      await postJSON("/api/live/reset", {
        session_id: sessionId,
        user_profile: { preference_tags: [] },
      });
    }
    const data = await postJSON("/api/live/respond", {
      session_id: sessionId,
      message: text,
      turn,
      top_k: 10,
    });
    const response = data.response || {};
    const usage = response.usage || {};
    updatePlayground({
      sessionId,
      turn,
      busy: false,
      certificate: null,
      last: { response, facts: data.catalog_facts || {} },
      log: [
        ...playground.log,
        { role: "shopper", turn, text },
        {
          role: "agent",
          turn,
          message: response.message || "",
          ask: response.ask_attribute,
          shown: (response.recommendations || []).length,
          tokens: Number(usage.prompt_tokens || 0) + Number(usage.completion_tokens || 0),
        },
      ],
    });
  } catch (error) {
    updatePlayground({ busy: false, error: error.message });
  }
  refocusLiveInput = true;
}

async function playgroundExplain() {
  const playground = store.getState().playground;
  if (!liveEnabled() || playground.busy || !playground.sessionId) return;
  updatePlayground({ busy: true, error: null });
  try {
    const data = await getJSON(
      `/api/live/explain?session_id=${encodeURIComponent(playground.sessionId)}`
    );
    updatePlayground({ busy: false, certificate: data.certificate || null });
  } catch (error) {
    updatePlayground({ busy: false, error: error.message });
  }
}

function playgroundReset() {
  store.update({ playground: structuredClone(emptyPlayground) });
  refocusLiveInput = true;
}

document.addEventListener("submit", (event) => {
  if (event.target.id !== "live-form") return;
  event.preventDefault();
  const input = event.target.querySelector("#live-input");
  const message = input?.value || "";
  if (input) input.value = "";
  playgroundSend(message);
});

document.addEventListener("change", (event) => {
  if (event.target.id === "session-filter") {
    const scenario = event.target.value;
    const sessions = filteredReplaySessions(scenario);
    const current = store.getState().sessionExplorer.sessionId;
    const sessionId = sessions.some((session) => session.sample_id === current)
      ? current
      : sessions[0]?.sample_id;
    updateSessionExplorer({ scenario, sessionId, turn: 1 });
  } else if (event.target.id === "session-select") {
    updateSessionExplorer({ sessionId: event.target.value, turn: 1 });
  }
});

function moveScene(delta) {
  const state = store.getState();
  const scene = Math.max(0, Math.min(SCENES.length - 1, state.scene + delta));
  store.update({ scene });
}

document.addEventListener("click", (event) => {
  const presetButton = event.target.closest("button[data-preset]");
  if (presetButton) {
    const presets = (bundle?.explore_trace?.turns || [])
      .map((turn) => turn.customer)
      .filter(Boolean);
    const message = presets[Number.parseInt(presetButton.dataset.preset, 10)];
    const input = document.querySelector("#live-input");
    if (input && message) {
      input.value = message;
      input.focus();
    }
    return;
  }
  const sceneButton = event.target.closest("button[data-scene]");
  if (sceneButton) {
    const scene = Number.parseInt(sceneButton.dataset.scene, 10);
    if (Number.isFinite(scene)) store.update({ scene, playing: false });
    return;
  }
  const actionButton = event.target.closest("[data-action]");
  if (!actionButton || actionButton.disabled) return;
  const state = store.getState();
  switch (actionButton.dataset.action) {
    case "home":
    case "restart":
      store.update({ scene: 0, playing: false });
      break;
    case "back":
      moveScene(-1);
      break;
    case "next":
      moveScene(1);
      break;
    case "play":
      store.update({ playing: !state.playing });
      break;
    case "fullscreen":
      if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
      else document.exitFullscreen?.();
      break;
    case "pg-reset":
      playgroundReset();
      break;
    case "pg-explain":
      playgroundExplain();
      break;
    case "session-mode-replay":
      updateSessionExplorer({ mode: "replay" });
      break;
    case "session-mode-live":
      updateSessionExplorer({ mode: "live" });
      break;
    case "session-prev-turn":
      updateSessionExplorer({
        turn: Math.max(1, state.sessionExplorer.turn - 1),
      });
      break;
    case "session-next-turn": {
      const selected = (bundle.session_replays?.sessions || []).find(
        (session) => session.sample_id === state.sessionExplorer.sessionId
      );
      updateSessionExplorer({
        turn: Math.min(selected?.turns.length || 1, state.sessionExplorer.turn + 1),
      });
      break;
    }
    case "session-set-turn":
      updateSessionExplorer({
        turn: Number.parseInt(actionButton.dataset.turn || "1", 10),
      });
      break;
    case "session-random": {
      const sessions = filteredReplaySessions();
      const alternatives = sessions.filter(
        (session) => session.sample_id !== state.sessionExplorer.sessionId
      );
      const pool = alternatives.length ? alternatives : sessions;
      const selected = pool[Math.floor(Math.random() * pool.length)];
      if (selected) updateSessionExplorer({ sessionId: selected.sample_id, turn: 1 });
      break;
    }
    default:
      break;
  }
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  const state = store.getState();
  if (event.key === "ArrowRight") moveScene(1);
  else if (event.key === "ArrowLeft") moveScene(-1);
  else if (event.key === " ") {
    event.preventDefault();
    store.update({ playing: !state.playing });
  } else if (event.key.toLowerCase() === "f") {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  }
});

async function start() {
  try {
    [bundle, health] = await Promise.all([
      getJSON(ENDPOINTS.demo),
      getJSON(ENDPOINTS.health),
    ]);
    render(store.getState());
  } catch (error) {
    root.innerHTML = renderFatal(error);
  }
}

start();
