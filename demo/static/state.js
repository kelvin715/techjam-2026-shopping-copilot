export const SCENES = [
  { id: "problem", label: "Formulation", duration: 18000 },
  { id: "method", label: "Method", duration: 34000 },
  { id: "headline", label: "Result", duration: 14000 },
  { id: "walk-1", label: "T1 · ASK", duration: 19000 },
  { id: "walk-2", label: "T2 · PIVOT", duration: 18000 },
  { id: "walk-3", label: "T3 · UPDATE", duration: 20000 },
  { id: "walk-4", label: "T4 · COMMIT", duration: 23000 },
  { id: "proof", label: "Evaluation", duration: 23000 },
  { id: "sessions", label: "Sessions", duration: 30000 },
];

const params = new URLSearchParams(window.location.search);
const requestedScene = Number.parseInt(params.get("scene") || "0", 10);
const requestedTurn = Number.parseInt(params.get("turn") || "1", 10);
const requestedScenario = params.get("scenario") || "all";
const requestedSessionMode = params.get("session_mode") === "live" ? "live" : "replay";

// The playground drives the real Agent over the local HTTP API, so its state is
// a live session rather than anything recorded in the bundle.
export const emptyPlayground = {
  sessionId: null,
  turn: 0,
  log: [],
  last: null,
  certificate: null,
  busy: false,
  error: null,
};

export const emptySessionExplorer = {
  mode: requestedSessionMode,
  scenario: ["all", "buying", "browsing", "intent_override", "boundary"].includes(requestedScenario)
    ? requestedScenario
    : "all",
  sessionId: /^public_\d{4}$/.test(params.get("session") || "")
    ? params.get("session")
    : "public_0187",
  turn: Number.isFinite(requestedTurn) ? Math.max(1, Math.min(10, requestedTurn)) : 1,
};

export const initialState = {
  scene: Number.isFinite(requestedScene)
    ? Math.max(0, Math.min(SCENES.length - 1, requestedScene))
    : 0,
  playing: params.get("autoplay") === "1",
  playground: structuredClone(emptyPlayground),
  sessionExplorer: structuredClone(emptySessionExplorer),
};

export function createStore(seed = initialState) {
  let state = structuredClone(seed);
  const listeners = new Set();

  function notify() {
    for (const listener of listeners) listener(state);
  }

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    update(patch) {
      state = { ...state, ...patch };
      notify();
    },
  };
}
