export const STAGES = [
  { id: "grounding", name: "Understand", hint: "Read the shopper’s message", working: "Understanding your request…" },
  { id: "evidence", name: "Update evidence", hint: "Remember what matters", working: "Updating your preferences…" },
  { id: "ranking", name: "Rank candidates", hint: "Find the closest matches", working: "Ranking matching products…" },
  { id: "mvoi", name: "Choose a question", hint: "Answerable MVOI", working: "Evaluating the next question…" },
  { id: "planner", name: "Plan recommendations", hint: "Batch Planner / output gate", working: "Deciding how many to recommend…" },
  { id: "response", name: "Reply", hint: "Ask + recommend", working: "Preparing your reply…" },
];

export function emptyView(turn = 0) {
  return { turn, seq: 0, failed: false, nodes: Object.fromEntries(STAGES.map(s => [s.id, { status: "waiting", data: {} }])) };
}

export function applyStage(view, event) {
  if (event.turn !== view.turn || event.seq <= view.seq) return false;
  view.seq = event.seq;
  if (event.stage === "error") {
    view.failed = true;
    for (const node of Object.values(view.nodes)) if (node.status === "running") node.status = "failed";
    return true;
  }
  const node = view.nodes[event.stage];
  if (!node) return false;
  if (event.status !== "result") node.status = event.status;
  node.data = { ...node.data, ...event.data };
  if (event.duration_ms != null) node.duration = event.duration_ms;
  return true;
}

// fetch-based SSE accepts POST bodies and handles UTF-8 / frame boundaries
// independently of network chunk boundaries. Each server event is ordered.
export async function readEventStream(response, onEvent) {
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  if (!response.body) throw new Error("This browser does not support streamed responses.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let match;
      while ((match = /\r?\n\r?\n/.exec(buffer))) {
        const packet = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        let type = "message";
        const data = [];
        for (const line of packet.split(/\r?\n/)) {
          if (line.startsWith("event:")) type = line.slice(6).trim();
          if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
        }
        if (data.length) await onEvent(type, JSON.parse(data.join("\n")));
      }
      if (done) break;
    }
    if (buffer.trim()) throw new Error("The event stream ended before the last event was complete.");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
