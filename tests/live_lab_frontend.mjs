import assert from "node:assert/strict";
import { emptyView, applyStage, readEventStream } from "../demo/static/lab-state.js";

const view = emptyView(2);
assert.equal(applyStage(view, { turn: 1, seq: 20, stage: "ranking", status: "complete", data: { count: 999 } }), false);
assert.equal(view.nodes.ranking.status, "waiting");
assert.equal(applyStage(view, { turn: 2, seq: 1, stage: "planner", status: "running", data: {} }), true);
applyStage(view, { turn: 2, seq: 2, stage: "planner", status: "result", data: { active: true, choices: [{ batch: 1, expected_value: .96 }] } });
assert.equal(view.nodes.planner.status, "running");
applyStage(view, { turn: 2, seq: 3, stage: "planner", status: "complete", data: { emitted_count: 1 } });
assert.equal(view.nodes.planner.data.choices[0].expected_value, .96);
assert.equal(view.nodes.planner.data.emitted_count, 1);
assert.equal(applyStage(view, { turn: 2, seq: 2, stage: "planner", status: "running", data: {} }), false);
assert.equal(view.nodes.planner.status, "complete");

const source = 'event: stage\r\ndata: {"message":"蓝色鞋子","turn":1}\r\n\r\nevent: done\ndata: {"ok":true}\n\n';
const bytes = new TextEncoder().encode(source);
const chunks = Array.from(bytes, b => Uint8Array.of(b));
const stream = new ReadableStream({ pull(controller) { if (chunks.length) controller.enqueue(chunks.shift()); else controller.close(); } });
const events = [];
await readEventStream(new Response(stream), (type, data) => events.push({ type, data }));
assert.deepEqual(events, [{ type: "stage", data: { message: "蓝色鞋子", turn: 1 } }, { type: "done", data: { ok: true } }]);
await assert.rejects(readEventStream(new Response('event: done\ndata: {"ok":true}'), () => {}), /ended before/);
console.log("Live lab state ordering and fragmented UTF-8 SSE checks passed.");
