import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
// Vite's environment expression is replaced for this dependency-free parser test.
const source = (await fs.readFile(new URL("../src/api.js", import.meta.url), "utf8"))
  .replace('import.meta.env.VITE_VECTOR_API_BASE_URL', '""');
const { consumeSse } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
function body(text) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({ start(controller) {
    for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
    controller.close();
  }});
}
test("handles byte-split UTF-8, CRLF, comments, and citations", async () => {
  const events = [];
  await consumeSse(body(': heartbeat\r\n\r\nevent: token\r\ndata: {"text":"你好"}\r\n\r\nevent: citations\ndata: {"citations":[]}\n\nevent: done\ndata: {}\n\n'), (...event) => events.push(event));
  assert.deepEqual(events, [["token", { text: "你好" }], ["citations", { citations: [] }], ["done", {}]]);
});
test("rejects a truncated response without done", async () => {
  await assert.rejects(consumeSse(body('event: token\ndata: {"text":"partial"}\n\n'), () => {}), /before the answer was saved/);
});
test("server errors are failures, never successful completion", async () => {
  await assert.rejects(consumeSse(body('event: error\ndata: {"message":"Failed"}\n\n'), () => {}), /Failed/);
});
