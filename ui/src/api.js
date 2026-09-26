export const API_BASE = import.meta.env.VITE_VECTOR_API_BASE_URL || "http://localhost:8001";

export async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : `Request failed (${response.status}).`);
  }
  return response.json();
}

export async function consumeSse(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;
  function parse(frame) {
    let event = "message";
    const data = [];
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) return;
    const payload = JSON.parse(data.join("\n"));
    if (event === "error") throw new Error(payload.message || "Response failed. Please try again.");
    onEvent(event, payload);
    if (event === "done") completed = true;
  }
  try {
    while (!completed) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let separator;
      while ((separator = /\r?\n\r?\n/.exec(buffer))) {
        parse(buffer.slice(0, separator.index));
        buffer = buffer.slice(separator.index + separator[0].length);
        if (completed) break;
      }
      if (done) break;
    }
    if (!completed) throw new Error("The connection ended before the answer was saved. Please try again.");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function streamChat(conversationId, message, signal, onEvent) {
  const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ conversation_id: conversationId, message }),
  });
  if (!response.ok || !response.body) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : "Unable to start the response.");
  }
  await consumeSse(response.body, onEvent);
}
