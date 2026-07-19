import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Loader2, Search, Send, Server, Sparkles } from "lucide-react";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function parseSseFrame(frame) {
  let event = "message";
  const dataLines = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  const rawData = dataLines.join("\n");
  return {
    event,
    data: rawData ? JSON.parse(rawData) : {},
  };
}

function SourceList({ sources }) {
  if (!sources.length) {
    return (
      <div className="emptySources">
        <Server size={16} />
        <span>No sources selected yet</span>
      </div>
    );
  }

  return (
    <div className="sourceList">
      {sources.map((source, index) => (
        <article className="sourceItem" key={`${source.source}-${index}`}>
          <div className="sourceMeta">
            <span className="sourceId">{source.source}</span>
            <span className="score">{source.score}</span>
          </div>
          <div className="heading">{source.heading}</div>
          <p>{source.content}</p>
        </article>
      ))}
    </div>
  );
}

function App() {
  const [query, setQuery] = useState("How long do refunds take?");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const canSubmit = useMemo(
    () => query.trim().length > 0 && status !== "streaming",
    [query, status],
  );

  async function submitQuestion(event) {
    event.preventDefault();
    if (!canSubmit) return;

    setAnswer("");
    setSources([]);
    setError("");
    setStatus("streaming");

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          if (!frame.trim()) continue;

          const message = parseSseFrame(frame);
          if (message.event === "sources") {
            setSources(message.data.sources || []);
          }
          if (message.event === "token") {
            setAnswer((current) => current + (message.data.text || ""));
          }
          if (message.event === "error") {
            setError(message.data.message || "Streaming request failed");
            setStatus("error");
          }
          if (message.event === "done") {
            setStatus((current) => (current === "error" ? current : "done"));
          }
        }
      }

      setStatus((current) => (current === "streaming" ? "done" : current));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Streaming request failed");
      setStatus("error");
    }
  }

  return (
    <main className="appShell">
      <section className="workspace">
        <div className="topbar">
          <div>
            <h1>Markdown KB Chat</h1>
            <p>Ask the indexed docs and watch the answer stream in.</p>
          </div>
          <div className={`status ${status}`}>
            {status === "streaming" ? <Loader2 className="spin" size={16} /> : <Sparkles size={16} />}
            <span>{status}</span>
          </div>
        </div>

        <form className="composer" onSubmit={submitQuestion}>
          <Search size={18} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask a question"
          />
          <button disabled={!canSubmit} type="submit" aria-label="Send question">
            <Send size={18} />
          </button>
        </form>

        {error ? <div className="error">{error}</div> : null}

        <div className="panels">
          <section className="sourcesPanel">
            <h2>Sources</h2>
            <SourceList sources={sources} />
          </section>

          <section className="answerPanel">
            <h2>Answer</h2>
            <div className="answerText">
              {answer || <span className="placeholder">Waiting for streamed tokens...</span>}
              {status === "streaming" ? <span className="cursor" /> : null}
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
