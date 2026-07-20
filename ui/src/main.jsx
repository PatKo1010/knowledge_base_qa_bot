import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Loader2, Search, Send, Server, Sparkles } from "lucide-react";
import "./styles.css";

const SERVICES = [
  {
    id: "markdown",
    title: "Markdown KB",
    description: "Keyword retrieval over parsed Markdown sections.",
    apiBaseUrl: import.meta.env.VITE_MARKDOWN_API_BASE_URL || "http://localhost:8000",
  },
  {
    id: "vector",
    title: "Vector RAG",
    description: "Embedding retrieval over chunked Markdown content.",
    apiBaseUrl: import.meta.env.VITE_VECTOR_API_BASE_URL || "http://localhost:8001",
  },
];

const emptyResult = {
  answer: "",
  sources: [],
  status: "idle",
  error: "",
};

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

function ResultBlock({ service, result }) {
  return (
    <section className="resultBlock">
      <div className="resultHeader">
        <div>
          <h2>{service.title}</h2>
          <p>{service.description}</p>
        </div>
        <div className={`status ${result.status}`}>
          {result.status === "streaming" ? <Loader2 className="spin" size={16} /> : <Sparkles size={16} />}
          <span>{result.status}</span>
        </div>
      </div>

      {result.error ? <div className="error">{result.error}</div> : null}

      <div className="panels">
        <section className="sourcesPanel">
          <h3>Sources</h3>
          <SourceList sources={result.sources} />
        </section>

        <section className="answerPanel">
          <h3>Answer</h3>
          <div className="answerText">
            {result.answer || <span className="placeholder">Waiting for streamed tokens...</span>}
            {result.status === "streaming" ? <span className="cursor" /> : null}
          </div>
        </section>
      </div>
    </section>
  );
}

function App() {
  const [query, setQuery] = useState("How long do refunds take?");
  const [results, setResults] = useState(() =>
    Object.fromEntries(SERVICES.map((service) => [service.id, emptyResult])),
  );

  const isStreaming = Object.values(results).some((result) => result.status === "streaming");

  const canSubmit = useMemo(
    () => query.trim().length > 0 && !isStreaming,
    [query, isStreaming],
  );

  function updateResult(serviceId, update) {
    setResults((current) => ({
      ...current,
      [serviceId]: {
        ...current[serviceId],
        ...(typeof update === "function" ? update(current[serviceId]) : update),
      },
    }));
  }

  async function streamService(service, question) {
    try {
      const response = await fetch(`${service.apiBaseUrl}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({ query: question }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`${service.title} request failed with status ${response.status}`);
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
            updateResult(service.id, { sources: message.data.sources || [] });
          }
          if (message.event === "token") {
            updateResult(service.id, (current) => ({
              answer: current.answer + (message.data.text || ""),
            }));
          }
          if (message.event === "error") {
            updateResult(service.id, {
              error: message.data.message || `${service.title} streaming request failed`,
              status: "error",
            });
          }
          if (message.event === "done") {
            updateResult(service.id, (current) => ({
              status: current.status === "error" ? current.status : "done",
            }));
          }
        }
      }

      updateResult(service.id, (current) => ({
        status: current.status === "streaming" ? "done" : current.status,
      }));
    } catch (err) {
      updateResult(service.id, {
        error: err instanceof Error ? err.message : `${service.title} streaming request failed`,
        status: "error",
      });
    }
  }

  async function submitQuestion(event) {
    event.preventDefault();
    if (!canSubmit) return;

    const question = query.trim();
    setResults(Object.fromEntries(
      SERVICES.map((service) => [
        service.id,
        {
          ...emptyResult,
          status: "streaming",
        },
      ]),
    ));

    await Promise.all(SERVICES.map((service) => streamService(service, question)));
  }

  return (
    <main className="appShell">
      <section className="workspace">
        <div className="topbar">
          <div>
            <h1>Knowledge Base Comparison</h1>
            <p>Send one question to both retrieval backends and compare their streamed answers.</p>
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

        <div className="resultsStack">
          {SERVICES.map((service) => (
            <ResultBlock key={service.id} service={service} result={results[service.id]} />
          ))}
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
