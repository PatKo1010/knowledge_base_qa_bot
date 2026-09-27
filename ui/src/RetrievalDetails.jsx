import React from "react";

export default function RetrievalDetails({ details }) {
  if (!details) return null;
  return <details className="retrievalDetails">
    <summary>Retrieval details</summary>
    <dl>
      <dt>Original question</dt><dd>{details.original_question}</dd>
      <dt>Search query</dt><dd>{details.rewritten_query}</dd>
      <dt>Rewrite</dt><dd>{details.rewrite_status === "skipped_no_history"
        ? "Skipped — no conversation history" : details.rewrite_model}</dd>
      <dt>Prompt version</dt><dd>{details.rewrite_prompt_version}</dd>
      <dt>Rewrite time</dt><dd>{details.rewrite_latency_ms} ms</dd>
      <dt>Distance threshold</dt><dd>{details.distance_threshold}</dd>
      <dt>Summary through message</dt><dd>{details.summary_through_sequence}</dd>
      <dt>Summary used</dt><dd>{details.summary || "None"}</dd>
      <dt>Recent message IDs</dt><dd>{details.recent_message_ids?.join("\n") || "None"}</dd>
    </dl>
    <p>Candidates after distance filtering, before reranking:</p>
    <ol>{details.retrieved_candidates?.map((candidate, index) => <li key={index}>
      {candidate.source} · Chunk {candidate.chunk_index ?? "—"} · Distance {candidate.score.toFixed(3)}
    </li>)}</ol>
  </details>;
}
