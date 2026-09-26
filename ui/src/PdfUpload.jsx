import React, { useState } from "react";

export default function PdfUpload({ apiBaseUrl }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function upload(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const file = form.elements.pdf.files[0];
    if (!file || busy) return;
    setMessage("");
    setError("");
    if (file.size > 20 * 1024 * 1024) {
      setError("PDF must be 20 MB or smaller.");
      return;
    }
    setBusy(true);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${apiBaseUrl}/documents/upload`, { method: "POST", body });
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "PDF upload failed.");
      const summary = result.status === "reindexed"
        ? `${result.filename}: ${result.pages} pages, ${result.chunks_added} chunks reindexed. Ready for questions.`
        : `${result.filename}: ${result.pages} pages, ${result.chunks_added} chunks indexed. Ready for questions.`;
      setMessage(summary + (result.skipped_pages.length
        ? ` Pages with no extractable text were skipped: ${result.skipped_pages.join(", ")}. Scanned pages need OCR.` : ""));
      form.reset();
    } catch (err) {
      setError(err instanceof Error ? err.message : "PDF upload failed. Please retry.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="pdfUpload" onSubmit={upload} aria-busy={busy}>
      <label htmlFor="pdf-file">Add a PDF to Vector RAG</label>
      <p>Selectable text only · Up to 20 MB and 200 pages</p>
      <div className="uploadControls">
        <input id="pdf-file" name="pdf" type="file" accept=".pdf,application/pdf" required disabled={busy} />
        <button type="submit" disabled={busy}>
          {busy ? "Processing PDF…" : "Upload and index"}
        </button>
      </div>
      <div role="status" aria-live="polite">{message}</div>
      {error ? <div className="error" role="alert">{error}</div> : null}
    </form>
  );
}

