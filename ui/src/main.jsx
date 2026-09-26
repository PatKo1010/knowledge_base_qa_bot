import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ArrowUp, BookOpen, ChevronRight, FileText, Layers, Loader2, MessageSquare,
  Plus, Search, Square, Upload, X } from "lucide-react";
import { api, API_BASE, streamChat } from "./api";
import PdfUpload from "./PdfUpload";
import "./styles.css";

function UploadDialog({ onClose }) {
  const dialogRef = useRef(null);
  useEffect(() => { dialogRef.current.showModal(); }, []);
  return (
    <dialog ref={dialogRef} className="uploadDialog" aria-labelledby="upload-title"
      onCancel={onClose} onClick={event => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="uploadModal">
        <header><h2 id="upload-title">Add to your knowledge base</h2>
          <button autoFocus aria-label="Close upload" onClick={onClose}><X size={18} /></button>
        </header>
        <PdfUpload apiBaseUrl={API_BASE} />
      </section>
    </dialog>
  );
}

function App() {
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [draft, setDraft] = useState("");
  const [filter, setFilter] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [hasOlder, setHasOlder] = useState(false);
  const [moreChats, setMoreChats] = useState(false);
  const [error, setError] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [activeSource, setActiveSource] = useState(null);
  const abortRef = useRef(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const followRef = useRef(true);
  const selected = messages.find(message => message.id === selectedId);
  const sources = selected?.citations || [];
  const title = conversations.find(item => item.id === conversationId)?.title || "New conversation";

  async function refreshHistory() {
    const result = await api("/api/v1/conversations?limit=50");
    setConversations(result.conversations);
    setMoreChats(result.conversations.length === 50);
  }

  useEffect(() => {
    refreshHistory().catch(err => setError(err.message)).finally(() => setLoading(false));
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (followRef.current) bottomRef.current?.scrollIntoView({ behavior: "instant", block: "end" });
  }, [messages]);

  async function openConversation(id) {
    if (busy || loading) return;
    setLoading(true); setError("");
    try {
      const result = await api(`/api/v1/conversations/${id}/messages`);
      setConversationId(id); setMessages(result.messages); setHasOlder(result.has_more);
      setSelectedId(result.messages.findLast(message => message.role === "assistant")?.id || null);
      setActiveSource(null); setDraft(""); followRef.current = true;
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  function newChat() {
    if (busy || loading) return;
    setConversationId(null); setMessages([]); setSelectedId(null); setHasOlder(false);
    setActiveSource(null); setError(""); setDraft(""); inputRef.current?.focus();
  }

  async function loadMoreChats() {
    setLoading(true);
    try {
      const result = await api(`/api/v1/conversations?limit=50&offset=${conversations.length}`);
      setConversations(current => [...current, ...result.conversations]);
      setMoreChats(result.conversations.length === 50);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  async function loadOlder() {
    setLoading(true); followRef.current = false;
    try {
      const result = await api(`/api/v1/conversations/${conversationId}/messages?before=${messages[0].sequence}`);
      setMessages(current => [...result.messages, ...current]); setHasOlder(result.has_more);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  async function send(event) {
    event.preventDefault();
    const question = draft.trim();
    if (!question || busy || loading) return;
    const temporaryUser = `user-${Date.now()}`;
    const temporaryAssistant = `assistant-${Date.now()}`;
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true); setDraft(""); setError(""); setSelectedId(temporaryAssistant);
    setActiveSource(null); followRef.current = true;
    setMessages(current => [...current.filter(message => !message.temporary),
      { id: temporaryUser, role: "user", content: question, citations: [], temporary: true },
      { id: temporaryAssistant, role: "assistant", content: "", citations: [], temporary: true, status: "streaming" }]);
    try {
      await streamChat(conversationId, question, controller.signal, (eventName, data) => {
        if (eventName === "conversation") setConversationId(data.conversation_id);
        if (eventName === "token" || eventName === "citations") {
          setMessages(current => current.map(message => message.id !== temporaryAssistant ? message : {
            ...message,
            ...(eventName === "token" ? { content: message.content + data.text } : { citations: data.citations }),
          }));
        }
        if (eventName === "done") {
          setMessages(current => [...current.filter(message => !message.temporary), ...data.messages]);
          setSelectedId(data.message_id);
        }
      });
    } catch (err) {
      const detail = err.name === "AbortError" ? "Response stopped. This turn has not been confirmed saved." : err.message;
      setError(detail); setDraft(question);
      setMessages(current => current.map(message => message.id === temporaryAssistant ?
        { ...message, status: "error" } : message));
    } finally {
      abortRef.current = null;
      setBusy(false);
      await refreshHistory().catch(() => setError(current => current || "Could not refresh chat history."));
      inputRef.current?.focus();
    }
  }

  function chooseSource(messageId, chunkId) {
    setSelectedId(messageId); setActiveSource(chunkId);
    requestAnimationFrame(() => document.getElementById(`source-${chunkId}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
  }

  return (
    <main className="appShell">
      <aside className="sidebar" aria-label="Chat history">
        <a className="brand" href="/" aria-label="Knowledge home"><span className="brandMark"><Layers size={21} /></span> knowledge<span className="brandDot">.</span></a>
        <button className="newChat" onClick={newChat} disabled={busy || loading}><Plus size={17} /> New conversation <span>↗</span></button>
        <label className="historySearch"><Search size={15} /><input aria-label="Search chat history" placeholder="Search conversations" value={filter} onChange={event => setFilter(event.target.value)} /></label>
        <div className="sectionLabel">YOUR CONVERSATIONS <span>{conversations.length}</span></div>
        <nav className="historyList">
          {conversations.filter(item => item.title.toLowerCase().includes(filter.toLowerCase())).map(item => (
            <button key={item.id} className={`historyItem ${item.id === conversationId ? "active" : ""}`} disabled={busy || loading} onClick={() => openConversation(item.id)}>
              <MessageSquare size={16} /><span className="historyText"><strong>{item.title}</strong><time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</time></span>{item.id === conversationId && <span className="activeDot" />}
            </button>
          ))}
          {!conversations.length && <p className="sidebarHint">Your conversations will appear here after you start a chat.</p>}
          {moreChats && <button className="textButton" disabled={busy || loading} onClick={loadMoreChats}>Load more conversations</button>}
        </nav>
        <div className="sidebarFooter"><button className="uploadButton" onClick={() => setUploadOpen(true)}><Upload size={16} /> Add a PDF <Plus size={14} /></button><div className="workspaceLabel"><span className="workspaceAvatar">K</span><div>My knowledge base<small>Vector RAG workspace</small></div></div></div>
      </aside>

      <section className="chatPanel" aria-label="Current conversation">
        <header className="chatHeader"><div className="breadcrumb">Workspace <ChevronRight size={13} /> <span>Chat</span></div><div className="headerActions"><button className="mobileUpload" aria-label="Add a PDF" onClick={() => setUploadOpen(true)}><Upload size={16} /></button><div className="modelBadge"><span /> Vector RAG</div></div></header>
        <div className="conversationTitle"><h1>{title}</h1><p>Answers grounded in your documents.</p></div>
        <div className="messageList" role="log" aria-label="Messages" aria-live="polite" onScroll={event => { const node = event.currentTarget; followRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 100; }}>
          {hasOlder && <button className="textButton olderButton" disabled={loading || busy} onClick={loadOlder}>Load earlier messages</button>}
          {loading && <div className="loading"><Loader2 className="spin" size={18} /> Loading conversations…</div>}
          {!messages.length && !loading && <div className="welcome"><div className="welcomeIcon"><BookOpen size={30} strokeWidth={1.5} /></div><div className="eyebrow">YOUR KNOWLEDGE, CONNECTED</div><h2>A little clarity starts<br />with a good question.</h2><p>Explore your documents, find the details that matter,<br className="desktopBreak" /> and keep the conversation going.</p><div className="suggestions">{["What are the key points in my documents?", "How long do refunds take?"].map(text => <button key={text} onClick={() => { setDraft(text); inputRef.current?.focus(); }}>{text}<ChevronRight size={16} /></button>)}</div></div>}
          {messages.map(message => <article key={message.id} className={`message ${message.role} ${selectedId === message.id ? "selected" : ""}`}>
            <div className="messageAvatar">{message.role === "user" ? "Y" : <Layers size={17} />}</div>
            <div className="messageBody"><div className="messageLabel">{message.role === "user" ? "You" : "Knowledge"}{message.status === "streaming" && <span className="streamLabel"><Loader2 size={12} className="spin" /> Responding</span>}</div>
              <div className="messageContent">{(message.content && message.content.split(/(\[\d+\])/g).map((part, index) => {
                const citation = message.role === "assistant" && message.citations.find(source => `[${source.index}]` === part);
                return citation ? <button key={index} className="inlineCitation" aria-label={`View source ${citation.index}`} onClick={() => chooseSource(message.id, citation.chunk_id)}>{part}</button> : part;
              })) || (message.status === "streaming" ? "Searching your knowledge base…" : "No completed response.")}{message.status === "streaming" && message.content && <span className="cursor" />}</div>
              {message.role === "assistant" && <div className="messageSources"><button className={`sourceToggle ${selectedId === message.id ? "chosen" : ""}`} onClick={() => { setSelectedId(message.id); setActiveSource(null); }}><BookOpen size={13} />{message.citations.length} sources<ChevronRight size={12} /></button>{message.citations.map(source => <button key={source.chunk_id} className="citationChip" aria-label={`View source ${source.index}: ${source.document_name}`} onClick={() => chooseSource(message.id, source.chunk_id)}>[{source.index}]</button>)}</div>}
              {message.status === "error" && <small className="unsaved">Incomplete response · not confirmed saved</small>}
            </div>
          </article>)}
          <div ref={bottomRef} />
        </div>
        <div className="composerArea">{error && <div className="error" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError("")}><X size={14} /></button></div>}
          <form className="composer" onSubmit={send}><textarea ref={inputRef} aria-label="Message" placeholder="Ask anything about your documents…" value={draft} maxLength={12000} rows={2} disabled={loading} onChange={event => setDraft(event.target.value)} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); send(event); } }} /><div className="composerBottom"><span><BookOpen size={13} /> Connected to your knowledge base</span>{busy ? <button type="button" className="sendButton" aria-label="Stop response" onClick={() => abortRef.current?.abort()}><Square size={14} fill="currentColor" /></button> : <button className="sendButton" type="submit" disabled={!draft.trim() || loading} aria-label="Send message"><ArrowUp size={19} /></button>}</div></form>
          <p className="composerHint">Check the sources for important details. <span>Enter to send · Shift + Enter for a new line</span></p>
        </div>
      </section>

      <aside className="resourcesPanel" aria-label="Response sources"><header className="resourcesHeader"><div><BookOpen size={17} /><h2>Sources</h2><span className="sourceCount">{sources.length}</span></div><p>Behind the answer</p></header>
        <div className="resourceScroll">{selected ? <><div className="resourceContext"><span className="sectionLabel">SOURCES FOR SELECTED RESPONSE</span><p>{selected.content.slice(0, 130) || "Finding relevant passages…"}</p></div>{!sources.length && <div className="emptyResources"><FileText size={28} strokeWidth={1.3} /><h3>{selected.status === "streaming" ? "Looking for sources" : "No supporting sources"}</h3><p>{selected.status === "streaming" ? "Relevant passages will appear here." : "This response has no retrieved passages. Try a more specific question or add a document."}</p></div>}{sources.map(source => <article tabIndex={0} id={`source-${source.chunk_id}`} key={source.chunk_id} className={`resourceCard ${activeSource === source.chunk_id ? "highlighted" : ""}`}><div className="resourceTop"><span className="resourceNumber">{source.index}</span><span>{source.page ? `PAGE ${source.page}` : "DOCUMENT"}</span><FileText size={15} /></div><h3>{source.document_name}</h3>{source.section && <p className="resourceSection">{source.section}</p>}<div className="resourceExcerpt">{source.content}</div><footer><span>{source.chunk_index != null ? `Chunk ${source.chunk_index}` : "Passage"}</span><span title="FAISS distance; lower is closer">Distance {source.score.toFixed(3)}</span></footer><details><summary>Source identifier</summary><code>{source.source}</code></details></article>)}</> : <div className="emptyResources initial"><div className="sourceIllustration"><FileText size={32} strokeWidth={1.2} /><Search size={18} /></div><h3>Every answer has a starting point.</h3><p>Source passages appear here as you chat. Select any response to revisit its references.</p><span className="emptyDivider" /><div className="sourcePromise"><span>01</span> Ask a question</div><div className="sourcePromise"><span>02</span> Explore the answer</div><div className="sourcePromise"><span>03</span> Go straight to the source</div></div>}</div>
        <div className="resourcesFooter"><Layers size={13} /> Retrieved from your knowledge base</div>
      </aside>
      {uploadOpen && <UploadDialog onClose={() => setUploadOpen(false)} />}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
