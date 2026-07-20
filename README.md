# Knowledge Base Q&A Bot

This repo compares two ways to retrieve context from the same Markdown knowledge base before asking an LLM to answer:

- `markdown_kb`: parses Markdown into heading-based sections and retrieves context with keyword/BM25 scoring.
- `vector_rag`: chunks Markdown, embeds chunks with OpenAI embeddings, and retrieves context with FAISS vector search.
- `ui`: a shared React/Vite UI that sends the same question to both backends through `POST /chat/stream` and displays sources plus streamed answers side by side.

The source documents live in `docs/*.md`.

## Retrieval Flow

### Markdown KB

```text
docs/*.md
  -> parse by Markdown headings
  -> build .kb/index.json
  -> BM25 keyword retrieval
  -> apply MIN_RETRIEVAL_SCORE
  -> put selected sections into the prompt
  -> stream grounded answer tokens
```

This strategy keeps the index inspectable. Each retrieved section maps directly to a Markdown heading and uses source IDs such as `refund_policy.md#refund-timeline`.

Use this when you want simple debugging, transparent source selection, and predictable behavior over structured Markdown.

### Vector RAG

```text
docs/*.md
  -> parse by Markdown headings
  -> split sections into chunks
  -> embed chunks with OpenAI
  -> persist FAISS index in .kb/faiss_index/
  -> vector similarity search
  -> apply MAX_RETRIEVAL_DISTANCE
  -> put selected chunks into the prompt
  -> stream grounded answer tokens
```

This strategy retrieves semantically similar chunks, which can help when user wording differs from the document wording. FAISS scores are distances here, so lower is better. Chunks with scores above `MAX_RETRIEVAL_DISTANCE` are excluded from both sources and prompt context.

## API

Both backends expose:

```text
GET  /health
POST /index
POST /chat
POST /chat/stream
```

`POST /chat/stream` returns Server-Sent Events:

```text
event: sources
data: {"sources": [...]}

event: token
data: {"text": "..."}

event: error
data: {"message": "..."}

event: done
data: {}
```

## Local Setup

Create `.env` files:

```bash
cp markdown_kb/.env.example markdown_kb/.env 2>/dev/null || true
cp vector_rag/.env.example vector_rag/.env 2>/dev/null || true
```

If those examples are not present, create these files manually:

```bash
# markdown_kb/.env
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-4o-mini"
MIN_RETRIEVAL_SCORE="1.0"
```

```bash
# vector_rag/.env
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-4o-mini"
MAX_RETRIEVAL_DISTANCE="0.8"
```

Install backend dependencies:

```bash
cd markdown_kb
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

cd ../vector_rag
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Install UI dependencies:

```bash
cd ../ui
npm install
```

## Run Locally

Start Markdown KB on port `8000`:

```bash
cd markdown_kb
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Start Vector RAG on port `8001`:

```bash
cd vector_rag
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Build indexes:

```bash
curl -X POST http://localhost:8000/index
curl -X POST http://localhost:8001/index
```

Start the UI:

```bash
cd ui
npm run dev
```

Open:

```text
http://localhost:5173
```

The UI sends each submitted query to:

```text
http://localhost:8000/chat/stream
http://localhost:8001/chat/stream
```

## Deployment

Deploy three services:

1. `markdown_kb` FastAPI service
2. `vector_rag` FastAPI service
3. `ui` static web app

### Backend Deployment

For each backend, deploy from its folder with Python dependencies installed from `requirements.txt`.

Markdown KB command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Vector RAG command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}
```

Set environment variables in the hosting platform:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

For Markdown KB, optionally set:

```bash
MIN_RETRIEVAL_SCORE=1.0
```

For Vector RAG, optionally set:

```bash
MAX_RETRIEVAL_DISTANCE=0.8
```

After deployment, call `/index` once for each backend. The generated indexes are written under `.kb/`. In production, use persistent disk/storage for `.kb/` if you want indexes to survive restarts. Otherwise, run `/index` during release/startup.

### UI Deployment

Build the UI with backend URLs configured:

```bash
cd ui
VITE_MARKDOWN_API_BASE_URL=https://your-markdown-api.example.com \
VITE_VECTOR_API_BASE_URL=https://your-vector-api.example.com \
npm run build
```

Deploy `ui/dist/` to any static host.

Make sure both backends allow the deployed UI origin in CORS. Locally, both services allow:

```text
http://localhost:5173
http://127.0.0.1:5173
```

For production, add your deployed UI URL to the CORS `allow_origins` list in each backend.

## Verification

Health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

Index checks:

```bash
curl -X POST http://localhost:8000/index
curl -X POST http://localhost:8001/index
```

Streaming checks:

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do refunds take?"}'

curl -N -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do refunds take?"}'
```

Expected behavior:

- `sources` arrives first.
- `token` events stream the answer.
- `done` ends the stream.
- If retrieval is weak, the service says it cannot confirm from the knowledge base.
- If OpenAI fails, the service emits an `error` event and then `done`.
