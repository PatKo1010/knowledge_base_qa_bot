# Conversational RAG V1 — Codex Implementation Instructions

## 1. Goal

Build the first version of a **multi-turn conversational RAG chatbot** with this exact runtime flow:

```text
User
 ↓
Load last 6 messages
 ↓
Load conversation summary
 ↓
LLM Query Rewrite
 ↓
Vector RAG retrieval
 ↓
Rerank top 20 → top 5
 ↓
LLM
 ↓
Answer + citations
 ↓
Store conversation
 ↓
Every ~10 messages:
update conversation summary
```

The system should support long conversations without sending the entire chat history to the LLM on every request.

The architecture should follow this structure:

```text
                         React
                           │
                     SSE / WebSocket
                           │
                           ▼
                        FastAPI
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     PostgreSQL          Redis          Vector DB
          │                │                │
   Conversation       Session/cache      Documents
      history         recent memory      embeddings
          │                                 │
          │                                 ▼
          │                              Retriever
          │                                 │
          └───────────────┐       ┌─────────┘
                          ▼       ▼
                         RAG Service
                              │
                        Query Rewriter
                              │
                          Reranker
                              │
                             LLM
```

---

# 2. Recommended Tech Stack

Use the following stack unless the existing repository already contains an equivalent component.

## Frontend

- React
- TypeScript
- Fetch API / EventSource for SSE
- WebSocket can be added later if required

For V1, prefer **SSE for streaming LLM responses** because the communication pattern is mainly server → client after the user sends a request.

## Backend

- Python 3.11+
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- Alembic
- asyncpg
- redis-py asyncio client

## Data

### PostgreSQL

Use PostgreSQL for durable application state:

- conversations
- messages
- conversation summaries
- documents metadata
- chunk metadata if needed

### Redis

Use Redis only for ephemeral / performance-related data:

- active session cache
- latest conversation state
- cached recent messages
- optional retrieval result cache

Redis must **not** be the system of record for conversation history.

### Vector database

Use the vector database already present in the repository if one exists.

If none exists, prefer one of:

1. PostgreSQL + pgvector
2. Qdrant

Keep vector retrieval behind an abstraction so the implementation can be replaced later.

---

# 3. Core Design Principle

Keep these concerns separate:

```text
Conversation memory
≠
Knowledge retrieval
```

Conversation history answers:

> What does the user mean in the current conversation?

The RAG knowledge base answers:

> What factual information should be used to answer?

Do not embed the entire conversation and use it directly as the document retrieval query.

Instead:

```text
recent messages
+
conversation summary
+
current user question
        ↓
query rewriter
        ↓
standalone retrieval query
        ↓
vector search
```

---

# 4. Runtime Request Flow

Implement the main chat endpoint using the following sequence.

## Step 1 — Receive user request

Example request:

```json
{
  "conversation_id": "uuid",
  "message": "What are its limitations?"
}
```

If no `conversation_id` is provided, create a new conversation.

---

## Step 2 — Load latest six messages

Read the latest six messages from PostgreSQL.

Prefer Redis cache if available.

The order passed to the LLM must be chronological.

Example:

```text
User: What is Redis?
Assistant: Redis is an in-memory datastore.

User: Can Redis be used for RAG?
Assistant: Yes. It can be used for caching...

User: What are its limitations?
```

Do not load the entire conversation.

Function interface:

```python
async def get_recent_messages(
    conversation_id: UUID,
    limit: int = 6,
) -> list[Message]:
    ...
```

---

## Step 3 — Load conversation summary

Each conversation has one rolling summary.

Example:

```text
The user is designing a RAG application.

Decisions made:
- Backend uses FastAPI.
- PostgreSQL stores conversation history.
- Redis is used as a cache.
- Vector retrieval is used for the knowledge base.
- The user is currently discussing Redis in the RAG architecture.
```

Function:

```python
async def get_conversation_summary(
    conversation_id: UUID,
) -> str | None:
    ...
```

For a new conversation, the summary may be empty.

---

## Step 4 — Rewrite the user query

Use an LLM to convert the latest user question into a standalone retrieval query.

Example input:

```text
Conversation summary:
The user is designing a RAG system and considering Redis.

Recent conversation:
User: Can Redis be used in this architecture?
Assistant: Yes...

Current question:
What are its limitations?
```

Expected rewritten query:

```text
What are the limitations of using Redis in a conversational RAG architecture?
```

The query rewriter should NOT answer the question.

It only produces the search query.

Suggested interface:

```python
async def rewrite_query(
    current_message: str,
    recent_messages: list[Message],
    conversation_summary: str | None,
) -> str:
    ...
```

Suggested prompt:

```text
You rewrite user questions for retrieval.

Using the conversation context, rewrite the current user message into a
standalone search query.

Rules:
- Preserve the user's original intent.
- Resolve pronouns and implicit references.
- Do not answer the question.
- Do not add information that is not present in the conversation.
- If the question is already standalone, return it unchanged.
- Return only the rewritten query.
```

---

# 5. Vector Retrieval

Use the rewritten query for semantic search.

Flow:

```text
rewritten query
    ↓
embedding model
    ↓
vector DB
    ↓
top 20 candidate chunks
```

Function:

```python
async def retrieve_chunks(
    query: str,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    ...
```

Each returned chunk should contain:

```python
class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict
```

Recommended metadata:

```json
{
  "document_name": "architecture-guide.pdf",
  "page": 12,
  "section": "Caching",
  "source": "uploaded_document"
}
```

---

# 6. Reranking

Retrieve 20 chunks, then rerank them and keep the best 5.

```text
Vector Search
top 20
    ↓
Reranker
    ↓
top 5
```

Create an abstraction:

```python
class Reranker(Protocol):

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        ...
```

Do not tightly couple the application to one reranking provider.

Possible implementations later:

- Cohere Rerank
- cross-encoder
- BGE reranker
- LLM-based reranking

For V1, use whichever reranker is easiest to configure in the repository.

---

# 7. Answer Generation

The answering LLM receives:

```text
system prompt
+
conversation summary
+
last 6 messages
+
top 5 retrieved chunks
+
current user message
```

Important:

Use the **original current user message** for the final answer.

Use the **rewritten query only for retrieval**.

Prompt structure:

```text
SYSTEM

You are a helpful assistant answering questions using a retrieval-augmented
knowledge base.

Use the retrieved context when answering factual questions.

If the retrieved context does not contain enough information, clearly say so.

Cite relevant sources.

Do not invent citations.


CONVERSATION SUMMARY

{summary}


RECENT CONVERSATION

{recent_messages}


RETRIEVED CONTEXT

[1]
Document: {document_name}
Page: {page}
Section: {section}

{chunk_content}

[2]
...


CURRENT USER MESSAGE

{user_message}
```

---

# 8. Citation Format

The backend should maintain structured citations independently from the rendered answer.

Suggested response model:

```python
class Citation(BaseModel):
    index: int
    document_id: str
    document_name: str
    page: int | None = None
    section: str | None = None
    chunk_id: str


class ChatResponse(BaseModel):
    conversation_id: UUID
    message_id: UUID
    answer: str
    citations: list[Citation]
```

The generated text can reference:

```text
Redis can reduce repeated database reads by caching session state [1].
```

The frontend can render `[1]` as an interactive source link.

Never allow the LLM to fabricate document IDs.

Citation metadata must come from retrieved chunks.

---

# 9. Store Conversation

After answer generation succeeds, persist:

```text
User message
Assistant message
```

into PostgreSQL.

Suggested schema:

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    title TEXT,
    summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_messages_conversation_created
ON messages(conversation_id, created_at DESC);
```

Optional:

```sql
ALTER TABLE messages
ADD COLUMN citations JSONB;
```

---

# 10. Conversation Summary Update

Update the rolling conversation summary approximately every 10 messages.

For V1:

```text
if message_count % 10 == 0:
    update_summary()
```

The summarization process should use:

```text
existing summary
+
messages since last summary
```

and create a new summary.

Do not summarize the entire conversation every time.

Suggested conversation fields:

```sql
ALTER TABLE conversations
ADD COLUMN summarized_until_message_id UUID;

ALTER TABLE conversations
ADD COLUMN message_count INTEGER NOT NULL DEFAULT 0;
```

Summary update function:

```python
async def update_conversation_summary(
    conversation_id: UUID,
) -> None:
    ...
```

Suggested prompt:

```text
Update the rolling summary of this conversation.

Preserve:
- important user goals
- important decisions
- architecture choices
- constraints
- unresolved questions
- important entities and references

Remove:
- greetings
- repetition
- irrelevant details
- verbose explanations

Do not invent information.

Existing summary:
{existing_summary}

New messages:
{messages}
```

Output should stay concise.

Target:

```text
300-700 tokens
```

Do not let the summary grow indefinitely.

---

# 11. Summary Update Timing

The answer should not wait for summary generation.

Preferred flow:

```text
generate answer
    ↓
persist user + assistant messages
    ↓
return / stream response to client
    ↓
if summary update required:
    trigger summary update
```

For V1, FastAPI `BackgroundTasks` is acceptable.

Later this can be replaced with:

```text
Celery
RQ
Arq
Kafka worker
RabbitMQ worker
```

Do not introduce a heavy job system in V1 unless one already exists.

---

# 12. Redis Strategy

Redis is an optimization layer.

Suggested keys:

```text
conversation:{conversation_id}:recent
conversation:{conversation_id}:summary
```

Recent messages:

```text
conversation:<uuid>:recent
```

Store at most six recent messages.

TTL example:

```text
30 minutes - 24 hours
```

Conversation summary can also be cached.

Cache miss:

```text
Redis
  ↓ miss
PostgreSQL
  ↓
write Redis
```

Never depend on Redis for durable conversation history.

---

# 13. Backend Folder Structure

Recommended structure:

```text
backend/
│
├── app/
│   │
│   ├── main.py
│   │
│   ├── config.py
│   │
│   ├── dependencies.py
│   │
│   ├── api/
│   │   ├── router.py
│   │   └── routes/
│   │       ├── chat.py
│   │       ├── conversations.py
│   │       └── documents.py
│   │
│   ├── models/
│   │   ├── conversation.py
│   │   ├── message.py
│   │   └── document.py
│   │
│   ├── schemas/
│   │   ├── chat.py
│   │   ├── conversation.py
│   │   └── retrieval.py
│   │
│   ├── services/
│   │   ├── rag_service.py
│   │   ├── conversation_service.py
│   │   ├── query_rewriter.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── llm_service.py
│   │   └── summary_service.py
│   │
│   ├── repositories/
│   │   ├── conversation_repository.py
│   │   ├── message_repository.py
│   │   └── document_repository.py
│   │
│   ├── infrastructure/
│   │   ├── postgres.py
│   │   ├── redis.py
│   │   ├── vector_store.py
│   │   └── llm_client.py
│   │
│   └── prompts/
│       ├── query_rewrite.txt
│       ├── answer.txt
│       └── conversation_summary.txt
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── alembic/
│
├── pyproject.toml
└── README.md
```

Avoid placing all RAG logic inside the FastAPI route.

The route should mainly orchestrate request/response handling.

---

# 14. RAG Service

Create one central service that orchestrates the pipeline.

Example:

```python
class RAGService:

    def __init__(
        self,
        conversation_service,
        query_rewriter,
        retriever,
        reranker,
        llm_service,
        summary_service,
    ):
        self.conversation_service = conversation_service
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.llm_service = llm_service
        self.summary_service = summary_service

    async def chat(
        self,
        conversation_id: UUID,
        user_message: str,
    ):
        recent_messages = (
            await self.conversation_service.get_recent_messages(
                conversation_id,
                limit=6,
            )
        )

        summary = (
            await self.conversation_service.get_summary(
                conversation_id
            )
        )

        rewritten_query = await self.query_rewriter.rewrite(
            current_message=user_message,
            recent_messages=recent_messages,
            conversation_summary=summary,
        )

        candidates = await self.retriever.search(
            query=rewritten_query,
            top_k=20,
        )

        context_chunks = await self.reranker.rerank(
            query=rewritten_query,
            chunks=candidates,
            top_k=5,
        )

        answer = await self.llm_service.answer(
            user_message=user_message,
            summary=summary,
            recent_messages=recent_messages,
            context_chunks=context_chunks,
        )

        await self.conversation_service.save_turn(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=answer.text,
            citations=answer.citations,
        )

        return answer
```

---

# 15. Streaming

Support streaming responses from FastAPI to React.

Recommended endpoint:

```text
POST /api/v1/chat/stream
```

SSE event format:

```text
event: token
data: {"text":"Redis"}

event: token
data: {"text":" can"}

event: token
data: {"text":" be"}

event: citations
data: {"citations":[...]}

event: done
data: {}
```

Do not persist the assistant message token-by-token.

Buffer the generated answer on the backend.

Persist only when generation completes successfully.

If generation fails:

- record an error in logs
- do not save a partial assistant answer as a completed message

---

# 16. API Design

## Create conversation

```text
POST /api/v1/conversations
```

Response:

```json
{
  "id": "uuid",
  "title": null
}
```

## Send chat message

```text
POST /api/v1/chat
```

Request:

```json
{
  "conversation_id": "uuid",
  "message": "What are the limitations?"
}
```

Response:

```json
{
  "conversation_id": "uuid",
  "message_id": "uuid",
  "answer": "...",
  "citations": []
}
```

## Streaming chat

```text
POST /api/v1/chat/stream
```

## Get conversation

```text
GET /api/v1/conversations/{conversation_id}
```

## Get messages

```text
GET /api/v1/conversations/{conversation_id}/messages
```

---

# 17. Frontend Behavior

The React UI should contain:

```text
Conversation sidebar

Main chat area

Message input

Streaming assistant output

Citation/source display
```

When the user sends a message:

```text
React
  ↓
POST /chat/stream
  ↓
display user message immediately
  ↓
consume SSE tokens
  ↓
append assistant tokens
  ↓
render citations
```

The frontend should not implement RAG logic.

All retrieval, rewriting, reranking and prompting belong on the backend.

---

# 18. Logging

Log the RAG pipeline in structured form.

Useful fields:

```json
{
  "conversation_id": "...",
  "request_id": "...",
  "rewrite_latency_ms": 120,
  "retrieval_latency_ms": 80,
  "rerank_latency_ms": 150,
  "generation_latency_ms": 2400,
  "retrieved_chunk_count": 20,
  "final_chunk_count": 5
}
```

Do not log:

- API keys
- secrets
- full sensitive documents
- authentication tokens

In development, optionally log the rewritten query.

---

# 19. Error Handling

Create explicit error types.

Examples:

```python
class ConversationNotFoundError(Exception):
    pass


class RetrievalError(Exception):
    pass


class RerankerError(Exception):
    pass


class LLMGenerationError(Exception):
    pass
```

Expected behavior:

```text
Vector retrieval failure
→ return controlled server error

Reranker failure
→ optionally fall back to top 5 vector results

Redis failure
→ fall back to PostgreSQL

Summary generation failure
→ log error but do not fail chat request

Citation metadata missing
→ omit invalid citation
```

Redis must never be a single point of failure.

---

# 20. Configuration

Use environment variables.

Example:

```env
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://localhost:6379

LLM_API_KEY=...
LLM_MODEL=...

EMBEDDING_MODEL=...

VECTOR_DB_URL=...

RETRIEVAL_TOP_K=20
RERANK_TOP_K=5

RECENT_MESSAGE_LIMIT=6
SUMMARY_UPDATE_INTERVAL=10
```

Centralize configuration using Pydantic Settings.

---

# 21. Testing Requirements

Implement tests for the following.

## Query rewriting

Given:

```text
User: Tell me about Redis.
Assistant: ...
User: What are its limitations?
```

Expect rewritten query to explicitly contain:

```text
Redis
```

---

## Recent history

Given 20 stored messages:

```text
get_recent_messages(limit=6)
```

must return exactly the latest six in chronological order.

---

## Retrieval

Mock vector DB and verify:

```text
top_k = 20
```

---

## Reranking

Verify:

```text
20 candidates
→ maximum 5 final chunks
```

---

## Summary

Given:

```text
message_count = 10
```

verify that summary update is scheduled.

Given:

```text
message_count = 9
```

verify that it is not.

---

## Cache fallback

Simulate Redis unavailable.

The chat request should still succeed using PostgreSQL.

---

## Citation integrity

Every citation returned to the frontend must correspond to a retrieved chunk.

Never accept arbitrary citation IDs generated by the LLM.

---

# 22. Important Constraints

Do NOT implement the following in V1 unless already present:

```text
agent framework
multi-agent architecture
semantic conversation memory
graph memory
query decomposition
web search routing
tool calling
complex intent router
knowledge graph
long-term user profile memory
```

Keep V1 deliberately simple.

The architecture should make it possible to add them later.

---

# 23. Future Extension Points

Design interfaces so these can be added later:

```text
Hybrid retrieval
BM25 + Vector Search

Semantic conversation memory

Retrieval router

Topic change detection

Query decomposition

Document permission filtering

Conversation memory embeddings

Agent/tool execution

Evaluation pipeline

LLM tracing

Prompt/version management
```

Do not implement them now.

---

# 24. Desired End-to-End Behavior

Example conversation:

```text
User:
What is Redis used for in this architecture?

System:
rewritten query =
"What is Redis used for in a conversational RAG architecture?"

retrieves 20 chunks
reranks to 5

Assistant:
Redis is primarily used as a temporary session and caching layer...
[1][2]
```

Next question:

```text
User:
What are its disadvantages?
```

Recent history + summary identifies `its = Redis`.

Rewrite:

```text
What are the disadvantages of using Redis as a cache and session layer
in a conversational RAG architecture?
```

Retrieve:

```text
20 chunks
```

Rerank:

```text
5 chunks
```

Generate answer with citations.

---

# 25. Final Acceptance Criteria

The V1 implementation is complete when all of the following work:

- [ ] React can create and continue a conversation.
- [ ] FastAPI accepts chat messages.
- [ ] The latest six messages are loaded for each request.
- [ ] A rolling conversation summary is loaded.
- [ ] The query rewriter resolves conversational references.
- [ ] The rewritten query is used for vector retrieval.
- [ ] Vector search returns the top 20 chunks.
- [ ] A reranker reduces them to the top five.
- [ ] The LLM receives summary + recent history + retrieved context.
- [ ] The original user question is used for answer generation.
- [ ] Answers stream to the frontend using SSE.
- [ ] Citations map to real retrieved chunks.
- [ ] User and assistant messages are persisted in PostgreSQL.
- [ ] Redis caches active conversation state.
- [ ] Redis failure does not break the application.
- [ ] Approximately every 10 messages, the rolling summary is updated.
- [ ] Summary updating does not block the chat response.
- [ ] Unit and integration tests cover the critical pipeline.
- [ ] RAG components are separated into services/interfaces and are not embedded directly in API routes.

---

# 26. Implementation Priority for Codex

Implement in this order:

```text
1. PostgreSQL models + migrations
2. conversation/message repositories
3. conversation service
4. vector retriever interface
5. query rewriter
6. reranker
7. answering LLM service
8. RAG service orchestration
9. normal chat endpoint
10. SSE streaming endpoint
11. Redis caching
12. background conversation summarization
13. React integration
14. tests
15. logging and cleanup
```

At each stage:

1. Inspect the existing repository before creating new abstractions.
2. Reuse existing models, clients and configuration where appropriate.
3. Avoid duplicate services.
4. Keep functions small and independently testable.
5. Use async I/O consistently throughout the FastAPI request path.
6. Run tests after each major change.
7. Do not redesign unrelated parts of the repository.

The final implementation should favor **clarity, modularity, and maintainability over premature optimization**.
