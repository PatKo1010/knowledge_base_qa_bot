"""Versioned conversation API. Legacy vector and document endpoints stay available."""
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .chat_service import stream_turn, update_summary
from .conversations import ConversationNotFoundError, get_repository
from .retrieval import sse_event

router = APIRouter(prefix="/api/v1")
# One backend worker, matching the existing FAISS writer requirement.
_active_conversations: set[str] = set()


class ConversationStreamingResponse(StreamingResponse):
    """Release a turn even if the client disconnects before iteration starts."""
    def __init__(self, *args, on_close, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_close = on_close

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.on_close()


class ConversationChatRequest(BaseModel):
    conversation_id: UUID | None = None
    message: str = Field(min_length=1, max_length=12000)

    @field_validator("message")
    @classmethod
    def trim_message(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


def find_conversation(repository, conversation_id):
    try:
        return repository.get(str(conversation_id))
    except ConversationNotFoundError as exc:
        raise HTTPException(404, "Conversation not found") from exc


@router.post("/conversations", status_code=201)
def create_conversation(repository=Depends(get_repository)):
    return repository.create()


@router.get("/conversations")
def list_conversations(limit: int = Query(50, ge=1, le=100),
                       offset: int = Query(0, ge=0), repository=Depends(get_repository)):
    return {"conversations": repository.list(limit, offset)}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: UUID, repository=Depends(get_repository)):
    return find_conversation(repository, conversation_id)


@router.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: UUID, limit: int = Query(50, ge=1, le=100),
                 before: int | None = Query(None, ge=1), repository=Depends(get_repository)):
    find_conversation(repository, conversation_id)
    rows = repository.messages(str(conversation_id), limit, before)
    return {"messages": rows, "has_more": bool(rows and rows[0]["sequence"] > 1)}


async def begin_turn(req, repository):
    if req.conversation_id:
        conversation = await run_in_threadpool(find_conversation, repository, req.conversation_id)
    else:
        conversation = await run_in_threadpool(repository.create)
    if conversation["id"] in _active_conversations:
        raise HTTPException(409, "A response is already being generated in this conversation")
    _active_conversations.add(conversation["id"])
    return conversation


@router.post("/chat/stream")
async def chat_stream(req: ConversationChatRequest, background_tasks: BackgroundTasks,
                      repository=Depends(get_repository)):
    conversation = await begin_turn(req, repository)
    released = False

    def release():
        nonlocal released
        if not released:
            _active_conversations.discard(conversation["id"])
            released = True

    async def events():
        try:
            yield sse_event("conversation", {"conversation_id": conversation["id"]})
            async for event, data in stream_turn(repository, conversation, req.message):
                if event == "done" and conversation["message_count"] + 2 - conversation["summarized_count"] >= 10:
                    background_tasks.add_task(update_summary, repository, conversation["id"])
                yield sse_event(event, data)
        finally:
            release()

    return ConversationStreamingResponse(
        events(), on_close=release, media_type="text/event-stream", background=background_tasks,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat")
async def chat(req: ConversationChatRequest, background_tasks: BackgroundTasks,
               repository=Depends(get_repository)):
    conversation = await begin_turn(req, repository)
    try:
        async for event, data in stream_turn(repository, conversation, req.message):
            if event == "error":
                raise HTTPException(500, data["message"])
            if event == "done":
                if conversation["message_count"] + 2 - conversation["summarized_count"] >= 10:
                    background_tasks.add_task(update_summary, repository, conversation["id"])
                answer = data["messages"][1]
                return {"conversation_id": conversation["id"], "message_id": answer["id"],
                        "answer": answer["content"], "citations": answer["citations"]}
    finally:
        _active_conversations.discard(conversation["id"])
