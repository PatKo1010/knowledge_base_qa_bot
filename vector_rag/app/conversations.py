"""Durable conversation storage; independent of the FAISS document index."""
from datetime import datetime, timezone
from functools import lru_cache
import os
from uuid import uuid4

from sqlalchemy import (JSON, Column, ForeignKey, Integer, MetaData, String, Table,
                        Text, UniqueConstraint, create_engine, insert, select, update)

from .indexer import REPO_DIR

metadata = MetaData()
conversations = Table(
    "conversations", metadata,
    Column("id", String(36), primary_key=True),
    Column("title", Text, nullable=False),
    Column("summary", Text, nullable=False, default=""),
    Column("summarized_count", Integer, nullable=False, default=0),
    Column("message_count", Integer, nullable=False, default=0),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
)
messages = Table(
    "messages", metadata,
    Column("id", String(36), primary_key=True),
    Column("conversation_id", String(36), ForeignKey("conversations.id"), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("role", String(20), nullable=False),
    Column("content", Text, nullable=False),
    Column("citations", JSON, nullable=False),
    Column("created_at", String(40), nullable=False),
    UniqueConstraint("conversation_id", "sequence"),
)


class ConversationNotFoundError(Exception):
    pass


class ConversationConflictError(Exception):
    pass


class ConversationRepository:
    def __init__(self, url: str):
        self.engine = create_engine(url, pool_pre_ping=True)
        # Idempotent initial schema. Further schema changes require migrations.
        metadata.create_all(self.engine)

    def create(self):
        now = datetime.now(timezone.utc).isoformat()
        row = dict(id=str(uuid4()), title="New conversation", summary="",
                   summarized_count=0, message_count=0, created_at=now, updated_at=now)
        with self.engine.begin() as connection:
            connection.execute(insert(conversations).values(**row))
        return row

    def get(self, conversation_id):
        with self.engine.connect() as connection:
            row = connection.execute(select(conversations).where(
                conversations.c.id == conversation_id)).mappings().first()
        if row is None:
            raise ConversationNotFoundError(conversation_id)
        return dict(row)

    def list(self, limit=50, offset=0):
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(conversations)
                    .order_by(conversations.c.created_at.desc(), conversations.c.id)
                    .limit(limit).offset(offset)).mappings()]

    def messages(self, conversation_id, limit=50, before=None):
        self.get(conversation_id)
        statement = select(messages).where(messages.c.conversation_id == conversation_id)
        if before is not None:
            statement = statement.where(messages.c.sequence < before)
        with self.engine.connect() as connection:
            rows = connection.execute(statement.order_by(messages.c.sequence.desc())
                                      .limit(limit)).mappings().all()
        return [dict(row) for row in reversed(rows)]

    def save_turn(self, conversation, question, answer, citations):
        now = datetime.now(timezone.utc).isoformat()
        count = conversation["message_count"]
        rows = [dict(id=str(uuid4()), conversation_id=conversation["id"],
                     sequence=count + index + 1, role=role, content=content,
                     citations=citations if role == "assistant" else [], created_at=now)
                for index, (role, content) in enumerate((("user", question), ("assistant", answer)))]
        with self.engine.begin() as connection:
            result = connection.execute(update(conversations).where(
                conversations.c.id == conversation["id"], conversations.c.message_count == count
            ).values(message_count=count + 2, updated_at=now,
                     title=question[:80] if count == 0 else conversation["title"]))
            if result.rowcount != 1:
                raise ConversationConflictError("Conversation changed during generation")
            connection.execute(insert(messages), rows)
        return rows

    def summary_input(self, conversation_id):
        conversation = self.get(conversation_id)
        if conversation["message_count"] - conversation["summarized_count"] < 10:
            return None
        with self.engine.connect() as connection:
            rows = connection.execute(select(messages).where(
                messages.c.conversation_id == conversation_id,
                messages.c.sequence > conversation["summarized_count"],
                messages.c.sequence <= conversation["message_count"],
            ).order_by(messages.c.sequence).limit(10)).mappings().all()
        return conversation, [dict(row) for row in rows]

    def save_summary(self, conversation, through, summary):
        with self.engine.begin() as connection:
            connection.execute(update(conversations).where(
                conversations.c.id == conversation["id"],
                conversations.c.summarized_count == conversation["summarized_count"],
            ).values(summary=summary, summarized_count=through))


@lru_cache(maxsize=1)
def get_repository():
    url = os.getenv("DATABASE_URL")
    if not url:
        directory = REPO_DIR / ".kb"
        directory.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{directory / 'conversations.sqlite3'}"
    return ConversationRepository(url)
