"""Offline verification of conversational retrieval and cache failure behavior."""
import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from langchain.schema import Document
from app.chat_pipeline import rewrite_query, LLMReranker
from app.chat_service import stream_turn, citations_for
from app.chat_memory import recent_messages
from app.conversations import ConversationRepository


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_rewrite_resolves_reference_using_memory(self):
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
            content="What are the limitations of Redis?")))
        recent = [{"role": "user", "content": "Tell me about Redis."}]
        with patch("app.chat_pipeline.retrieval.get_llm", return_value=llm):
            result = await rewrite_query("What are its limitations?", recent, "Discussing cache design.")
        self.assertIn("Redis", result)
        supplied = json.loads(llm.ainvoke.call_args.args[0][1].content)
        self.assertEqual(supplied["recent_messages"], recent)
        self.assertEqual(supplied["current_question"], "What are its limitations?")

    async def test_first_question_needs_no_rewrite_call(self):
        with patch("app.chat_pipeline.retrieval.get_llm") as llm:
            self.assertEqual(await rewrite_query("Redis?", [], ""), "Redis?")
        llm.assert_not_called()

    async def test_reranker_selects_real_chunks_and_rejects_fabricated_ids(self):
        chunks = [(Document(page_content=f"Passage {i}"), i / 100) for i in range(20)]
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="[9, 5, 8, 2, 1]")))
        with patch("app.chat_pipeline.retrieval.get_llm", return_value=llm):
            self.assertEqual(await LLMReranker().rerank("query", chunks), [chunks[i] for i in [9, 5, 8, 2, 1]])
            for invalid in ("[100, 2, 3, 4, 5]", "[1, 1, 2, 3, 4]", "not json"):
                llm.ainvoke.return_value.content = invalid
                with self.assertLogs("app.chat_pipeline", level="WARNING"):
                    self.assertEqual(await LLMReranker().rerank("query", chunks), chunks[:5])

    async def test_original_question_answers_rewritten_query_retrieves(self):
        conversation = dict(id="example", summary="Redis cache", message_count=0)
        repository = SimpleNamespace(messages=lambda *args: [], save_turn=lambda *args: [{"id": "u"}, {"id": "a"}])
        chunk = (Document(page_content="Redis uses memory.", metadata={"source": "redis.md"}), 0.1)
        async def tokens(prompt):
            self.assertIn("What are its limitations?", prompt[-1].content)
            self.assertIn("[1] Document: redis.md", prompt[-1].content)
            yield SimpleNamespace(content="Memory constraints [1].")
        with patch("app.chat_service.rewrite_query", new=AsyncMock(return_value="Redis limitations")), \
             patch("app.chat_service.retrieval.retrieve_context", return_value={"fallback": False, "ranked_items": [chunk]}) as retrieve, \
             patch("app.chat_service.retrieval.get_streaming_llm", return_value=SimpleNamespace(astream=tokens)):
            events = [item async for item in stream_turn(repository, conversation, "What are its limitations?")]
        retrieve.assert_called_once_with("Redis limitations", k=20)
        self.assertEqual(events[-1][0], "done")
        self.assertEqual(events[0][1]["citations"][0]["source"], "redis.md")

    async def test_redis_unavailable_falls_back_to_sql(self):
        rows = [{"role": "user", "content": "Redis?"}]
        repository = SimpleNamespace(messages=lambda *args: rows)
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}), \
             patch("redis.asyncio.Redis.from_url", side_effect=ConnectionError("offline")), \
             self.assertLogs("app.chat_memory", level="WARNING"):
            self.assertEqual(await recent_messages(repository, {"id": "x", "message_count": 1}), rows)

    async def test_citation_numbers_are_contiguous_after_missing_metadata(self):
        chunks = [(Document(page_content="Missing source"), 0.1),
                  (Document(page_content="Evidence", metadata={"source": "a.md#heading"}), 0.2)]
        citations = citations_for(chunks)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["index"], 1)
        self.assertTrue(citations[0]["document_id"])


class HistoryOrderTests(unittest.TestCase):
    def test_recent_reply_does_not_reorder_created_history(self):
        with TemporaryDirectory() as temp:
            repository = ConversationRepository(f"sqlite:///{Path(temp) / 'chat.db'}")
            try:
                first = repository.create()
                second = repository.create()
                repository.save_turn(first, "Older chat continued", "Answer", [])
                self.assertEqual([row["id"] for row in repository.list()], [second["id"], first["id"]])
            finally:
                repository.engine.dispose()
