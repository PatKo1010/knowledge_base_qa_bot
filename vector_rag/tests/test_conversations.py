"""Offline conversation API tests using real SQL storage and stubbed inference."""
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient
from langchain.schema import Document

from app.chat_routes import router, _active_conversations, chat_stream, ConversationChatRequest
from app.chat_service import stream_turn, update_summary
from app.conversations import ConversationRepository, get_repository


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.url = f"sqlite:///{Path(self.temp.name) / 'chat.db'}"
        self.repository = ConversationRepository(self.url)
        self.addCleanup(self.repository.engine.dispose)
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_repository] = lambda: self.repository
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(_active_conversations.clear)
        patcher = patch("app.chat_service.rewrite_query", new_callable=AsyncMock)
        self.rewrite = patcher.start()
        self.rewrite.side_effect = lambda question, recent, summary: question
        self.addCleanup(patcher.stop)
        patcher = patch("app.chat_service.retrieval.get_llm",
                        return_value=SimpleNamespace(model_name="test-rewrite-model"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.context = {"fallback": False, "ranked_items": [(Document(
            page_content="Refunds take seven days.", metadata={"source": "policy.pdf#page-2",
            "file": "policy.pdf", "page": 2, "heading1": "Refunds", "chunk_index": 4,
            "document_id": "document-1"}), 0.2)], "sources": []}
        patcher = patch("app.chat_service.retrieval.retrieve_context", return_value=self.context)
        self.retrieve = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("app.chat_service.retrieval.get_streaming_llm")
        self.llm = patcher.start().return_value
        self.addCleanup(patcher.stop)
        async def tokens(*args, **kwargs):
            yield SimpleNamespace(content="Seven ")
            yield SimpleNamespace(content="days [policy.pdf#page-2].")
        self.llm.astream.side_effect = tokens

    def create(self):
        response = self.client.post("/api/v1/conversations")
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    def test_stream_saves_pair_with_sources_and_reopens_after_restart(self):
        response = self.client.post("/api/v1/chat/stream", json={"message": "How long?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        frames = [frame.splitlines() for frame in response.text.strip().split("\n\n")]
        self.assertEqual([frame[0] for frame in frames], ["event: conversation", "event: citations",
                         "event: token", "event: token", "event: done"])
        done = json.loads(frames[-1][1][6:])
        conversation_id = done["conversation_id"]
        self.retrieve.assert_called_once_with("How long?", k=20)
        rows = self.client.get(f"/api/v1/conversations/{conversation_id}/messages").json()["messages"]
        self.assertEqual([row["role"] for row in rows], ["user", "assistant"])
        self.assertEqual(rows[1]["content"], "Seven days [policy.pdf#page-2].")
        citation = rows[1]["citations"][0]
        self.assertEqual(citation["section"], "Refunds")
        self.assertEqual((citation["page"], citation["chunk_index"], citation["content"]),
                         (2, 4, "Refunds take seven days."))
        fresh = ConversationRepository(self.url)
        self.addCleanup(fresh.engine.dispose)
        self.assertEqual(fresh.messages(conversation_id), rows)
        self.assertEqual(self.client.get("/api/v1/conversations").json()["conversations"][0]["title"], "How long?")

    def test_recent_six_chronological_and_message_pagination(self):
        conversation_id = self.create()
        for index in range(10):
            self.repository.save_turn(self.repository.get(conversation_id), f"question {index}", f"answer {index}", [])
        rows = self.repository.messages(conversation_id, 6)
        self.assertEqual([row["sequence"] for row in rows], list(range(15, 21)))
        page = self.client.get(f"/api/v1/conversations/{conversation_id}/messages?limit=6&before=15").json()
        self.assertEqual([row["sequence"] for row in page["messages"]], list(range(9, 15)))
        self.assertTrue(page["has_more"])
        with patch("app.chat_routes.update_summary", new_callable=AsyncMock):
            response = self.client.post("/api/v1/chat", json={"conversation_id": conversation_id, "message": "Next"})
        self.assertEqual(response.status_code, 200)
        prompt = self.llm.astream.call_args.args[0]
        self.assertEqual([item.content for item in prompt[2:-1]], [row["content"] for row in rows])
        self.retrieve.assert_called_once_with("Next", k=20)

    def test_rewrite_trace_survives_restart_without_replacing_user_message(self):
        conversation_id = self.create()
        prior = self.repository.save_turn(self.repository.get(conversation_id), "Refund policy?", "Seven days.", [])
        self.repository.save_summary(self.repository.get(conversation_id), 2, "Discussing refunds.")
        self.rewrite.side_effect = lambda *args: "Refund application deadline"
        result = self.client.post("/api/v1/chat", json={
            "conversation_id": conversation_id, "message": "What deadline?"}).json()
        trace = result["retrieval_details"]
        self.assertEqual(trace["rewritten_query"], "Refund application deadline")
        self.assertEqual(trace["original_question"], "What deadline?")
        self.assertEqual(trace["recent_message_ids"], [row["id"] for row in prior])
        self.assertEqual(trace["summary"], "Discussing refunds.")
        self.assertEqual(trace["summary_through_sequence"], 2)
        self.assertEqual(trace["rewrite_model"], "test-rewrite-model")
        self.assertEqual(trace["retrieved_candidates"][0]["score"], 0.2)
        self.retrieve.assert_called_once_with("Refund application deadline", k=20)
        self.repository.save_summary(self.repository.get(conversation_id), 4, "A newer summary.")
        fresh = ConversationRepository(self.url)
        self.addCleanup(fresh.engine.dispose)
        rows = fresh.messages(conversation_id)
        self.assertEqual(rows[-2]["content"], "What deadline?")
        self.assertIsNone(rows[-2]["retrieval_details"])
        self.assertEqual(rows[-1]["retrieval_details"], trace)
        prompt = self.llm.astream.call_args.args[0]
        self.assertNotIn("Refund application deadline", "\n".join(m.content for m in prompt))

    def test_first_turn_trace_records_rewrite_skipped(self):
        result = self.client.post("/api/v1/chat", json={"message": "Hello"}).json()
        trace = result["retrieval_details"]
        self.assertEqual(trace["rewrite_status"], "skipped_no_history")
        self.assertIsNone(trace["rewrite_model"])
        self.assertEqual(trace["recent_message_ids"], [])

    def test_existing_database_gets_trace_table_without_losing_messages(self):
        from app.conversations import retrieval_traces
        conversation_id = self.create()
        self.repository.save_turn(self.repository.get(conversation_id), "Old question", "Old answer", [])
        retrieval_traces.drop(self.repository.engine)
        fresh = ConversationRepository(self.url)
        self.addCleanup(fresh.engine.dispose)
        rows = fresh.messages(conversation_id)
        self.assertEqual([row["content"] for row in rows], ["Old question", "Old answer"])
        self.assertTrue(all(row["retrieval_details"] is None for row in rows))

    def test_generation_or_retrieval_error_never_persists_partial_turn(self):
        conversation_id = self.create()
        async def failure(*args):
            yield SimpleNamespace(content="Partial")
            raise RuntimeError("provider failure")
        self.llm.astream.side_effect = failure
        for retrieval_fails in (False, True):
            if retrieval_fails:
                self.retrieve.side_effect = RuntimeError("retrieval failure")
            with self.assertLogs("app.chat_service", level="ERROR"):
                response = self.client.post("/api/v1/chat/stream", json={
                    "conversation_id": conversation_id, "message": "question"})
            self.assertIn("event: error", response.text)
            self.assertNotIn("event: done", response.text)
            self.assertEqual(self.repository.messages(conversation_id), [])
            self.assertNotIn(conversation_id, _active_conversations)

    def test_cancelled_generation_does_not_save(self):
        conversation_id = self.create()
        async def cancel():
            generator = stream_turn(self.repository, self.repository.get(conversation_id), "Question")
            await anext(generator)  # citations
            await anext(generator)  # first token
            await generator.aclose()
        asyncio.run(cancel())
        self.assertEqual(self.repository.messages(conversation_id), [])

    def test_validation_missing_conversation_and_busy_conflict(self):
        self.assertEqual(self.client.post("/api/v1/chat/stream", json={"message": "  "}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/chat/stream", json={"message": "x" * 12001}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/chat/stream", json={
            "message": "Hello", "conversation_id": "00000000-0000-0000-0000-000000000000"}).status_code, 404)
        conversation_id = self.create()
        _active_conversations.add(conversation_id)
        self.assertEqual(self.client.post("/api/v1/chat/stream", json={
            "message": "Hello", "conversation_id": conversation_id}).status_code, 409)

    def test_fallback_is_saved_without_model(self):
        self.retrieve.return_value = {"fallback": True, "answer": "No matching documents.", "ranked_items": []}
        result = self.client.post("/api/v1/chat", json={"message": "Hello"}).json()
        self.assertEqual(result["answer"], "No matching documents.")
        self.assertEqual(result["citations"], [])
        self.llm.astream.assert_not_called()

    def test_summary_is_scheduled_after_fifth_completed_turn(self):
        conversation_id = self.create()
        with patch("app.chat_routes.update_summary", new_callable=AsyncMock) as summary:
            for index in range(4):
                response = self.client.post("/api/v1/chat/stream", json={
                    "conversation_id": conversation_id, "message": f"question {index}"})
                self.assertIn("event: done", response.text)
            summary.assert_not_called()
            self.client.post("/api/v1/chat/stream", json={
                "conversation_id": conversation_id, "message": "question 4"})
            summary.assert_awaited_once_with(self.repository, conversation_id)

    def test_disconnect_before_first_event_releases_conversation(self):
        conversation_id = self.create()
        async def disconnect():
            response = await chat_stream(ConversationChatRequest(
                conversation_id=conversation_id, message="Question"),
                BackgroundTasks(), self.repository)
            async def send(message):
                raise OSError("client disconnected before response headers")
            async def receive():
                await asyncio.Event().wait()
            with self.assertRaises(Exception):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        asyncio.run(disconnect())
        self.assertNotIn(conversation_id, _active_conversations)
        self.assertEqual(self.repository.messages(conversation_id), [])

    def test_summary_interval_and_only_unsummarized_batch(self):
        conversation_id = self.create()
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="A short summary.")))
        with patch("app.chat_service.retrieval.get_llm", return_value=model):
            for index in range(4):
                self.repository.save_turn(self.repository.get(conversation_id), f"q{index}", "answer", [])
            asyncio.run(update_summary(self.repository, conversation_id))
            model.ainvoke.assert_not_called()
            self.repository.save_turn(self.repository.get(conversation_id), "q4", "answer", [])
            asyncio.run(update_summary(self.repository, conversation_id))
            self.assertEqual(self.repository.get(conversation_id)["summarized_count"], 10)
            for index in range(5, 10):
                self.repository.save_turn(self.repository.get(conversation_id), f"q{index}", "answer", [])
            asyncio.run(update_summary(self.repository, conversation_id))
            prompt = model.ainvoke.call_args.args[0][1].content
            self.assertIn("A short summary.", prompt)
            self.assertIn("q5", prompt)
            self.assertNotIn("q4", prompt)
            self.assertEqual(self.repository.get(conversation_id)["summarized_count"], 20)

    def test_summary_failure_does_not_undo_messages(self):
        conversation_id = self.create()
        for index in range(5):
            self.repository.save_turn(self.repository.get(conversation_id), f"q{index}", "answer", [])
        with patch("app.chat_service.retrieval.get_llm", side_effect=RuntimeError("offline")):
            with self.assertLogs("app.chat_service", level="ERROR"):
                asyncio.run(update_summary(self.repository, conversation_id))
        self.assertEqual(self.repository.get(conversation_id)["message_count"], 10)
        self.assertEqual(self.repository.get(conversation_id)["summarized_count"], 0)


if __name__ == "__main__":
    unittest.main()
