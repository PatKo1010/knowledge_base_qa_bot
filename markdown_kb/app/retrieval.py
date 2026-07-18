import json
import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer


SYSTEM_PROMPT = """
# TODO: Write the system prompt for the knowledge base Q&A assistant.
#
# Design decision: Hallucination defense for raw Markdown context.
#
# Hints:
# 1. Only answer using the provided CONTEXT.
# 2. Cite only exact source IDs shown in [Source: ...].
#    Each source ID uses filename#heading format.
# 3. Define fallback behavior when the context lacks the answer.
# 4. Explicitly prohibit guessing or outside knowledge.
"""

_llm = None
_streaming_llm = None

MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "1.0"))
UNINDEXED_ANSWER = "The knowledge base has not been indexed yet. Call POST /index first."
FALLBACK_ANSWER = "I cannot confirm from the knowledge base."


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm

def get_streaming_llm():
    global _streaming_llm
    if _streaming_llm is None:
        _streaming_llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            streaming=True,
            request_timeout=20,
            max_retries=1,
        )
    return _streaming_llm


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def build_prompt(query: str, ranked_sections: list) -> str:
    context_parts = []
    for section, _score in ranked_sections:
        heading_path = " > ".join(section.heading_path)
        context_parts.append(
            f"[Source: {section.id}]\n"
            f"Heading path: {heading_path}\n\n"
            f"{section.content}"
        )

    context = "\n\n---\n\n".join(context_parts) if context_parts else "(no context)"
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def build_sources(ranked_sections: list) -> list[dict]:
    return [
        {
            "source": section.id,
            "heading": " > ".join(section.heading_path),
            "score": round(score, 3),
            "content": section.content[:240],
        }
        for section, score in ranked_sections
    ]


def retrieve_context(question: str, k: int = 3) -> dict:
    if not indexer.sections:
        return {
            "fallback": True,
            "answer": UNINDEXED_ANSWER,
            "ranked_items": [],
            "sources": [],
        }

    ranked_sections = indexer.search(question, k=k)
    if not ranked_sections or ranked_sections[0][1] < MIN_RETRIEVAL_SCORE:
        return {
            "fallback": True,
            "answer": FALLBACK_ANSWER,
            "ranked_items": ranked_sections,
            "sources": [],
        }

    return {
        "fallback": False,
        "answer": None,
        "ranked_items": ranked_sections,
        "sources": build_sources(ranked_sections),
    }


def query(question: str) -> dict:
    retrieval = retrieve_context(question)
    if retrieval["fallback"]:
        return {
            "answer": retrieval["answer"],
            "sources": [],
        }

    ranked_sections = retrieval["ranked_items"]
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_sections)),
    ])

    return {
        "answer": response.content,
        "sources": retrieval["sources"],
    }

def stream_query(question: str):
    retrieval = retrieve_context(question)

    if retrieval["fallback"]:
        yield sse_event("sources", {"sources": []})
        yield sse_event("token", {"text": retrieval["answer"]})
        yield sse_event("done", {})
        return

    yield sse_event("sources", {"sources": retrieval["sources"]})

    for chunk in get_streaming_llm().stream([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, retrieval["ranked_items"])),
    ]):
        if chunk.content:
            yield sse_event("token", {"text": chunk.content})

    yield sse_event("done", {})
