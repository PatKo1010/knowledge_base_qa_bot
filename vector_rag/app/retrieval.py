import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer


SYSTEM_PROMPT = """
# TODO: Write the system prompt for the knowledge base Q&A assistant.
#
# Design decision: Hallucination defense for retrieved chunks.
#
# Hints:
# 1. Only answer using the provided CONTEXT.
# 2. Cite only exact source IDs shown in [Source: ...].
#    Each source ID uses filename#heading format.
# 3. Define fallback behavior when the context lacks the answer.
# 4. Explicitly prohibit guessing or outside knowledge.
"""

_llm = None
MAX_RETRIEVAL_DISTANCE = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "0.8"))
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


def build_prompt(query: str, ranked_chunks: list) -> str:
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n"
        f"Heading path: {doc.metadata.get('heading', 'unknown')}\n\n"
        f"{doc.page_content}"
        for doc, _score in ranked_chunks
    )
    return f"CONTEXT:\n{context or '(no context)'}\n\nQUESTION:\n{query}"


def build_sources(ranked_chunks: list) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked_chunks
    ]


def retrieve_context(question: str, k: int = 3) -> dict:
    if indexer.vectorstore is None:
        return {
            "fallback": True,
            "answer": UNINDEXED_ANSWER,
            "ranked_items": [],
            "sources": [],
        }

    ranked_chunks = indexer.search(question, k=k)
    if not ranked_chunks or ranked_chunks[0][1] > MAX_RETRIEVAL_DISTANCE:
        return {
            "fallback": True,
            "answer": FALLBACK_ANSWER,
            "ranked_items": ranked_chunks,
            "sources": [],
        }

    return {
        "fallback": False,
        "answer": None,
        "ranked_items": ranked_chunks,
        "sources": build_sources(ranked_chunks),
    }


def query(question: str) -> dict:
    retrieval = retrieve_context(question)
    if retrieval["fallback"]:
        return {
            "answer": retrieval["answer"],
            "sources": [],
        }

    ranked_chunks = retrieval["ranked_items"]
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_chunks)),
    ])

    return {
        "answer": response.content,
        "sources": retrieval["sources"],
    }
