from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


REPO_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_DIR / "docs"
INDEX_DIR = REPO_DIR / ".kb" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# TODO: Configure chunking parameters for traditional RAG.
#
# Design decision: Balance semantic recall against context noise.
#
# Hints:
# 1. chunk_size around 500 chars is a reasonable prototype default.
# 2. chunk_overlap helps avoid cutting facts at boundaries.
# 3. separators should prefer Markdown structure before individual words.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " "],
)

vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def get_embeddings():
    global _embeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            request_timeout=20,
            max_retries=1,
        )
    return _embeddings


def load_markdown_sections(path: Path) -> list[Document]:
    documents: list[Document] = []
    heading_stack: list[str] = []
    current_heading: str | None = None
    current_path = ""
    current_lines: list[str] = []

    def flush_section() -> None:
        if current_heading is None:
            return
        content = "\n".join(current_lines).strip()
        if not content:
            return
        documents.append(
            Document(
                page_content=f"{current_path}\n\n{content}",
                metadata={
                    "source": f"{path.name}#{slugify(current_heading)}",
                    "heading": current_path,
                    "file": path.name,
                },
            )
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            del heading_stack[level - 1 :]
            heading_stack.append(heading)
            current_heading = heading
            current_path = " > ".join(heading_stack)
            current_lines = []
        else:
            current_lines.append(line)

    flush_section()
    return documents


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    markdown_files = sorted(docs_dir.glob("*.md"))
    documents: list[Document] = []
    for path in markdown_files:
        documents.extend(load_markdown_sections(path))

    chunks = splitter.split_documents(documents) if documents else []
    vectorstore = FAISS.from_documents(chunks, get_embeddings()) if chunks else None
    files_indexed = len(markdown_files)
    sections_indexed = len(chunks)
    save_vector_index()
    return files_indexed, sections_indexed


def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    if vectorstore is None:
        if sections_indexed == 0 and index_dir.exists():
            shutil.rmtree(index_dir)
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed,
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    index_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    metadata_file = index_dir / "metadata.json"
    if not (index_file.exists() and pkl_file.exists() and metadata_file.exists()):
        return 0, 0

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        return 0, 0

    vectorstore = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    files_indexed = int(metadata.get("files_indexed", 0))
    sections_indexed = int(metadata.get("sections_indexed", 0))
    return files_indexed, sections_indexed


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
