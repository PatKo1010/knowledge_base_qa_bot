from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid

import faiss
import os
import re
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .pdf_ingestion import PDF_PIPELINE_VERSION, chunk_documents, chunk_pages, extract_pdf


REPO_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_DIR / "docs"
INDEX_DIR = REPO_DIR / ".kb" / "faiss_index"
DOCUMENTS_DIR = REPO_DIR / ".kb" / "vector_documents"
_WRITE_LOCK = threading.RLock()
documents_indexed: dict = {}
EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

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


def chunk_manifest(store: FAISS | None = None) -> list[dict]:
    active_store = store if store is not None else vectorstore
    if active_store is None:
        return []
    return [
        {
            "vector_id": vector_id,
            "content": document.page_content,
            "metadata": document.metadata,
        }
        for vector_id, document in active_store.docstore._dict.items()
    ]


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


def _commit(store: FAISS | None, files: int, chunks: int, registry: dict,
            index_dir: Path | None = None) -> None:
    """Publish a complete snapshot with one atomic pointer replacement.

    The previous snapshot remains available to readers and for recovery. Run one
    backend worker: the lock and live vectorstore are process-local.
    """
    global vectorstore, files_indexed, sections_indexed, documents_indexed
    index_dir = index_dir or INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    snapshot = index_dir / generation
    snapshot.mkdir()
    pointer = index_dir / f".current-{generation}"
    try:
        if store is not None:
            store.save_local(str(snapshot))
        (snapshot / "chunks.json").write_text(
            json.dumps(chunk_manifest(store), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (snapshot / "metadata.json").write_text(json.dumps({
            "embedding_model": EMBEDDING_MODEL,
            "files_indexed": files, "sections_indexed": chunks,
            "documents": registry,
        }, indent=2), encoding="utf-8")
        pointer.write_text(generation, encoding="utf-8")
        os.replace(pointer, index_dir / "current")
    except Exception:
        pointer.unlink(missing_ok=True)
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
    vectorstore = store
    files_indexed = files
    sections_indexed = chunks
    documents_indexed = registry


def build_index(docs_dir: Path | None = None) -> tuple[int, int]:
    docs_dir = docs_dir or DOCS_DIR
    with _WRITE_LOCK:
        markdown_files = sorted(docs_dir.glob("*.md"))
        documents = []
        for path in markdown_files:
            documents.extend(load_markdown_sections(path))
        chunks = chunk_documents(documents) if documents else []
        registry = {}
        pdf_files = sorted(path for path in (docs_dir / "pdf").glob("*")
                           if path.is_file() and path.suffix.lower() == ".pdf")
        for path in pdf_files:
            data = path.read_bytes()
            document_id = hashlib.sha256(data).hexdigest()
            if document_id in registry:
                continue
            pdf_chunks, info = prepare_pdf(data, path.name, document_id, docs_dir)
            chunks.extend(pdf_chunks)
            registry[document_id] = info
        store = FAISS.from_documents(chunks, get_embeddings()) if chunks else None
        _commit(store, len(markdown_files) + len(registry), len(chunks), registry)
        return files_indexed, sections_indexed


def save_pdf_markdown(document_dir: Path, pages: list[Document],
                      docs_dir: Path | None = None) -> None:
    # Markdown derivatives are inspectable; original.pdf remains the rebuild source.
    (document_dir / "markdown_pages.json").write_text(json.dumps([
        {"page_content": page.page_content, "metadata": page.metadata} for page in pages
    ]), encoding="utf-8")
    markdown = "\n\n".join(
        f"<!-- page: {page.metadata['page']} -->\n\n{page.page_content}" for page in pages
    )
    (document_dir / "document.md").write_text(markdown, encoding="utf-8")
    # Keep generated exports below docs/pdf so docs/*.md does not index them again.
    export_dir = (docs_dir or DOCS_DIR) / "pdf"
    export_dir.mkdir(parents=True, exist_ok=True)
    filename = pages[0].metadata["file"]
    stem = re.sub(r"[^\w.-]+", "_", Path(filename).stem).strip("._")[:80] or "document"
    target = export_dir / f"{stem}-{document_dir.name}.md"
    temporary = export_dir / f".{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(markdown, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def save_pdf_chunks(document_dir: Path, chunks: list[Document]) -> None:
    """Persist the exact chunk text and metadata used for embedding."""
    payload = [
        {
            "chunk_index": chunk.metadata.get("chunk_index"),
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    (document_dir / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_pdf(data: bytes, filename: str, document_id: str,
                docs_dir: Path | None = None) -> tuple[list[Document], dict]:
    """Convert and persist the same PDF artifacts for uploads and rebuilds."""
    pages, page_count, skipped = extract_pdf(data, filename, document_id)
    chunks = chunk_pages(pages)
    info = {"document_id": document_id, "filename": filename, "pages": page_count,
            "chunks_added": len(chunks), "skipped_pages": skipped,
            "status": "reindexed" if document_id in documents_indexed else "indexed",
            "pipeline_version": PDF_PIPELINE_VERSION}
    document_dir = DOCUMENTS_DIR / document_id
    document_dir.mkdir(parents=True, exist_ok=True)
    (document_dir / "original.pdf").write_bytes(data)
    save_pdf_markdown(document_dir, pages, docs_dir)
    save_pdf_chunks(document_dir, chunks)
    return chunks, info


def ingest_pdf(data: bytes, filename: str) -> dict:
    document_id = hashlib.sha256(data).hexdigest()
    # Never use the client filename as a filesystem path.
    filename = filename.replace("\\", "/").split("/")[-1]
    filename = "".join(c for c in filename if c.isprintable())[:180] or "document.pdf"
    with _WRITE_LOCK:
        was_indexed = document_id in documents_indexed
        chunks, info = prepare_pdf(data, filename, document_id)
        # Embedding and persistence failures must not mutate the serving index.
        addition = FAISS.from_documents(chunks, get_embeddings())
        candidate = addition
        if vectorstore is not None:
            candidate = FAISS(
                embedding_function=get_embeddings(),
                index=faiss.clone_index(vectorstore.index),
                docstore=copy.deepcopy(vectorstore.docstore),
                index_to_docstore_id=vectorstore.index_to_docstore_id.copy(),
            )
            old_ids = [
                vector_id
                for vector_id, document in candidate.docstore._dict.items()
                if document.metadata.get("document_id") == document_id
            ]
            if old_ids:
                candidate.delete(old_ids)
            candidate.merge_from(addition)
        registry = {**documents_indexed, document_id: info}
        old_chunk_count = sum(
            1 for document in (vectorstore.docstore._dict.values() if vectorstore else [])
            if document.metadata.get("document_id") == document_id
        )
        file_count = files_indexed if was_indexed else files_indexed + 1
        chunk_count = sections_indexed - old_chunk_count + len(chunks)
        _commit(candidate, file_count, chunk_count, registry)
        return info


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed, documents_indexed
    with _WRITE_LOCK:
        pointer = index_dir / "current"
        snapshot = index_dir
        if pointer.exists():
            generation = pointer.read_text(encoding="utf-8").strip()
            if not re.fullmatch(r"[a-f0-9]{32}", generation):
                raise RuntimeError("Invalid index snapshot pointer")
            snapshot = index_dir / generation
        metadata_file = snapshot / "metadata.json"
        if not metadata_file.exists():
            if pointer.exists():
                raise RuntimeError("Index snapshot metadata is missing")
            return 0, 0
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("embedding_model") != EMBEDDING_MODEL:
            raise RuntimeError("Saved embedding model does not match the configured model")
        store = None
        if int(metadata.get("sections_indexed", 0)):
            store = FAISS.load_local(str(snapshot), get_embeddings(), allow_dangerous_deserialization=True)
        vectorstore = store
        files_indexed = int(metadata.get("files_indexed", 0))
        sections_indexed = int(metadata.get("sections_indexed", 0))
        documents_indexed = metadata.get("documents", {})
        return files_indexed, sections_indexed


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    store = vectorstore
    if store is None:
        return []
    return store.similarity_search_with_score(query, k=k)
