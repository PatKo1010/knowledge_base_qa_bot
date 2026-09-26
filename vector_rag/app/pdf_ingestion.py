"""PDF -> Docling Markdown -> heading sections -> bounded overlapping chunks."""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import re

import tiktoken
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 200
MAX_TEXT_CHARS = 2_000_000
PDF_PIPELINE_VERSION = "docling-markdown-v1"
CHUNK_TARGET_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 75

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#" * level, f"header_{level}") for level in range(1, 7)],
    strip_headers=False,
)
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_SENTENCE_RE = re.compile(r"(?<=[。！？；])(?:\s+|(?=[^\s]))|(?<=[.!?;])\s+")
_TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_TOKEN_ENCODER.encode(text))


def _is_table(block: str) -> bool:
    lines = block.splitlines()
    return len(lines) >= 2 and any("|" in line for line in lines) and any(
        re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", line)
        for line in lines
    )


def _is_list(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return bool(lines) and any(_LIST_LINE_RE.match(line) for line in lines)


def _markdown_units(text: str) -> list[str]:
    """Return sentence units while keeping Markdown tables and lists atomic."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    units: list[str] = []
    for block in blocks:
        if _is_table(block) or _is_list(block):
            units.append(block)
            continue
        units.extend(part.strip() for part in _SENTENCE_RE.split(block) if part.strip())
    return units


def chunk_documents(documents: list[Document], target_tokens: int = CHUNK_TARGET_TOKENS,
                    overlap_target_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[Document]:
    """Chunk Markdown sections by tokens without breaking tables or lists."""
    chunks: list[Document] = []
    for section_index, document in enumerate(documents):
        units = _markdown_units(document.page_content)
        section_chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = token_count(unit)
            if current and current_tokens + unit_tokens > target_tokens:
                section_chunks.append("\n\n".join(current))
                overlap: list[str] = []
                current_overlap_tokens = 0
                for previous in reversed(current):
                    previous_tokens = token_count(previous)
                    if current_overlap_tokens + previous_tokens > overlap_target_tokens:
                        break
                    overlap.insert(0, previous)
                    current_overlap_tokens += previous_tokens
                current = overlap
                current_tokens = current_overlap_tokens

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            section_chunks.append("\n\n".join(current))

        for content in section_chunks:
            metadata = {
                **document.metadata,
                "section_index": section_index,
                "chunk_index": len(chunks),
            }
            chunks.append(Document(page_content=content, metadata=metadata))
    return chunks


@lru_cache(maxsize=1)
def get_converter():
    # Load Docling lazily: Markdown search/startup does not require model downloads.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=False, do_table_structure=True,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=2),
    )
    return DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
    })


def extract_pdf(data: bytes, filename: str, document_id: str) -> tuple[list[Document], int, list[int]]:
    """Validate the PDF, then use Docling for page-preserving Markdown conversion."""
    if not data.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a PDF.")
    # Lightweight validation happens before loading Docling's inference models.
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported. Upload an unlocked PDF.")
        page_count = len(reader.pages)
        if page_count > MAX_PAGES:
            raise ValueError(f"PDFs must contain at most {MAX_PAGES} pages.")
        if page_count == 0:
            raise ValueError("PDF contains no pages.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Unable to read this PDF. It may be damaged or unsupported.") from exc

    from docling.datamodel.base_models import ConversionStatus, DocumentStream

    # Infrastructure/model download failures propagate as server errors, not bad PDFs.
    result = get_converter().convert(
        DocumentStream(name=filename, stream=BytesIO(data)),
        max_num_pages=MAX_PAGES, max_file_size=MAX_UPLOAD_BYTES,
        raises_on_error=False,
    )
    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError("Docling could not completely convert the PDF. Please retry.")
    pages = []
    skipped = []
    total_chars = 0
    for number in range(1, page_count + 1):
        markdown = result.document.export_to_markdown(
            page_no=number, image_placeholder="",
        ).replace("\x00", "").strip()
        total_chars += len(markdown)
        if total_chars > MAX_TEXT_CHARS:
            raise ValueError("Converted PDF contains too much text. Split it into smaller documents.")
        if not markdown:
            skipped.append(number)
            continue
        pages.append(Document(page_content=markdown, metadata={
            "document_id": document_id, "file": filename, "page": number,
            "source": f"{filename}#{document_id[:12]}-page-{number}",
            "heading": f"Page {number}", "format": "markdown",
            "pipeline_version": PDF_PIPELINE_VERSION,
        }))
    if not pages:
        raise ValueError("No extractable text found. Scanned PDFs require OCR before upload.")
    return pages, page_count, skipped


def chunk_pages(pages: list[Document]) -> list[Document]:
    sections = []
    for page in pages:
        for section in header_splitter.split_text(page.page_content):
            metadata = {**page.metadata, **section.metadata}
            headings = [metadata[f"header_{level}"] for level in range(1, 7)
                        if f"header_{level}" in metadata]
            metadata["heading"] = " > ".join(headings) or page.metadata["heading"]
            sections.append(Document(page_content=section.page_content, metadata=metadata))
    return chunk_documents(sections, target_tokens=100, overlap_target_tokens=10)
