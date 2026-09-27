"""PDF -> Docling Markdown -> heading sections -> bounded overlapping chunks."""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import re
from uuid import uuid4

import tiktoken
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 200
MAX_TEXT_CHARS = 2_000_000
PDF_PIPELINE_VERSION = "docling-markdown-v2"
CHUNK_TARGET_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 75

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#" * level, f"header_{level}") for level in range(1, 7)],
    strip_headers=False,
)
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_SENTENCE_RE = re.compile(r"(?<=[。！？；])(?:\s+|(?=[^\s]))|(?<=[.!?;])\s+")
_TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
_LAW_HEADING_RE = re.compile(r"^[一二三四五六七八九十百千零〇]+、\s*\S")


def fix_taiwan_law_headings(text: str) -> str:
    """Normalize Chinese-numbered provisions to H2, preserving fenced code."""
    lines = []
    fence_char = ""
    fence_length = 0
    for line in text.split("\n"):
        stripped = line.strip()
        fence = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)
        if fence:
            marker, suffix = fence.groups()
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length and not suffix.strip():
                fence_char = ""
            lines.append(line)
            continue
        # Docling may classify a numbered provision as a Markdown list item.
        candidate = _LIST_LINE_RE.sub("", stripped, count=1)
        candidate = re.sub(r"^#{1,6}\s+", "", candidate, count=1)
        if not fence_char and _LAW_HEADING_RE.match(candidate):
            lines.append(f"## {candidate}")
        else:
            lines.append(line)
    return "\n".join(lines)


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
        section_chunks: list[list[tuple[str, list[int]]]] = []
        current: list[tuple[str, list[int]]] = []
        current_tokens = 0
        cursor = 0
        page_spans = document.metadata.get("_page_spans", [])

        for unit in units:
            start = document.page_content.index(unit, cursor)
            cursor = start + len(unit)
            unit_pages = sorted({page for left, right, page in page_spans
                                 if left < cursor and right > start})
            unit_tokens = token_count(unit)
            if current and current_tokens + unit_tokens > target_tokens:
                section_chunks.append(current)
                overlap: list[tuple[str, list[int]]] = []
                current_overlap_tokens = 0
                for previous in reversed(current):
                    previous_tokens = token_count(previous[0])
                    if current_overlap_tokens + previous_tokens > overlap_target_tokens:
                        break
                    overlap.insert(0, previous)
                    current_overlap_tokens += previous_tokens
                current = overlap
                current_tokens = current_overlap_tokens

            current.append((unit, unit_pages))
            current_tokens += unit_tokens

        if current:
            section_chunks.append(current)

        for chunk_units in section_chunks:
            content = "\n\n".join(unit for unit, _ in chunk_units)
            metadata = {
                **{key: value for key, value in document.metadata.items() if key != "_page_spans"},
                "section_index": section_index,
                "chunk_index": len(chunks),
            }
            chunk_pages = sorted({page for _, pages in chunk_units for page in pages})
            if chunk_pages:
                metadata.update(page=chunk_pages[0], page_end=chunk_pages[-1], pages=chunk_pages)
                source_base = metadata["source"].rsplit("-page-", 1)[0]
                suffix = str(chunk_pages[0])
                if len(chunk_pages) > 1:
                    suffix += f"-{chunk_pages[-1]}"
                metadata["source"] = f"{source_base}-page-{suffix}"
            chunks.append(Document(page_content=content, metadata=metadata))
    return chunks


@lru_cache(maxsize=1)
def get_converter():
    # Load Docling lazily: Markdown search/startup does not require model downloads.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=False, do_table_structure=True,
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
        generate_parsed_pages=True,  # Retain font/style signals for heading hierarchy inference.
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
        markdown = fix_taiwan_law_headings(markdown)
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
    """Split the whole PDF by headings; page markers track citations only."""
    if not pages:
        return []
    # Unique markers survive the Markdown splitter and are removed before tokenization.
    marker = f"PDF_PAGE_{uuid4().hex}_"
    marker_re = re.compile(re.escape(marker) + r"(\d+)_END")
    markdown = "\n\n".join(
        f"{marker}{page.metadata['page']}_END\n\n{page.page_content}" for page in pages
    )
    sections = []
    current_page = pages[0].metadata["page"]
    for section in header_splitter.split_text(markdown):
        parts = []
        spans = []
        offset = 0
        cursor = 0
        for match in marker_re.finditer(section.page_content):
            text = section.page_content[cursor:match.start()]
            parts.append(text)
            if text.strip():
                spans.append((offset, offset + len(text), current_page))
            offset += len(text)
            current_page = int(match.group(1))
            cursor = match.end()
        text = section.page_content[cursor:]
        parts.append(text)
        if text.strip():
            spans.append((offset, offset + len(text), current_page))
        content = "".join(parts)
        if not content.strip():
            continue
        metadata = {**pages[0].metadata, **section.metadata, "_page_spans": spans}
        headings = [metadata[f"header_{level}"] for level in range(1, 7)
                    if f"header_{level}" in metadata]
        metadata["heading"] = " > ".join(headings) or "Document"
        sections.append(Document(page_content=content, metadata=metadata))
    return chunk_documents(sections, target_tokens=100, overlap_target_tokens=10)
