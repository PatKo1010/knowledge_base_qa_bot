import math
import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[3] / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    parsed_sections: list[Section] = []
    heading_stack: list[str] = []
    current_heading = path.stem.replace("_", " ").title()
    current_heading_path = [current_heading]
    current_lines: list[str] = []
    seen_ids: Counter[str] = Counter()

    def flush_current_section() -> None:
        content = "\n".join(current_lines).strip()
        if not content:
            return

        base_slug = slugify(current_heading)
        seen_ids[base_slug] += 1
        slug = base_slug if seen_ids[base_slug] == 1 else f"{base_slug}-{seen_ids[base_slug]}"
        section_text = " ".join(current_heading_path + [content])
        parsed_sections.append(
            Section(
                id=f"{path.name}#{slug}",
                file=path.name,
                heading=current_heading,
                heading_path=current_heading_path.copy(),
                content=content,
                tokens=tokenize(section_text),
            )
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if not match:
            current_lines.append(line)
            continue

        flush_current_section()

        level = len(match.group(1))
        heading = match.group(2).strip().strip("#").strip()
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(heading)
        current_heading = heading
        current_heading_path = heading_stack.copy()
        current_lines = []

    flush_current_section()
    return parsed_sections


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": [section.to_dict() for section in sections],
        "stats": {
            "files_indexed": files_indexed,
            "sections_indexed": len(sections),
            "avg_doc_len": avg_doc_len,
        },
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rebuild_stats() -> None:
    global doc_freq, avg_doc_len, files_indexed

    token_doc_freq: Counter[str] = Counter()
    indexed_files: set[str] = set()
    total_tokens = 0

    for section in sections:
        indexed_files.add(section.file)
        total_tokens += len(section.tokens)
        token_doc_freq.update(set(section.tokens))

    doc_freq = token_doc_freq
    files_indexed = len(indexed_files)
    avg_doc_len = total_tokens / len(sections) if sections else 0.0


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    global sections

    if not index_path.exists():
        return 0, 0

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    sections = [Section(**item) for item in payload["sections"]]
    rebuild_stats()
    return files_indexed, len(sections)


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global sections

    markdown_files = sorted(docs_dir.glob("*.md"))
    sections = [
        section
        for path in markdown_files
        for section in parse_markdown(path)
    ]
    rebuild_stats()
    write_index_json()
    return files_indexed, len(sections)


def bm25_score(query_tokens: list[str], section: Section, k1: float = 1.5, b: float = 0.75) -> float:
    # Score one section for the query using BM25.
    if not query_tokens or not section.tokens or not avg_doc_len:
        return 0.0

    term_freq = Counter(section.tokens)
    doc_count = len(sections)
    section_len = len(section.tokens)
    length_norm = k1 * (1 - b + b * section_len / avg_doc_len)
    heading_tokens = set(tokenize(" ".join(section.heading_path)))
    score = 0.0

    for token in set(query_tokens):
        freq = term_freq.get(token, 0)
        if not freq:
            continue

        idf = math.log(1 + (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
        score += idf * (freq * (k1 + 1)) / (freq + length_norm)
        if token in heading_tokens:
            score += idf * 0.2

    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
