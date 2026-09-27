"""Offline ingestion tests with stubbed Docling inference and local embeddings."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json
import sys
from io import BytesIO
from types import SimpleNamespace, ModuleType
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain.schema import Document

from app.pdf_ingestion import chunk_documents, chunk_pages, token_count, fix_taiwan_law_headings

from app import indexer
from app.routes import router


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float("refund" in text.lower()), float("holiday" in text.lower()), 0.1]


def pdf_bytes(texts, encrypted=False):
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("secret")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def fake_convert(source, **kwargs):
    pages = [page.extract_text() or "" for page in PdfReader(source.stream).pages]
    return SimpleNamespace(status="success", document=SimpleNamespace(
        export_to_markdown=lambda *, page_no, image_placeholder: pages[page_no - 1],
    ))


class PdfUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index_dir = self.root / "index"
        self.docs = self.root / "docs"
        self.docs.mkdir()
        for name, value in {
            "INDEX_DIR": self.index_dir, "DOCUMENTS_DIR": self.root / "pdfs",
            "DOCS_DIR": self.docs,
            "vectorstore": None, "files_indexed": 0, "sections_indexed": 0,
            "documents_indexed": {},
        }.items():
            patcher = patch.object(indexer, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(indexer, "get_embeddings", return_value=LocalEmbeddings())
        patcher.start()
        self.addCleanup(patcher.stop)
        # Inference is stubbed, so offline tests do not need the Docling package.
        base_models = ModuleType("docling.datamodel.base_models")
        base_models.ConversionStatus = SimpleNamespace(SUCCESS="success")
        base_models.DocumentStream = SimpleNamespace
        models_patch = patch.dict(sys.modules, {"docling.datamodel.base_models": base_models})
        models_patch.start()
        self.addCleanup(models_patch.stop)
        converter_patch = patch("app.pdf_ingestion.get_converter")
        self.converter = converter_patch.start().return_value
        self.converter.convert.side_effect = fake_convert
        self.addCleanup(converter_patch.stop)
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def upload(self, data, filename="policy.pdf"):
        return self.client.post("/documents/upload", files={"file": (filename, data, "application/pdf")})

    def test_search_duplicate_restart_and_rebuild(self):
        data = pdf_bytes(["# Refunds\nRefunds take seven days.", "", "# Holidays\nHoliday allowance is twenty days."])
        response = self.upload(data, "../../policy.pdf")
        self.assertEqual(response.status_code, 200, response.text)
        info = response.json()
        self.assertEqual(info["filename"], "policy.pdf")
        self.assertEqual(info["skipped_pages"], [2])
        match, _ = indexer.search("holiday", 1)[0]
        self.assertEqual(match.metadata["page"], 3)
        self.assertIn("page-3", match.metadata["source"])
        count = indexer.sections_indexed
        self.assertEqual(self.upload(data).json()["status"], "reindexed")
        self.assertEqual(indexer.sections_indexed, count)
        indexer.vectorstore = None
        indexer.documents_indexed = {}
        indexer.load_vector_index(self.index_dir)
        reindexed = self.upload(data).json()
        self.assertEqual(reindexed["status"], "reindexed")
        self.assertGreater(reindexed["chunks_added"], 0)
        self.docs.joinpath("faq.md").write_text("# Refund\nRefund policy details")
        files, chunks = indexer.build_index(self.docs)
        self.assertEqual((files, chunks), (1, 1))
        self.assertEqual(indexer.documents_indexed, {})
        self.assertTrue(all("document_id" not in doc.metadata
                            for doc in indexer.vectorstore.docstore._dict.values()))
        indexer.load_vector_index(self.index_dir)
        self.assertEqual(indexer.documents_indexed, {})
        self.assertEqual(indexer.sections_indexed, 1)
        self.assertTrue((indexer.DOCUMENTS_DIR / info["document_id"] / "original.pdf").exists())
        # A previously removed PDF can still be explicitly uploaded again.
        self.assertEqual(self.upload(data).json()["status"], "indexed")
        self.assertEqual(indexer.sections_indexed, count + 1)

    def test_index_pdf_pipeline_matches_upload_and_survives_restart(self):
        data = pdf_bytes(["# Refunds\n\n" + "Refund details. " * 180,
                          "# Holidays\n\nHoliday allowance is twenty days."])
        upload_response = self.upload(data)
        self.assertEqual(upload_response.status_code, 200, upload_response.text)
        uploaded = upload_response.json()
        upload_chunks = [(c.page_content, c.metadata) for c in
                         indexer.vectorstore.docstore._dict.values()]
        self.assertEqual([metadata["chunk_index"] for _, metadata in upload_chunks],
                         list(range(len(upload_chunks))))
        pdf_dir = self.docs / "pdf"
        (pdf_dir / "policy.pdf").write_bytes(data)
        (pdf_dir / "duplicate.PDF").write_bytes(data)
        self.docs.joinpath("faq.md").write_text("# FAQ\nGeneral information")
        response = self.client.post("/index")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"files_indexed": 2,
                                          "sections_indexed": len(upload_chunks) + 1})
        chunks = [c for c in indexer.vectorstore.docstore._dict.values()
                  if c.metadata.get("document_id") == uploaded["document_id"]]
        self.assertEqual([c.metadata["chunk_index"] for c in chunks], list(range(len(chunks))))
        self.assertEqual([c.page_content for c in chunks], [c[0] for c in upload_chunks])
        self.assertEqual({c.metadata["page"] for c in chunks}, {1, 2})
        artifact = indexer.DOCUMENTS_DIR / uploaded["document_id"]
        saved = json.loads((artifact / "chunks.json").read_text())
        self.assertEqual([c["metadata"] for c in saved], [c.metadata for c in chunks])
        indexer.load_vector_index(self.index_dir)
        self.assertEqual(len(indexer.documents_indexed), 1)
        self.assertEqual(self.client.post("/index").json(), response.json())
        self.assertEqual(self.upload(data).json()["status"], "reindexed")

    def test_failed_pdf_rebuild_preserves_snapshot(self):
        self.upload(pdf_bytes(["Refund policy"]))
        old_store = indexer.vectorstore
        old_pointer = (self.index_dir / "current").read_text()
        (self.docs / "pdf" / "broken.pdf").write_bytes(b"not a PDF")
        self.assertEqual(self.client.post("/index").status_code, 500)
        self.assertIs(indexer.vectorstore, old_store)
        self.assertEqual((self.index_dir / "current").read_text(), old_pointer)

    def test_invalid_empty_encrypted_and_oversized(self):
        for data in [b"not a pdf", b"%PDF-broken", pdf_bytes([""]), pdf_bytes(["secret"], True)]:
            self.assertEqual(self.upload(data).status_code, 422)
        with patch("app.routes.MAX_UPLOAD_BYTES", 10):
            self.assertEqual(self.upload(b"x" * 11).status_code, 413)
        self.assertIsNone(indexer.vectorstore)

    def test_failures_preserve_live_and_persisted_index(self):
        self.upload(pdf_bytes(["Refund policy"]))
        old_store = indexer.vectorstore
        old_pointer = (self.index_dir / "current").read_text()
        for target in ["app.indexer.FAISS.from_documents", "app.indexer.os.replace"]:
            with patch(target, side_effect=RuntimeError("simulated failure")):
                with self.assertLogs("app.routes", level="ERROR"):
                    response = self.upload(pdf_bytes(["Holiday policy"]))
                self.assertEqual(response.status_code, 500)
            self.assertIs(indexer.vectorstore, old_store)
            self.assertEqual((self.index_dir / "current").read_text(), old_pointer)
            self.assertEqual(len(indexer.documents_indexed), 1)
        indexer.load_vector_index(self.index_dir)
        self.assertEqual(indexer.sections_indexed, 1)
        self.assertEqual(self.upload(pdf_bytes(["Holiday policy"])).status_code, 200)

    def test_concurrent_duplicate_uploads_commit_once(self):
        data = pdf_bytes(["Refund policy"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: indexer.ingest_pdf(data, "policy.pdf"), range(2)))
        self.assertEqual(sorted(result["status"] for result in results), ["indexed", "reindexed"])
        self.assertEqual(indexer.sections_indexed, 1)

    def test_legacy_index_can_load_and_migrate(self):
        self.upload(pdf_bytes(["Refund policy"]))
        legacy = self.root / "legacy"
        indexer.vectorstore.save_local(str(legacy))
        (legacy / "metadata.json").write_text(
            '{"embedding_model":"text-embedding-3-small","files_indexed":1,"sections_indexed":1}')
        indexer.load_vector_index(legacy)
        self.assertEqual(indexer.sections_indexed, 1)
        self.assertEqual(indexer.documents_indexed, {})
        self.assertEqual(indexer.search("refund", 1)[0][0].metadata["page"], 1)

    def test_markdown_heading_boundaries_and_metadata(self):
        page = Document(page_content=(
            "Preamble stays.\n\n# Policy\n\n## Refunds\n\n" + "Refund details. " * 180
            + "\n\n## Holidays\n\nHoliday allowance is twenty days."
        ), metadata={"page": 4, "heading": "Page 4", "source": "policy.pdf#page-4"})
        chunks = chunk_pages([page])
        self.assertEqual(chunks[0].page_content, "Preamble stays.")
        refund_chunks = [c for c in chunks if c.metadata.get("header_2") == "Refunds"]
        self.assertGreater(len(refund_chunks), 1)
        self.assertTrue(all(c.metadata["heading"] == "Policy > Refunds" for c in refund_chunks))
        self.assertTrue(all("Holiday allowance" not in c.page_content for c in refund_chunks))
        self.assertEqual(chunks[-1].metadata["heading"], "Policy > Holidays")
        self.assertTrue(all(c.metadata["page"] == 4 for c in chunks))
        self.assertTrue(all(len(c.page_content) <= 1000 for c in chunks))

    def test_heading_section_can_cross_pages(self):
        pages = [Document(page_content=text, metadata={
            "page": number, "heading": f"Page {number}", "source": f"policy.pdf#abc-page-{number}",
        }) for number, text in [
            (1, "# Policy\n\n## Refunds\n\nRefund requests are accepted."),
            (2, "Submit your receipt.\n\n## Holidays\n\nTwenty days per year."),
        ]]
        chunks = chunk_pages(pages)
        refund = next(c for c in chunks if c.metadata.get("header_2") == "Refunds")
        self.assertIn("Submit your receipt.", refund.page_content)
        self.assertEqual(refund.metadata["heading"], "Policy > Refunds")
        self.assertEqual(refund.metadata["pages"], [1, 2])
        self.assertEqual(refund.metadata["source"], "policy.pdf#abc-page-1-2")
        holiday = next(c for c in chunks if c.metadata.get("header_2") == "Holidays")
        self.assertEqual(holiday.metadata["pages"], [2])
        self.assertNotIn("Submit your receipt.", holiday.page_content)
        self.assertTrue(all("PDF_PAGE_" not in c.page_content for c in chunks))
        self.assertTrue(all("_page_spans" not in c.metadata for c in chunks))

    def test_long_cross_page_section_keeps_heading_and_actual_chunk_pages(self):
        pages = [Document(page_content=text, metadata={
            "page": number, "heading": f"Page {number}", "source": f"policy.pdf#abc-page-{number}",
        }) for number, text in [
            (1, "## Refunds\n\n" + "Refunds require receipts. " * 100),
            (3, "Submit the application. " * 100),
        ]]
        chunks = chunk_pages(pages)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(c.metadata["heading"] == "Refunds" for c in chunks))
        self.assertTrue(all(c.metadata["section_index"] == 0 for c in chunks))
        self.assertEqual(chunks[0].metadata["pages"], [1])
        self.assertEqual(chunks[-1].metadata["pages"], [3])
        self.assertTrue(any(c.metadata["pages"] == [1, 3] for c in chunks))
        self.assertTrue(all(2 not in c.metadata["pages"] for c in chunks))

    def test_document_without_headings_can_cross_pages(self):
        pages = [Document(page_content=f"Text on page {number}.", metadata={
            "page": number, "source": f"policy.pdf#abc-page-{number}", "heading": f"Page {number}",
        }) for number in [1, 2]]
        chunks = chunk_pages(pages)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["pages"], [1, 2])
        self.assertEqual(chunks[0].metadata["heading"], "Document")

    def test_docling_headings_are_split_and_persisted(self):
        data = pdf_bytes(["# Company Policies\n\n## Refund Rules\n\nRefunds take seven days."
                          "\n\n## Holiday Rules\n\nHoliday allowance is twenty days."])
        response = self.upload(data)
        self.assertEqual(response.status_code, 200, response.text)
        document_dir = indexer.DOCUMENTS_DIR / response.json()["document_id"]
        markdown = (document_dir / "document.md").read_text()
        exports = list((self.docs / "pdf").glob("*.md"))
        self.assertEqual(len(exports), 1)
        self.assertEqual(exports[0].read_text(), markdown)
        self.assertIn(response.json()["document_id"], exports[0].name)
        self.assertIn("# Company Policies", markdown)
        self.assertIn("## Refund Rules", markdown)
        chunks = list(indexer.vectorstore.docstore._dict.values())
        self.assertTrue(any(c.metadata["heading"] == "Company Policies > Refund Rules" for c in chunks))
        self.assertTrue(any(c.metadata["heading"] == "Company Policies > Holiday Rules" for c in chunks))
        counts = indexer.build_index(self.docs)
        self.assertEqual(counts, (0, 0))  # docs/pdf exports are excluded.
        self.assertIsNone(indexer.vectorstore)
        self.assertEqual(indexer.documents_indexed, {})
        self.assertEqual(exports[0].read_text(), markdown)
        self.assertTrue((document_dir / "original.pdf").exists())

    def test_taiwan_law_heading_normalization(self):
        text = ("# 作業要點\n一、目的\n  十一、申請資格\n二十六、評審基準\n"
                "## 二、既有標題\n（一）細項\n內文提到三、其他規定\n"
                "| 四、表格內容 | 說明 |\n```text\n五、程式碼\n```\n"
                "~~~\n六、範例\n~~~\n七、附則")
        expected = text.replace("\n一、", "\n## 一、").replace(
            "\n  十一、", "\n## 十一、").replace(
            "\n二十六、", "\n## 二十六、").replace("\n七、", "\n## 七、")
        self.assertEqual(fix_taiwan_law_headings(text), expected)
        self.assertEqual(fix_taiwan_law_headings(expected), expected)

    def test_taiwan_law_headings_are_persisted_and_chunked(self):
        markdown = "# 作業要點\n\n- 一、目的\n\n支持文化發展。\n\n### 十一、評審基準\n\n依計畫內容評分。"
        self.converter.convert.side_effect = lambda *args, **kwargs: SimpleNamespace(
            status="success", document=SimpleNamespace(export_to_markdown=lambda **kwargs: markdown))
        response = self.upload(pdf_bytes(["Law document"]))
        self.assertEqual(response.status_code, 200, response.text)
        artifact = indexer.DOCUMENTS_DIR / response.json()["document_id"]
        self.assertIn("## 一、目的", (artifact / "document.md").read_text())
        self.assertIn("\n## 十一、評審基準", (artifact / "document.md").read_text())
        pages = json.loads((artifact / "markdown_pages.json").read_text())
        self.assertIn("\n## 十一、評審基準", pages[0]["page_content"])
        chunks = json.loads((artifact / "chunks.json").read_text())
        headings = [c["metadata"].get("header_2") for c in chunks]
        self.assertIn("一、目的", headings)
        self.assertIn("十一、評審基準", headings)
        self.assertTrue(all(c["metadata"]["page"] == 1 for c in chunks))

    def test_taiwan_law_headings_with_docling_list_prefixes(self):
        for prefix in ("- ", "* ", "+ ", "5. ", "19. ", "2) "):
            with self.subTest(prefix=prefix):
                self.assertEqual(fix_taiwan_law_headings(prefix + "十一、評審基準"),
                                 "## 十一、評審基準")
        unchanged = "- （一）細項\n1. 一般清單\n### 其他標題\n```\n- 六、範例\n```"
        self.assertEqual(fix_taiwan_law_headings(unchanged), unchanged)

    def test_existing_chinese_numbered_headings_normalize_to_h2(self):
        for level in range(1, 7):
            text = "#" * level + " 十、違反本要點規定之處置"
            expected = "## 十、違反本要點規定之處置"
            self.assertEqual(fix_taiwan_law_headings(text), expected)
            self.assertEqual(fix_taiwan_law_headings(expected), expected)
        code = "```markdown\n### 十一、範例\n```"
        self.assertEqual(fix_taiwan_law_headings(code), code)

    def test_docling_failure_does_not_publish_partial_content(self):
        self.converter.convert.side_effect = None
        self.converter.convert.return_value = SimpleNamespace(status="partial_success")
        with self.assertLogs("app.routes", level="ERROR"):
            response = self.upload(pdf_bytes(["Refund policy"]))
        self.assertEqual(response.status_code, 500)
        self.assertIsNone(indexer.vectorstore)
        self.converter.convert.side_effect = RuntimeError("model download unavailable")
        with self.assertLogs("app.routes", level="ERROR"):
            response = self.upload(pdf_bytes(["Refund policy"]))
        self.assertEqual(response.status_code, 500)
        self.assertIsNone(indexer.vectorstore)

    def test_chunk_overlap_and_physical_page_metadata(self):
        response = self.upload(pdf_bytes(["Refund details. " * 180]))
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["chunks_added"], 1)
        chunks = list(indexer.vectorstore.docstore._dict.values())
        self.assertTrue(all(len(chunk.page_content) <= 1000 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["page"] == 1 for chunk in chunks))
        self.assertTrue(any(
            chunks[0].page_content[-size:] == chunks[1].page_content[:size]
            for size in range(20, 151)
        ))

    def test_chinese_boundaries_keep_markdown_blocks_atomic(self):
        section = Document(page_content=(
            "# 退費規定\n\n"
            + "申請人應於期限內提出申請。受理後將進行資料確認。" * 80 + "\n\n"
            "| 費用 | 條件 |\n| --- | --- |\n| 停車費 | 符合資格 |\n\n"
            "- 準備申請書\n- 提供付款證明\n"
        ), metadata={"source": "policy.md#refund", "heading": "退費規定", "page": 2})

        chunks = chunk_documents([section])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(any("| 費用 | 條件 |\n| --- | --- |\n| 停車費 | 符合資格 |" in chunk.page_content
                            for chunk in chunks))
        self.assertTrue(any("- 準備申請書\n- 提供付款證明" in chunk.page_content
                            for chunk in chunks))
        self.assertFalse(any("| 費用 | 條件 |\n| --- | --- |\n| 停車費 |" in chunk.page_content
                           and "| 符合資格 |" not in chunk.page_content for chunk in chunks))
        self.assertTrue(all(chunk.metadata["heading"] == "退費規定" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["page"] == 2 for chunk in chunks))
        self.assertEqual([chunk.metadata["chunk_index"] for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(token_count(chunk.page_content) <= 600 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
