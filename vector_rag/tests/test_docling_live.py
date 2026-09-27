"""Opt-in real Docling inference smoke test; downloads models on first run."""
from io import BytesIO
import os
import unittest

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.pdf_ingestion import extract_pdf, chunk_pages


@unittest.skipUnless(os.getenv("DOCLING_INTEGRATION") == "1", "Set DOCLING_INTEGRATION=1 for model inference")
class DoclingLiveTests(unittest.TestCase):
    def test_real_pdf_to_markdown_with_page_citations(self):
        writer = PdfWriter()
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        for title, body in [("Refund policy", "Refund requests must be made within seven days."),
                            ("Holiday policy", "Employees receive twenty days of holiday each year.")]:
            page = writer.add_blank_page(width=612, height=792)
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
            stream = DecodedStreamObject()
            stream.set_data((f"BT /F1 24 Tf 50 700 Td ({title}) Tj ET\n"
                             f"BT /F1 11 Tf 50 650 Td ({body}) Tj ET\n"
                             f"BT /F1 11 Tf 50 625 Td ({body}) Tj ET").encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
        output = BytesIO()
        writer.write(output)
        pages, count, skipped = extract_pdf(output.getvalue(), "policies.pdf", "live-test")
        self.assertEqual(count, 2)
        self.assertEqual(skipped, [])
        self.assertIn("seven days", pages[0].page_content)
        self.assertIn("twenty days", pages[1].page_content)
        chunks = chunk_pages(pages)
        self.assertEqual({chunk.metadata["page"] for chunk in chunks}, {1, 2})
        self.assertTrue(any("header_1" in chunk.metadata or "header_2" in chunk.metadata for chunk in chunks))
        self.assertTrue(all(chunk.metadata["pipeline_version"] == "docling-markdown-v2" for chunk in chunks))
