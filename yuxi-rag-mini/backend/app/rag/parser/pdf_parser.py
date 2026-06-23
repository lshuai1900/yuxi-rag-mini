import asyncio
from app.rag.parser.base import BaseParser, ParseResult


class PDFParser(BaseParser):
    async def parse(self, file_path: str, filename: str = "") -> ParseResult:
        return await asyncio.to_thread(self._parse_sync, file_path, filename)

    def _parse_sync(self, file_path: str, filename: str = "") -> ParseResult:
        try:
            import fitz  # PyMuPDF
            return self._parse_with_fitz(file_path, filename)
        except ImportError:
            # Fallback to pypdf if PyMuPDF is not available
            return self._parse_with_pypdf(file_path, filename)

    def _parse_with_fitz(self, file_path: str, filename: str) -> ParseResult:
        """Parse PDF using PyMuPDF (fitz) - preferred for better quality."""
        import fitz
        doc = fitz.open(file_path)
        pages = []
        all_text_parts = []
        for page in doc:
            page_num = page.number + 1
            text = page.get_text("text")
            if text.strip():
                pages.append({"page_number": page_num, "text": text.strip()})
                all_text_parts.append(text.strip())
        doc.close()

        full_text = "\n\n".join(all_text_parts)
        if not full_text.strip():
            raise ValueError(
                "No extractable text found in PDF. OCR is not enabled in this version."
            )

        return ParseResult(
            text=full_text,
            filename=filename,
            file_type="pdf",
            pages=pages,
            metadata={"total_pages": len(pages), "parser": "pymupdf"},
        )

    def _parse_with_pypdf(self, file_path: str, filename: str) -> ParseResult:
        """Fallback: parse PDF using pypdf."""
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        all_text_parts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_number": i + 1, "text": text.strip()})
                all_text_parts.append(text.strip())
        full_text = "\n\n".join(all_text_parts)

        if not full_text.strip():
            raise ValueError(
                "No extractable text found in PDF. OCR is not enabled in this version."
            )

        return ParseResult(
            text=full_text,
            filename=filename,
            file_type="pdf",
            pages=pages,
            metadata={"total_pages": len(reader.pages), "parser": "pypdf"},
        )

    def supported_extensions(self) -> list[str]:
        return [".pdf"]
