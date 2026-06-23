import asyncio
from app.rag.parser.base import BaseParser, ParseResult


class PDFParser(BaseParser):
    async def parse(self, file_path: str, filename: str = "") -> ParseResult:
        return await asyncio.to_thread(self._parse_sync, file_path, filename)

    def _parse_sync(self, file_path: str, filename: str = "") -> ParseResult:
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
        return ParseResult(
            text=full_text,
            filename=filename,
            file_type="pdf",
            pages=pages,
            metadata={"total_pages": len(reader.pages)},
        )

    def supported_extensions(self) -> list[str]:
        return [".pdf"]
