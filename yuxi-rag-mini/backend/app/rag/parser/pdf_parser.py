import asyncio
from app.rag.parser.base import BaseParser


class PDFParser(BaseParser):
    async def parse(self, file_path: str) -> str:
        return await asyncio.to_thread(self._parse_sync, file_path)

    def _parse_sync(self, file_path: str) -> str:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)

    def supported_extensions(self) -> list[str]:
        return [".pdf"]
