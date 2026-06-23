import asyncio
from app.rag.parser.base import BaseParser


class DocxParser(BaseParser):
    async def parse(self, file_path: str) -> str:
        return await asyncio.to_thread(self._parse_sync, file_path)

    def _parse_sync(self, file_path: str) -> str:
        from docx import Document
        document = Document(file_path)
        blocks = []
        for para in document.paragraphs:
            text = para.text.strip()
            if text:
                blocks.append(text)
        for table in document.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            header = rows[0]
            blocks.append(f"| {' | '.join(header)} |")
            blocks.append(f"| {' | '.join(['---'] * len(header))} |")
            for row in rows[1:]:
                normalized = row + [""] * (len(header) - len(row))
                blocks.append(f"| {' | '.join(normalized[:len(header)])} |")
            blocks.append("")
        return "\n\n".join(blocks).strip()

    def supported_extensions(self) -> list[str]:
        return [".docx"]
