from app.rag.parser.base import BaseParser


class MarkdownParser(BaseParser):
    async def parse(self, file_path: str) -> str:
        with open(file_path, encoding="utf-8") as f:
            return f.read()

    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]
