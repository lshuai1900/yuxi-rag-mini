from app.rag.parser.base import BaseParser


class TextParser(BaseParser):
    async def parse(self, file_path: str) -> str:
        with open(file_path, encoding="utf-8") as f:
            return f.read()

    def supported_extensions(self) -> list[str]:
        return [".txt", ".text", ".log", ".csv", ".json", ".xml", ".html", ".htm"]
