from app.rag.parser.base import BaseParser, ParseResult


class TextParser(BaseParser):
    async def parse(self, file_path: str, filename: str = "") -> ParseResult:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return ParseResult(
            text=text,
            filename=filename,
            file_type="txt",
            pages=[],
            metadata={},
        )

    def supported_extensions(self) -> list[str]:
        return [".txt", ".text", ".log", ".csv", ".json", ".xml", ".html", ".htm"]
