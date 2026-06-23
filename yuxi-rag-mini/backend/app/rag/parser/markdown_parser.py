from app.rag.parser.base import BaseParser, ParseResult


class MarkdownParser(BaseParser):
    async def parse(self, file_path: str, filename: str = "") -> ParseResult:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return ParseResult(
            text=text,
            filename=filename,
            file_type="markdown",
            pages=[],
            metadata={},
        )

    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]
