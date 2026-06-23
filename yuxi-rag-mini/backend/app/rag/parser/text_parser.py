from app.rag.parser.base import BaseParser, ParseResult


class TextParser(BaseParser):
    async def parse(self, file_path: str, filename: str = "") -> ParseResult:
        # Try utf-8 first, then utf-8-sig (BOM), then gbk as fallback
        text = None
        for encoding in ["utf-8", "utf-8-sig", "gbk"]:
            try:
                with open(file_path, encoding=encoding) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if text is None:
            raise ValueError(f"Cannot decode {filename or 'text file'} with supported encodings (utf-8, gbk).")

        if not text.strip():
            raise ValueError(f"No extractable content found in {filename or 'text file'}. The file may be empty.")

        return ParseResult(
            text=text,
            filename=filename,
            file_type="txt",
            pages=[],
            metadata={},
        )

    def supported_extensions(self) -> list[str]:
        return [".txt", ".text", ".log", ".csv", ".json", ".xml", ".html", ".htm"]
