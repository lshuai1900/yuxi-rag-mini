from pathlib import Path
from app.rag.parser.pdf_parser import PDFParser
from app.rag.parser.docx_parser import DocxParser
from app.rag.parser.markdown_parser import MarkdownParser
from app.rag.parser.text_parser import TextParser
from app.core.logging import logger


_parsers = {}


def _init_parsers():
    global _parsers
    if _parsers:
        return
    for parser_cls in [PDFParser, DocxParser, MarkdownParser, TextParser]:
        instance = parser_cls()
        for ext in instance.supported_extensions():
            _parsers[ext] = instance


def get_parser(file_path: str):
    _init_parsers()
    ext = Path(file_path).suffix.lower()
    parser = _parsers.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser


async def parse_file(file_path: str) -> str:
    parser = get_parser(file_path)
    content = await parser.parse(file_path)
    logger.info(f"Parsed {file_path}: {len(content)} chars")
    return content
