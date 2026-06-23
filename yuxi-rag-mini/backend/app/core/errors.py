from fastapi import HTTPException


class KBNotFoundError(Exception):
    pass


class KBError(Exception):
    pass


class FileNotFoundError(Exception):
    pass


class ParseError(Exception):
    pass


class IndexError(Exception):
    pass
