import zlib
from typing import Iterator, TextIO

def exact_div(a, b):
    assert a % b == 0
    return a // b

def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))



