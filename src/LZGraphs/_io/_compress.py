"""Transparent decompression driven by magic bytes, not file extensions."""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import sys

from ._spec import FormatError

_MAGIC = (
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
)

_PEEK = 8


def detect_compression(head: bytes) -> str:
    """Classify a codec from the leading bytes of a stream."""
    for magic, name in _MAGIC:
        if head.startswith(magic):
            return name
    return "none"


def _open_zstd(raw):
    try:
        import zstandard
    except ImportError:
        raise FormatError(
            "input is zstd-compressed but the 'zstandard' package is not installed\n"
            "  install it with: pip install zstandard"
        ) from None
    reader = zstandard.ZstdDecompressor().stream_reader(raw)
    return io.TextIOWrapper(reader, encoding="utf-8")


def open_text(path: str):
    """Open ``path`` as text, decompressing if needed.

    Returns ``(stream, codec_name)``. The caller is responsible for closing
    the stream unless it is stdin.
    """
    if path == "-":
        raw = sys.stdin.buffer
    else:
        raw = open(path, "rb")
    if not isinstance(raw, io.BufferedReader):
        raw = io.BufferedReader(raw)

    codec = detect_compression(raw.peek(_PEEK)[:_PEEK])
    if codec == "gzip":
        return gzip.open(raw, "rt", encoding="utf-8"), codec
    if codec == "bzip2":
        return bz2.open(raw, "rt", encoding="utf-8"), codec
    if codec == "xz":
        return lzma.open(raw, "rt", encoding="utf-8"), codec
    if codec == "zstd":
        return _open_zstd(raw), codec
    return io.TextIOWrapper(raw, encoding="utf-8"), codec
