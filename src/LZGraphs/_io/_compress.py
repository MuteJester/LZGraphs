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


class _ClosingTextIOWrapper(io.TextIOWrapper):
    """Text wrapper that also closes an extra underlying handle on close.

    gzip.GzipFile does not close an externally supplied fileobj, so without
    this the raw handle survives the caller closing the stream.
    """

    def __init__(self, buffer, *args, extra_close=None, **kwargs):
        super().__init__(buffer, *args, **kwargs)
        self._extra_close = extra_close

    def close(self):
        try:
            super().close()
        finally:
            extra = self._extra_close
            if extra is not None and not extra.closed:
                extra.close()


def _open_zstd(raw):
    try:
        import zstandard
    except ImportError:
        raise FormatError(
            "input is zstd-compressed but the 'zstandard' package is not installed\n"
            "  install it with: pip install zstandard"
        ) from None
    reader = zstandard.ZstdDecompressor().stream_reader(raw)
    return _ClosingTextIOWrapper(reader, encoding="utf-8-sig", extra_close=raw)


def open_text(path: str):
    """Open ``path`` as text, decompressing if needed.

    Returns ``(stream, codec_name)``. The caller is responsible for closing
    the stream; closing it is always safe, even when path is "-" (stdin).
    """
    if path == "-":
        raw = io.BufferedReader(io.FileIO(sys.stdin.fileno(), mode="rb", closefd=False))
    else:
        raw = open(path, "rb")
    if not isinstance(raw, io.BufferedReader):
        raw = io.BufferedReader(raw)

    try:
        codec = detect_compression(raw.peek(_PEEK)[:_PEEK])
        if codec == "gzip":
            binary = gzip.GzipFile(fileobj=raw)
        elif codec == "bzip2":
            binary = bz2.BZ2File(raw)
        elif codec == "xz":
            binary = lzma.LZMAFile(raw)
        elif codec == "zstd":
            return _open_zstd(raw), codec
        else:
            binary = raw
        return (
            _ClosingTextIOWrapper(binary, encoding="utf-8-sig", extra_close=raw),
            codec,
        )
    except Exception:
        raw.close()
        raise
