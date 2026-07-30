from __future__ import annotations

import bz2
import gzip
import lzma

import pytest

from LZGraphs._io._compress import detect_compression, open_text

PAYLOAD = "CASSLGQAYEQYF\nCASSPGTGVYGYTF\n"


def test_detect_compression_by_magic():
    assert detect_compression(gzip.compress(b"x")) == "gzip"
    assert detect_compression(bz2.compress(b"x")) == "bzip2"
    assert detect_compression(lzma.compress(b"x")) == "xz"
    assert detect_compression(b"CASSLG") == "none"


@pytest.mark.parametrize(
    "suffix,compressor",
    [
        ("", lambda b: b),
        (".gz", gzip.compress),
        (".bz2", bz2.compress),
        (".xz", lzma.compress),
    ],
)
def test_open_text_roundtrips_every_codec(tmp_path, suffix, compressor):
    path = tmp_path / f"seqs.txt{suffix}"
    path.write_bytes(compressor(PAYLOAD.encode()))
    stream, codec = open_text(str(path))
    try:
        assert stream.read() == PAYLOAD
    finally:
        stream.close()
    assert codec == {"": "none", ".gz": "gzip", ".bz2": "bzip2", ".xz": "xz"}[suffix]


def test_open_text_ignores_a_lying_extension(tmp_path):
    # A gzip file misnamed .txt must still decompress.
    path = tmp_path / "actually_gzip.txt"
    path.write_bytes(gzip.compress(PAYLOAD.encode()))
    stream, codec = open_text(str(path))
    try:
        assert stream.read() == PAYLOAD
    finally:
        stream.close()
    assert codec == "gzip"
