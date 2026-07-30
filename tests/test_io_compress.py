from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os

import pytest

from LZGraphs._io import _compress
from LZGraphs._io._compress import detect_compression, open_text
from LZGraphs._io._spec import FormatError

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


def test_open_text_close_does_not_close_real_stdin(tmp_path, monkeypatch):
    # Write the payload to a temporary file.
    stdin_file = tmp_path / "stdin_mock"
    stdin_file.write_bytes(PAYLOAD.encode())

    # Open the file with a real file handle so it has a fileno().
    with open(stdin_file, "rb") as handle:
        # Wrap it in TextIOWrapper to make it look like sys.stdin.
        fake_stdin = io.TextIOWrapper(handle, encoding="utf-8")

        # Monkeypatch sys.stdin.
        monkeypatch.setattr("sys.stdin", fake_stdin)

        # Call open_text("-").
        stream, codec = open_text("-")

        # Read from it.
        assert stream.read() == PAYLOAD

        # Close the returned stream.
        stream.close()

        # Assert the original handle is still open.
        assert not handle.closed


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
@pytest.mark.parametrize(
    "suffix,compressor",
    [
        ("", lambda b: b),
        (".gz", gzip.compress),
        (".bz2", bz2.compress),
        (".xz", lzma.compress),
    ],
)
def test_open_text_close_releases_the_underlying_fd(tmp_path, suffix, compressor):
    # Verify that closing the returned stream releases the underlying fd.
    path = tmp_path / f"seqs.txt{suffix}"
    path.write_bytes(compressor(PAYLOAD.encode()))

    fd_count_before = len(os.listdir("/proc/self/fd"))
    stream, codec = open_text(str(path))
    stream.read()
    stream.close()
    fd_count_after = len(os.listdir("/proc/self/fd"))

    assert fd_count_after == fd_count_before


def test_open_text_closes_raw_when_codec_construction_fails(tmp_path, monkeypatch):
    # Verify that raw is closed when codec construction raises an exception.
    path = tmp_path / "fake_zstd.bin"
    path.write_bytes(b"\x28\xb5\x2f\xfdsome data")

    # Track which handle was opened via monkeypatching builtins.open.
    opened_handle = None
    original_open = open

    def mock_open_func(p, *args, **kwargs):
        nonlocal opened_handle
        opened_handle = original_open(p, *args, **kwargs)
        return opened_handle

    monkeypatch.setattr("builtins.open", mock_open_func)

    # Monkeypatch _open_zstd to raise FormatError.
    def mock_open_zstd(raw):
        raise FormatError("mocked error")

    monkeypatch.setattr(_compress, "_open_zstd", mock_open_zstd)

    # Call open_text and catch FormatError.
    with pytest.raises(FormatError):
        open_text(str(path))

    # Assert the opened handle is closed.
    assert opened_handle is not None
    assert opened_handle.closed
