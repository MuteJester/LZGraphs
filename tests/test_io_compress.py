from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os

import pytest

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


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_open_text_does_not_leak_fd_when_zstd_unavailable(tmp_path):
    # Skip if zstandard is actually installed; we need the ImportError path.
    try:
        import zstandard  # noqa: F401
        pytest.skip("zstandard is installed; cannot test ImportError path")
    except ImportError:
        pass

    # Write a file with zstd magic bytes.
    path = tmp_path / "fake_zstd.bin"
    path.write_bytes(b"\x28\xb5\x2f\xfdsome data")

    # Count open fds before.
    fd_count_before = len(os.listdir("/proc/self/fd"))

    # Call open_text and catch FormatError.
    with pytest.raises(FormatError):
        open_text(str(path))

    # Count open fds after. Should be unchanged.
    fd_count_after = len(os.listdir("/proc/self/fd"))
    assert fd_count_after == fd_count_before


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
