from __future__ import annotations

import io

from LZGraphs._io._readers import iter_fasta
from LZGraphs._io._sniff import looks_like_fasta

SIMPLE = ">seq1\nCASSLGQAYEQYF\n>seq2\nCASSPGTGVYGYTF\n"
WRAPPED = ">seq1\nCASSLG\nQAYEQYF\n\n>seq2\nCASSPG\n"


def test_looks_like_fasta():
    assert looks_like_fasta([">seq1", "CASSLG"])
    assert looks_like_fasta(["", "  ", ">seq1"])
    assert not looks_like_fasta(["CASSLG", "CASSPG"])
    assert not looks_like_fasta(["@seq1", "ACGT", "+"])


def test_iter_fasta_never_yields_a_header():
    got = list(iter_fasta(io.StringIO(SIMPLE)))
    assert got == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert not any(s.startswith(">") for s in got)


def test_iter_fasta_joins_wrapped_records_and_skips_blanks():
    assert list(iter_fasta(io.StringIO(WRAPPED))) == ["CASSLGQAYEQYF", "CASSPG"]


def test_iter_fasta_handles_final_record_without_trailing_newline():
    assert list(iter_fasta(io.StringIO(">a\nCASS"))) == ["CASS"]


def test_iter_fasta_ignores_semicolon_comments():
    assert list(iter_fasta(io.StringIO(";note\n>a\nCASS\n"))) == ["CASS"]


def test_looks_like_fasta_skips_semicolon_comments():
    # Detector and reader must agree: if iter_fasta accepts it, looks_like_fasta must detect it.
    # This verifies they both skip semicolon-prefixed comments.
    assert looks_like_fasta([";note", ">seq1", "CASS"])


def test_iter_fasta_with_utf8_bom(tmp_path):
    # A FASTA file with UTF-8 BOM written by Windows/Excel must yield clean sequences.
    # This is the same bug that corrupts TSV columns: BOM merges into the first record.
    path = tmp_path / "bom.fasta"
    fasta_with_bom = "﻿>seq1\nCASSLG\n>seq2\nCASSPG\n"
    path.write_bytes(fasta_with_bom.encode("utf-8"))

    from LZGraphs._io._compress import open_text

    stream, codec = open_text(str(path))
    try:
        got = list(iter_fasta(stream))
        assert got == ["CASSLG", "CASSPG"]
        assert not any(">" in s or "﻿" in s for s in got)
    finally:
        stream.close()
