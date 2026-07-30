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
