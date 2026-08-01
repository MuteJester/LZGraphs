from __future__ import annotations

import io

import pytest

from LZGraphs._io._readers import iter_plain, iter_seqcount
from LZGraphs._io._sniff import looks_like_seqcount


def test_looks_like_seqcount():
    assert looks_like_seqcount(["CASSLG\t4", "CASSPG\t9"])
    assert not looks_like_seqcount(["CASSLG", "CASSPG"])
    assert not looks_like_seqcount(["junction_aa\tduplicate_count"])


def test_iter_plain_skips_blank_lines():
    assert list(iter_plain(io.StringIO("CASSLG\n\n  \nCASSPG\n"))) == ["CASSLG", "CASSPG"]


def test_iter_seqcount_parses_counts():
    assert list(iter_seqcount(io.StringIO("CASSLG\t4\nCASSPG\t9\n"))) == [
        ("CASSLG", 4),
        ("CASSPG", 9),
    ]


def test_iter_seqcount_defaults_missing_count_to_one():
    assert list(iter_seqcount(io.StringIO("CASSLG\nCASSPG\t3\n"))) == [
        ("CASSLG", 1),
        ("CASSPG", 3),
    ]


@pytest.mark.parametrize(
    "raw,expected_count",
    [
        ("3", 3),
        ("3.0", 3),
        ("1e3", 1000),
        ("3.7", 1),
        ("NA", 1),
        ("-5", 1),
        ("12345678901234567890123", 12345678901234567890123),
        ("9007199254740993.0", 9007199254740993),
    ],
)
def test_iter_seqcount_parses_counts_like_tabular(raw, expected_count):
    """Ensure iter_seqcount uses the same count parsing as iter_tabular_rows."""
    result = list(iter_seqcount(io.StringIO(f"SEQ\t{raw}\n")))
    assert result == [("SEQ", expected_count)]


def test_iter_seqcount_skips_empty_sequence():
    """A record with an empty sequence field is skipped entirely."""
    assert list(iter_seqcount(io.StringIO("\t5\n"))) == []


def test_iter_seqcount_skips_empty_sequence_in_stream():
    """Empty sequence records are skipped without desyncing adjacent records."""
    result = list(iter_seqcount(io.StringIO("CASSLG\t3\n\t5\nCASSPG\t9\n")))
    assert result == [("CASSLG", 3), ("CASSPG", 9)]


def test_iter_plain_strips_tabs():
    # In a plain format with no count column, tabs are just whitespace.
    # A line like "\t5\n" after stripping all whitespace yields "5".
    # This is correct because the plain format has no structural meaning for tabs.
    assert list(iter_plain(io.StringIO("\t5\n"))) == ["5"]
