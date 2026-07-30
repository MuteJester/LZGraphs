from __future__ import annotations

import io

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
