from __future__ import annotations

import io

import pytest

from LZGraphs._io._readers import iter_fastq
from LZGraphs._io._sniff import looks_like_fastq
from LZGraphs._io._spec import FormatError

RECORD = "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nTTGGCCAA\n+\nIIIIIIII\n"


def test_looks_like_fastq():
    assert looks_like_fastq(["@r1", "ACGT", "+", "IIII"])
    assert not looks_like_fastq([">r1", "ACGT"])
    assert not looks_like_fastq(["@r1", "ACGT", "ACGT", "ACGT"])


def test_iter_fastq_yields_only_sequence_lines():
    assert list(iter_fastq(io.StringIO(RECORD))) == ["ACGTACGT", "TTGGCCAA"]


def test_iter_fastq_rejects_a_truncated_record():
    with pytest.raises(FormatError, match="FASTQ"):
        list(iter_fastq(io.StringIO("@r1\nACGT\nnot-a-plus\nIIII\n")))


def test_iter_fastq_skips_blank_sequence_lines():
    """A record with blank sequence yields no empty string; matches iter_fasta behavior."""
    result = list(iter_fastq(io.StringIO("@r1\n\n+\nIIIIIIII\n@r2\nTTGGCCAA\n+\nIIIIIIII\n")))
    assert result == ["TTGGCCAA"]
    assert "" not in result


def test_looks_like_fastq_rejects_blank_line_inside_record():
    """Blank line inside a record makes detector say False, matching reader's refusal."""
    assert not looks_like_fastq(["@r1", "", "ACGTACGT", "+", "IIIIIIII"])


def test_iter_fastq_accepts_blank_lines_between_records():
    """Blank lines between records do not affect detection or reading."""
    assert looks_like_fastq(["@r1", "ACGT", "+", "IIII"])
    result = list(iter_fastq(io.StringIO("@r1\nACGT\n+\nIIII\n\n@r2\nTTGG\n+\nIIII\n")))
    assert result == ["ACGT", "TTGG"]


def test_iter_fastq_accepts_truncation_after_separator():
    """A record truncated after separator still yields the sequence."""
    result = list(iter_fastq(io.StringIO("@r1\nACGTACGT\n+\n")))
    assert result == ["ACGTACGT"]
