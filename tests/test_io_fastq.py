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
