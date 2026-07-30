"""Every supported format times every supported codec yields the same data."""
from __future__ import annotations

import bz2
import gzip
import lzma

import pytest

from LZGraphs._io import read_sequences

SEQUENCES = ["CASSLGQAYEQYF", "CASSPGTGVYGYTF", "CASSQDSSYEQYF"]
COUNTS = [7, 2, 5]

RENDERERS = {
    "fasta": lambda: "".join(f">s{i}\n{s}\n" for i, s in enumerate(SEQUENCES)),
    "fastq": lambda: "".join(
        f"@s{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(SEQUENCES)
    ),
    "tsv": lambda: "junction_aa\tduplicate_count\n"
    + "".join(f"{s}\t{c}\n" for s, c in zip(SEQUENCES, COUNTS)),
    "csv": lambda: "junction_aa,duplicate_count\n"
    + "".join(f"{s},{c}\n" for s, c in zip(SEQUENCES, COUNTS)),
    "plain": lambda: "".join(f"{s}\n" for s in SEQUENCES),
    "seqcount": lambda: "".join(f"{s}\t{c}\n" for s, c in zip(SEQUENCES, COUNTS)),
}
CODECS = {
    "none": (lambda b: b, ""),
    "gzip": (gzip.compress, ".gz"),
    "bzip2": (bz2.compress, ".bz2"),
    "xz": (lzma.compress, ".xz"),
}
CARRIES_COUNTS = {"tsv", "csv", "seqcount"}


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
@pytest.mark.parametrize("codec", sorted(CODECS))
def test_format_codec_matrix(tmp_path, fmt, codec):
    compress, suffix = CODECS[codec]
    path = tmp_path / f"data_{fmt}{suffix}"
    path.write_bytes(compress(RENDERERS[fmt]().encode()))

    got = read_sequences(str(path))
    assert got["sequences"] == SEQUENCES, f"{fmt}/{codec} changed the sequences"
    expected_counts = COUNTS if fmt in CARRIES_COUNTS else [1] * len(SEQUENCES)
    assert got["abundances"] == expected_counts, f"{fmt}/{codec} changed the counts"


def test_all_formats_agree_on_the_sequence_multiset(tmp_path):
    results = {}
    for fmt, render in RENDERERS.items():
        path = tmp_path / f"agree_{fmt}"
        path.write_text(render())
        results[fmt] = read_sequences(str(path))["sequences"]
    assert len(set(map(tuple, results.values()))) == 1, results
