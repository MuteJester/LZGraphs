from __future__ import annotations

import pytest

from LZGraphs._io._sniff import infer_alphabet, reject_binary
from LZGraphs._io._spec import FormatError, InputSpec


def test_input_spec_is_frozen():
    spec = InputSpec(path="x.txt", format="plain", compression="none")
    assert spec.seq_column is None
    assert spec.alphabet == "ambiguous"
    with pytest.raises(Exception):
        spec.format = "fasta"


def test_reject_binary_names_the_file_and_shows_bytes():
    with pytest.raises(FormatError) as exc:
        reject_binary("PK\x03\x04\x00\x00garbage", "archive.zip")
    message = str(exc.value)
    assert "archive.zip" in message
    assert "binary" in message.lower()


def test_reject_binary_allows_ordinary_text():
    reject_binary("CASSLGQAYEQYF\nCASSPG\n", "seqs.txt")


@pytest.mark.parametrize(
    "samples,expected",
    [
        (["ACGTACGT", "ACGTTTGA"], "nucleotide"),
        (["CASSLGQAYEQYF", "CASSPGTGVYGYTF"], "amino_acid"),
        ([], "ambiguous"),
        (["----"], "ambiguous"),
        (["ACGTRACGT"], "nucleotide"),
        (["ACGTNNRYKMACGTACGTACGT"], "nucleotide"),
        (["CASSY"], "amino_acid"),
        (["CARDYW"], "amino_acid"),
    ],
)
def test_infer_alphabet(samples, expected):
    assert infer_alphabet(samples) == expected
