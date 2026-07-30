"""Defects found in the 3.2 CLI review, locked out permanently.

Before this suite, a FASTA build ingested '>seq10' as a sequence and a CSV
build ingested whole comma-joined rows, both without any error.
"""
from __future__ import annotations

import pytest

from LZGraphs import FlashBackGraph
from LZGraphs._io import read_sequences

FASTA = ">seq10\nCASSLGQAYEQYF\n>seq11\nCASSPGTGVYGYTF\n"
CSV = (
    "sequence_id,v_call,j_call,junction_aa,duplicate_count,productive\n"
    "seq0,IGHV1-2*02,IGHJ4*02,CASSLGQAYEQYF,47,T\n"
    "seq1,IGHV3-7*01,IGHJ6*02,CASSPGTGVYGYTF,12,T\n"
)
SEQCOUNT = "CASSLGQAYEQYF\t47\nCASSPGTGVYGYTF\t12\n"
FORBIDDEN = set(">@+,\t;")


@pytest.mark.parametrize("text,name", [(FASTA, "a.fa"), (CSV, "a.csv")])
def test_no_structural_character_survives_into_sequences(tmp_path, text, name):
    path = tmp_path / name
    path.write_text(text)
    for sequence in read_sequences(str(path))["sequences"]:
        assert not (FORBIDDEN & set(sequence)), f"structural char leaked: {sequence!r}"


def test_fasta_headers_never_become_sequences(tmp_path):
    path = tmp_path / "a.fa"
    path.write_text(FASTA)
    assert read_sequences(str(path))["sequences"] == [
        "CASSLGQAYEQYF",
        "CASSPGTGVYGYTF",
    ]


def test_csv_rows_never_become_sequences(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text(CSV)
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert got["abundances"] == [47, 12]


def test_seqcount_counts_never_become_sequences(tmp_path):
    path = tmp_path / "a.seqcount"
    path.write_text(SEQCOUNT)
    got = read_sequences(str(path))
    # Exact equality catches leaked counts regardless of which characters they contain.
    # A real defect here was iter_seqcount yielding "\t5" as ('5', 1).
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert got["abundances"] == [47, 12]


@pytest.mark.parametrize("text,name", [(FASTA, "a.fa"), (CSV, "a.csv"), (SEQCOUNT, "a.seqcount")])
def test_simulated_sequences_contain_only_residues(tmp_path, text, name):
    path = tmp_path / name
    path.write_text(text)
    graph = FlashBackGraph(read_sequences(str(path))["sequences"])
    for sequence in graph.simulate(25, seed=1).sequences:
        assert not (FORBIDDEN & set(sequence)), f"structural char leaked: {sequence!r}"
