from __future__ import annotations

import io

import pytest

from LZGraphs._io._readers import iter_tabular_rows
from LZGraphs._io._sniff import resolve_columns, sniff_delimiter
from LZGraphs._io._spec import FormatError, InputSpec

AIRR_TSV = (
    "sequence_id\tv_call\tj_call\tjunction_aa\tduplicate_count\n"
    "s1\tIGHV1-2*02\tIGHJ4*02\tCASSLGQAYEQYF\t7\n"
    "s2\tIGHV3-7*01\tIGHJ6*02\tCASSPGTGVYGYTF\t2\n"
)
QUOTED_CSV = (
    'junction_aa,note,duplicate_count\n'
    'CASSLGQAYEQYF,"comma, inside",3\n'
)


def test_sniff_delimiter():
    assert sniff_delimiter("a\tb\tc") == "\t"
    assert sniff_delimiter("a,b,c") == ","
    assert sniff_delimiter("justonecolumn") is None


def test_resolve_columns_finds_airr_names():
    header = ["sequence_id", "v_call", "j_call", "junction_aa", "duplicate_count"]
    seq, abund, v, j = resolve_columns(header, None, "aap")
    assert (seq, abund, v, j) == ("junction_aa", "duplicate_count", "v_call", "j_call")


def test_resolve_columns_reports_available_names_when_missing():
    with pytest.raises(FormatError) as exc:
        resolve_columns(["id", "count"], "junction_aa", "aap")
    message = str(exc.value)
    assert "junction_aa" in message
    assert "id" in message and "count" in message


def test_iter_tabular_yields_only_the_sequence_column():
    spec = InputSpec(
        path="x.tsv", format="tabular", compression="none", delimiter="\t",
        seq_column="junction_aa", abundance_column="duplicate_count",
        v_column="v_call", j_column="j_call",
    )
    got = list(iter_tabular_rows(io.StringIO(AIRR_TSV), spec))
    assert got == [
        ("CASSLGQAYEQYF", 7, "IGHV1-2*02", "IGHJ4*02", None),
        ("CASSPGTGVYGYTF", 2, "IGHV3-7*01", "IGHJ6*02", None),
    ]
    assert not any("\t" in row[0] or "," in row[0] for row in got)


def test_iter_tabular_rows_surfaces_the_productive_column():
    text = "junction_aa\tproductive\nCASSLG\tT\nCASSPG\tF\n"
    spec = InputSpec(
        path="x.tsv", format="tabular", compression="none", delimiter="\t",
        seq_column="junction_aa",
    )
    assert [row[4] for row in iter_tabular_rows(io.StringIO(text), spec)] == ["T", "F"]


def test_iter_tabular_respects_csv_quoting():
    spec = InputSpec(
        path="x.csv", format="tabular", compression="none", delimiter=",",
        seq_column="junction_aa", abundance_column="duplicate_count",
    )
    assert list(iter_tabular_rows(io.StringIO(QUOTED_CSV), spec)) == [
        ("CASSLGQAYEQYF", 3, None, None, None)
    ]


def test_iter_tabular_defaults_bad_abundance_to_one():
    text = "junction_aa\tduplicate_count\nCASSLG\tnot-a-number\n"
    spec = InputSpec(
        path="x.tsv", format="tabular", compression="none", delimiter="\t",
        seq_column="junction_aa", abundance_column="duplicate_count",
    )
    assert list(iter_tabular_rows(io.StringIO(text), spec)) == [
        ("CASSLG", 1, None, None, None)
    ]
