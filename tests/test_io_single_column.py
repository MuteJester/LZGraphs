"""Regression tests for the single-column header-leak defect (Plan 1b, Task 1).

A file whose only column was named ``junction``, ``sequence``, or
``aminoAcid`` used to detect as ``plain``, so its header line became a
leaked sequence in the built graph. ``cdr3`` and ``junction_aa`` escaped
only by accident, because they contain a digit/underscore that
``_is_wellformed`` happens to reject downstream -- an incidental protection,
not a designed one. See ``_sniff.detect_format``'s lone-header
reclassification, which makes all of these -- and ``seq`` -- resolve as a
single-column table by design instead.
"""
from __future__ import annotations

import io

import pytest

from LZGraphs._io._public import read_sequences
from LZGraphs._io._readers import iter_tabular_rows
from LZGraphs._io._sniff import detect_format
from LZGraphs._io._spec import InputSpec

_DATA_ROWS = ["CASSLGQAYEQYF", "CASSPGTGVYGYTF", "CASSQGATNTGQLYF"]


@pytest.mark.parametrize(
    "header", ["junction", "sequence", "aminoAcid", "cdr3", "junction_aa", "seq"]
)
def test_lone_header_is_classified_as_tabular_not_plain(tmp_path, header):
    """The designed fix: sniffing must recognise the header, not merely fail
    to leak it by accident (which is all the old code did for cdr3/junction_aa).
    """
    path = tmp_path / "single.txt"
    path.write_text(header + "\n" + "\n".join(_DATA_ROWS) + "\n")

    spec = detect_format(str(path))

    assert spec.format == "tabular"
    assert spec.seq_column == header


@pytest.mark.parametrize(
    "header", ["junction", "sequence", "aminoAcid", "cdr3", "junction_aa", "seq"]
)
def test_lone_header_never_leaks_as_a_sequence(tmp_path, header):
    path = tmp_path / "single.txt"
    path.write_text(header + "\n" + "\n".join(_DATA_ROWS) + "\n")

    result = read_sequences(str(path))

    assert result["sequences"] == _DATA_ROWS
    assert header.lower() not in {s.lower() for s in result["sequences"]}


@pytest.mark.parametrize("header", ["Junction", "JUNCTION", "AminoAcid", "AMINOACID"])
def test_lone_header_match_is_case_insensitive(tmp_path, header):
    path = tmp_path / "single.txt"
    path.write_text(header + "\n" + "\n".join(_DATA_ROWS) + "\n")

    spec = detect_format(str(path))
    result = read_sequences(str(path))

    assert spec.format == "tabular"
    assert spec.seq_column == header  # original casing preserved as the DictReader key
    assert result["sequences"] == _DATA_ROWS


def test_iter_tabular_rows_handles_a_single_column_directly():
    """Verify (not assume) that csv.DictReader, given a delimiter that never
    occurs in single-column data, still yields one field per row keyed by
    the header -- the mechanism the fix relies on.
    """
    text = "junction\nCASSLG\nCASSPG\n"
    spec = InputSpec(
        path="x.txt", format="tabular", compression="none",
        delimiter=None, seq_column="junction",
    )
    got = list(iter_tabular_rows(io.StringIO(text), spec))
    assert got == [
        ("CASSLG", 1, None, None, None),
        ("CASSPG", 1, None, None, None),
    ]


def test_plain_file_with_real_sequence_first_line_stays_plain(tmp_path):
    """Hazard 1: a genuine first sequence must not be mistaken for a header."""
    path = tmp_path / "plain.txt"
    path.write_text("\n".join(_DATA_ROWS) + "\n")

    spec = detect_format(str(path))
    result = read_sequences(str(path))

    assert spec.format == "plain"
    assert result["sequences"] == _DATA_ROWS


def test_lone_header_with_no_data_rows_yields_nothing(tmp_path):
    """Hazard 3: a header-only file must not emit the header as a sequence.

    Chosen behaviour: yields an empty sequence list rather than raising.
    This matches existing precedent -- an ordinary multi-column TSV with a
    header but no data rows already yields nothing, never raises -- so a
    single-column file is not treated as a special case.
    """
    path = tmp_path / "single.txt"
    path.write_text("junction\n")

    spec = detect_format(str(path))
    result = read_sequences(str(path))

    assert spec.format == "tabular"
    assert result["sequences"] == []


# An explicit --format/expect_format override is this codebase's established
# philosophy for beating content sniffing: a user who forces "plain" has
# asked for exactly that, header line included. This is deliberate, not a
# leak -- the override short-circuits detect_format's auto-detect branch
# (and therefore the lone-header reclassification) entirely, so the header
# is read back as an ordinary data line. Pinned here so a future change to
# the elif ordering in detect_format cannot alter this silently.
def test_expect_format_plain_override_beats_lone_header_reclassification(tmp_path):
    path = tmp_path / "single.txt"
    path.write_text("junction\nCASSLGQAYEQYF\nCASSPGTGVYGYTF\n")

    result = read_sequences(str(path), expect_format="plain")

    assert result["sequences"] == ["junction", "CASSLGQAYEQYF", "CASSPGTGVYGYTF"]


def test_no_override_still_reclassifies_the_same_file_as_tabular(tmp_path):
    """Companion to the override test above: without expect_format, the same
    file is auto-detected and the header is resolved as the seq column, not
    read back as data.
    """
    path = tmp_path / "single.txt"
    path.write_text("junction\nCASSLGQAYEQYF\nCASSPGTGVYGYTF\n")

    result = read_sequences(str(path))

    assert result["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]


def test_seqcount_format_unaffected_by_column_like_first_token(tmp_path):
    """Hazard 5: seq<TAB>count lines whose first token looks like a column
    name must still be classified as plain_seqcount, not reclassified.
    """
    path = tmp_path / "counts.tsv"
    path.write_text("junction\t3\nCASSPG\t5\n")

    spec = detect_format(str(path))
    result = read_sequences(str(path))

    assert spec.format == "plain_seqcount"
    assert result["sequences"] == ["junction", "CASSPG"]
    assert result["abundances"] == [3, 5]


def test_multi_column_tabular_files_are_unaffected(tmp_path):
    """Hazard 2: normal multi-column files must resolve exactly as before."""
    path = tmp_path / "airr.tsv"
    path.write_text(
        "junction_aa\tv_call\tduplicate_count\n"
        "CASSLGQAYEQYF\tIGHV1-2*02\t7\n"
    )

    spec = detect_format(str(path))

    assert spec.format == "tabular"
    assert spec.seq_column == "junction_aa"
    assert spec.v_column == "v_call"
    assert spec.abundance_column == "duplicate_count"
    assert spec.delimiter == "\t"
