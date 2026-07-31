"""A header with duplicate column names must refuse, not silently pick one.

Measured defect: ``csv.DictReader`` keeps only the *last* occurrence of a
repeated header key, so a TSV with two ``junction_aa`` columns builds a graph
entirely from the second column while the user believes it read the first.
This is total silent data substitution, not a dropped record, and it must be
impossible to reach with exit 0.

``detect_format`` is the single place both ``read_sequences`` (build) and
``validate_input`` (validate-input) route through, so the guard belongs there
and both callers inherit it for free.
"""
from __future__ import annotations

import pytest

from LZGraphs._io import FormatError, read_sequences, validate_input
from LZGraphs._io._sniff import detect_format


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_duplicate_sequence_column_raises_and_names_it(tmp_path):
    text = (
        "junction_aa\tjunction_aa\n"
        "CASSLGQAYEQYF\tWRONGVALUE\n"
        "CASSPGTGVYGYTF\tALSOWRONG\n"
    )
    path = _write(tmp_path, "dup_seq.tsv", text)

    with pytest.raises(FormatError) as exc:
        detect_format(path)
    message = str(exc.value)
    assert "junction_aa" in message
    assert "2" in message


def test_duplicate_sequence_column_blocks_read_sequences(tmp_path):
    """The exact measured defect: this must no longer return WRONGVALUE/ALSOWRONG."""
    text = (
        "junction_aa\tjunction_aa\n"
        "CASSLGQAYEQYF\tWRONGVALUE\n"
        "CASSPGTGVYGYTF\tALSOWRONG\n"
    )
    path = _write(tmp_path, "dup_seq_read.tsv", text)

    with pytest.raises(FormatError):
        read_sequences(path)


def test_duplicate_abundance_column_raises(tmp_path):
    text = (
        "junction_aa\tduplicate_count\tduplicate_count\n"
        "CASSLGQAYEQYF\t7\t99\n"
    )
    path = _write(tmp_path, "dup_abundance.tsv", text)

    with pytest.raises(FormatError) as exc:
        detect_format(path)
    assert "duplicate_count" in str(exc.value)


def test_duplicate_irrelevant_column_raises(tmp_path):
    """Ambiguity in a column nobody resolves against is still ambiguity."""
    text = (
        "junction_aa\tnote\tnote\n"
        "CASSLGQAYEQYF\tfoo\tbar\n"
    )
    path = _write(tmp_path, "dup_irrelevant.tsv", text)

    with pytest.raises(FormatError) as exc:
        detect_format(path)
    assert "note" in str(exc.value)


def test_duplicate_columns_case_insensitive(tmp_path):
    """Column matching is case-insensitive elsewhere in this module (_pick),
    so Junction_AA and junction_aa must be treated as the same duplicated name.
    """
    text = (
        "junction_aa\tJunction_AA\n"
        "CASSLGQAYEQYF\tWRONGVALUE\n"
    )
    path = _write(tmp_path, "dup_case.tsv", text)

    with pytest.raises(FormatError) as exc:
        detect_format(path)
    assert "junction_aa" in str(exc.value).lower()


def test_no_duplicates_is_unaffected(tmp_path):
    text = (
        "junction_aa\tduplicate_count\tv_call\n"
        "CASSLGQAYEQYF\t7\tIGHV1-2*02\n"
    )
    path = _write(tmp_path, "no_dup.tsv", text)

    spec = detect_format(path)
    assert spec.format == "tabular"
    assert spec.seq_column == "junction_aa"


def test_error_message_names_column_and_mentions_column_flag(tmp_path):
    text = (
        "junction_aa\tjunction_aa\n"
        "CASSLGQAYEQYF\tWRONGVALUE\n"
    )
    path = _write(tmp_path, "dup_msg.tsv", text)

    with pytest.raises(FormatError) as exc:
        detect_format(path)
    message = str(exc.value)
    assert "junction_aa" in message
    assert "--column" in message


def test_single_trailing_empty_header_cell_is_not_a_duplicate(tmp_path):
    """A trailing delimiter produces an extra "" column name in real exports.
    A single empty cell must not be flagged: only one column is nameless.
    """
    text = "junction_aa\t\nCASSLGQAYEQYF\textra\n"
    path = _write(tmp_path, "trailing_empty.tsv", text)

    spec = detect_format(path)
    assert spec.format == "tabular"
    assert spec.seq_column == "junction_aa"


def test_multiple_empty_header_cells_are_not_flagged_either(tmp_path):
    """Empty names are excluded from duplicate detection entirely, not just
    tolerated once: no candidate list or --column value in this module ever
    resolves to "", so two nameless columns are not the ambiguity this
    guard exists to catch, and flagging them would only add noise for messy
    but harmless real-world exports (e.g. multiple trailing tabs).
    """
    text = "junction_aa\t\t\nCASSLGQAYEQYF\textra\tmore\n"
    path = _write(tmp_path, "trailing_empties.tsv", text)

    spec = detect_format(path)
    assert spec.format == "tabular"
    assert spec.seq_column == "junction_aa"


def test_single_column_reclassified_file_is_unaffected(tmp_path):
    """The lone-header reclassification branch (plain -> tabular for a file
    whose only line is a recognised sequence-column name) has exactly one
    header cell, so it can never trigger a duplicate-name error.
    """
    text = "junction_aa\nCASSLGQAYEQYF\nCASSPGTGVYGYTF\n"
    path = _write(tmp_path, "lone_header.tsv", text)

    spec = detect_format(path)
    assert spec.format == "tabular"
    assert spec.seq_column == "junction_aa"


def test_validate_input_reports_duplicate_columns_cleanly(tmp_path):
    text = (
        "junction_aa\tjunction_aa\n"
        "CASSLGQAYEQYF\tWRONGVALUE\n"
    )
    path = _write(tmp_path, "dup_validate.tsv", text)

    report = validate_input(path)
    assert report["ok"] is False
    assert report["error_count"] >= 1
    messages = " ".join(issue["message"] for issue in report["issues"])
    assert "junction_aa" in messages
