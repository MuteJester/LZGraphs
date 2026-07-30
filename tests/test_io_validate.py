"""``validate_input`` must agree with ``read_sequences``/``build`` on format
and record count, for every supported format -- not run its own, older,
first-line-only classifier.

Regression target: a 2-record FASTA used to be reported by ``validate_input``
as ``detected_kind='plain'`` with ``records=4`` (the two ``>`` header lines
counted as sequences), while ``read_sequences`` on the same file correctly
read 2 sequences. See ``test_fasta_case_reports_fasta_not_plain`` below.
"""
from __future__ import annotations

import pytest
from test_io_matrix import RENDERERS

from LZGraphs._io import detect_input_kind, read_sequences, validate_input

_REPORT_KEY_TYPES = {
    "path": str,
    "variant": str,
    "detected_kind": str,
    "expect_format": (str, type(None)),
    "strict_input": bool,
    "ok": bool,
    "mode": (str, type(None)),
    "total_lines": int,
    "records": int,
    "blank_lines": int,
    "plain_records": int,
    "seqcount_records": int,
    "tabular_rows": int,
    "warning_count": int,
    "error_count": int,
    "issues": list,
    "seq_column": (str, type(None)),
    "v_column": (str, type(None)),
    "j_column": (str, type(None)),
    "abundance_column": (str, type(None)),
    "has_header": bool,
    "summary": str,
}

# fasta, fastq, tsv, csv, plain, seqcount -- the six RENDERERS formats from
# test_io_matrix.py, reused rather than re-specified here so the two test
# files can never quietly drift apart on what each format's content looks
# like.
_EXPECTED_KIND = {
    "fasta": "fasta",
    "fastq": "fastq",
    "tsv": "tabular",
    "csv": "tabular",
    "plain": "plain",
    "seqcount": "plain_seqcount",
}


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
def test_detected_kind_matches_detect_input_kind(tmp_path, fmt):
    path = tmp_path / f"data_{fmt}.txt"
    path.write_text(RENDERERS[fmt]())

    report = validate_input(str(path))
    assert report["detected_kind"] == _EXPECTED_KIND[fmt]
    assert report["detected_kind"] == detect_input_kind(str(path))


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
def test_records_matches_read_sequences_count(tmp_path, fmt):
    path = tmp_path / f"data_{fmt}.txt"
    path.write_text(RENDERERS[fmt]())

    report = validate_input(str(path))
    expected = len(read_sequences(str(path))["sequences"])
    assert report["records"] == expected
    assert report["ok"] is True
    assert report["error_count"] == 0


def test_fasta_case_reports_fasta_not_plain(tmp_path):
    """The exact regression: a 2-record FASTA must not be seen as 4 plain lines."""
    path = tmp_path / "two_records.fasta"
    path.write_text(">seq1\nCASSLGQAYEQYF\n>seq2\nCASSPGTGVYGYTF\n")

    report = validate_input(str(path))
    assert report["detected_kind"] == "fasta"
    assert report["records"] == 2
    assert report["plain_records"] == 0


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
def test_format_specific_counters_are_zero_elsewhere(tmp_path, fmt):
    """plain_records/seqcount_records/tabular_rows are 0 for formats they don't describe."""
    path = tmp_path / f"data_{fmt}.txt"
    path.write_text(RENDERERS[fmt]())
    report = validate_input(str(path))

    counters = {
        "plain": report["plain_records"],
        "plain_seqcount": report["seqcount_records"],
        "tabular": report["tabular_rows"],
    }
    kind = _EXPECTED_KIND[fmt]
    for name, value in counters.items():
        if name == kind:
            assert value == report["records"]
        else:
            assert value == 0


def test_all_22_keys_present_with_right_types(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("CASSLGQAYEQYF\nCASSPGTGVYGYTF\n")
    report = validate_input(str(path))

    assert len(report) == 22, sorted(report)
    for key, expected_type in _REPORT_KEY_TYPES.items():
        assert key in report, f"missing key {key!r}"
        assert isinstance(report[key], expected_type), (
            f"{key!r} has type {type(report[key]).__name__}, expected {expected_type}"
        )
    for issue in report["issues"]:
        assert isinstance(issue, dict)
        assert "level" in issue and "message" in issue


class TestMalformedRecordsBecomeIssues:
    def test_plain_malformed_sequence_is_a_warning_by_default(self, tmp_path):
        path = tmp_path / "plain_malformed.txt"
        path.write_text("CASSLGQAYEQYF\n1234\nCASSPGTGVYGYTF\n")
        report = validate_input(str(path))
        assert report["ok"] is True
        assert report["warning_count"] >= 1
        assert any(i["level"] == "warning" for i in report["issues"])
        assert report["records"] == 2

    def test_plain_malformed_sequence_is_an_error_under_strict(self, tmp_path):
        path = tmp_path / "plain_malformed_strict.txt"
        path.write_text("CASSLGQAYEQYF\n1234\nCASSPGTGVYGYTF\n")
        report = validate_input(str(path), strict_input=True)
        assert report["ok"] is False
        assert report["error_count"] >= 1
        assert report["records"] == 2

    def test_tabular_malformed_sequence_is_an_issue(self, tmp_path):
        path = tmp_path / "tabular_malformed.tsv"
        path.write_text(
            "junction_aa\tduplicate_count\n"
            "CASSLGQAYEQYF\t3\n"
            "1234\t5\n"
        )
        report = validate_input(str(path))
        assert report["ok"] is True
        assert report["warning_count"] >= 1
        assert report["tabular_rows"] == 1

        strict_report = validate_input(str(path), strict_input=True)
        assert strict_report["ok"] is False
        assert strict_report["error_count"] >= 1

    def test_fastq_malformed_sequence_is_an_issue(self, tmp_path):
        path = tmp_path / "fastq_malformed.fastq"
        path.write_text(
            "@s0\nCASSLGQAYEQYF\n+\nIIIIIIIIIIIII\n"
            "@s1\n1234\n+\nIIII\n"
        )
        report = validate_input(str(path))
        assert report["ok"] is True
        assert report["warning_count"] >= 1
        assert report["records"] == 1


class TestExpectFormatIsAnAssertion:
    """expect_format must record a mismatch as an error, not silently override it."""

    def test_declared_plain_but_content_is_tabular_is_an_error(self, tmp_path):
        path = tmp_path / "actually_tabular.tsv"
        path.write_text(RENDERERS["tsv"]())
        report = validate_input(str(path), expect_format="plain")
        assert report["ok"] is False
        assert report["error_count"] >= 1
        assert report["records"] == 0

    def test_declared_tabular_but_content_is_plain_is_an_error(self, tmp_path):
        path = tmp_path / "actually_plain.txt"
        path.write_text(RENDERERS["plain"]())
        report = validate_input(str(path), expect_format="tabular")
        assert report["ok"] is False
        assert report["error_count"] >= 1
        assert report["records"] == 0

    def test_declared_format_matching_content_is_ok(self, tmp_path):
        path = tmp_path / "seqcount.txt"
        path.write_text(RENDERERS["seqcount"]())
        report = validate_input(str(path), expect_format="plain_seqcount")
        assert report["ok"] is True
        assert report["error_count"] == 0


class TestEmptyAndBinaryInput:
    def test_empty_file_is_a_clean_report_not_a_crash(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("")
        report = validate_input(str(path))
        assert report["ok"] is False
        assert report["error_count"] >= 1
        assert report["records"] == 0

    def test_binary_file_is_a_clean_report_not_a_crash(self, tmp_path):
        path = tmp_path / "binary.dat"
        path.write_bytes(b"PK\x03\x04\x00\x00\x00\x00binarystuff")
        report = validate_input(str(path))
        assert report["ok"] is False
        assert report["error_count"] >= 1
        assert report["records"] == 0
