"""``validate_input`` must agree with ``read_sequences``/``build`` on format
and record count, for every supported format, not run its own, older,
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

# fasta, fastq, tsv, csv, plain, seqcount: the six RENDERERS formats from
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


# Plan 1c, Task 6: `_PRODUCTIVE_TRUE` and the column-override block used to
# be implemented once in `_public.py` (for `read_sequences`) and again,
# separately, in `_validate.py` (for `validate_input`). Both are now imported
# from one place (`_public.py`) instead. These two tests pin the actual
# agreement that duplication risked losing silently: a productive-column
# edge case (mixed-case/whitespace truthy values, several falsy spellings)
# and an explicit column override, both read identically by build and
# validate against the same fixture.
def test_build_and_validate_agree_on_productive_column_edge_cases(tmp_path):
    path = tmp_path / "productive_edge_cases.tsv"
    path.write_text(
        "junction_aa\tduplicate_count\tproductive\n"
        "CASSLGQAYEQYF\t3\tTrue\n"      # truthy, mixed case
        "CASSPGTGVYGYTF\t2\t 1 \n"      # truthy, whitespace + numeric
        "CASSQGATNTGQLYF\t1\tYES\n"     # truthy, all caps
        "CASSIRSSYEQYF\t4\tfalse\n"     # falsy
        "CASSDRVGNTIYF\t5\t0\n"         # falsy
        "CASSEGQGSDTQYF\t6\t\n"         # blank: not in _PRODUCTIVE_TRUE, dropped
    )

    report = validate_input(str(path))
    result = read_sequences(str(path))

    # 3 productive rows kept, 3 nonproductive dropped, by both paths.
    assert report["records"] == 3
    assert report["records"] == len(result["sequences"])
    assert result["stats"].nonproductive == 3
    assert result["sequences"] == [
        "CASSLGQAYEQYF", "CASSPGTGVYGYTF", "CASSQGATNTGQLYF",
    ]


def test_build_and_validate_agree_on_column_overrides(tmp_path):
    path = tmp_path / "override_target.tsv"
    path.write_text(
        "junction_aa\tv_call\tj_call\treads\talt_v\talt_j\talt_count\n"
        "CASSLGQAYEQYF\tIGHV1-2*02\tIGHJ1*01\t9\tIGHV3-1*01\tIGHJ2*01\t42\n"
    )

    report = validate_input(
        str(path), v_column="alt_v", j_column="alt_j", abundance_column="alt_count",
    )
    result = read_sequences(
        str(path), v_column="alt_v", j_column="alt_j", abundance_column="alt_count",
    )

    assert report["v_column"] == "alt_v"
    assert report["j_column"] == "alt_j"
    assert report["abundance_column"] == "alt_count"
    assert result["v_genes"] == ["IGHV3-1*01"]
    assert result["j_genes"] == ["IGHJ2*01"]
    assert result["abundances"] == [42]


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


class TestIssueLineLocators:
    """Every issue must carry a real position, honest about what it means.

    Line-oriented formats (plain, plain_seqcount, tabular) get a true 1-based
    file line in ``line``. FASTA/FASTQ records span several physical lines,
    so a record ordinal is not a line number: ``line`` is left unset and the
    record's 1-based position is folded into the message text instead. See
    ``_validate_reader``'s docstring in ``_validate.py`` for why.
    """

    def test_plain_malformed_record_reports_its_true_line(self, tmp_path):
        path = tmp_path / "plain_known_line.txt"
        # line 1: good, line 2: good, line 3: malformed (digits only).
        path.write_text("CASSLGQAYEQYF\nCASSPGTGVYGYTF\n1234\n")

        report = validate_input(str(path))
        bad = [i for i in report["issues"] if "1234" in i["message"]]
        assert len(bad) == 1
        assert bad[0]["line"] == 3

    def test_plain_blank_lines_before_the_bad_record_still_count(self, tmp_path):
        path = tmp_path / "plain_blanks_then_bad.txt"
        # line 1: good, lines 2-3: blank, line 4: malformed.
        path.write_text("CASSLGQAYEQYF\n\n\n1234\n")

        report = validate_input(str(path))
        bad = [i for i in report["issues"] if "1234" in i["message"]]
        assert len(bad) == 1
        assert bad[0]["line"] == 4
        assert report["blank_lines"] == 2

    def test_tabular_bad_row_reports_true_file_line_including_header(self, tmp_path):
        path = tmp_path / "tabular_known_bad_row.tsv"
        # line 1: header, line 2: good, line 3: bad row, line 4: good.
        path.write_text(
            "junction_aa\tduplicate_count\n"
            "CASSLGQAYEQYF\t3\n"
            "1234\t5\n"
            "CASSPGTGVYGYTF\t2\n"
        )

        report = validate_input(str(path))
        bad = [i for i in report["issues"] if "1234" in i["message"]]
        assert len(bad) == 1
        assert bad[0]["line"] == 3
        assert report["tabular_rows"] == 2

    def test_fasta_malformed_record_names_the_record_ordinal_not_a_line(self, tmp_path):
        path = tmp_path / "fasta_known_bad_record.fasta"
        # record 1: good, record 2: malformed, record 3: good.
        path.write_text(">s0\nCASSLGQAYEQYF\n>s1\n1234\n>s2\nCASSPGTGVYGYTF\n")

        report = validate_input(str(path))
        bad = [i for i in report["issues"] if "1234" in i["message"]]
        assert len(bad) == 1
        assert "line" not in bad[0]
        assert "record 2" in bad[0]["message"]

    def test_cli_render_shows_line_for_plain_family(self, tmp_path, capsys):
        from types import SimpleNamespace

        from LZGraphs.cli import _emit_validation_report

        path = tmp_path / "plain_known_line_cli.txt"
        path.write_text("CASSLGQAYEQYF\n1234\n")
        report = validate_input(str(path))

        _emit_validation_report(SimpleNamespace(json=False), report)
        out = capsys.readouterr().out
        assert "line=2" in out
