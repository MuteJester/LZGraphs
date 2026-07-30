from __future__ import annotations

import pytest

from LZGraphs._io import FormatError, read_sequences

MIXED = ">good1\nCASSLGQAYEQYF\n>bad\nCASS!!LG\n>good2\nCASSPGTGVYGYTF\n"
AIRR_PRODUCTIVE = (
    "junction_aa\tduplicate_count\tproductive\n"
    "CASSLGQAYEQYF\t7\tT\n"
    "CASSBROKEN\t2\tF\n"
    "CASSPGTGVYGYTF\t5\tT\n"
)


def test_malformed_records_are_skipped_and_counted(tmp_path):
    path = tmp_path / "a.fa"
    path.write_text(MIXED)
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert got["stats"].malformed == 1
    assert got["stats"].kept == 2
    assert got["stats"].total == 3


def test_strict_mode_raises_on_the_first_malformed_record(tmp_path):
    path = tmp_path / "a.fa"
    path.write_text(MIXED)
    with pytest.raises(FormatError, match="CASS"):
        read_sequences(str(path), strict_input=True)


def test_nonproductive_rows_are_dropped_by_default(tmp_path):
    path = tmp_path / "a.tsv"
    path.write_text(AIRR_PRODUCTIVE)
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert got["stats"].nonproductive == 1


def test_nonproductive_rows_are_kept_on_request(tmp_path):
    path = tmp_path / "a.tsv"
    path.write_text(AIRR_PRODUCTIVE)
    got = read_sequences(str(path), keep_nonproductive=True)
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSBROKEN", "CASSPGTGVYGYTF"]
    assert got["stats"].nonproductive == 0


def test_clean_input_reports_zero_losses(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("CASSLGQAYEQYF\nCASSPGTGVYGYTF\n")
    stats = read_sequences(str(path))["stats"]
    assert (stats.total, stats.kept, stats.malformed, stats.nonproductive) == (2, 2, 0, 0)
