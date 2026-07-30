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


# ---------------------------------------------------------------------------
# Fix round 1: '*' (stop codon) and '-' (gap) are legitimate in real AIRR
# amino-acid data. A stop codon is the single most common reason a row is
# non-productive, so a malformed check that rejects '*' silently defeats
# keep_nonproductive=True for its primary real-world purpose.
# ---------------------------------------------------------------------------

NONPRODUCTIVE_STOP_CODON = (
    "junction_aa\tduplicate_count\tproductive\n"
    "CASSLGQAYEQYF\t7\tT\n"
    "CASSLG*QAYEQYF\t2\tF\n"
    "CASSPGTGVYGYTF\t5\tT\n"
)


def test_stop_codon_and_gap_characters_are_kept_not_malformed(tmp_path):
    """Real AIRR amino-acid sequences can carry '*' (stop codon) and '-'/'.'
    (alignment gap); those characters must not be flagged as malformed."""
    path = tmp_path / "a.fa"
    path.write_text(
        ">clean\nCASSPGTGVYGYTF\n>stop\nCASSLG*QAYEQYF\n>gap\nCASSLG-QAYEQYF\n"
    )
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSPGTGVYGYTF", "CASSLG*QAYEQYF", "CASSLG-QAYEQYF"]
    assert got["stats"].malformed == 0


def test_keep_nonproductive_preserves_a_stop_codon_row(tmp_path):
    """The row keep_nonproductive=True exists to preserve is exactly the
    kind of row a stop-codon-blind malformed check would silently re-drop."""
    path = tmp_path / "b.tsv"
    path.write_text(NONPRODUCTIVE_STOP_CODON)
    got = read_sequences(str(path), keep_nonproductive=True)
    assert got["sequences"] == [
        "CASSLGQAYEQYF", "CASSLG*QAYEQYF", "CASSPGTGVYGYTF",
    ]
    assert got["stats"].nonproductive == 0
    assert got["stats"].malformed == 0


def test_same_file_without_the_flag_drops_the_stop_codon_row_once(tmp_path):
    """Without the flag, the same row is dropped -- but as nonproductive,
    not malformed: it is counted once, in exactly one category."""
    path = tmp_path / "b.tsv"
    path.write_text(NONPRODUCTIVE_STOP_CODON)
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    stats = got["stats"]
    assert stats.nonproductive == 1
    assert stats.malformed == 0


def test_a_row_both_nonproductive_and_malformed_is_counted_once(tmp_path):
    """A row can fail both checks: non-productive AND not a usable sequence.
    It must land in exactly one bucket. The productive filter runs first, so
    it is counted as nonproductive, not malformed, and total always equals
    the sum of the three outcome buckets.

    A literally empty sequence field can't demonstrate this: iter_tabular_rows
    already discards those upstream, before read_sequences ever sees them
    (see the RecordStats accounting-gap docstring) -- so a non-empty but
    invalid sequence (here, one containing a digit) is the only way to
    actually observe the overlap.
    """
    path = tmp_path / "c.tsv"
    path.write_text(
        "junction_aa\tduplicate_count\tproductive\n"
        "CASSLGQAYEQYF\t7\tT\n"
        "CASS1LG\t2\tF\n"
        "CASSPGTGVYGYTF\t5\tT\n"
    )
    got = read_sequences(str(path))
    stats = got["stats"]
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert stats.nonproductive == 1
    assert stats.malformed == 0
    assert stats.total == stats.kept + stats.malformed + stats.nonproductive


def test_structural_corruption_in_a_sequence_column_is_still_malformed(tmp_path):
    """Widening the malformed check to accept '*', '-', and '.' must not let
    real corruption through: a FASTA-header-shaped value, a comma-joined
    row's worth of fields collapsed into one value, and a bare leaked
    numeric count must all still be rejected as malformed when they turn up
    in a legitimately tabular file's sequence column."""
    path = tmp_path / "d.tsv"
    path.write_text(
        "junction_aa\tduplicate_count\tproductive\n"
        ">seq10\t1\tT\n"
        "seq2370,IGHV1-2*02,IGHJ4*02,CASSLG,27,T\t1\tT\n"
        "47\t1\tT\n"
        "CASSPGTGVYGYTF\t1\tT\n"
    )
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSPGTGVYGYTF"]
    assert got["stats"].malformed == 3
    assert got["stats"].kept == 1


# ---------------------------------------------------------------------------
# Fix round 2: round 1's fix widened _is_wellformed to accept '*', '-', '.'
# without requiring an actual letter, so those residue marks ALONE -- the
# usual missing-value sentinels in delimited exports -- qualified as
# "sequences". They must be rejected; a real sequence carrying one of them
# alongside actual letters (round 1's concern) must still be accepted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["-", ".", "*", "--", "...", "*-."])
def test_residue_marks_alone_are_malformed_not_kept(tmp_path, bad):
    """A value made up entirely of residue marks (no letters at all) is a
    missing-value sentinel, not a sequence, and must be rejected."""
    path = tmp_path / "sentinel.tsv"
    path.write_text(
        "junction_aa\tproductive\n"
        f"{bad}\tT\n"
        "CASSLGQAYEQYF\tT\n"
    )
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLGQAYEQYF"]
    assert got["stats"].malformed == 1
    assert got["stats"].kept == 1


def test_missing_value_sentinels_alongside_real_sequences_are_dropped(tmp_path):
    """'-' and '.' are VCF-style / general missing-value sentinels. When a
    junction_aa column holds them on their own, those rows must be dropped
    and counted malformed, while real sequences elsewhere in the same file
    are unaffected."""
    path = tmp_path / "sentinels.tsv"
    path.write_text(
        "junction_aa\tduplicate_count\tproductive\n"
        "CASSLGQAYEQYF\t7\tT\n"
        "-\t2\tT\n"
        ".\t3\tT\n"
        "CASSPGTGVYGYTF\t5\tT\n"
    )
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert got["stats"].malformed == 2
