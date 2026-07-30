from __future__ import annotations

import gzip

import pytest

from LZGraphs._io import detect_input_kind, read_sequences, read_sequences_simple

AIRR = (
    "sequence_id\tv_call\tj_call\tjunction_aa\tduplicate_count\n"
    "s1\tIGHV1-2*02\tIGHJ4*02\tCASSLGQAYEQYF\t7\n"
)
FASTA = ">s1\nCASSLGQAYEQYF\n>s2\nCASSPGTGVYGYTF\n"
FASTQ = (
    "@s1\nCASSLGQAYEQYF\n+\nIIIIIIIIIIIIII\n"
    "@s2\nCASSPGTGVYGYTF\n+\nIIIIIIIIIIIIIII\n"
)


def test_read_sequences_keeps_its_dict_contract(tmp_path):
    path = tmp_path / "a.tsv"
    path.write_text(AIRR)
    got = read_sequences(str(path))
    assert set(got) == {"sequences", "abundances", "v_genes", "j_genes", "stats"}
    assert got["sequences"] == ["CASSLGQAYEQYF"]
    assert got["abundances"] == [7]
    assert got["v_genes"] == ["IGHV1-2*02"]


def test_read_sequences_now_handles_fasta(tmp_path):
    path = tmp_path / "a.fa"
    path.write_text(FASTA)
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert got["abundances"] == [1, 1]


def test_read_sequences_now_handles_gzip(tmp_path):
    path = tmp_path / "a.tsv.gz"
    path.write_bytes(gzip.compress(AIRR.encode()))
    assert read_sequences(str(path))["sequences"] == ["CASSLGQAYEQYF"]


def test_read_sequences_simple_returns_a_list_of_strings(tmp_path):
    path = tmp_path / "a.fa"
    path.write_text(FASTA)
    result = read_sequences_simple(str(path))
    assert result == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert all(isinstance(item, str) for item in result)


@pytest.mark.parametrize(
    "text,kind",
    [
        (FASTA, "fasta"),
        ("CASSLG\nCASSPG\n", "plain"),
        ("CASSLG\t3\n", "plain_seqcount"),
        (AIRR, "tabular"),
    ],
)
def test_detect_input_kind_reports_the_new_vocabulary(tmp_path, text, kind):
    path = tmp_path / "x"
    path.write_text(text)
    assert detect_input_kind(str(path)) == kind


# ---------------------------------------------------------------------------
# Hazard checks (Task 9 brief): every reader path must round-trip through
# read_sequences with the correct 5-tuple unpacking, gene lists must stay
# aligned with sequences (or be None, never a desynced/empty list), and
# no_genes must suppress gene population entirely.
# ---------------------------------------------------------------------------


def test_read_sequences_handles_fastq(tmp_path):
    path = tmp_path / "a.fastq"
    path.write_text(FASTQ)
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert got["abundances"] == [1, 1]
    assert got["v_genes"] is None
    assert got["j_genes"] is None


def test_read_sequences_handles_plain_seqcount(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("CASSLG\t3\nCASSPG\t9\n")
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLG", "CASSPG"]
    assert got["abundances"] == [3, 9]
    assert got["v_genes"] is None
    assert got["j_genes"] is None


def test_read_sequences_handles_plain(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("CASSLG\nCASSPG\n")
    got = read_sequences(str(path))
    assert got["sequences"] == ["CASSLG", "CASSPG"]
    assert got["abundances"] == [1, 1]
    assert got["v_genes"] is None
    assert got["j_genes"] is None


def test_no_genes_suppresses_gene_lists_for_tabular(tmp_path):
    path = tmp_path / "a.tsv"
    path.write_text(AIRR)
    got = read_sequences(str(path), no_genes=True)
    assert got["v_genes"] is None
    assert got["j_genes"] is None
    assert got["sequences"] == ["CASSLGQAYEQYF"]


def test_gene_lists_stay_aligned_when_some_rows_lack_gene_calls(tmp_path):
    airr_partial = (
        "sequence_id\tv_call\tj_call\tjunction_aa\tduplicate_count\n"
        "s1\tIGHV1-2*02\tIGHJ4*02\tCASSLGQAYEQYF\t7\n"
        "s2\t\tIGHJ5*01\tCASSPGTGVYGYTF\t1\n"
    )
    path = tmp_path / "partial.tsv"
    path.write_text(airr_partial)
    got = read_sequences(str(path))
    assert len(got["v_genes"]) == len(got["sequences"]) == 2
    assert len(got["j_genes"]) == len(got["sequences"]) == 2
    assert got["v_genes"] == ["IGHV1-2*02", ""]
    assert got["j_genes"] == ["IGHJ4*02", "IGHJ5*01"]


def test_fasta_fastq_plain_never_populate_gene_lists(tmp_path):
    for suffix, text in (("fa", FASTA), ("fastq", FASTQ), ("txt", "CASSLG\nCASSPG\n")):
        path = tmp_path / f"a.{suffix}"
        path.write_text(text)
        got = read_sequences(str(path))
        assert got["v_genes"] is None, suffix
        assert got["j_genes"] is None, suffix


def test_stream_is_closed_on_success(tmp_path, monkeypatch):
    import LZGraphs._io._public as public

    path = tmp_path / "a.fa"
    path.write_text(FASTA)

    closed = []
    real_open_text = public.open_text

    def spying_open_text(p):
        stream, codec = real_open_text(p)
        real_close = stream.close
        stream.close = lambda: (closed.append(True), real_close())[1]
        return stream, codec

    monkeypatch.setattr(public, "open_text", spying_open_text)
    read_sequences(str(path))
    assert closed == [True]


def test_stream_is_closed_when_a_reader_raises_mid_iteration(tmp_path, monkeypatch):
    import LZGraphs._io._public as public

    path = tmp_path / "a.fastq"
    # First record is well-formed (so content-sniffing still classifies the
    # file as fastq), second record's separator is not '+', so iter_fastq
    # raises FormatError partway through iteration.
    path.write_text(
        "@s1\nCASSLGQAYEQYF\n+\nIIIIIIIIIIIIII\n"
        "@s2\nCASSPGTGVYGYTF\nBADSEP\nIIIIIIIIIIIIIII\n"
    )

    closed = []
    real_open_text = public.open_text

    def spying_open_text(p):
        stream, codec = real_open_text(p)
        real_close = stream.close
        stream.close = lambda: (closed.append(True), real_close())[1]
        return stream, codec

    monkeypatch.setattr(public, "open_text", spying_open_text)
    with pytest.raises(Exception):
        read_sequences(str(path))
    assert closed == [True]


def test_v_column_override_is_honored_for_tabular(tmp_path):
    text = (
        "sequence_id\tvgene_custom\tjunction_aa\n"
        "s1\tIGHV3-23*01\tCASSLGQAYEQYF\n"
    )
    path = tmp_path / "custom.tsv"
    path.write_text(text)
    got = read_sequences(str(path), v_column="vgene_custom")
    assert got["v_genes"] == ["IGHV3-23*01"]


def test_v_column_override_falls_back_when_absent(tmp_path):
    # v_column requested but not present in this file's header: legacy
    # semantics are fail-soft here (unlike seq_column), so this must not
    # raise, and should simply yield no V gene data.
    path = tmp_path / "no_vcol.tsv"
    path.write_text("sequence_id\tjunction_aa\ns1\tCASSLGQAYEQYF\n")
    got = read_sequences(str(path), v_column="nonexistent_column")
    assert got["sequences"] == ["CASSLGQAYEQYF"]
    assert got["v_genes"] is None


def test_read_sequences_strict_input_still_rejects_mixed_plain_records(tmp_path):
    # Documents the same contract as
    # tests/test_io.py::test_read_sequences_strict_plain_seqcount_fails_on_mixed:
    # forcing the format via expect_format bypasses content sniffing, so this
    # is the one piece of strict-mode behaviour read_sequences must still
    # perform itself (see _iter_plain_strict's docstring in _public.py).
    path = tmp_path / "mixed.txt"
    path.write_text("CASSLGIRRT\t5\nCASSLGYEQYF\n")
    with pytest.raises(ValueError, match="mixed"):
        read_sequences(
            str(path), variant="aap", strict_input=True,
            expect_format="plain_seqcount",
        )
