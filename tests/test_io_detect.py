from __future__ import annotations

import gzip

import pytest

from LZGraphs._io._sniff import detect_format
from LZGraphs._io._spec import FormatError

CASES = {
    "a.fa": ">s1\nCASSLGQAYEQYF\n",
    "b.fq": "@r1\nACGTACGT\n+\nIIIIIIII\n",
    "c.tsv": "junction_aa\tduplicate_count\nCASSLG\t3\n",
    "d.csv": "junction_aa,duplicate_count\nCASSLG,3\n",
    "e.txt": "CASSLG\nCASSPG\n",
    "f.txt": "CASSLG\t3\nCASSPG\t9\n",
}
EXPECTED = {
    "a.fa": "fasta", "b.fq": "fastq", "c.tsv": "tabular",
    "d.csv": "tabular", "e.txt": "plain", "f.txt": "plain_seqcount",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_detect_format_by_content(tmp_path, name):
    path = tmp_path / name
    path.write_text(CASES[name])
    assert detect_format(str(path)).format == EXPECTED[name]


def test_detect_format_sees_through_gzip(tmp_path):
    path = tmp_path / "a.fa.gz"
    path.write_bytes(gzip.compress(CASES["a.fa"].encode()))
    spec = detect_format(str(path))
    assert spec.format == "fasta"
    assert spec.compression == "gzip"


def test_detect_format_resolves_tabular_columns(tmp_path):
    path = tmp_path / "c.tsv"
    path.write_text(CASES["c.tsv"])
    spec = detect_format(str(path))
    assert spec.seq_column == "junction_aa"
    assert spec.abundance_column == "duplicate_count"
    assert spec.delimiter == "\t"


def test_detect_format_override_wins(tmp_path):
    path = tmp_path / "a.fa"
    path.write_text(CASES["a.fa"])
    assert detect_format(str(path), override="plain").format == "plain"


def test_detect_format_rejects_binary(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"PK\x03\x04\x00\x00\x00\x00binarystuff")
    with pytest.raises(FormatError, match="binary"):
        detect_format(str(path))


def test_detect_format_rejects_empty(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    with pytest.raises(FormatError, match="empty"):
        detect_format(str(path))


# --- Mandatory fix: looks_like_seqcount must agree with _parse_count -------
#
# _readers._parse_count accepts the float forms pandas/R emit (e.g. "3.0"),
# but looks_like_seqcount used to gate on parts[1].isdigit(), which rejects
# them. That mismatch let such files fall through to "plain", where
# iter_plain yields the whole "seq\t3.0" line (tab and all) as a sequence.


def test_detect_format_seqcount_accepts_pandas_float_counts(tmp_path):
    """A pandas/R-emitted seq<TAB>3.0 file must be detected as plain_seqcount."""
    path = tmp_path / "counts.tsv"
    path.write_text("CASSLG\t3.0\nCASSPG\t9.0\n")
    spec = detect_format(str(path))
    assert spec.format == "plain_seqcount"


def test_detect_format_amino_acid_lines_without_tab_stay_plain(tmp_path):
    """Ordinary amino-acid lines with no tab must still be plain, not seqcount."""
    path = tmp_path / "plain.txt"
    path.write_text("CASSLGQAYEQYF\nCASSPGTGVYGYTF\n")
    assert detect_format(str(path)).format == "plain"


def test_detect_format_fastq_alphabet_ignores_quality_string(tmp_path):
    """FASTQ alphabet inference must use the sequence line, not the quality line.

    Filtering samples by "does not start with @/+" still leaves the quality
    string in the sample set. Phred-quality characters can include letters
    (like 'I') that are valid amino acids but not valid IUPAC nucleotide
    codes, which corrupts infer_alphabet's verdict for ordinary nucleotide
    reads.
    """
    path = tmp_path / "reads.fq"
    path.write_text("@r1\nACGTACGTACGT\n+\nIIIIIIIIIIII\n")
    spec = detect_format(str(path))
    assert spec.format == "fastq"
    assert spec.alphabet == "nucleotide"
