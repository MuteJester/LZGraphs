"""Plan 1c, Tasks 3 and 4: ``expect_format``/``--expect-format`` contract.

Task 3 (Defect A): three vocabularies used to disagree about which formats
``--expect-format``/``expect_format`` accept. ``cli.py``'s argparse
``choices`` and ``_validate.py``'s ``_VALID_EXPECT_FORMATS`` both excluded
``fasta``/``fastq``, while ``_sniff.py``'s ``_VALID_OVERRIDES`` already
included them, so a flag the tool fully supported (``lzg build data.fa
--expect-format fasta``) was refused at the argparse layer or reported as a
validation error. ``VALID_FORMATS`` in ``_spec.py`` is now the single source
of truth all three sites reference.

Task 4 (Defect B): ``read_sequences`` used to thread ``expect_format``
straight into ``detect_format``'s ``override=``, which *coerces* the
classification unconditionally. A tabular file declared
``expect_format="plain"`` was read line-for-line as plain text, so every
row failed ``_is_wellformed`` and ``read_sequences`` silently returned an
empty list, with no error and no warning, while ``validate_input`` correctly
reported ``ok=False`` for the identical input. ``expect_format`` is now an
assertion in both: a genuine mismatch raises ``FormatError`` naming both the
declared and detected format, in ``read_sequences`` exactly as it already
did in ``validate_input``.
"""
from __future__ import annotations

import subprocess
import sys

import pytest
from test_io_matrix import RENDERERS, SEQUENCES

from LZGraphs._io import (
    VALID_FORMATS,
    FormatError,
    _sniff,
    _validate,
    read_sequences,
    validate_input,
)
from LZGraphs._io._spec import format_family

LZG = [sys.executable, "-m", "LZGraphs.cli"]

# fasta, fastq, tsv, csv, plain, seqcount: the six RENDERERS formats from
# test_io_matrix.py. tsv and csv both detect as "tabular", so both map to the
# same declared value here. Kept in sync with test_io_validate.py's
# _EXPECTED_KIND on purpose (that dict is the more authoritative copy) rather
# than imported, since importing a private, underscore-prefixed name across
# a third test file starts to stretch the "reuse a fixture" precedent this
# project otherwise follows.
_EXPECTED_KIND = {
    "fasta": "fasta",
    "fastq": "fastq",
    "tsv": "tabular",
    "csv": "tabular",
    "plain": "plain",
    "seqcount": "plain_seqcount",
}


def run_lzg(*args, input_text=None):
    """Run the lzg CLI and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        LZG + list(args),
        capture_output=True, text=True, input=input_text, timeout=30,
    )
    return result.stdout, result.stderr, result.returncode


# ── Task 3: one vocabulary, three sites ─────────────────────────────


def test_valid_formats_includes_fasta_and_fastq():
    # This is the measured defect itself: fasta/fastq are formats the tool
    # fully supports (FASTA/FASTQ readers exist and work), so the vocabulary
    # every expect_format site validates against must include them.
    assert set(VALID_FORMATS) == {"fasta", "fastq", "plain", "plain_seqcount", "tabular"}


def test_sniff_and_validate_reference_the_same_constant_object():
    """The three call sites (_sniff.py, _validate.py, cli.py) must not each
    keep their own tuple: they must all resolve to the one object defined in
    _spec.py, or a future edit to one could silently drift from the others
    again, which is exactly how this defect happened the first time.
    """
    assert _sniff.VALID_FORMATS is VALID_FORMATS
    assert _validate.VALID_FORMATS is VALID_FORMATS


@pytest.mark.parametrize("subcommand", ["build", "validate-input"])
def test_cli_help_lists_every_valid_format(subcommand):
    out, _err, rc = run_lzg(subcommand, "--help")
    assert rc == 0
    for fmt in VALID_FORMATS:
        assert fmt in out, f"{subcommand} --help is missing {fmt!r} from --expect-format choices"


@pytest.mark.parametrize("subcommand", ["build", "validate-input"])
def test_cli_unknown_expect_format_value_lists_the_valid_set(tmp_path, subcommand):
    path = tmp_path / "whatever.txt"
    path.write_text("CASSLGQAYEQYF\n")
    args = [subcommand, str(path)]
    if subcommand == "build":
        args += ["-o", str(tmp_path / "out.lzg")]
    args += ["--expect-format", "bogus"]

    _out, err, rc = run_lzg(*args)
    assert rc != 0
    assert "invalid choice: 'bogus'" in err
    for fmt in VALID_FORMATS:
        assert fmt in err


def test_cli_build_accepts_expect_format_fasta_on_a_real_fasta_file(tmp_path):
    # The exact repro from the defect report: `lzg build data.fa
    # --expect-format fasta` used to fail at argparse before ever reading the
    # file.
    path = tmp_path / "data.fa"
    path.write_text(">s0\nCASSLGQAYEQYF\n>s1\nCASSPGTGVYGYTF\n")
    out_path = tmp_path / "out.lzg"

    _out, err, rc = run_lzg(
        "build", str(path), "-o", str(out_path), "--expect-format", "fasta"
    )
    assert rc == 0, err
    assert out_path.exists()


def test_cli_validate_input_accepts_expect_format_fastq_on_a_real_fastq_file(tmp_path):
    path = tmp_path / "data.fastq"
    path.write_text("@s0\nCASSLGQAYEQYF\n+\nIIIIIIIIIIIII\n")

    out, _err, rc = run_lzg(
        "validate-input", str(path), "--expect-format", "fastq"
    )
    assert rc == 0
    assert "VL\tok\tyes" in out


# ── Task 4: expect_format is an assertion, not a coercion ───────────


def test_read_sequences_tabular_declared_plain_raises_naming_both_formats(tmp_path):
    """The exact measured defect: read_sequences(tabular_file,
    expect_format="plain") used to return [] silently. It must now raise
    FormatError naming both the declared and the detected format.
    """
    path = tmp_path / "data.tsv"
    path.write_text(RENDERERS["tsv"]())

    with pytest.raises(FormatError) as excinfo:
        read_sequences(str(path), expect_format="plain")

    message = str(excinfo.value)
    assert "plain" in message
    assert "tabular" in message


def test_read_sequences_and_validate_input_agree_on_the_same_mismatch(tmp_path):
    """Companion to the defect report: before the fix, read_sequences
    coerced (silently empty) while validate_input asserted (ok=False) for
    the identical file and expect_format. They must now agree that this is a
    genuine, loud mismatch.
    """
    path = tmp_path / "data.tsv"
    path.write_text(RENDERERS["tsv"]())

    with pytest.raises(FormatError):
        read_sequences(str(path), expect_format="plain")

    report = validate_input(str(path), expect_format="plain")
    assert report["ok"] is False


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
def test_matching_expect_format_still_works(tmp_path, fmt):
    """Hazard 2: a declared format that matches the content must still work,
    for all five supported formats, through both read_sequences and
    validate_input.
    """
    path = tmp_path / f"match_{fmt}.txt"
    path.write_text(RENDERERS[fmt]())
    declared = _EXPECTED_KIND[fmt]

    result = read_sequences(str(path), expect_format=declared)
    assert result["sequences"] == SEQUENCES

    report = validate_input(str(path), expect_format=declared)
    assert report["ok"] is True
    assert report["error_count"] == 0


def _mismatched_pairs():
    # One representative per detected kind (skip "csv": it detects as
    # "tabular" too, same as "tsv", so it would only duplicate that row).
    representative_fmts = ["fasta", "fastq", "tsv", "plain", "seqcount"]
    pairs = []
    for fmt in representative_fmts:
        detected = _EXPECTED_KIND[fmt]
        for declared in VALID_FORMATS:
            if format_family(declared) != format_family(detected):
                pairs.append((fmt, declared))
    return pairs


@pytest.mark.parametrize("fmt,declared", _mismatched_pairs())
def test_genuine_mismatch_fails_loudly_in_every_path(tmp_path, fmt, declared):
    """Hazard 3: a genuine mismatch must fail loudly everywhere, with a
    message naming both formats, for every combination of detected content
    and declared expect_format that is a real mismatch (not same-family, like
    plain vs. plain_seqcount).
    """
    path = tmp_path / f"mismatch_{fmt}_{declared}.txt"
    path.write_text(RENDERERS[fmt]())
    detected = _EXPECTED_KIND[fmt]

    with pytest.raises(FormatError) as excinfo:
        read_sequences(str(path), expect_format=declared)
    message = str(excinfo.value)
    assert declared in message
    assert detected in message

    report = validate_input(str(path), expect_format=declared)
    assert report["ok"] is False
    assert report["error_count"] >= 1


def test_read_sequences_rejects_unknown_expect_format_value():
    with pytest.raises(ValueError, match="plain_seqcount"):
        read_sequences("does-not-need-to-exist.txt", expect_format="bogus")
