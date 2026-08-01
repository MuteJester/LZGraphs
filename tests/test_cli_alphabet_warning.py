"""Plan 1c, Task 7: wire ``InputSpec.alphabet`` into ``cmd_build`` as a warning.

``detect_format`` classifies every file's residues into ``'amino_acid'``,
``'nucleotide'``, or ``'ambiguous'`` on every call already, but nothing read
that field: it was populated everywhere and consumed only by two tests in
``tests/test_io_sniff.py`` / ``tests/test_io_detect.py``. Its documented
purpose was warning a user who feeds nucleotide data to an amino-acid engine
(or the reverse), so this wires it into ``cli.cmd_build``.

Engine/alphabet mapping (see ``cli.py``'s ``_VARIANT_ALPHABET``, and
``docs/concepts/graph-types.md``'s "Input" row, confirmed independently by
``_io/_sniff.py``'s ``_SEQ_COLUMNS["naive"]`` candidate list mixing both
amino-acid and nucleotide column names):

- ``aap`` expects amino acids.
- ``ndp`` expects nucleotides.
- ``naive`` accepts either and is never checked.
- ``flashback`` (``lzg flashback build``) is amino-acid only, but it never
  calls ``detect_format``/``read_sequences`` at all (see ``cli._fb_build``,
  which reads plain lines directly), so it has no ``InputSpec.alphabet`` to
  check and is out of scope here.

Every test below drives ``cli.cmd_build`` directly (the exact function
``lzg build`` dispatches to), the same pattern ``test_cli_build_matrix.py``
and ``test_cli_compressed_build.py`` already use, and reads the warning off
real stderr via ``capsys``, not by inspecting internal state, so a future
change that silently drops the warning (e.g. reverts to computing
``alphabet`` and never reading it) would be caught here.
"""
from __future__ import annotations

import argparse

from LZGraphs import LZGraph
from LZGraphs.cli import cmd_build

# Unambiguous amino-acid residues (contain letters like E/L/Q/F that are not
# valid IUPAC nucleotide codes), reused from the established fixture in
# tests/test_io_single_column.py so this file does not invent a second,
# possibly-drifting definition of "looks like amino acid data".
_AMINO_ACID_SEQS = ["CASSLGQAYEQYF", "CASSPGTGVYGYTF", "CASSQGATNTGQLYF"]

# Unambiguous nucleotide residues: every character is a core base (A/C/G/T),
# well above infer_alphabet's 0.65 core-fraction threshold.
_NUCLEOTIDE_SEQS = ["ACGTACGTACGTACGT", "GGCCTTAAGGCCTTAA", "ACGTGGCCAATTGGCC"]


def _fasta(seqs):
    return "".join(f">seq{i}\n{s}\n" for i, s in enumerate(seqs))


def _namespace(input_path, output_path, *, variant, quiet=False, log_level=None):
    """The subset of argparse.Namespace fields cmd_build reads.

    Mirrors test_cli_build_matrix.py's _build_namespace, the working
    reference for the exact attribute set cmd_build requires; quiet/
    log_level are left as explicit parameters here since that is exactly
    what this module varies.
    """
    return argparse.Namespace(
        input=input_path,
        output=output_path,
        variant=variant,
        seq_column=None,
        v_column="v_call",
        j_column="j_call",
        abundance_column=None,
        no_genes=False,
        smoothing=0.0,
        strict_input=False,
        expect_format=None,
        quiet=quiet,
        log_level=log_level,
    )


def test_nucleotide_data_on_amino_acid_engine_warns(tmp_path, capsys):
    path = tmp_path / "nucleotide.fasta"
    path.write_text(_fasta(_NUCLEOTIDE_SEQS))
    out_path = tmp_path / "out.lzg"

    cmd_build(_namespace(str(path), str(out_path), variant="aap"))

    err = capsys.readouterr().err
    assert "alphabet=nucleotide" in err
    assert "variant=aap" in err
    assert "expected=amino_acid" in err
    # Never an error: the build must still have produced a graph.
    assert LZGraph.load(str(out_path)).n_nodes > 0


def test_amino_acid_data_on_nucleotide_engine_warns(tmp_path, capsys):
    """The reverse direction: aap-shaped data fed to the ndp engine."""
    path = tmp_path / "amino_acid.fasta"
    path.write_text(_fasta(_AMINO_ACID_SEQS))
    out_path = tmp_path / "out.lzg"

    cmd_build(_namespace(str(path), str(out_path), variant="ndp"))

    err = capsys.readouterr().err
    assert "alphabet=amino_acid" in err
    assert "variant=ndp" in err
    assert "expected=nucleotide" in err


def test_amino_acid_data_on_amino_acid_engine_does_not_warn(tmp_path, capsys):
    """The matching case: no mismatch, so nothing to warn about."""
    path = tmp_path / "amino_acid.fasta"
    path.write_text(_fasta(_AMINO_ACID_SEQS))
    out_path = tmp_path / "out.lzg"

    cmd_build(_namespace(str(path), str(out_path), variant="aap"))

    err = capsys.readouterr().err
    assert "alphabet" not in err


def test_nucleotide_data_on_nucleotide_engine_does_not_warn(tmp_path, capsys):
    path = tmp_path / "nucleotide.fasta"
    path.write_text(_fasta(_NUCLEOTIDE_SEQS))
    out_path = tmp_path / "out.lzg"

    cmd_build(_namespace(str(path), str(out_path), variant="ndp"))

    err = capsys.readouterr().err
    assert "alphabet" not in err


def test_naive_engine_never_warns_regardless_of_alphabet(tmp_path, capsys):
    """naive is documented to accept either alphabet (see module docstring):
    a mismatch check would be actively wrong here, not merely unnecessary.
    """
    path = tmp_path / "nucleotide.fasta"
    path.write_text(_fasta(_NUCLEOTIDE_SEQS))
    out_path = tmp_path / "out.lzg"

    cmd_build(_namespace(str(path), str(out_path), variant="naive"))

    err = capsys.readouterr().err
    assert "alphabet" not in err


def test_mismatch_warning_is_suppressed_by_quiet(tmp_path, capsys):
    path = tmp_path / "nucleotide.fasta"
    path.write_text(_fasta(_NUCLEOTIDE_SEQS))
    out_path = tmp_path / "out.lzg"

    cmd_build(_namespace(str(path), str(out_path), variant="aap", quiet=True))

    err = capsys.readouterr().err
    assert "alphabet" not in err
    # Quiet must not have blocked the build itself.
    assert LZGraph.load(str(out_path)).n_nodes > 0


def test_mismatch_warning_is_suppressed_by_log_level_below_warn(tmp_path, capsys):
    path = tmp_path / "nucleotide.fasta"
    path.write_text(_fasta(_NUCLEOTIDE_SEQS))
    out_path = tmp_path / "out.lzg"

    cmd_build(_namespace(str(path), str(out_path), variant="aap", log_level="error"))

    err = capsys.readouterr().err
    assert "alphabet" not in err


def test_mismatch_warning_shown_at_default_log_level(tmp_path, capsys):
    """Explicit log_level='warn' (not just the quiet=False default) still shows it."""
    path = tmp_path / "nucleotide.fasta"
    path.write_text(_fasta(_NUCLEOTIDE_SEQS))
    out_path = tmp_path / "out.lzg"

    cmd_build(_namespace(str(path), str(out_path), variant="aap", log_level="warn"))

    err = capsys.readouterr().err
    assert "alphabet=nucleotide" in err
