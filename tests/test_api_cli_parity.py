"""``LZGraph.from_file``/``FlashBackGraph.from_file`` must agree with ``lzg build``.

Plan 1d, Task 1: the safety gate that decides whether a file's raw bytes are
safe to stream straight into the C builder used to live only in
``cli.py::cmd_build`` (see ``tests/test_cli_fast_path_parity.py``). Before
this fix, ``LZGraph.from_file`` and ``FlashBackGraph.from_file`` never ran
that gate at all: they streamed every file unconditionally through the C
reader, bypassing ``detect_format`` entirely. A user who wrote
``LZGraph.from_file(path)`` instead of using ``lzg build`` therefore got a
silently different (and, for FASTA/CSV input, silently *wrong*) graph than
the CLI would have built from the exact same file, with no warning and exit
code 0.

The fix moves the routing decision into
``LZGraphs._io.plan_streaming_read``, a single implementation shared by
``cmd_build``, ``LZGraph.from_file``, and ``FlashBackGraph.from_file``. This
module drives every case from the defect table in the Plan 1d task
description (clean plain, BOM plain, FASTA, CSV, a lone single-column
header, a ``sequence<TAB>0`` edge case, and bzip2), plus FASTQ, gzip, xz, and
an ordinary ``sequence<TAB>count`` file, through all three entry points --
``lzg build`` (``cmd_build``), ``LZGraph.from_file``, and
``LZGraph(**read_sequences(path))`` -- and asserts they all agree on node
count. The same is repeated for ``FlashBackGraph``.

The FASTA and CSV cases are this project's two chartered corruption
defects (fixed for the CLI in earlier plans, but still live in the public
Python API until this fix): before the fix in this module's target commit,
``LZGraph.from_file`` and ``FlashBackGraph.from_file`` disagreed with
``cmd_build``/``read_sequences`` on both (see
``test_fasta_and_csv_diverge_before_the_fix`` below, which reproduces that
divergence directly against the pre-fix code path by calling the C builder
the same way the old ``from_file`` did, so this module does not merely
assert a property that was never wired up to fail).
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import lzma
import os
from unittest import mock

import pytest

from LZGraphs import FlashBackGraph, LZGraph
from LZGraphs import _clzgraph as _c
from LZGraphs._io import read_sequences
from LZGraphs.cli import _fb_build, cmd_build

SEQUENCES = [
    "CASSLGQAYEQYF", "CASSPGTGVYGYTF", "CASSIRSSYEQYF", "CASSDTSGGTDTQYF",
    "CASSFGQGSYEQYF", "CASSQDSSYEQYF", "CASSEGTGNTIYF", "CASSLAGGYNEQFF",
    "CASSPWGQGAFF", "CASSYSTDTQYF", "CASSLGTGGYEQYF", "CASSFRDRGNTIYF",
    "CASSQEGSTDTQYF", "CASSIGTSGYTF", "CASSLNRDTQYF", "CASSDRGQGYEQYF",
    "CASSPGQGATEAFF", "CASSLWGNTEAFF", "CASSQGTSGSYNEQFF", "CASSPRDSSYEQYF",
]
COUNTS = list(range(1, len(SEQUENCES) + 1))


def _plain_text():
    return "".join(f"{s}\n" for s in SEQUENCES)


def _bom_plain_bytes():
    return b"\xef\xbb\xbf" + _plain_text().encode()


def _fasta_bytes():
    return "".join(f">s{i}\n{s}\n" for i, s in enumerate(SEQUENCES)).encode()


def _csv_bytes():
    header = "junction_aa,duplicate_count\n"
    rows = "".join(f"{s},{c}\n" for s, c in zip(SEQUENCES, COUNTS))
    return (header + rows).encode()


def _fastq_bytes():
    return "".join(
        f"@s{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(SEQUENCES)
    ).encode()


def _lone_header_bytes():
    # A header naming no known sequence column (see
    # ``_io._sniff._ALL_SEQ_COLUMN_NAMES``), so ``detect_format`` sniffs the
    # file as ordinary ``plain`` and only well-formedness filtering (or the
    # streaming gate) catches the header as not itself a real sequence.
    return ("amino_acid_sequence\n" + _plain_text()).encode()


def _seqcount_zero_bytes():
    # The first record's count is the pathological "0" edge case; the rest
    # are "1", keeping the file safely recognized as plain_seqcount shape
    # (mirrors tests/test_cli_fast_path_parity.py's abundance-parity cases).
    lines = [f"{SEQUENCES[0]}\t0\n"]
    lines += [f"{s}\t1\n" for s in SEQUENCES[1:]]
    return "".join(lines).encode()


def _seqcount_bytes():
    return "".join(f"{s}\t{c}\n" for s, c in zip(SEQUENCES, COUNTS)).encode()


# name -> (raw-bytes renderer, file suffix)
_CASES = {
    "clean_plain": (lambda: _plain_text().encode(), ".txt"),
    "bom_plain": (_bom_plain_bytes, ".txt"),
    "fasta": (_fasta_bytes, ".fasta"),
    "csv": (_csv_bytes, ".csv"),
    "lone_header": (_lone_header_bytes, ".txt"),
    "seqcount_zero_count": (_seqcount_zero_bytes, ".txt"),
    "fastq": (_fastq_bytes, ".fastq"),
    "seqcount": (_seqcount_bytes, ".txt"),
}

# name -> (compressor, extra suffix), applied to the clean plain-text bytes,
# closing the specific hole this defect was measured against: bzip2/xz/gzip
# input has no fast-path suffix to catch it, yet was streamed unconditionally
# through the C builder by the pre-fix from_file methods (bzip2 raised
# UnicodeDecodeError; gzip/xz built a silently wrong graph from compressed
# bytes).
_COMPRESSED_CASES = {
    "gzip": (gzip.compress, ".gz"),
    "bzip2": (bz2.compress, ".bz2"),
    "xz": (lzma.compress, ".xz"),
}


def _lzgraph_namespace(input_path, output_path):
    return argparse.Namespace(
        input=input_path, output=output_path, variant="aap", seq_column=None,
        v_column="v_call", j_column="j_call", abundance_column=None,
        no_genes=False, smoothing=0.0, strict_input=False, expect_format=None,
        quiet=True, log_level="none",
    )


def _fb_namespace(input_path, output_path):
    return argparse.Namespace(
        input=input_path, output=output_path, smoothing=0.0,
        quiet=True, log_level="none",
    )


def _lzgraph_cli_nodes(path, tmp_path, tag):
    out_path = os.path.join(str(tmp_path), f"{tag}.lzg")
    cmd_build(_lzgraph_namespace(str(path), out_path))
    return LZGraph.load(out_path).n_nodes


def _lzgraph_from_file_nodes(path):
    return LZGraph.from_file(str(path), variant="aap").n_nodes


def _lzgraph_read_sequences_nodes(path):
    data = read_sequences(str(path), variant="aap")
    g = LZGraph(
        data["sequences"], variant="aap", abundances=data["abundances"],
        v_genes=data["v_genes"], j_genes=data["j_genes"],
    )
    return g.n_nodes


def _fb_cli_nodes(path, tmp_path, tag):
    out_path = os.path.join(str(tmp_path), f"{tag}.lzg")
    _fb_build(_fb_namespace(str(path), out_path))
    return FlashBackGraph.load(out_path).n_nodes


def _fb_from_file_nodes(path):
    return FlashBackGraph.from_file(str(path)).n_nodes


def _fb_read_sequences_nodes(path):
    data = read_sequences(str(path), variant="aap", no_genes=True)
    g = FlashBackGraph(data["sequences"], abundances=data["abundances"])
    return g.n_nodes


@pytest.mark.parametrize("case", sorted(_CASES))
def test_lzgraph_entry_points_agree(tmp_path, case):
    """``lzg build``, ``LZGraph.from_file``, and ``LZGraph(**read_sequences(...))``
    must build the identical graph (same node count) for every case in the
    defect table, uncompressed.
    """
    render, suffix = _CASES[case]
    path = tmp_path / f"{case}{suffix}"
    path.write_bytes(render())

    n_cli = _lzgraph_cli_nodes(path, tmp_path, f"{case}_cli")
    n_from_file = _lzgraph_from_file_nodes(path)
    n_read_sequences = _lzgraph_read_sequences_nodes(path)

    assert n_from_file == n_cli, (
        f"{case}: LZGraph.from_file has {n_from_file} nodes, "
        f"lzg build has {n_cli} nodes"
    )
    assert n_read_sequences == n_cli, (
        f"{case}: LZGraph(**read_sequences(...)) has {n_read_sequences} "
        f"nodes, lzg build has {n_cli} nodes"
    )


@pytest.mark.parametrize("codec", sorted(_COMPRESSED_CASES))
def test_lzgraph_entry_points_agree_compressed(tmp_path, codec):
    """The same three-way agreement, for the clean sequence set compressed
    with each supported codec. Closes the specific hole this defect was
    measured against: compressed input has no fast-path suffix to catch it
    (unlike a ``.gz``-named file, a bzip2/xz file looks exactly like any
    other file to a filename-based check), yet the pre-fix ``from_file``
    methods streamed it unconditionally through the C builder anyway
    (bzip2 raised ``UnicodeDecodeError``; gzip/xz silently built a wrong
    graph from the still-compressed bytes).
    """
    compress, suffix = _COMPRESSED_CASES[codec]
    path = tmp_path / f"clean_{codec}{suffix}"
    path.write_bytes(compress(_plain_text().encode()))

    n_cli = _lzgraph_cli_nodes(path, tmp_path, f"{codec}_cli")
    n_from_file = _lzgraph_from_file_nodes(path)
    n_read_sequences = _lzgraph_read_sequences_nodes(path)

    assert n_from_file == n_cli, (
        f"{codec}: LZGraph.from_file has {n_from_file} nodes, "
        f"lzg build has {n_cli} nodes"
    )
    assert n_read_sequences == n_cli


@pytest.mark.parametrize("case", sorted(_CASES))
def test_flashbackgraph_entry_points_agree(tmp_path, case):
    """The same three-way agreement for ``FlashBackGraph``: ``lzg flashback
    build`` (``_fb_build``), ``FlashBackGraph.from_file``, and
    ``FlashBackGraph(**read_sequences(..., no_genes=True))``.
    """
    render, suffix = _CASES[case]
    path = tmp_path / f"fb_{case}{suffix}"
    path.write_bytes(render())

    n_cli = _fb_cli_nodes(path, tmp_path, f"fb_{case}_cli")
    n_from_file = _fb_from_file_nodes(path)
    n_read_sequences = _fb_read_sequences_nodes(path)

    assert n_from_file == n_cli, (
        f"{case}: FlashBackGraph.from_file has {n_from_file} nodes, "
        f"lzg flashback build has {n_cli} nodes"
    )
    assert n_read_sequences == n_cli, (
        f"{case}: FlashBackGraph(**read_sequences(...)) has "
        f"{n_read_sequences} nodes, lzg flashback build has {n_cli} nodes"
    )


@pytest.mark.parametrize("codec", sorted(_COMPRESSED_CASES))
def test_flashbackgraph_entry_points_agree_compressed(tmp_path, codec):
    compress, suffix = _COMPRESSED_CASES[codec]
    path = tmp_path / f"fb_clean_{codec}{suffix}"
    path.write_bytes(compress(_plain_text().encode()))

    n_cli = _fb_cli_nodes(path, tmp_path, f"fb_{codec}_cli")
    n_from_file = _fb_from_file_nodes(path)
    n_read_sequences = _fb_read_sequences_nodes(path)

    assert n_from_file == n_cli, (
        f"{codec}: FlashBackGraph.from_file has {n_from_file} nodes, "
        f"lzg flashback build has {n_cli} nodes"
    )
    assert n_read_sequences == n_cli


# ---------------------------------------------------------------------------
# Prove the FASTA/CSV rows actually fail before the fix, not merely in
# principle. This calls the C builder directly the same unconditional way
# the pre-fix ``LZGraph.from_file``/``FlashBackGraph.from_file`` did (no
# ``plan_streaming_read`` gate at all), so it stays a faithful reproduction
# of the old code path even after the source fix, instead of quietly
# becoming a no-op once ``from_file`` itself is patched.
# ---------------------------------------------------------------------------


def test_fasta_and_csv_diverge_before_the_fix(tmp_path):
    """Reproduces the pre-fix defect directly: streaming FASTA/CSV bytes
    unconditionally through the C plain-text builder (what ``from_file`` did
    before ``plan_streaming_read`` existed) ingests header lines and
    delimiters as literal sequence data, producing a DIFFERENT node count
    than the correctly-parsed graph. If this ever stops diverging, the
    scenario has stopped being a faithful reproduction of the historical bug
    and this test (not the fix) should be revisited.
    """
    fasta_path = tmp_path / "reproduce.fasta"
    fasta_path.write_bytes(_fasta_bytes())
    csv_path = tmp_path / "reproduce.csv"
    csv_path.write_bytes(_csv_bytes())

    correct_fasta = _lzgraph_read_sequences_nodes(fasta_path)
    correct_csv = _lzgraph_read_sequences_nodes(csv_path)

    unconditional_fasta = LZGraph._from_capsule(
        _c.graph_build_file(str(fasta_path), "aap", 0.0)
    ).n_nodes
    unconditional_csv = LZGraph._from_capsule(
        _c.graph_build_file(str(csv_path), "aap", 0.0)
    ).n_nodes

    assert unconditional_fasta != correct_fasta, (
        "the raw C builder no longer diverges from the correct FASTA parse; "
        "this scenario has stopped reproducing the historical defect"
    )
    assert unconditional_csv != correct_csv, (
        "the raw C builder no longer diverges from the correct CSV parse; "
        "this scenario has stopped reproducing the historical defect"
    )


# ---------------------------------------------------------------------------
# Constant-memory streaming must survive the routing fix: a clean,
# uncompressed plain file is why from_file exists, and the new
# plan_streaming_read gate must still choose the C builder for it, not
# silently fall back to materializing a Python list for every large file.
# ---------------------------------------------------------------------------


def _many_clean_lines(n):
    return "".join(f"{SEQUENCES[i % len(SEQUENCES)]}\n" for i in range(n))


def test_lzgraph_from_file_still_streams_clean_plain_input(tmp_path):
    """A large, clean, uncompressed plain file must still reach the C
    builder (``_c.graph_build_file``) directly from ``LZGraph.from_file``,
    not the ``read_sequences``-plus-``LZGraph(...)`` fallback: that constant-
    memory path is the reason ``from_file`` exists, and the routing fix must
    narrow when the fast path is taken, not remove it for the common case.
    """
    path = tmp_path / "large_clean.txt"
    path.write_text(_many_clean_lines(500))

    with mock.patch.object(_c, "graph_build_file", wraps=_c.graph_build_file) as spy:
        g = LZGraph.from_file(str(path), variant="aap")
        assert spy.called, (
            "a large, clean, uncompressed plain file should still reach "
            "the C streaming builder directly"
        )
    assert g.n_nodes > 0


def test_flashbackgraph_from_file_still_streams_clean_plain_input(tmp_path):
    """The same guarantee for ``FlashBackGraph.from_file``."""
    path = tmp_path / "fb_large_clean.txt"
    path.write_text(_many_clean_lines(500))

    with mock.patch.object(_c, "fb_graph_build_file", wraps=_c.fb_graph_build_file) as spy:
        g = FlashBackGraph.from_file(str(path))
        assert spy.called, (
            "a large, clean, uncompressed plain file should still reach "
            "the C streaming builder directly"
        )
    assert g.n_nodes > 0


def test_cmd_build_still_streams_clean_plain_input_via_from_file(tmp_path):
    """End-to-end: ``lzg build`` on a large clean plain file must still
    reach ``LZGraph.from_file``, mirroring
    ``tests/test_cli_fast_path_parity.py::test_clean_file_longer_than_the_sniff_prefix_still_streams``,
    which pins the same guarantee for ``cmd_build`` itself.
    """
    path = tmp_path / "cli_large_clean.txt"
    path.write_text(_many_clean_lines(500))
    out_path = tmp_path / "out.lzg"

    with mock.patch.object(LZGraph, "from_file", wraps=LZGraph.from_file) as spy:
        cmd_build(_lzgraph_namespace(str(path), str(out_path)))
        assert spy.called, (
            "a large, clean, uncompressed plain file should still take "
            "cmd_build's streaming fast path"
        )
