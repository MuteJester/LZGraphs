"""Format x codec matrix coverage for ``cli.cmd_build``'s streaming fast path.

``tests/test_io_matrix.py`` exercises 6 formats x 4 codecs, but every cell
goes through ``read_sequences``. ``cmd_build`` has a *separate* code path:
``can_stream_plain`` (``cli.py``, near line 174) is true only when
``detect_format`` reports ``compression == 'none'`` and
``format in ('plain', 'plain_seqcount')``; only then does ``cmd_build``
stream straight into the C builder via ``LZGraph.from_file`` instead of
going through ``read_sequences``. That gate used to be decided from the
filename (``endswith('.gz')``) instead of the real detected compression, so
a bzip2 or xz file, having neither a ``.gz`` suffix nor any other
disqualifying name, passed the gate and had its raw compressed bytes
streamed straight into the C builder: exit code 0, no warning, a small
graph built from binary garbage instead of the real sequences. No test
drove ``cmd_build`` itself across the format/codec matrix, so twelve
individual task reviews stayed green; only a whole-branch review caught it.

This module drives ``cli.cmd_build`` (the exact function ``lzg build``
dispatches to) over the same 24-cell matrix as ``test_io_matrix.py``, and
asserts the built graph's node count equals the node count built from the
same sequences as plain, uncompressed, unwrapped text. Node-count equality
is the assertion that matters: merely checking that no exception was
raised would NOT have caught the original bug, because the corrupted
build succeeded with exit code 0.

Be precise about what each cell actually exercises, since the matrix shape
invites the same false confidence that let the original bug through
twelve reviews. Not every cell reaches the streaming path:

- ``plain``/``none`` and ``seqcount``/``none`` (2 cells) actually call
  ``LZGraph.from_file``: these are the only two cells where
  ``can_stream_plain`` is true today.
- ``plain`` and ``seqcount`` under ``gzip``/``bzip2``/``xz`` (6 cells) are
  negative-space regression tests of the compression gate itself: they
  confirm compressed plain/plain_seqcount input is correctly *refused* the
  fast path and falls back to ``read_sequences``. These are exactly the
  cells that fail (wrong, non-crashing node counts) if the gate regresses
  to a filename check, since bzip2/xz files carry no ``.gz`` suffix.
- ``fasta``, ``fastq``, ``tsv``, ``csv`` under all four codecs (16 cells)
  never touch ``can_stream_plain`` under any codec, because ``detect_format``
  never reports their format as ``plain``/``plain_seqcount``, so they are
  end-to-end ``cmd_build`` coverage that duplicates what
  ``test_io_matrix.py`` already covers through ``read_sequences``. They are
  kept because they are cheap and because they would turn into genuine
  seam coverage the moment the gate is ever widened to recognize more
  formats.

So 8 of the 24 cells (the first two bullets) are what closes the seam this
module exists for; the remaining 16 are breadth, not seam coverage.
"""
from __future__ import annotations

import argparse
import os

import pytest
from test_io_matrix import CODECS, RENDERERS

from LZGraphs import LZGraph
from LZGraphs.cli import cmd_build


def _build_namespace(input_path, output_path):
    """The subset of argparse.Namespace fields cmd_build reads.

    Mirrors tests/test_cli_compressed_build.py's namespace, which is the
    working reference for the exact attribute set cmd_build requires.
    """
    return argparse.Namespace(
        input=input_path,
        output=output_path,
        variant="aap",
        seq_column=None,
        v_column="v_call",
        j_column="j_call",
        abundance_column=None,
        no_genes=False,
        smoothing=0.0,
        strict_input=False,
        expect_format=None,
        quiet=True,
        log_level="none",
    )


def _build_and_load(path, tmp_path, name):
    out_path = os.path.join(str(tmp_path), f"{name}.lzg")
    cmd_build(_build_namespace(str(path), out_path))
    return LZGraph.load(out_path)


@pytest.fixture(scope="module")
def reference_n_nodes(tmp_path_factory):
    """n_nodes from building RENDERERS["plain"]() as truly unwrapped text.

    This is deliberately NOT one of the matrix cells below (not even the
    "plain"/"none" cell): it is built from a freshly rendered plain-text
    file in its own tmp directory, independent of anything the matrix loop
    touches, so a bug specific to one cell can never contaminate the
    reference it is compared against. Calling ``RENDERERS["plain"]()``
    (rather than re-inlining the same sequences by hand) keeps this
    reference from silently drifting out of sync if that renderer ever
    changes.
    """
    tmp_path = tmp_path_factory.mktemp("reference")
    plain_path = tmp_path / "reference.txt"
    plain_path.write_text(RENDERERS["plain"]())
    g = _build_and_load(plain_path, tmp_path, "reference")
    assert g.n_nodes > 0
    return g.n_nodes


def test_count_carrying_format_matches_the_plain_reference(tmp_path, reference_n_nodes):
    """Hazard check: tsv/csv/seqcount carry per-sequence abundances, which
    feed edge weights. Confirm empirically that carrying counts does not
    change *which nodes exist* before relying on a single reference node
    count for the count-carrying cells in the matrix below.
    """
    path = tmp_path / "counts_reference.tsv"
    path.write_text(RENDERERS["tsv"]())
    g = _build_and_load(path, tmp_path, "counts_reference")
    assert g.n_nodes == reference_n_nodes, (
        "abundances changed the node count, so the matrix below can no "
        "longer compare count-carrying cells against the plain reference"
    )


# See the module docstring for which of these 24 cells actually reach
# LZGraph.from_file (plain/none, seqcount/none), which exist to catch a
# regression of the compression gate (plain, seqcount x gzip/bzip2/xz),
# and which are end-to-end cmd_build breadth that duplicates
# test_io_matrix.py (fasta/fastq/tsv/csv x all codecs). If this cell is
# failing, check which category it falls into before assuming the gate
# itself is broken.
@pytest.mark.parametrize("fmt", sorted(RENDERERS))
@pytest.mark.parametrize("codec", sorted(CODECS))
def test_build_matrix_node_count(tmp_path, reference_n_nodes, fmt, codec):
    """Every format x codec cell, built through cli.cmd_build, must produce
    a graph with the same node count as the same sequences built from
    plain, uncompressed, unwrapped text.

    Not every cell exercises cmd_build's ``can_stream_plain`` gate the same
    way; see the module docstring for the exact 2/6/16 breakdown. What
    all 24 cells share is this: if the gate (or anything upstream of it)
    ever lets compressed or wrongly-parsed bytes reach the C builder
    without raising, the build still "succeeds" (exit code 0, no
    exception) but yields a small, wrong node count, which is exactly what
    this assertion catches.
    """
    compress, suffix = CODECS[codec]
    path = tmp_path / f"data_{fmt}{suffix}"
    path.write_bytes(compress(RENDERERS[fmt]().encode()))

    g = _build_and_load(path, tmp_path, f"built_{fmt}_{codec}")

    assert g.n_nodes == reference_n_nodes, (
        f"{fmt}/{codec}: built graph has {g.n_nodes} nodes, "
        f"reference (plain, uncompressed) has {reference_n_nodes}"
    )
