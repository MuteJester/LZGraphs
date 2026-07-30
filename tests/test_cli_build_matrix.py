"""Format x codec matrix coverage for ``cli.cmd_build``'s streaming fast path.

``tests/test_io_matrix.py`` exercises 6 formats x 4 codecs, but every cell
goes through ``read_sequences``. ``cmd_build`` has a *separate* code path:
when the input is uncompressed plain or plain_seqcount text, it streams
straight into the C builder via ``LZGraph.from_file`` instead of going
through ``read_sequences`` first. That path used to be gated by a filename
check (``endswith('.gz')``), so a bzip2 or xz file -- having neither a
``.gz`` suffix nor any other disqualifying name -- passed the gate and had
its raw compressed bytes streamed straight into the C builder: exit code 0,
no warning, a small graph built from binary garbage instead of the real
sequences. No test drove ``cmd_build`` itself across the format/codec
matrix, so twelve individual task reviews stayed green; only a whole-branch
review caught it.

This module closes that hole: it drives ``cli.cmd_build`` -- the exact
function ``lzg build`` dispatches to -- over the same matrix as
``test_io_matrix.py``, and asserts the built graph's node count equals the
node count built from the same sequences as plain, uncompressed,
unwrapped text. Node-count equality is the assertion that matters: merely
checking that no exception was raised would NOT have caught the original
bug, because the corrupted build succeeded with exit code 0.
"""
from __future__ import annotations

import argparse
import os

import pytest
from test_io_matrix import CODECS, RENDERERS, SEQUENCES

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
    """n_nodes from building SEQUENCES as truly unwrapped plain text.

    This is deliberately NOT one of the matrix cells below (not even the
    "plain"/"none" cell): it is built from a freshly rendered plain-text
    file in its own tmp directory, independent of anything the matrix loop
    touches, so a bug specific to one cell can never contaminate the
    reference it is compared against.
    """
    tmp_path = tmp_path_factory.mktemp("reference")
    plain_path = tmp_path / "reference.txt"
    plain_path.write_text("".join(f"{s}\n" for s in SEQUENCES))
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
        "abundances changed the node count -- the matrix below can no "
        "longer compare count-carrying cells against the plain reference"
    )


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
@pytest.mark.parametrize("codec", sorted(CODECS))
def test_build_matrix_node_count(tmp_path, reference_n_nodes, fmt, codec):
    """Every format x codec cell, built through cli.cmd_build, must produce
    a graph with the same node count as the same sequences built from
    plain, uncompressed, unwrapped text.

    This specifically exercises cmd_build's ``can_stream_plain`` gate: for
    the "plain"/"plain_seqcount"-shaped renderers ("plain" and "seqcount")
    under the "none" codec, cmd_build takes the raw-streaming
    ``LZGraph.from_file`` fast path. Every other cell -- any compressed
    codec, or a format the fast path does not recognize (fasta, fastq,
    tsv, csv) -- must fall back to the safe ``read_sequences`` path. If the
    gate ever regresses to trusting the filename instead of the real
    detected compression, a compressed file streamed raw into the C
    builder still "succeeds" (no exception) but yields a small, wrong node
    count, which is exactly what this assertion catches.
    """
    compress, suffix = CODECS[codec]
    path = tmp_path / f"data_{fmt}{suffix}"
    path.write_bytes(compress(RENDERERS[fmt]().encode()))

    g = _build_and_load(path, tmp_path, f"built_{fmt}_{codec}")

    assert g.n_nodes == reference_n_nodes, (
        f"{fmt}/{codec}: built graph has {g.n_nodes} nodes, "
        f"reference (plain, uncompressed) has {reference_n_nodes}"
    )
