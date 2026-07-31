"""Plan 1d, Task 2: make the from_file/cmd_build parity property permanent.

``tests/test_api_cli_parity.py`` pins the specific cases from the Plan 1d
defect table by hand. A hand-written table can go stale: a newly supported
format or codec has no reason to be added to that list, and silently isn't.

This module instead reuses the exact ``RENDERERS``/``CODECS`` matrix
``tests/test_io_matrix.py`` already defines for ``read_sequences`` (6 formats
x 4 codecs) and asserts, for every cell, that ``lzg build``
(``cli.cmd_build``), ``LZGraph.from_file``, and
``LZGraph(**read_sequences(path))`` all agree on the built graph's node
count. Because the matrix is imported rather than re-declared, a format or
codec added to ``test_io_matrix.py`` in the future is automatically exercised
here too, with no separate update needed to keep this test current.

It also covers ``FlashBackGraph``: since the README steers users toward
FlashBackGraph as the primary engine, ``FlashBackGraph.from_file`` and
``FlashBackGraph(**read_sequences(path, no_genes=True))`` must agree across
the same matrix (``FlashBackGraph`` has no CLI-independent ``lzg build``
counterpart worth checking separately: ``lzg flashback build`` calls
``FlashBackGraph.from_file`` directly for any real file, so it can never
diverge from it on its own).
"""
from __future__ import annotations

import argparse
import os

import pytest
from test_io_matrix import CODECS, RENDERERS

from LZGraphs import FlashBackGraph, LZGraph
from LZGraphs._io import read_sequences
from LZGraphs.cli import cmd_build


def _lzgraph_namespace(input_path, output_path):
    return argparse.Namespace(
        input=input_path, output=output_path, variant="aap", seq_column=None,
        v_column="v_call", j_column="j_call", abundance_column=None,
        no_genes=False, smoothing=0.0, strict_input=False, expect_format=None,
        quiet=True, log_level="none",
    )


def _cli_nodes(path, tmp_path, tag):
    out_path = os.path.join(str(tmp_path), f"{tag}.lzg")
    cmd_build(_lzgraph_namespace(str(path), out_path))
    return LZGraph.load(out_path).n_nodes


def _from_file_nodes(path):
    return LZGraph.from_file(str(path), variant="aap").n_nodes


def _read_sequences_lzgraph_nodes(path):
    data = read_sequences(str(path), variant="aap")
    g = LZGraph(
        data["sequences"], variant="aap", abundances=data["abundances"],
        v_genes=data["v_genes"], j_genes=data["j_genes"],
    )
    return g.n_nodes


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
@pytest.mark.parametrize("codec", sorted(CODECS))
def test_all_three_entry_points_agree_across_the_matrix(tmp_path, fmt, codec):
    """``lzg build``, ``LZGraph.from_file``, and
    ``LZGraph(**read_sequences(...))`` must build the same graph (same node
    count) for every format x codec cell ``test_io_matrix.py`` defines.
    """
    compress, suffix = CODECS[codec]
    path = tmp_path / f"data_{fmt}{suffix}"
    path.write_bytes(compress(RENDERERS[fmt]().encode()))

    n_cli = _cli_nodes(path, tmp_path, f"cli_{fmt}_{codec}")
    n_from_file = _from_file_nodes(path)
    n_read_sequences = _read_sequences_lzgraph_nodes(path)

    assert n_from_file == n_cli, (
        f"{fmt}/{codec}: LZGraph.from_file has {n_from_file} nodes, "
        f"lzg build has {n_cli} nodes"
    )
    assert n_read_sequences == n_cli, (
        f"{fmt}/{codec}: LZGraph(**read_sequences(...)) has "
        f"{n_read_sequences} nodes, lzg build has {n_cli} nodes"
    )


def _fb_from_file_nodes(path):
    return FlashBackGraph.from_file(str(path)).n_nodes


def _fb_read_sequences_nodes(path):
    data = read_sequences(str(path), variant="aap", no_genes=True)
    g = FlashBackGraph(data["sequences"], abundances=data["abundances"])
    return g.n_nodes


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
@pytest.mark.parametrize("codec", sorted(CODECS))
def test_flashbackgraph_from_file_matches_read_sequences_across_the_matrix(
    tmp_path, fmt, codec
):
    """``FlashBackGraph.from_file`` and
    ``FlashBackGraph(**read_sequences(path, no_genes=True))`` must agree on
    node count for every format x codec cell: FlashBackGraph is the engine
    the README steers users toward as the primary file-based entry point, so
    its ``from_file`` must not silently disagree with the same input read
    through the general-purpose pipeline.
    """
    compress, suffix = CODECS[codec]
    path = tmp_path / f"fb_data_{fmt}{suffix}"
    path.write_bytes(compress(RENDERERS[fmt]().encode()))

    n_from_file = _fb_from_file_nodes(path)
    n_read_sequences = _fb_read_sequences_nodes(path)

    assert n_from_file == n_read_sequences, (
        f"{fmt}/{codec}: FlashBackGraph.from_file has {n_from_file} nodes, "
        f"FlashBackGraph(**read_sequences(...)) has {n_read_sequences} nodes"
    )
