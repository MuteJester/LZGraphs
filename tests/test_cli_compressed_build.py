"""Regression test for the bzip2/xz/misnamed-gzip streaming-corruption bug.

``cmd_build``'s raw-streaming fast path (``LZGraph.from_file``) reads bytes
straight off disk with no decompression. Before the fix, the gate that
decides whether to take that fast path (``can_stream_plain`` in
``LZGraphs/cli.py``) trusted the filename: it excluded only names ending in
``.gz``. A bzip2 file, an xz file, or a gzip file with a misleading ``.txt``
name all passed the gate, so the raw compressed bytes were streamed straight
into the C graph builder, silently producing a small, wrong graph built
from binary garbage instead of raising an error. Exit code 0, no warning.

This test builds a graph from each of those three inputs through
``LZGraphs.cli.cmd_build`` (the exact function ``lzg build`` dispatches to)
and checks the resulting graph's ``n_nodes`` against the graph built from
the same content uncompressed. Before the fix this assertion fails (fewer,
wrong nodes from garbage bytes); after the fix it passes because
``cmd_build`` now decides streaming eligibility from the real detected
compression codec (via ``detect_format``), not the filename, and falls back
to the decompressing ``read_sequences`` path for anything actually
compressed.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import lzma
import os

import pytest

from LZGraphs import LZGraph
from LZGraphs.cli import cmd_build

_CONTENT = (
    "CASSLGQAYEQYF\n"
    "CASSPGTGVYGYTF\n"
    "CASSIRSSYEQYF\n"
    "CASSDTSGGTDTQYF\n"
    "CASSFGQGSYEQYF\n"
)


def _build_namespace(input_path, output_path):
    """The subset of argparse.Namespace fields cmd_build reads."""
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


def _build_and_load(path, out_dir, name):
    out_path = os.path.join(str(out_dir), f"{name}.lzg")
    cmd_build(_build_namespace(path, out_path))
    return LZGraph.load(out_path)


@pytest.fixture
def reference_n_nodes(tmp_path):
    """n_nodes from building the same content uncompressed."""
    plain_path = tmp_path / "reference.txt"
    plain_path.write_text(_CONTENT)
    g = _build_and_load(str(plain_path), tmp_path, "reference")
    assert g.n_nodes > 0
    return g.n_nodes


@pytest.mark.parametrize(
    "filename,compressor",
    [
        ("sequences.txt.bz2", bz2.compress),
        ("sequences.txt.xz", lzma.compress),
        # A gzip file with a misleading ".txt" name: the OLD gate excluded
        # compressed input by checking for a literal ".gz" suffix, so this
        # case slipped through even though gzip decompression existed.
        ("misnamed.txt", gzip.compress),
    ],
    ids=["bzip2", "xz", "misnamed_gzip"],
)
def test_compressed_input_builds_the_real_sequences(
    tmp_path, reference_n_nodes, filename, compressor
):
    compressed_path = tmp_path / filename
    compressed_path.write_bytes(compressor(_CONTENT.encode("utf-8")))

    g = _build_and_load(str(compressed_path), tmp_path, "compressed")

    # This is the assertion that actually catches the corruption: streaming
    # raw compressed bytes into the C builder produces some small graph (it
    # doesn't crash: arbitrary bytes still "decode" into a handful of bogus
    # nodes), so only comparing against the true node count from the same
    # content built uncompressed distinguishes "built the real sequences"
    # from "built garbage that happens to load".
    assert g.n_nodes == reference_n_nodes
