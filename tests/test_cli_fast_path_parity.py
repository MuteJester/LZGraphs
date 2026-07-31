"""``cmd_build``'s streaming fast path must agree with the Python pipeline.

``cmd_build``'s raw-streaming fast path (``LZGraph.from_file``) reads bytes
straight off disk with no decompression and none of ``read_sequences``'s
per-line normalisation: no ``utf-8-sig`` BOM stripping, no universal-newline
translation, and no ``_is_wellformed`` filtering. Before the fix in this
module's target commit, ``can_stream_plain`` (``cli.py``, near line 174)
allowed any uncompressed ``plain``/``plain_seqcount`` file onto that path
regardless of what its raw bytes actually looked like, so a file that needed
any of that normalisation silently built a *different, wrong* graph while
exiting 0 with no warning, and its gzip-compressed twin, which cannot use
the fast path at all, built the *correct* graph. That made the CLI's answer
depend on whether the input happened to be compressed.

The fix narrows ``can_stream_plain`` with ``_io.raw_prefix_is_streamable``,
which declines the fast path whenever the file's raw prefix has a leading
UTF-8 BOM, a lone carriage return, or a first meaningful line that is not
itself a usable sequence. Part (a) below proves each of those scenarios
actually reaches a *different* node count than the clean reference before
that gate change (confirmed by hand against the pre-fix ``cli.py`` snapshot
at commit HEAD: BOM+plain, lone-header, and CR-only builds all diverged from
the clean-plain reference; see the task report for the exact before/after
numbers). Part (b) pins that the gzip counterpart of each scenario, which
never had a fast path to begin with, still agrees with the reference, so
the two paths agree going forward instead of merely "both happening to work
by accident."

This module also covers the second fix: ``read_sequences``'s ``RecordStats``
(``total``/``kept``/``malformed``/``nonproductive``) used to be computed and
then silently discarded by ``cmd_build``, so a file that dropped every record
(e.g. an AIRR file with a blank ``productive`` column) failed with the bare,
uninformative ``LZGraph.__init__`` message ``sequences must be a non-empty
list``. Part (c) covers the fixed, explanatory error; part (d) covers the
warning ``cmd_build`` now emits when only *some* records are dropped.

Round 2 (see the bottom section) closes a gap in the round-1 fix: the
original ``raw_prefix_is_streamable`` only checked the FIRST meaningful line
against ``_is_wellformed``, so a malformed record anywhere after line 1 still
slipped onto the fast path and was ingested (a 3-line file with a bad second
line built 24 nodes uncompressed, via the fast path, versus 19 nodes gzipped,
via ``read_sequences``; see ``test_malformed_record_on_line_two_matches``).
The fix checks every meaningful line within the same bounded sniff prefix
``detect_format`` already reads (``_SNIFF_LINES``, 32 lines), which is
strictly O(prefix) so it does not undermine why the fast path exists:
constant-memory streaming of foundation-scale plain files. That bound is
also why a malformed record beyond the prefix is a documented, accepted
residual rather than a bug: see
``test_malformed_record_beyond_prefix_is_documented_residual`` below, which
pins today's behaviour so a future change to it is a deliberate decision,
not a silent drift.
"""
from __future__ import annotations

import argparse
import gzip
import os
from unittest import mock

import pytest

from LZGraphs import LZGraph
from LZGraphs.cli import cmd_build

SEQUENCES = [
    "CASSLGQAYEQYF", "CASSPGTGVYGYTF", "CASSIRSSYEQYF", "CASSDTSGGTDTQYF",
    "CASSFGQGSYEQYF", "CASSQDSSYEQYF", "CASSEGTGNTIYF", "CASSLAGGYNEQFF",
    "CASSPWGQGAFF", "CASSYSTDTQYF", "CASSLGTGGYEQYF", "CASSFRDRGNTIYF",
    "CASSQEGSTDTQYF", "CASSIGTSGYTF", "CASSLNRDTQYF", "CASSDRGQGYEQYF",
    "CASSPGQGATEAFF", "CASSLWGNTEAFF", "CASSQGTSGSYNEQFF", "CASSPRDSSYEQYF",
]


def _build_namespace(input_path, output_path, *, quiet=True, log_level="none"):
    """The subset of argparse.Namespace fields cmd_build reads.

    Mirrors tests/test_cli_build_matrix.py and
    tests/test_cli_compressed_build.py's namespace helper, the working
    reference for the exact attribute set cmd_build requires.
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
        quiet=quiet,
        log_level=log_level,
    )


def _build_and_load(path, tmp_path, name, **kwargs):
    out_path = os.path.join(str(tmp_path), f"{name}.lzg")
    cmd_build(_build_namespace(str(path), out_path, **kwargs))
    return LZGraph.load(out_path)


def _plain_text():
    return "".join(f"{s}\n" for s in SEQUENCES)


def _bom_plain_bytes():
    return b"\xef\xbb\xbf" + _plain_text().encode()


def _bom_seqcount_bytes():
    body = "".join(f"{s}\t{i + 1}\n" for i, s in enumerate(SEQUENCES))
    return b"\xef\xbb\xbf" + body.encode()


def _lone_header_bytes():
    # A single-column header that is NOT one of the small set of names
    # cli.py's ``detect_format`` special-cases for reclassification as
    # tabular (see ``_io._sniff._ALL_SEQ_COLUMN_NAMES``), so this file is
    # sniffed as ordinary ``plain`` and only ``raw_prefix_is_streamable``'s
    # ``_is_wellformed`` check on the first line catches it.
    return ("amino_acid_sequence\n" + _plain_text()).encode()


def _cr_only_bytes():
    # Old-Mac-style line endings: bare '\r' with no '\n' at all.
    return ("\r".join(SEQUENCES) + "\r").encode()


_SCENARIOS = {
    "bom_plain": _bom_plain_bytes,
    "bom_seqcount": _bom_seqcount_bytes,
    "lone_header_plain": _lone_header_bytes,
    "cr_only": _cr_only_bytes,
}


@pytest.fixture(scope="module")
def reference_n_nodes(tmp_path_factory):
    """n_nodes from building the same sequences as clean, uncompressed,
    unwrapped plain text, independent of anything the scenarios below
    write, so a bug specific to one scenario can never contaminate the
    reference it is compared against.
    """
    tmp_path = tmp_path_factory.mktemp("reference")
    path = tmp_path / "reference.txt"
    path.write_bytes(_plain_text().encode())
    g = _build_and_load(path, tmp_path, "reference")
    assert g.n_nodes > 0
    return g.n_nodes


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_awkward_uncompressed_input_matches_the_clean_reference(
    tmp_path, reference_n_nodes, scenario
):
    """(a) Each awkward-but-uncompressed variant must build the SAME graph
    as the clean reference, through cmd_build's real fast-path gate.

    Before the ``can_stream_plain`` narrowing, every one of these four
    scenarios reached ``LZGraph.from_file`` and built a different (wrong)
    node count than the reference: verified by hand against the pre-fix
    ``cli.py`` (a snapshot of this branch's merge-base commit) with this
    exact SEQUENCES list, where BOM+plain, lone-header, and CR-only all
    diverged from the clean-plain reference's node count while their gzip
    twins (see ``test_gzip_counterpart_still_agrees`` below) already
    matched it. That divergence with no exception and exit code 0 is
    exactly what this assertion catches.
    """
    path = tmp_path / f"{scenario}.txt"
    path.write_bytes(_SCENARIOS[scenario]())

    g = _build_and_load(path, tmp_path, scenario)

    assert g.n_nodes == reference_n_nodes, (
        f"{scenario}: built graph has {g.n_nodes} nodes, "
        f"clean reference has {reference_n_nodes}"
    )


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_gzip_counterpart_still_agrees(tmp_path, reference_n_nodes, scenario):
    """(b) The gzipped twin of each scenario never had a fast path (it fails
    the ``spec.compression == 'none'`` leg of the gate), so it already went
    through the correct, decompressing ``read_sequences`` path before this
    fix and must continue to do so afterwards. Pinning this alongside (a)
    is what proves the two paths now agree, rather than each merely
    happening to produce some graph independently.
    """
    path = tmp_path / f"{scenario}.txt.gz"
    path.write_bytes(gzip.compress(_SCENARIOS[scenario]()))

    g = _build_and_load(path, tmp_path, f"{scenario}_gz")

    assert g.n_nodes == reference_n_nodes, (
        f"{scenario} (gzipped): built graph has {g.n_nodes} nodes, "
        f"clean reference has {reference_n_nodes}"
    )


# ---------------------------------------------------------------------------
# RecordStats surfacing (the second fix)
# ---------------------------------------------------------------------------

_AIRR_ALL_BLANK_PRODUCTIVE = (
    "junction_aa\tduplicate_count\tproductive\n"
    "CASSLGQAYEQYF\t7\t\n"
    "CASSPGTGVYGYTF\t5\t\n"
    "CASSIRSSYEQYF\t2\t\n"
)

_AIRR_SOME_MALFORMED = (
    "junction_aa\tduplicate_count\n"
    "CASSLGQAYEQYF\t7\n"
    "1234567\t3\n"
    "CASSPGTGVYGYTF\t5\n"
)


def test_all_records_dropped_raises_an_explanatory_error(tmp_path):
    """(c) An AIRR file whose ``productive`` column is blank on every row
    drops every record as non-productive. Before the fix, ``cmd_build``
    never looked at ``RecordStats`` at all, so this reached
    ``LZGraph(sequences=[])`` and failed with the bare, context-free
    ``sequences must be a non-empty list`` (verified by hand against the
    pre-fix ``cli.py``: this exact file produces exactly that message).
    The fixed error must instead say how many records were read and why
    none survived.
    """
    path = tmp_path / "all_nonproductive.tsv"
    path.write_text(_AIRR_ALL_BLANK_PRODUCTIVE)
    out_path = tmp_path / "out.lzg"

    with pytest.raises(ValueError) as excinfo:
        cmd_build(_build_namespace(str(path), str(out_path)))

    message = str(excinfo.value)
    assert message != "sequences must be a non-empty list", (
        "error regressed to the bare, uninformative LZGraph message"
    )
    assert "3" in message, "message should name the total records read"
    assert "nonproductive" in message, (
        "message should name why the records were dropped"
    )


def test_some_malformed_records_emit_a_warning_and_the_build_succeeds(
    tmp_path, capsys
):
    """(d) A file with a mix of usable and malformed records must still
    build (the malformed rows are dropped, not fatal), but must no longer
    drop them silently: a warning naming the count goes to stderr.
    """
    path = tmp_path / "some_malformed.tsv"
    path.write_text(_AIRR_SOME_MALFORMED)
    out_path = tmp_path / "out.lzg"

    cmd_build(_build_namespace(
        str(path), str(out_path), quiet=False, log_level=None,
    ))

    assert os.path.exists(out_path)
    g = LZGraph.load(str(out_path))
    assert g.n_nodes > 0

    stderr = capsys.readouterr().err
    assert "malformed=1" in stderr, (
        f"expected a warning naming 1 malformed record, got: {stderr!r}"
    )


# ---------------------------------------------------------------------------
# Round 2: the gate must validate the WHOLE sniff prefix, not just line 1
# ---------------------------------------------------------------------------
#
# raw_prefix_is_streamable originally checked only the first meaningful line
# against _is_wellformed. A malformed record anywhere else in the file (even
# on line 2, even well within the small sniff prefix detect_format already
# reads) still passed the gate and reached LZGraph.from_file, which has no
# _is_wellformed check of its own and ingests it as real data. The fix checks
# every meaningful line within the bounded _SNIFF_LINES (32) prefix.


def _n_clean_lines(n):
    """``n`` well-formed lines, cycling through SEQUENCES as needed."""
    return "".join(f"{SEQUENCES[i % len(SEQUENCES)]}\n" for i in range(n))


_MALFORMED_ON_LINE_TWO = "CASSLGQAYEQYF\nCASS!!BAD\nCASSPGTGVYGYTF\n"
_MALFORMED_AT_THE_END = "CASSLGQAYEQYF\nCASSPGTGVYGYTF\nCASS!!BAD\n"


def _build_plain_and_gz(tmp_path, content, tag):
    plain_path = tmp_path / f"{tag}.txt"
    plain_path.write_bytes(content.encode())
    gz_path = tmp_path / f"{tag}.txt.gz"
    gz_path.write_bytes(gzip.compress(content.encode()))
    return (
        _build_and_load(plain_path, tmp_path, f"{tag}_plain").n_nodes,
        _build_and_load(gz_path, tmp_path, f"{tag}_gz").n_nodes,
    )


def test_malformed_record_on_line_two_matches(tmp_path):
    """(round-2 a) A malformed record on line 2 of a 3-line file must build
    the SAME node count whether the file is plain or gzipped.

    Before this fix, the plain build took the fast path (line 1 alone was
    well-formed) and ingested ``CASS!!BAD`` as real data, building 24 nodes
    (including graph nodes derived straight from the ``!`` characters);
    the gzipped build never had a fast path, dropped the malformed record
    via ``_is_wellformed``, and built 19 nodes. Verified by hand: calling
    ``LZGraph.from_file`` directly on this exact plain file (which is what
    round 1's gate would have chosen) gives 24 nodes, while ``cmd_build`` on
    the gzip twin gives 19, reproducing the reported 24-vs-19 split exactly.
    """
    n_plain, n_gz = _build_plain_and_gz(tmp_path, _MALFORMED_ON_LINE_TWO, "line2")
    assert n_plain == n_gz, (
        f"plain build has {n_plain} nodes, gzip build has {n_gz} nodes; "
        "the malformed second line is disagreeing between the two paths"
    )


def test_malformed_record_at_the_end_is_also_caught(tmp_path):
    """(round-2 b) A malformed record at the END of a short file (still well
    within the 32-line sniff prefix) must also be caught, proving the fix
    checks every line in the prefix and not merely "line 1 and line 2".
    """
    n_plain, n_gz = _build_plain_and_gz(tmp_path, _MALFORMED_AT_THE_END, "end")
    assert n_plain == n_gz, (
        f"plain build has {n_plain} nodes, gzip build has {n_gz} nodes; "
        "a malformed line at the end of the file is disagreeing between "
        "the two paths"
    )


def test_clean_file_longer_than_the_sniff_prefix_still_streams(tmp_path):
    """(round-2 c) A clean plain file with MORE than _SNIFF_LINES (32) lines
    must still take the fast path: the gate was narrowed, not disabled for
    anything longer than one line's worth of checking.
    """
    path = tmp_path / "long_clean.txt"
    path.write_text(_n_clean_lines(40))
    out_path = tmp_path / "out.lzg"

    with mock.patch.object(LZGraph, "from_file", wraps=LZGraph.from_file) as spy:
        cmd_build(_build_namespace(str(path), str(out_path)))
        assert spy.called, (
            "a clean 40-line file should still take the streaming fast path"
        )


def test_malformed_record_beyond_prefix_is_documented_residual(tmp_path):
    """(round-2 d) A malformed record beyond line _SNIFF_LINES (32) is a
    KNOWN, ACCEPTED limitation, not a bug: catching it would require reading
    the whole file up front, defeating the constant-memory streaming the
    fast path exists to provide for large files. This pins TODAY's actual
    behaviour (the fast path is still taken, and the out-of-prefix malformed
    record IS ingested, so the plain and gzip builds disagree) precisely so
    that if this behaviour ever changes, this test fails and tells someone,
    instead of the change drifting in silently.
    """
    content = _n_clean_lines(40) + "CASS!!BAD\n"
    plain_path = tmp_path / "long_then_bad.txt"
    plain_path.write_bytes(content.encode())
    gz_path = tmp_path / "long_then_bad.txt.gz"
    gz_path.write_bytes(gzip.compress(content.encode()))

    with mock.patch.object(LZGraph, "from_file", wraps=LZGraph.from_file) as spy:
        n_plain = _build_and_load(plain_path, tmp_path, "long_then_bad_plain").n_nodes
        assert spy.called, (
            "expected the fast path to be taken for this file; if it is no "
            "longer taken, the residual documented here may already be "
            "fixed and this test should be revisited"
        )

    n_gz = _build_and_load(gz_path, tmp_path, "long_then_bad_gz").n_nodes

    assert n_plain != n_gz, (
        "the plain and gzip builds now agree even with a malformed record "
        "beyond the sniff prefix: either the fast path started validating "
        "beyond _SNIFF_LINES (reconsider the O(prefix) trade-off this test "
        "documents) or it stopped being taken for this file (update this "
        "test and the residual-gap documentation in cli.py / "
        "raw_prefix_is_streamable accordingly)"
    )
