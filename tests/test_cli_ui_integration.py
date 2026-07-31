"""Task 6: wiring the ``_term`` rendering layer into ``cli.cmd_build``.

``cmd_build`` used to report every fact (input format, read/construct/save
phase lines, the alphabet-mismatch warning, the "all records dropped" error)
via bare ``print(..., file=sys.stderr)`` calls gated by ``_cli_log_enabled``.
This module drives the real CLI (``python -m LZGraphs.cli``, exactly like
``tests/test_cli.py`` does) across ``--ui {auto,rich,plain,quiet}`` and
confirms three things a rendering-layer migration could silently break
without any of it showing up as an exception:

1. **stdout purity.** ``build`` writes progress to stderr and the graph to
   the ``-o`` file; stdout must stay empty (and therefore byte-identical)
   regardless of which renderer is doing the reporting.
2. **Fact survival.** Every fact ``cmd_build``'s module docstring/the task
   spec calls out by name still appears somewhere in ``--ui plain``'s
   scrolling output, just reformatted through ``LZGraphs._term.PlainRenderer``
   instead of a literal f-string.
3. **Result parity.** The graph ``build`` produces (node/edge count) is
   identical across all four modes: rendering must never perturb the
   result, only how it is reported.
4. **C-library severity routing (Task 6 fix round 1).** The C library's own
   ``LZG_WARN``/``LZG_ERROR`` messages (a mixed plain/seqcount file the
   streaming reader had to recover from mid-file, for instance) must reach
   a real warning in rich mode too, not just plain: ``_bridge_c_log_to_ui``
   used to drop every C message lacking a ``pct=`` progress field,
   regardless of its actual severity, silently swallowing real warnings.
   ``TestCLogBridge`` covers the routing logic directly (unit-level, driven
   by synthetic ``(level, message)`` pairs); ``TestRichModeOnRealTty``'s
   mixed-format test covers it end to end through a real pty.

The rich-mode checks that need a genuine ``isatty()`` (cursor hide/show,
the boxed panel, no raw ``[LZGraph/INFO]`` leaking into a live redraw) drive
a real pty via ``subprocess.Popen``, mirroring ``tests/test_term_rich.py``'s
own convention of gating pty-only tests individually rather than skipping
the whole module.
"""
from __future__ import annotations

import argparse
import os
import random
import select
import subprocess
import sys

import pytest

from LZGraphs import LZGraph
from LZGraphs._term import NullRenderer, PlainRenderer, RichRenderer
from LZGraphs.cli import (
    _bridge_c_log_to_ui,
    _effective_log_level,
    _resolve_build_ui,
    cmd_build,
)

try:
    import pty

    _HAVE_PTY = True
except ImportError:  # pragma: no cover - exercised only on Windows
    _HAVE_PTY = False

_NO_PTY_REASON = "pty is POSIX-only and unavailable on this platform"

LZG = [sys.executable, '-m', 'LZGraphs.cli']

_UI_MODES = ('auto', 'rich', 'plain', 'quiet')


def run_lzg(*args, input_text=None, env=None):
    """Run the CLI as a real subprocess; stdout/stderr are pipes (non-tty)."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    result = subprocess.run(
        LZG + list(args),
        capture_output=True, text=True, input=input_text, timeout=30,
        env=full_env,
    )
    return result.stdout, result.stderr, result.returncode


def _drain_pty(master_fd, idle=0.2, timeout=5.0):
    """Read whatever a pty produces until a short quiet period, bounded
    overall by ``timeout``. Mirrors ``tests/test_term_rich.py``'s own
    ``_drain`` helper.
    """
    import time as _time

    data = b""
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        r, _, _ = select.select([master_fd], [], [], idle)
        if not r:
            if data:
                break
            continue
        try:
            chunk = os.read(master_fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        data += chunk
    return data


def _run_lzg_on_pty(args, columns=100, lines=30):
    """Run the CLI with stdin/stdout/stderr all attached to one real pty.

    Returns ``(output_bytes, returncode)``. A single pty (rather than a
    stdout pipe plus a pty for stderr) keeps this simple and is enough for
    what these tests check: the rich renderer never writes to stdout at
    all, so nothing here needs to tell the two apart.
    """
    master_fd, slave_fd = pty.openpty()
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src'
    env['COLUMNS'] = str(columns)
    env['LINES'] = str(lines)
    proc = subprocess.Popen(
        LZG + args, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        env=env, preexec_fn=os.setsid,
    )
    os.close(slave_fd)
    try:
        data = _drain_pty(master_fd)
        proc.wait(timeout=10)
    finally:
        os.close(master_fd)
    return data, proc.returncode


@pytest.fixture
def seq_file(tmp_path):
    p = tmp_path / 'seqs.txt'
    p.write_text('CASSLGIRRT\nCASSLGYEQYF\nCASSLEPSGGTDTQYF\n'
                 'CASSDTSGGTDTQYF\nCASSFGQGSYEQYF\nCASSQETQYF\n')
    return str(p)


@pytest.fixture
def facts_file(tmp_path):
    """A tabular AIRR file exercising every ``phase=read`` fact at once:
    kept records, one malformed record (an all-digit ``junction_aa``, which
    ``_is_wellformed`` rejects for having no letters), and one non-productive
    record.
    """
    p = tmp_path / 'facts.tsv'
    p.write_text(
        'sequence_id\tjunction_aa\tv_call\tj_call\tduplicate_count\tproductive\n'
        'seq0\tCASSLGIRRT\tTRBV1\tTRBJ1\t5\tT\n'
        'seq1\tCASSLGYEQYF\tTRBV1\tTRBJ1\t2\tT\n'
        'seq2\tCASSQETQYF\tTRBV1\tTRBJ1\t1\tT\n'
        'seq3\t123456\tTRBV1\tTRBJ1\t1\tT\n'
        'seq4\tCASSDTSGGTDTQYF\tTRBV1\tTRBJ1\t1\tF\n'
    )
    return str(p)


@pytest.fixture
def nucleotide_file(tmp_path):
    """Unambiguous nucleotide data fed to the (amino-acid) ``aap`` engine."""
    p = tmp_path / 'nucleotide.fasta'
    seqs = ["ACGTACGTACGTACGT", "GGCCTTAAGGCCTTAA", "ACGTGGCCAATTGGCC"]
    p.write_text("".join(f">seq{i}\n{s}\n" for i, s in enumerate(seqs)))
    return str(p)


@pytest.fixture
def all_dropped_file(tmp_path):
    p = tmp_path / 'all_dropped.tsv'
    p.write_text(
        'sequence_id\tjunction_aa\tproductive\n'
        'seq0\tCASSLGIRRT\tF\n'
        'seq1\tCASSLGYEQYF\tF\n'
    )
    return str(p)


@pytest.fixture
def mixed_format_file(tmp_path):
    """A file that genuinely reaches the C library's own
    ``mixed_input_format`` warning (fix round 1, Finding 2), rather than
    being caught by the Python-side gate first.

    ``detect_format``/``raw_prefix_is_streamable`` (``_io/_sniff.py``'s
    ``_SNIFF_LINES``/``cmd_build``'s streaming gate) only ever sample the
    first 32 lines, so 40 plain (no-tab) lines establish this as
    unambiguously ``plain`` and streamable; the switch to ``seq<TAB>count``
    lines only happens afterwards, at line 41, past every prefix check, so
    ``can_stream_plain`` is still ``True`` and the actual C ingest loop
    (``lib/graph_core/graph_build_ingest.c``'s ``lzg_update_stream_mode``)
    is the one that notices the format changed mid-file and logs its own
    ``LZG_WARN`` (``issue=mixed_input_format ... continuing=true``) rather
    than raising: this is a *recoverable* warning, not a fatal error, so the
    build still succeeds (``rc == 0``) with a graph built from both parts.
    """
    p = tmp_path / 'mixed_format.txt'
    aas = "ACDEFGHIKLMNPQRSTVWY"
    rng = random.Random(0)
    lines = [
        "".join(rng.choice(aas) for _ in range(rng.randint(10, 15)))
        for _ in range(40)
    ]
    lines += [
        f"{''.join(rng.choice(aas) for _ in range(rng.randint(10, 15)))}\t3"
        for _ in range(5)
    ]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


# ── 1. stdout purity / byte-identity ─────────────────────────────


class TestStdoutPurity:
    def test_stdout_is_empty_and_identical_across_every_ui_mode_streaming_path(
        self, seq_file, tmp_path
    ):
        """``build`` never writes to stdout in any mode: it writes progress
        to stderr and the graph to ``-o``. Byte-identity across modes (and
        with today's pre-``_term`` behaviour, which also never touched
        stdout) reduces to: stdout is empty, in every one of them.

        ``seq_file`` is a bare plain-text file, so this exercises
        ``cmd_build``'s streaming fast path (``can_stream_plain``), which
        returns early; see the sibling ``..._in_memory_path`` test below for
        the other branch, which has its own separate ``return``/fall-through
        and is not exercised by this fixture at all.
        """
        outputs = []
        for mode in _UI_MODES:
            out_path = str(tmp_path / f'out_{mode}.lzg')
            stdout, stderr, rc = run_lzg('--ui', mode, 'build', seq_file, '-o', out_path)
            assert rc == 0, stderr
            outputs.append(stdout)
        assert all(o == '' for o in outputs)
        assert len(set(outputs)) == 1

    def test_stdout_is_empty_and_identical_across_every_ui_mode_in_memory_path(
        self, facts_file, tmp_path
    ):
        """Same invariant as above, over ``cmd_build``'s *other* branch: a
        tabular file (``facts_file``) never qualifies for the streaming fast
        path, so this exercises the ``read_sequences``/in-memory
        construction code path instead, all the way down to its own final
        ``term.done()`` call.
        """
        outputs = []
        for mode in _UI_MODES:
            out_path = str(tmp_path / f'out_{mode}.lzg')
            stdout, stderr, rc = run_lzg(
                '--ui', mode, 'build', facts_file, '-o', out_path,
                '--seq-column', 'junction_aa',
            )
            assert rc == 0, stderr
            outputs.append(stdout)
        assert all(o == '' for o in outputs)
        assert len(set(outputs)) == 1

    def test_stdout_empty_with_no_ui_flag_at_all(self, seq_file, tmp_path):
        """The pre-Task-6 invocation shape (no ``--ui`` at all) is exactly
        ``--ui auto``'s behaviour; confirm it independently.
        """
        out_path = str(tmp_path / 'out.lzg')
        stdout, stderr, rc = run_lzg('build', seq_file, '-o', out_path)
        assert rc == 0, stderr
        assert stdout == ''


# ── 2. every documented fact survives in --ui plain ──────────────


class TestFactsSurviveInPlainMode:
    def test_source_and_format_are_reported(self, facts_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('--ui', 'plain', 'build', facts_file, '-o', out_path,
                                 '--seq-column', 'junction_aa')
        assert rc == 0, stderr
        assert facts_file in stderr  # input provenance
        assert 'format=tabular' in stderr  # detected format
        assert 'engine=aap' in stderr

    def test_phase_read_reports_total_malformed_nonproductive(self, facts_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('--ui', 'plain', 'build', facts_file, '-o', out_path,
                                 '--seq-column', 'junction_aa')
        assert rc == 0, stderr
        assert 'phase=read' in stderr
        assert 'total=5' in stderr
        assert 'malformed=1' in stderr
        assert 'nonproductive=1' in stderr

    def test_dropped_warning_shown_when_records_are_dropped(self, facts_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('--ui', 'plain', 'build', facts_file, '-o', out_path,
                                 '--seq-column', 'junction_aa')
        assert rc == 0, stderr
        assert 'status=warn' in stderr
        assert 'dropped=2' in stderr

    def test_alphabet_mismatch_warning_shown(self, nucleotide_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('--ui', 'plain', 'build', nucleotide_file, '-o', out_path,
                                 '-V', 'aap')
        assert rc == 0, stderr
        assert 'alphabet=nucleotide' in stderr
        assert 'variant=aap' in stderr
        assert 'expected=amino_acid' in stderr

    def test_c_library_mixed_format_warning_shown_in_plain_mode(
        self, mixed_format_file, tmp_path
    ):
        """The plain-mode half of fix round 1, Finding 2's reproduction:
        confirms the fixture genuinely reaches the C library's own warning
        (unchanged in plain mode, via ``set_log_level``) before relying on
        it in the rich-mode test.
        """
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('--ui', 'plain', 'build', mixed_format_file, '-o', out_path,
                                 '-V', 'aap')
        assert rc == 0, stderr
        assert 'issue=mixed_input_format' in stderr

    def test_phase_construct_reports_node_and_edge_counts(self, facts_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('--ui', 'plain', 'build', facts_file, '-o', out_path,
                                 '--seq-column', 'junction_aa')
        assert rc == 0, stderr
        assert 'phase=construct' in stderr
        g = LZGraph.load(out_path)
        assert f'nodes={g.n_nodes}' in stderr
        assert f'edges={g.n_edges}' in stderr

    def test_phase_save_reports_output_path_and_size(self, facts_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('--ui', 'plain', 'build', facts_file, '-o', out_path,
                                 '--seq-column', 'junction_aa')
        assert rc == 0, stderr
        assert 'phase=save' in stderr
        # `stage`, not `status`: `status` is the plain renderer's own
        # reserved token (fix round 1, Finding 1), so cmd_build's own
        # sub-step field is named `stage` to avoid colliding with it.
        assert 'stage=start' in stderr
        assert 'stage=done' in stderr
        assert out_path in stderr
        assert 'size_kb=' in stderr

    def test_explanatory_error_when_every_record_dropped(self, all_dropped_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('--ui', 'plain', 'build', all_dropped_file, '-o', out_path,
                                 '--seq-column', 'junction_aa')
        assert rc != 0
        assert 'read 2 record(s), 0 kept' in stderr
        assert 'nothing left to build a graph from' in stderr
        assert not os.path.exists(out_path)

    def test_facts_also_appear_under_plain_auto_detection(self, facts_file, tmp_path):
        """Same as the above, but with no ``--ui`` flag: a subprocess's
        stderr is a pipe (never a tty), so auto-detection already resolves
        to plain, exactly as every pre-Task-6 test in test_cli.py relies on.
        """
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('build', facts_file, '-o', out_path,
                                 '--seq-column', 'junction_aa')
        assert rc == 0, stderr
        assert 'total=5' in stderr
        assert 'malformed=1' in stderr
        assert 'dropped=2' in stderr


# ── 3. --ui quiet emits nothing but errors ────────────────────────


class TestQuietMode:
    def test_quiet_emits_nothing_on_success(self, seq_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        stdout, stderr, rc = run_lzg('--ui', 'quiet', 'build', seq_file, '-o', out_path)
        assert rc == 0
        assert stdout == ''
        assert stderr == ''

    def test_quiet_still_suppresses_c_library_log_lines(self, seq_file, tmp_path):
        """The C library's own ``[LZGraph/INFO] ...`` lines (driven by
        ``set_log_level``) must not leak through even though ``--ui quiet``
        alone (no ``-q``) does not touch ``--log-level`` directly.
        """
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('--ui', 'quiet', 'build', seq_file, '-o', out_path)
        assert rc == 0
        assert 'LZGraph/INFO' not in stderr

    def test_quiet_still_reports_the_fatal_error(self, all_dropped_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        stdout, stderr, rc = run_lzg('--ui', 'quiet', 'build', all_dropped_file, '-o', out_path,
                                      '--seq-column', 'junction_aa')
        assert rc != 0
        assert stdout == ''
        assert 'Error:' in stderr
        assert 'nothing left to build a graph from' in stderr

    def test_bare_quiet_flag_forces_quiet_regardless_of_explicit_ui(self, seq_file, tmp_path):
        """``-q`` must keep suppressing everything even if ``--ui rich`` is
        also given (precedence rule 1: quiet always wins over an explicit
        rendering style)."""
        out_path = str(tmp_path / 'out.lzg')
        stdout, stderr, rc = run_lzg('-q', '--ui', 'rich', 'build', seq_file, '-o', out_path)
        assert rc == 0
        assert stdout == ''
        assert stderr == ''

    def test_quiet_wins_over_ui_on_the_error_path_too(self, all_dropped_file, tmp_path):
        """Same rule as above (``-q`` beats ``--ui``), checked on the error
        path instead of the success path.

        This is deliberately a *different* scenario from the test above,
        not a restatement of it: on the success path, ``-q`` alone already
        drives ``_effective_log_level`` to ``'none'``, which independently
        keeps ``cmd_build``'s own ``show_info``/``show_warn``-gated calls
        silent even if ``_resolve_build_ui``'s quiet-precedence rule were
        broken and let ``--ui rich`` win (a non-tty stderr never draws a
        rich frame either way), so that test alone would not actually
        catch a regression there. ``term.error()`` fires *unconditionally*
        (see ``cmd_build``'s docstring: an error is never gated by log
        level), so it is not shielded by that same coincidence: if quiet
        ever stopped winning here, a real ``PlainRenderer``/``RichRenderer``
        would still print its own ``[build] status=error ...`` line or card
        for this failure, on top of ``main()``'s own unconditional
        ``Error: ...`` printer, and this test would see two differently
        shaped copies of the failure instead of one.
        """
        out_path = str(tmp_path / 'out.lzg')
        stdout, stderr, rc = run_lzg(
            '-q', '--ui', 'rich', 'build', all_dropped_file, '-o', out_path,
            '--seq-column', 'junction_aa',
        )
        assert rc != 0
        assert stdout == ''
        assert stderr.count('nothing left to build a graph from') == 1
        assert '[build]' not in stderr


# ── 4. NO_COLOR / --no-color strip colour ─────────────────────────


class TestColourStripping:
    def test_no_color_env_strips_colour_in_plain_mode(self, nucleotide_file, tmp_path):
        """Force colour on via ``FORCE_COLOR`` first (a piped stderr would
        otherwise never be coloured at all, making this test vacuous), then
        confirm ``NO_COLOR`` overrides it back off, exactly as
        ``_caps.py``'s documented precedence (``NO_COLOR`` beats
        ``FORCE_COLOR``) requires.
        """
        out_path = str(tmp_path / 'coloured.lzg')
        _, stderr_coloured, rc = run_lzg(
            '--ui', 'plain', 'build', nucleotide_file, '-o', out_path, '-V', 'aap',
            env={'FORCE_COLOR': '1'},
        )
        assert rc == 0, stderr_coloured
        assert '\x1b[' in stderr_coloured  # sanity: colour really was on

        out_path2 = str(tmp_path / 'plain2.lzg')
        _, stderr_plain, rc = run_lzg(
            '--ui', 'plain', 'build', nucleotide_file, '-o', out_path2, '-V', 'aap',
            env={'FORCE_COLOR': '1', 'NO_COLOR': '1'},
        )
        assert rc == 0, stderr_plain
        assert '\x1b[' not in stderr_plain

    def test_no_color_flag_strips_colour_even_under_force_color(self, nucleotide_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg(
            '--ui', 'plain', '--no-color', 'build', nucleotide_file, '-o', out_path, '-V', 'aap',
            env={'FORCE_COLOR': '1'},
        )
        assert rc == 0, stderr
        assert '\x1b[' not in stderr

    @pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
    def test_no_color_flag_strips_sgr_but_keeps_cursor_control_in_rich_mode(
        self, seq_file, tmp_path
    ):
        out_path = str(tmp_path / 'out.lzg')
        data, rc = _run_lzg_on_pty(
            ['--ui', 'rich', '--no-color', 'build', seq_file, '-o', out_path]
        )
        assert rc == 0
        assert b'\x1b[?25l' in data  # cursor hide: still a rendered rich session
        assert b'\x1b[?25h' in data  # cursor restored
        assert b'38;5' not in data  # no 256-colour SGR codes
        assert b'\x1b[1m' not in data  # no bold
        assert b'\x1b[2m' not in data  # no dim/muted


# ── 5. node count is identical no matter which renderer drew it ──


class TestResultParityAcrossModes:
    def test_node_and_edge_counts_identical_across_all_four_modes(self, facts_file, tmp_path):
        counts = {}
        for mode in _UI_MODES:
            out_path = str(tmp_path / f'g_{mode}.lzg')
            _, stderr, rc = run_lzg('--ui', mode, 'build', facts_file, '-o', out_path,
                                     '--seq-column', 'junction_aa')
            assert rc == 0, stderr
            g = LZGraph.load(out_path)
            counts[mode] = (g.n_nodes, g.n_edges)
        assert len(set(counts.values())) == 1, counts

    @pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
    def test_node_count_identical_between_plain_and_real_rich_tty(self, seq_file, tmp_path):
        """The subprocess matrix above never exercises a genuine ``isatty()``
        rich session; do that once here, over a real pty, and compare
        against the same input built in plain mode.
        """
        plain_out = str(tmp_path / 'plain.lzg')
        _, stderr, rc = run_lzg('--ui', 'plain', 'build', seq_file, '-o', plain_out)
        assert rc == 0, stderr

        rich_out = str(tmp_path / 'rich.lzg')
        _data, rc = _run_lzg_on_pty(['--ui', 'rich', 'build', seq_file, '-o', rich_out])
        assert rc == 0

        g_plain = LZGraph.load(plain_out)
        g_rich = LZGraph.load(rich_out)
        assert g_plain.n_nodes == g_rich.n_nodes
        assert g_plain.n_edges == g_rich.n_edges


# ── 6. rich mode: real pty behaviour ──────────────────────────────


class TestRichModeOnRealTty:
    @pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
    def test_rich_mode_draws_a_panel_and_restores_the_cursor(self, seq_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        data, rc = _run_lzg_on_pty(['--ui', 'rich', 'build', seq_file, '-o', out_path])
        assert rc == 0
        assert b'\x1b[?25l' in data  # cursor hidden at some point
        assert data.rindex(b'\x1b[?25h') > data.rindex(b'\x1b[?25l')  # shown last
        assert b'lzg build' in data
        assert b'done' in data

    @pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
    def test_rich_mode_never_leaks_raw_c_log_lines(self, seq_file, tmp_path):
        """Hazard 2: the C library's raw ``fprintf``-to-stderr logging must
        never interleave with the live redraw; see ``_bridge_c_log_to_ui``.
        """
        out_path = str(tmp_path / 'out.lzg')
        data, rc = _run_lzg_on_pty(['--ui', 'rich', 'build', seq_file, '-o', out_path])
        assert rc == 0
        assert b'LZGraph/INFO' not in data

    @pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
    def test_rich_mode_error_card_and_cursor_still_restored(self, all_dropped_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        data, rc = _run_lzg_on_pty(
            ['--ui', 'rich', 'build', all_dropped_file, '-o', out_path,
             '--seq-column', 'junction_aa']
        )
        assert rc != 0
        assert data.rindex(b'\x1b[?25h') > data.rindex(b'\x1b[?25l')
        assert b'error' in data
        assert b'nothing left to build a graph from' in data
        assert not os.path.exists(out_path)

    @pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
    def test_rich_mode_surfaces_a_real_c_library_warning(self, mixed_format_file, tmp_path):
        """Fix round 1, Finding 2: a genuine C-library warning (the
        streaming reader's own ``mixed_input_format`` recovery) must reach
        the live panel in rich mode, not just plain.

        The message is long (it embeds the full file path) and the panel is
        capped at 100 columns, so the *tail* of the C message
        (``issue=mixed_input_format ...``) is not guaranteed to survive
        the row's own truncation; ``"stream build: warning"`` and
        ``"phase=ingest"`` are the prefix that is, and together they
        unambiguously identify this specific warning (nothing else in this
        renderer emits that phrase). The build still succeeds (``rc == 0``):
        this is a recoverable warning, not a fatal error.
        """
        out_path = str(tmp_path / 'out.lzg')
        data, rc = _run_lzg_on_pty(
            ['--ui', 'rich', 'build', mixed_format_file, '-o', out_path, '-V', 'aap']
        )
        assert rc == 0
        assert b'stream build: warning' in data
        assert b'phase=ingest' in data
        assert b'LZGraph/WARN' not in data  # via term.warn(), never raw stderr text
        assert data.rindex(b'\x1b[?25h') > data.rindex(b'\x1b[?25l')
        assert b'done' in data
        assert os.path.exists(out_path)

    @pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
    def test_rich_mode_never_leaks_raw_c_log_lines_even_with_a_warning(
        self, mixed_format_file, tmp_path
    ):
        """The routing fix must not regress hazard 2: the C library must
        still never be a second, uncoordinated writer to the live-redraw
        stream, warning included.
        """
        out_path = str(tmp_path / 'out.lzg')
        data, rc = _run_lzg_on_pty(
            ['--ui', 'rich', 'build', mixed_format_file, '-o', out_path, '-V', 'aap']
        )
        assert rc == 0
        assert b'[LZGraph/' not in data


# ── 7. unit-level checks on the cli.py wiring itself ─────────────


class TestUiResolutionHelpers:
    def _namespace(self, **overrides):
        base = dict(quiet=False, ui=None, no_color=False)
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_quiet_flag_forces_quiet_mode_even_with_explicit_ui_rich(self):
        term = _resolve_build_ui(self._namespace(quiet=True, ui='rich'))
        assert isinstance(term, NullRenderer)

    def test_explicit_ui_plain_is_honoured_without_quiet(self):
        term = _resolve_build_ui(self._namespace(ui='plain'))
        assert isinstance(term, PlainRenderer)

    def test_missing_ui_and_no_color_attributes_default_safely(self):
        """Namespaces built before Task 6 (``test_cli_alphabet_warning.py``,
        ``test_cli_build_matrix.py``, ...) have no ``ui``/``no_color``
        attribute at all; this must not raise.
        """
        bare = argparse.Namespace(quiet=False)
        term = _resolve_build_ui(bare)
        assert isinstance(term, (PlainRenderer, RichRenderer))

    def test_no_color_flag_merges_no_color_into_env(self, monkeypatch):
        """``_resolve_build_ui`` does ``from ._term import ui as _make_ui``
        freshly on every call, so patching the ``ui`` name on the live
        ``LZGraphs._term`` module before calling it is enough to intercept
        the call and inspect the ``env`` mapping it was actually given.
        """
        captured = {}

        def _fake_ui(requested=None, stream=None, env=None):
            captured['env'] = env
            return NullRenderer()

        monkeypatch.setattr('LZGraphs._term.ui', _fake_ui)

        _resolve_build_ui(self._namespace(no_color=True))
        assert captured['env'].get('NO_COLOR') == '1'

    def test_effective_log_level_ui_quiet_defaults_to_none(self):
        assert _effective_log_level(self._namespace(ui='quiet')) == 'none'

    def test_effective_log_level_explicit_log_level_overrides_ui_quiet(self):
        ns = self._namespace(ui='quiet', log_level='info')
        assert _effective_log_level(ns) == 'info'

    def test_effective_log_level_explicit_log_level_overrides_bare_quiet(self):
        """Pre-existing precedent (see
        tests/test_cli.py::test_build_quiet_overridden_by_explicit_log_level):
        an explicit --log-level always outranks -q too.
        """
        ns = self._namespace(quiet=True, log_level='info')
        assert _effective_log_level(ns) == 'info'


class TestCLogBridge:
    """Unit-level coverage of the severity-based C-log router feeding the
    rich renderer (fix round 1, Finding 2): progress for INFO-level
    streaming-ingest ticks, ``term.warn()`` for everything at WARN/ERROR
    (or an unrecognised level), silence for the rest, all decided from the
    real ``LZGLogLevel`` integer the C library passes rather than from
    message text. No multi-gigabyte fixture is needed to exercise the
    progress path for real (the C ingest loop only logs a periodic tick
    every 1,000,000 lines or 5 seconds; see
    ``lib/graph_core/graph_build_ingest_internal.h``); driving the callback
    directly with synthetic ``(level, message)`` pairs is both faster and a
    more precise probe of the routing logic than waiting for a real one.
    """

    class _FakeTerm:
        def __init__(self):
            self.calls = []

        def progress(self, label, fraction, detail=None):
            self.calls.append(('progress', label, fraction, detail))

        def warn(self, message):
            self.calls.append(('warn', message))

    def test_pct_field_at_info_level_drives_a_progress_call(self):
        term = self._FakeTerm()
        callback = _bridge_c_log_to_ui(term)
        message = (
            "stream build: phase=ingest_progress file=x mode=plain lines=100 "
            "sequences=90 blank=0 pct=42.50 bytes=1.0/2.0MB nodes=5 edges=5 "
            "rate=100 inst_rate=100 avg_mbps=1 inst_mbps=1 eta=00:00:01"
        )
        callback(3, message)  # LZG_LOG_INFO
        assert len(term.calls) == 1
        kind, label, fraction, detail = term.calls[0]
        assert kind == 'progress'
        assert label == 'ingest'
        assert fraction == pytest.approx(0.425)
        assert '90' in detail
        assert '00:00:01' in detail

    def test_info_messages_without_pct_are_dropped(self):
        term = self._FakeTerm()
        callback = _bridge_c_log_to_ui(term)
        callback(3, "graph ready: 5 nodes, 5 edges, root=0")
        callback(3, "building graph: 5 sequences, variant=0, genes=no")
        assert term.calls == []

    def test_fraction_is_clamped_and_never_negative_or_over_one(self):
        term = self._FakeTerm()
        callback = _bridge_c_log_to_ui(term)
        callback(3, "phase=ingest_progress pct=150.00 sequences=1")
        assert term.calls[0][2] <= 1.0 + 1e-9

    def test_warn_level_message_is_surfaced_via_term_warn(self):
        """Fix round 1, Finding 2: a real C-library warning (e.g. the
        streaming reader's own ``mixed_input_format`` recovery, or
        ``graph_finalize.c``'s "no @ root node found") must never be
        dropped just because it lacks a ``pct=`` field.
        """
        term = self._FakeTerm()
        callback = _bridge_c_log_to_ui(term)
        message = (
            "stream build: warning phase=ingest file=x line=7 "
            "issue=mixed_input_format previous_mode=plain "
            "current_record=seqcount continuing=true"
        )
        callback(2, message)  # LZG_LOG_WARN
        assert term.calls == [('warn', message)]

    def test_error_level_message_is_surfaced_via_term_warn(self):
        term = self._FakeTerm()
        callback = _bridge_c_log_to_ui(term)
        callback(1, "something went wrong")  # LZG_LOG_ERROR
        assert term.calls == [('warn', 'something went wrong')]

    def test_warn_level_wins_over_message_text_that_looks_like_progress(self):
        """Severity, not message content, decides routing: a WARN-level
        message that happens to contain a `pct=`-shaped substring must
        still be surfaced as a warning, never misrouted to progress().
        """
        term = self._FakeTerm()
        callback = _bridge_c_log_to_ui(term)
        message = "issue=mixed_input_format pct=42.00 looks like progress but is not"
        callback(2, message)
        assert term.calls == [('warn', message)]

    def test_debug_and_trace_level_chatter_is_dropped(self):
        term = self._FakeTerm()
        callback = _bridge_c_log_to_ui(term)
        callback(4, "some debug-level internal detail")  # LZG_LOG_DEBUG
        callback(5, "some trace-level per-item detail")  # LZG_LOG_TRACE
        assert term.calls == []

    def test_unrecognised_level_is_surfaced_rather_than_dropped(self):
        """"If a C message arrives that you cannot classify, surface it
        rather than dropping it" (fix round 1): a level outside the C
        library's own known enum (0-5) is not assumed safe to drop.
        """
        term = self._FakeTerm()
        callback = _bridge_c_log_to_ui(term)
        callback(99, "an unrecognised severity")
        assert term.calls == [('warn', 'an unrecognised severity')]


# ── 8. cmd_build called directly (no subprocess), matching the
#      argparse.Namespace convention test_cli_build_matrix.py already uses ──


class TestCmdBuildDirect:
    def _namespace(self, input_path, output_path, **overrides):
        base = dict(
            input=input_path, output=output_path, variant='aap',
            seq_column=None, v_column='v_call', j_column='j_call',
            abundance_column=None, no_genes=False, smoothing=0.0,
            strict_input=False, expect_format=None,
            quiet=False, log_level=None, ui=None, no_color=False,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_cmd_build_direct_call_still_works_with_new_ui_fields(self, seq_file, tmp_path, capsys):
        out_path = str(tmp_path / 'out.lzg')
        cmd_build(self._namespace(seq_file, out_path))
        err = capsys.readouterr().err
        assert 'phase=construct' in err
        assert LZGraph.load(out_path).n_nodes > 0

    def test_cmd_build_direct_call_without_ui_attribute_at_all(self, seq_file, tmp_path, capsys):
        """Exactly the shape test_cli_build_matrix.py / test_cli_alphabet_warning.py
        use: no ``ui``/``no_color`` attribute present at all.
        """
        ns = argparse.Namespace(
            input=seq_file, output=str(tmp_path / 'out.lzg'), variant='aap',
            seq_column=None, v_column='v_call', j_column='j_call',
            abundance_column=None, no_genes=False, smoothing=0.0,
            strict_input=False, expect_format=None,
            quiet=True, log_level='none',
        )
        cmd_build(ns)
        assert LZGraph.load(ns.output).n_nodes > 0
