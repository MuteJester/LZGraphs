"""Task 7: cross-cutting quality suite for the ``_term`` rendering layer.

Tasks 1 to 6 each shipped a per-module test file (``test_term_caps.py``,
``test_term_ansi.py``, ``test_term_widgets.py``, ``test_term_plain.py``,
``test_term_rich.py``, ``test_cli_ui_integration.py``), written by five
different implementers, each of whom only ever saw their own module. This
file is deliberately different in kind, not just in name: every test below
is a *property of the whole layer* (stdout purity, rendering overhead,
environment-to-mode wiring, width safety, encoding safety, terminal-state
restoration) rather than one more example of a single module's own
contract, and several of them are driven end to end through a real ``lzg``
subprocess specifically because the point being tested is the *wiring*
between modules, not any one module in isolation.

Building this file surfaced two real, previously uncaught bugs in
``_rich.py`` (both fixed as part of this task, see its module docstring's
new "Encoding safety" and "Embedded newlines in caller-supplied values"
sections):

1. ``RichRenderer`` had no protection against ``UnicodeEncodeError``: every
   other renderer (``PlainRenderer``) already guarded its writes, but
   ``_draw`` wrote straight to ``stream.write(...)``, so a build reporting a
   non-ASCII path/identifier under a strict ASCII stream crashed instead of
   degrading. See ``TestEncodingSafety`` below.
2. ``RichRenderer`` never sanitized embedded newlines in field/row/warning
   values, unlike ``PlainRenderer`` (which explicitly does, precisely to
   keep one call to one physical line). An embedded ``\\n`` in a rendered
   row desynchronises ``self._lines_drawn`` from how many terminal rows the
   row actually occupies, corrupting every subsequent redraw's
   ``cursor_up`` arithmetic for the rest of the session. See
   ``TestEmbeddedNewlineSafety`` below.

Both are exactly the kind of "two modules disagree about a contract"
finding the task asks for: ``_plain.py``'s author reasoned carefully about
both hazards (its own docstring devotes paragraphs to each); ``_rich.py``'s
author, writing to the same ``Ui`` protocol, did neither, and nothing in
the shared protocol's own docstring made either guarantee explicit either
way. See this task's report for the full "seams" writeup.
"""
from __future__ import annotations

import json
import os
import re
import select
import signal
import statistics
import struct
import subprocess
import sys
import time

import pytest

try:
    import fcntl
    import pty
    import termios

    _HAVE_PTY = True
except ImportError:  # pragma: no cover - exercised only on Windows
    _HAVE_PTY = False

_NO_PTY_REASON = "pty/fcntl/termios are POSIX-only and unavailable on this platform"

# ── Subprocess plumbing ──────────────────────────────────────────────
#
# Every subprocess launched here explicitly sets PYTHONPATH to this
# worktree's src/, rather than relying on it being inherited from whatever
# invoked pytest. This project's editable install can point at a sibling
# checkout (a different worktree of the same repo) rather than this one, so
# a bare `python -m LZGraphs.cli` with no explicit PYTHONPATH can silently
# exercise the wrong copy of the source tree; explicit PYTHONPATH here
# removes that ambiguity, matching the defensive convention
# test_cli_ui_integration.py's own `_run_lzg_on_pty` already uses for the
# same reason.

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
LZG = [sys.executable, "-m", "LZGraphs.cli"]

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SGR_RE = re.compile(r"\x1b\[[0-9;]+m")  # colour/style codes specifically (excludes cursor/clear)
_HIDE = "\x1b[?25l"
_SHOW = "\x1b[?25h"


def _env(extra=None):
    e = dict(os.environ)
    e["PYTHONPATH"] = _SRC
    if extra:
        e.update(extra)
    return e


def run_lzg(*args, input_text=None, env=None, timeout=60):
    """Run the CLI as a real subprocess; stdout/stderr are pipes (non-tty)."""
    result = subprocess.run(
        LZG + list(args),
        capture_output=True, text=True, input=input_text,
        timeout=timeout, env=_env(env),
    )
    return result.stdout, result.stderr, result.returncode


def _open_sized_pty(columns, lines=30):
    """Open a real pty pair sized via TIOCSWINSZ (not COLUMNS/LINES env),
    so ``os.get_terminal_size`` on the slave fd reports the exact
    configured geometry directly rather than depending on the env-var
    fallback chain ``_caps._terminal_size`` only reaches for an
    *unconfigured* (0x0) pty. Mirrors test_term_rich.py's helper of the
    same shape.
    """
    master_fd, slave_fd = pty.openpty()
    winsize = struct.pack("HHHH", lines, columns, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
    return master_fd, slave_fd


def _drain(master_fd, idle=0.2, timeout=10.0):
    """Read whatever a pty produces until a short quiet period, bounded
    overall by ``timeout``. Mirrors test_term_rich.py's/
    test_cli_ui_integration.py's own draining helpers.
    """
    chunks = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master_fd], [], [], idle)
        if not ready:
            if chunks:
                break
            continue
        try:
            data = os.read(master_fd, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def _run_on_pty(args, columns=100, lines=30, env=None, timeout=15.0):
    """Run the CLI with stdin/stdout/stderr all attached to one real,
    sized pty. Returns ``(data: bytes, returncode)``.
    """
    master_fd, slave_fd = _open_sized_pty(columns, lines)
    proc = subprocess.Popen(
        LZG + list(args), stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        env=_env(env), preexec_fn=os.setsid,
    )
    os.close(slave_fd)
    try:
        data = _drain(master_fd, timeout=timeout)
        proc.wait(timeout=timeout)
    finally:
        os.close(master_fd)
    return data, proc.returncode


def _visible_lines(text):
    """Split ``text`` into physical lines, stripping the trailing ``\\r`` a
    pty's line discipline inserts before each ``\\n`` (ONLCR translation),
    which is not something any renderer in this layer emits itself. Without
    stripping it, every measured line would appear exactly one column wider
    than what was actually rendered.
    """
    lines = text.split("\n")
    return [ln[:-1] if ln.endswith("\r") else ln for ln in lines]


def _visible_len(s):
    """Local, dependency-free re-implementation of _ansi.visible_len, so
    this cross-cutting file measures what a pty actually produced without
    importing the very module whose correctness the rest of the suite
    already pins (test_term_ansi.py). Same CSI-stripping rule.
    """
    return len(_ANSI_RE.sub("", s))


# ── Shared fixtures ──────────────────────────────────────────────────


@pytest.fixture
def seq_file(tmp_path):
    p = tmp_path / "seqs.txt"
    seqs = [
        "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
        "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
        "CASSPGQGAYEQYF", "CASSDAGGYTF",
    ]
    p.write_text("\n".join(seqs) + "\n")
    return str(p)


@pytest.fixture
def graph_file(tmp_path, seq_file):
    out = str(tmp_path / "g.lzg")
    _, stderr, rc = run_lzg("-q", "build", seq_file, "-o", out)
    assert rc == 0, stderr
    return out


@pytest.fixture
def score_input_file(tmp_path):
    p = tmp_path / "score_in.txt"
    seqs = ["CASSLGIRRT", "CASSLGYEQYF", "CASSQETQYF", "CASSDAGGYTF"]
    p.write_text("\n".join(seqs) + "\n")
    return str(p), len(seqs)


def _generate_plain_seq_file(path, n, seed):
    """Write ``n`` deterministic amino-acid sequences, one per line, fast
    enough to build as a fixture rather than a checked-in asset. A
    dedicated ``random.Random(seed)`` instance is used (never the global
    ``random`` module) so this never perturbs any other test's RNG state.
    """
    import random

    rng = random.Random(seed)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    with open(path, "w") as f:
        for _ in range(n):
            length = rng.randint(10, 20)
            f.write("".join(rng.choice(alphabet) for _ in range(length)))
            f.write("\n")


def _generate_tabular_file(path, n, seed):
    """Write an AIRR-shaped tabular file with ``n`` records. Tabular input
    never qualifies for ``cmd_build``'s streaming fast path, so building
    from this file exercises the in-memory ``read_sequences`` branch, which
    spends a meaningful share of its wall time in a plain Python loop
    (rather than a fast C ingest loop), which is exactly the property the
    SIGINT test below needs: a real, promptly-interruptible window between
    ``term.start()`` (which fires before this loop) and the build's finish.
    """
    import random

    rng = random.Random(seed)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    with open(path, "w") as f:
        f.write("sequence_id\tjunction_aa\tv_call\tj_call\tduplicate_count\tproductive\n")
        for i in range(n):
            length = rng.randint(10, 20)
            seq = "".join(rng.choice(alphabet) for _ in range(length))
            f.write(f"seq{i}\t{seq}\tTRBV1\tTRBJ1\t1\tT\n")


@pytest.fixture(scope="module")
def big_plain_seq_file(tmp_path_factory):
    """A large plain-text sequence file for the overhead benchmark (Property
    2): big enough that the build itself takes a fraction of a second of
    real work, small enough that many repetitions stay affordable.
    """
    d = tmp_path_factory.mktemp("quality_overhead")
    p = d / "big_plain.txt"
    _generate_plain_seq_file(str(p), n=150_000, seed=42)
    return str(p)


@pytest.fixture(scope="module")
def big_tabular_file(tmp_path_factory):
    """A large tabular file for the SIGINT-mid-build test (Property 6)."""
    d = tmp_path_factory.mktemp("quality_sigint")
    p = d / "big_tabular.tsv"
    _generate_tabular_file(str(p), n=100_000, seed=7)
    return str(p)


# ══════════════════════════════════════════════════════════════════
# Property 1: stdout purity
# ══════════════════════════════════════════════════════════════════


class TestStdoutPurity:
    """For every data-producing command, redirected stdout must contain
    zero ANSI escape sequences and exactly the expected record count,
    regardless of UI mode. This is the property that makes
    ``lzg simulate g.lzg -n 1000 | head -3`` work; the last test in this
    class drives that literal pipeline.

    Expected counts are derived from each command's own documented output
    contract (see the comment above each assertion for exactly which
    ``cli.py`` lines it corresponds to), not observed by running the
    command first: a test that only checks "stdout has no escapes" would
    miss an accidental extra/missing line (a stray debug print, a
    suppressed record); checking the exact count catches both.
    """

    def test_build_writes_nothing_to_stdout_in_any_ui_mode(self, seq_file, tmp_path):
        # `-o` is required for `build`, so the graph goes to a file and
        # stdout's only correct content is nothing at all, in every mode.
        for mode in ("auto", "rich", "plain", "quiet"):
            out = str(tmp_path / f"b_{mode}.lzg")
            stdout, stderr, rc = run_lzg("--ui", mode, "build", seq_file, "-o", out)
            assert rc == 0, stderr
            assert stdout == ""

    def test_simulate_bare_mode_emits_exactly_n_lines_no_ansi(self, graph_file):
        n = 500
        stdout, stderr, rc = run_lzg("simulate", graph_file, "-n", str(n), "--seed", "7")
        assert rc == 0, stderr
        assert "\x1b[" not in stdout
        lines = stdout.split("\n")
        assert lines[-1] == "", "expected a trailing newline after the last sequence"
        assert len(lines) - 1 == n

    def test_simulate_detailed_mode_emits_header_plus_n_lines(self, graph_file):
        n = 500
        stdout, stderr, rc = run_lzg(
            "simulate", graph_file, "-n", str(n), "--seed", "7", "--with-details"
        )
        assert rc == 0, stderr
        assert "\x1b[" not in stdout
        lines = stdout.split("\n")
        assert lines[-1] == ""
        assert len(lines) - 1 == n + 1  # header + n data rows

    def test_simulate_json_mode_emits_exactly_n_records(self, graph_file):
        n = 500
        stdout, stderr, rc = run_lzg(
            "simulate", graph_file, "-n", str(n), "--seed", "7", "--json"
        )
        assert rc == 0, stderr
        assert "\x1b[" not in stdout
        records = json.loads(stdout)
        assert len(records) == n

    def test_score_emits_header_plus_one_row_per_input_sequence(
        self, graph_file, score_input_file
    ):
        path, n = score_input_file
        stdout, stderr, rc = run_lzg("score", graph_file, path)
        assert rc == 0, stderr
        assert "\x1b[" not in stdout
        lines = stdout.split("\n")
        assert lines[-1] == ""
        assert len(lines) - 1 == n + 1  # "sequence\tlzpgen" header + n rows

    def test_diversity_emits_exactly_the_documented_tag_count(self, graph_file):
        # cli.py's cmd_diversity (default, no --json): one "# lzg diversity
        # ..." comment line, one HL line per Hill order in --hill's default
        # ('0,1,2,5,inf', 5 orders), then 4 fixed DV tags
        # (effective_diversity, entropy_nats, entropy_bits, uniformity) and
        # 3 fixed DR tags (dynamic_range_decades, pgen_mean, pgen_std).
        default_hill = "0,1,2,5,inf"
        n_orders = len(default_hill.split(","))
        expected = 1 + n_orders + 4 + 3
        stdout, stderr, rc = run_lzg("diversity", graph_file)
        assert rc == 0, stderr
        assert "\x1b[" not in stdout
        lines = stdout.split("\n")
        assert lines[-1] == ""
        assert len(lines) - 1 == expected

    def test_info_emits_exactly_the_documented_tag_count(self, graph_file):
        # cli.py's cmd_info (default, no --genes/--all): one "# lzg info
        # ..." comment line, 8 fixed GR tags (variant, nodes, edges,
        # initial_states, terminal_states, is_dag, has_gene_data,
        # path_count), 4 fixed DV tags, and 4 fixed PR tags (pgen_mean,
        # pgen_std, dynamic_range_decades, is_proper).
        expected = 1 + 8 + 4 + 4
        stdout, stderr, rc = run_lzg("info", graph_file)
        assert rc == 0, stderr
        assert "\x1b[" not in stdout
        lines = stdout.split("\n")
        assert lines[-1] == ""
        assert len(lines) - 1 == expected

    def test_pipe_to_head_yields_clean_truncated_output(self, graph_file):
        """The literal motivating scenario: `lzg simulate ... -n 1000 |
        head -3` must produce exactly 3 clean lines and must not hang.
        """
        p1 = subprocess.Popen(
            LZG + ["simulate", graph_file, "-n", "1000", "--seed", "3"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_env(),
        )
        p2 = subprocess.Popen(
            ["head", "-3"], stdin=p1.stdout, stdout=subprocess.PIPE, text=True,
        )
        p1.stdout.close()  # let p1 receive SIGPIPE once p2/head is done reading
        out, _ = p2.communicate(timeout=30)
        p1.wait(timeout=30)

        lines = out.splitlines()
        assert len(lines) == 3
        assert not any("\x1b[" in ln for ln in lines)
        assert p2.returncode == 0


# ══════════════════════════════════════════════════════════════════
# Property 2: rendering overhead
# ══════════════════════════════════════════════════════════════════


class TestRenderingOverhead:
    """Rendering overhead should be a small fraction of total wall time on
    a build large enough that per-call formatting/print cost is not the
    dominant term.

    Measured in-process (direct ``cmd_build`` calls, not subprocesses) to
    remove interpreter-startup noise (~50-150ms per subprocess launch on
    this machine) from the comparison entirely, since that noise is larger
    than the effect being measured and has nothing to do with the
    rendering layer. ``--ui quiet`` (``NullRenderer``, true no-op) is the
    baseline; ``--ui plain`` is measured against it, both writing to a real
    file (not a tty), matching the common CI/redirected-log scenario this
    layer is built for.

    Honesty about the noise floor: on this development machine, 9
    repetitions of a ~0.3s build measured a plain/quiet median difference
    of well under 1%, sometimes negative (plain faster than quiet), i.e.
    indistinguishable from measurement noise. The true cost of the layer's
    handful of ``print()``-and-flush calls per build (a fixed, O(1) cost
    independent of input size, thanks to ``PlainRenderer``'s own
    throttling) is almost certainly a small fraction of a percent at this
    scale, but asserting a literal "< 2%" bound here would be asserting a
    number the measurement cannot actually distinguish from zero, which is
    exactly what the task warns against. The threshold below
    (``_OVERHEAD_ASSERTION_THRESHOLD``, 25%) is deliberately generous: not a
    claim that real overhead is anywhere near 25%, but a bound wide enough
    to never trip on ordinary noise while still catching a real regression
    (verified by mutation: injecting a mere 20ms of artificial per-call
    latency into ``PlainRenderer._write`` (a small change, and far below
    what a naive unthrottled per-record print would cost) pushes the
    measured ratio to over 40%, comfortably tripping this bound). The
    actual measured numbers are always included in the assertion message,
    so a CI failure (or a manual run with ``-s``) reports real evidence
    rather than a bare "assertion failed".
    """

    _OVERHEAD_ASSERTION_THRESHOLD = 0.25

    def _bench(self, mode, path, tmp_path, reps):
        import argparse

        from LZGraphs.cli import cmd_build

        def make_ns(out):
            return argparse.Namespace(
                input=path, output=out, variant="aap",
                seq_column=None, v_column="v_call", j_column="j_call",
                abundance_column=None, no_genes=False, smoothing=0.0,
                strict_input=False, expect_format=None,
                quiet=False, log_level=None, ui=mode, no_color=False,
            )

        times = []
        for i in range(reps):
            out = str(tmp_path / f"bench_{mode}_{i}.lzg")
            log_path = tmp_path / f"bench_{mode}_{i}.log"
            with open(log_path, "w") as log_stream:
                old_stderr = sys.stderr
                sys.stderr = log_stream
                try:
                    t0 = time.perf_counter()
                    cmd_build(make_ns(out))
                    times.append(time.perf_counter() - t0)
                finally:
                    sys.stderr = old_stderr
            os.remove(out)
        return times

    def test_plain_mode_overhead_over_quiet_is_small_and_reported(
        self, big_plain_seq_file, tmp_path
    ):
        reps = 9
        # Interleave quiet/plain repetitions (rather than all of one mode
        # then all of the other) so a slow monotonic drift across the run
        # (thermal throttling, background load ramping up) affects both
        # samples roughly equally instead of biasing one mode's mean.
        quiet_times = []
        plain_times = []
        for i in range(reps):
            quiet_times += self._bench("quiet", big_plain_seq_file, tmp_path, 1)
            plain_times += self._bench("plain", big_plain_seq_file, tmp_path, 1)

        quiet_median = statistics.median(quiet_times)
        plain_median = statistics.median(plain_times)
        overhead_ratio = (plain_median - quiet_median) / quiet_median

        message = (
            f"quiet median={quiet_median:.4f}s ({[round(t, 4) for t in quiet_times]}) "
            f"plain median={plain_median:.4f}s ({[round(t, 4) for t in plain_times]}) "
            f"overhead_ratio={overhead_ratio:.1%} "
            f"(generous assertion threshold={self._OVERHEAD_ASSERTION_THRESHOLD:.0%}; "
            f"see this test's docstring for why 2% itself is not asserted)"
        )
        assert overhead_ratio < self._OVERHEAD_ASSERTION_THRESHOLD, message


# ══════════════════════════════════════════════════════════════════
# Property 3: environment matrix, driven end to end
# ══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
class TestEnvironmentMatrix:
    """``NO_COLOR``, ``TERM=dumb``, ``CI``, non-TTY, and ``FORCE_COLOR``
    each select the documented mode/colour depth, driven end to end
    through the real CLI rather than by calling ``_caps.detect`` directly
    (that unit-level coverage already exists in ``test_term_caps.py``): the
    point here is that ``cli.py``'s wiring (``_resolve_build_ui`` ->
    ``LZGraphs._term.ui`` -> ``_caps.detect``/``resolve_mode``) actually
    honours what ``_caps.py`` promises, end to end, not just that
    ``_caps.py`` itself is correct in isolation.

    A real pty is used wherever the test needs a genuine ``isatty() ==
    True`` (``NO_COLOR``, ``TERM=dumb``, ``CI``, the CI-with-explicit-rich
    exception); the non-TTY and ``FORCE_COLOR``-on-non-TTY cases use a
    plain pipe instead, since "not a tty" is exactly the condition under
    test there.
    """

    def test_no_color_on_a_tty_strips_colour_but_keeps_rich_mode(self, seq_file, tmp_path):
        out = str(tmp_path / "out.lzg")
        data, rc = _run_on_pty(["build", seq_file, "-o", out], env={"NO_COLOR": "1"})
        assert rc == 0
        text = data.decode("utf-8", "replace")
        assert _HIDE in text, "NO_COLOR must not affect mode selection: rich should still draw"
        assert not _SGR_RE.search(text), "NO_COLOR must strip every colour/style code"

    def test_no_color_empty_string_still_disables_colour_end_to_end(self, seq_file, tmp_path):
        """The published NO_COLOR convention: presence, not truthiness, is
        what matters, so an *empty* value must still disable colour. Driven
        through the real CLI/env, complementing test_term_caps.py's
        unit-level version of the same rule.
        """
        out = str(tmp_path / "out.lzg")
        data, rc = _run_on_pty(["build", seq_file, "-o", out], env={"NO_COLOR": ""})
        assert rc == 0
        assert not _SGR_RE.search(data.decode("utf-8", "replace"))

    def test_term_dumb_on_a_tty_forces_plain_and_no_colour(self, seq_file, tmp_path):
        out = str(tmp_path / "out.lzg")
        data, rc = _run_on_pty(["build", seq_file, "-o", out], env={"TERM": "dumb"})
        assert rc == 0
        text = data.decode("utf-8", "replace")
        assert _HIDE not in text, "TERM=dumb must downgrade auto-mode to plain (no cursor control)"
        assert not _SGR_RE.search(text)

    def test_ci_on_a_tty_forces_auto_mode_to_plain(self, seq_file, tmp_path):
        out = str(tmp_path / "out.lzg")
        data, rc = _run_on_pty(["build", seq_file, "-o", out], env={"CI": "1"})
        assert rc == 0
        assert _HIDE not in data.decode("utf-8", "replace")

    def test_ci_does_not_override_an_explicit_rich_request(self, seq_file, tmp_path):
        """Policy default (CI) vs. explicit intent (--ui rich): explicit
        intent wins, per _caps.py's own documented distinction.
        """
        out = str(tmp_path / "out.lzg")
        data, rc = _run_on_pty(
            ["--ui", "rich", "build", seq_file, "-o", out], env={"CI": "1"}
        )
        assert rc == 0
        assert _HIDE in data.decode("utf-8", "replace")

    def test_non_tty_auto_mode_selects_plain_with_no_colour(self, seq_file, tmp_path):
        out = str(tmp_path / "out.lzg")
        stdout, stderr, rc = run_lzg("build", seq_file, "-o", out)
        assert rc == 0, stderr
        assert _HIDE not in stderr
        assert not _SGR_RE.search(stderr)

    def test_force_color_on_non_tty_enables_colour_but_stays_plain(self, seq_file, tmp_path):
        out = str(tmp_path / "out.lzg")
        stdout, stderr, rc = run_lzg("build", seq_file, "-o", out, env={"FORCE_COLOR": "1"})
        assert rc == 0, stderr
        assert _HIDE not in stderr, "FORCE_COLOR must not itself force rich/cursor-control mode"
        assert _SGR_RE.search(stderr), "FORCE_COLOR must enable colour even off a tty"


# ══════════════════════════════════════════════════════════════════
# Property 4: width matrix, via a real pty
# ══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
class TestWidthMatrix:
    """At 40, 60, 80, and 200 columns, no line the rich renderer ever wrote
    (not just the final visible frame: every frame drawn during the whole
    session, including ones later overwritten by a redraw) exceeds the
    configured terminal width. Ragged wrapping is the failure this catches:
    a widget that pads/truncates incorrectly at one width but happens to
    look right at another.

    At 200 columns, ``_caps.py`` deliberately clamps ``Capabilities.width``
    to 100 (so panels do not stretch absurdly on an ultra-wide monitor);
    this is asserted explicitly below as a second, complementary check, not
    just folded into "no line exceeds the terminal width" (which a
    100-column line trivially satisfies against a 200-column terminal
    regardless of whether the clamp is working).
    """

    @pytest.mark.parametrize("width", [40, 60, 80, 200])
    def test_no_rendered_line_exceeds_the_terminal_width(self, width, seq_file, tmp_path):
        out = str(tmp_path / f"w_{width}.lzg")
        data, rc = _run_on_pty(
            ["--ui", "rich", "build", seq_file, "-o", out], columns=width, lines=30
        )
        assert rc == 0
        text = data.decode("utf-8", "replace")
        lines = _visible_lines(text)
        visible_lens = [_visible_len(ln) for ln in lines]
        violations = [
            (i, vis, ln) for i, (vis, ln) in enumerate(zip(visible_lens, lines))
            if vis > width
        ]
        assert violations == [], f"lines exceeding width={width}: {violations[:5]}"
        # "no line exceeds width" is satisfied vacuously if the renderer
        # drew nothing at all (empty lines trivially never exceed
        # anything): confirmed by mutating _rich._draw to write "" instead
        # of the assembled frame, which left every parametrisation above
        # green. A real rich session must draw a visible panel (the boxed
        # title border alone is dozens of columns), so pin that something
        # was actually rendered, not merely that nothing rendered too wide.
        assert max(visible_lens, default=0) > 0, (
            "expected at least one visibly non-empty rendered line; a "
            "renderer that draws nothing would satisfy the width check "
            "above vacuously"
        )

    def test_200_columns_still_clamps_panel_width_to_100(self, seq_file, tmp_path):
        out = str(tmp_path / "w_200_clamp.lzg")
        data, rc = _run_on_pty(
            ["--ui", "rich", "build", seq_file, "-o", out], columns=200, lines=30
        )
        assert rc == 0
        lines = _visible_lines(data.decode("utf-8", "replace"))
        max_seen = max((_visible_len(ln) for ln in lines), default=0)
        assert max_seen == 100, (
            f"expected the 100-column clamp (_caps._MAX_WIDTH) to produce a line of "
            f"exactly 100 visible columns somewhere in the session, saw max={max_seen}"
        )


# ══════════════════════════════════════════════════════════════════
# Property 5: encoding safety
# ══════════════════════════════════════════════════════════════════


class TestEncodingSafety:
    """An ASCII-only, strict-error stream must never raise
    ``UnicodeEncodeError`` for any renderer, with non-ASCII data in
    sequence identifiers or file paths.

    This is deliberately a unit-level property (constructing an explicit
    ``ascii``/``errors="strict"`` stream), not a subprocess one, for a
    documented reason: CPython hard-codes ``sys.stderr``'s own error
    handler to ``"backslashreplace"`` regardless of ``PYTHONIOENCODING``
    (see ``PYTHONIOENCODING``'s own documentation: "the error handler for
    sys.stderr is always 'backslashreplace'"), so a real CLI subprocess's
    stderr can *never* actually raise ``UnicodeEncodeError`` no matter how
    aggressively its encoding is constrained from the environment, as
    verified empirically while building this test (``PYTHONIOENCODING=
    ascii:strict`` still reports ``errors == "backslashreplace"`` on
    ``sys.stderr``). That is precisely *why* this exact bug (see this
    file's module docstring) went uncaught by every subprocess-driven test
    in ``test_cli_ui_integration.py``: the hazard is invisible to anything
    that writes through a real, interpreter-owned ``sys.stderr``, and is
    only reachable by constructing a stream with a stricter error handler
    directly, exactly as ``test_term_plain.py`` already does for
    ``PlainRenderer`` (this file adds the equivalent for ``RichRenderer``,
    which had no such test at all before this task).

    A best-effort end-to-end sanity check is still included
    (``test_end_to_end_non_ascii_path_does_not_crash_the_build``) for
    documentation purposes, but it does not exercise the true hazard, for
    the reason above; it would pass even without the ``_rich.py`` fix.
    """

    def _ascii_strict_stream(self):
        import io

        raw = io.BytesIO()
        return raw, io.TextIOWrapper(raw, encoding="ascii", errors="strict")

    def test_rich_renderer_never_raises_on_non_ascii_field_values(self):
        from LZGraphs._term._caps import Capabilities
        from LZGraphs._term._rich import RichRenderer

        caps = Capabilities(
            is_tty=True, colours=0, width=60, real_width=60, height=24,
            unicode=True, interactive=True, supports_cursor_control=True,
        )
        _raw, stream = self._ascii_strict_stream()
        r = RichRenderer(caps, stream=stream)
        r.start("lzg build", {"source": "café_input.tsv"})
        r.progress("ingest", 0.5, detail="naïve rate")
        r.update(output="résumé.lzg")
        r.warn("séquence dropped: 12")
        r.done("lzg build", {"output": "café.lzg", "nodes": "1,234"})

    def test_rich_renderer_never_raises_on_non_ascii_error_lines(self):
        from LZGraphs._term._caps import Capabilities
        from LZGraphs._term._rich import RichRenderer

        caps = Capabilities(
            is_tty=True, colours=0, width=60, real_width=60, height=24,
            unicode=True, interactive=True, supports_cursor_control=True,
        )
        _raw, stream = self._ascii_strict_stream()
        r = RichRenderer(caps, stream=stream)
        r.start("lzg build", {})
        r.error("lzg build", ["mauvaise séquence: café_input.tsv"])

    def test_plain_renderer_never_raises_on_non_ascii_field_values(self):
        from LZGraphs._term._caps import Capabilities
        from LZGraphs._term._plain import PlainRenderer

        caps = Capabilities(
            is_tty=False, colours=0, width=80, real_width=80, height=24,
            unicode=False, interactive=False, supports_cursor_control=False,
        )
        _raw, stream = self._ascii_strict_stream()
        p = PlainRenderer(caps, stream=stream)
        p.start("lzg build", {"source": "café_input.tsv"})
        p.warn("séquence dropped")
        p.error("lzg build", ["café"])
        p.done("lzg build", {"output": "café.lzg"})

    def test_end_to_end_non_ascii_path_does_not_crash_the_build(self, tmp_path):
        """Documentation-level sanity check; see the class docstring for
        why this cannot exercise the real hazard through a genuine
        ``sys.stderr``.
        """
        src_dir = tmp_path / "café"
        src_dir.mkdir()
        seq_path = src_dir / "café_input.txt"
        seq_path.write_text("CASSLGIRRT\nCASSLGYEQYF\nCASSQETQYF\n")
        out_path = src_dir / "café_output.lzg"

        stdout, stderr, rc = run_lzg(
            "--ui", "plain", "build", str(seq_path), "-o", str(out_path),
            env={"PYTHONIOENCODING": "ascii:strict"},
        )
        assert rc == 0, stderr
        assert "UnicodeEncodeError" not in stderr
        assert out_path.exists()


# ══════════════════════════════════════════════════════════════════
# Property 6: cursor restoration, end to end
# ══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
class TestCursorRestorationEndToEnd:
    """Through a real subprocess with a pty: after a normal build, after a
    build that errors, and after SIGINT mid-build, the final cursor
    operation is show-cursor. This is the property users notice when it
    breaks.

    The normal-build and error-path cases are already covered end to end
    by ``test_cli_ui_integration.py``'s ``TestRichModeOnRealTty``; they are
    included again here, compactly, so this file states the full three-case
    property in one place rather than splitting it across files by
    accident. The SIGINT case is new: nothing in any per-module suite sends
    a real signal to a real, running ``lzg build`` subprocess and inspects
    what its terminal is left in. Every existing ``KeyboardInterrupt``
    test (in ``test_term_rich.py``) raises the exception *programmatically*
    inside a script that already imports ``RichRenderer`` directly, which
    exercises the same ``atexit``/``__exit__`` machinery but never proves
    that a real ``SIGINT`` delivered to a real ``lzg build`` process (going
    through the OS's own signal delivery, argparse, ``cmd_build``, and
    ``main()``'s own ``except KeyboardInterrupt`` handler) actually reaches
    that machinery in time.
    """

    def test_normal_build_restores_the_cursor(self, seq_file, tmp_path):
        out = str(tmp_path / "out.lzg")
        data, rc = _run_on_pty(["--ui", "rich", "build", seq_file, "-o", out])
        assert rc == 0
        text = data.decode("utf-8", "replace")
        assert text.count(_HIDE) == 1
        assert text.count(_SHOW) == 1
        assert text.rindex(_SHOW) > text.rindex(_HIDE)

    def test_build_that_errors_restores_the_cursor(self, tmp_path):
        bad = tmp_path / "all_dropped.tsv"
        bad.write_text(
            "sequence_id\tjunction_aa\tproductive\n"
            "seq0\tCASSLGIRRT\tF\n"
            "seq1\tCASSLGYEQYF\tF\n"
        )
        out = str(tmp_path / "out.lzg")
        data, rc = _run_on_pty(
            ["--ui", "rich", "build", str(bad), "-o", out, "--seq-column", "junction_aa"]
        )
        assert rc != 0
        text = data.decode("utf-8", "replace")
        assert text.count(_HIDE) == 1
        assert text.count(_SHOW) == 1
        assert text.rindex(_SHOW) > text.rindex(_HIDE)
        assert not os.path.exists(out)

    def test_sigint_mid_build_still_restores_the_cursor(self, big_tabular_file, tmp_path):
        """Send a real ``SIGINT`` to a real, running ``lzg build`` process
        while it is genuinely mid-build (confirmed, not assumed: we wait
        for the hide-cursor escape to actually appear in the pty's output
        before sending the signal, proving a rich session is open), and
        confirm the cursor still ends up shown, as the very last cursor
        operation, exactly as it must for a normal or an error exit.

        ``big_tabular_file`` forces cmd_build's in-memory
        ``read_sequences`` path (a tabular file never qualifies for the
        streaming fast path), which spends real wall time in a plain
        Python loop. Unlike the fast C-backed streaming path, a signal
        arriving there is delivered promptly rather than deferred until a
        long C call returns.
        """
        master_fd, slave_fd = _open_sized_pty(100, 30)
        env = _env()
        proc = subprocess.Popen(
            LZG + [
                "--ui", "rich", "build", big_tabular_file,
                "-o", str(tmp_path / "sigint_out.lzg"),
                "--seq-column", "junction_aa",
            ],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env=env, preexec_fn=os.setsid,
        )
        os.close(slave_fd)

        buf = b""
        hide_seen = False
        deadline = time.monotonic() + 15.0
        try:
            while time.monotonic() < deadline:
                ready, _, _ = select.select([master_fd], [], [], 0.05)
                if ready:
                    try:
                        chunk = os.read(master_fd, 65536)
                    except OSError:
                        break
                    buf += chunk
                    if _HIDE.encode() in buf:
                        hide_seen = True
                        break
                elif proc.poll() is not None:
                    break  # the build finished before we ever saw hide_cursor

            assert hide_seen, (
                "the build finished (or timed out) before a rich session was ever "
                "observed to open; increase big_tabular_file's record count"
            )

            os.killpg(os.getpgid(proc.pid), signal.SIGINT)

            drain_deadline = time.monotonic() + 15.0
            while time.monotonic() < drain_deadline:
                ready, _, _ = select.select([master_fd], [], [], 0.2)
                if ready:
                    try:
                        chunk = os.read(master_fd, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                elif proc.poll() is not None:
                    break
            proc.wait(timeout=15)
        finally:
            os.close(master_fd)

        assert proc.returncode != 0, "an interrupted build must not report success"
        text = buf.decode("utf-8", "replace")
        assert text.count(_HIDE) == 1
        assert text.count(_SHOW) == 1
        assert text.rindex(_SHOW) > text.rindex(_HIDE), (
            "show_cursor must be the last cursor-visibility operation after SIGINT"
        )


# ══════════════════════════════════════════════════════════════════
# Seam finding: embedded newlines in caller-supplied values (RichRenderer)
# ══════════════════════════════════════════════════════════════════


class TestEmbeddedNewlineSafety:
    """A cross-module contract disagreement found while building this
    suite (item B, "judge the seams"): ``PlainRenderer`` sanitizes embedded
    newlines out of every field/row/message value it renders (its own
    module docstring explains why: one call must always produce exactly
    one physical line); ``RichRenderer`` did not, despite implementing the
    identical ``Ui`` protocol. An unsanitized ``\\n`` in a rendered row
    desynchronises ``RichRenderer``'s own ``self._lines_drawn`` tracking
    from how many terminal rows that row actually consumed, which corrupts
    every subsequent redraw's ``cursor_up`` arithmetic for the rest of the
    session (not just the one frame that carried it): a real, silent
    display-corruption bug, not merely a cosmetic one. Fixed in ``_rich.py``
    as part of this task (``_sanitize_line``, applied to every field, row,
    warning, and error-line value); tested here so the fix cannot regress.
    """

    def _caps(self):
        from LZGraphs._term._caps import Capabilities

        return Capabilities(
            is_tty=True, colours=0, width=60, real_width=60, height=24,
            unicode=False, interactive=True, supports_cursor_control=True,
        )

    def _renderer(self, buf):
        """A ``RichRenderer`` whose clock advances by a full second on
        every call. Several ``Ui`` methods (``progress``/``update``/
        ``warn``) redraw through the throttled ``_redraw()``, which is a
        silent no-op if the clock has not advanced past
        ``_MIN_REDRAW_INTERVAL`` since the previous draw; a fixed clock
        (e.g. ``lambda: 0.0``) would make such a call *never actually
        redraw*, which would make a test built around it pass vacuously
        (the assertion would hold trivially, because nothing new was ever
        drawn) rather than actually exercising the newline-sanitising code
        path. ``done()``/``error()`` always draw unconditionally regardless
        of the clock (they call ``_draw`` directly via ``_finish``, not
        through the throttle), so they are unaffected either way, but using
        the same advancing clock everywhere keeps every test in this class
        exercising a real draw, not an accidentally-skipped one.
        """
        import itertools

        from LZGraphs._term._rich import RichRenderer

        counter = itertools.count(1)
        return RichRenderer(self._caps(), stream=buf, clock=lambda: float(next(counter)))

    def _assert_last_frame_lines_drawn_matches(self, r, buf, before_len):
        """``self._lines_drawn`` must equal the number of terminal rows the
        *most recently drawn frame* actually occupies, not the cumulative
        newline count of the whole session's buffer (an earlier ``start()``
        frame is still sitting in ``buf`` too). Isolating the suffix written
        since ``before_len`` is what makes this comparison meaningful for a
        renderer that has already drawn more than once.
        """
        new_text = buf.getvalue()[before_len:]
        assert r._lines_drawn == new_text.count("\n"), (
            "self._lines_drawn must match the real number of terminal rows the "
            "latest frame consumed, or the next redraw's cursor_up() will be wrong; "
            f"lines_drawn={r._lines_drawn} but the frame just written has "
            f"{new_text.count(chr(10))} newlines: {new_text!r}"
        )

    def test_embedded_newline_in_a_field_value_does_not_desync_lines_drawn(self):
        import io

        buf = io.StringIO()
        r = self._renderer(buf)
        before = len(buf.getvalue())
        r.start("lzg build", {"source": "multi\nline\nvalue"})
        self._assert_last_frame_lines_drawn_matches(r, buf, before)

    def test_embedded_newline_in_a_warning_does_not_desync_lines_drawn(self):
        import io

        buf = io.StringIO()
        r = self._renderer(buf)
        r.start("lzg build", {})
        before = len(buf.getvalue())
        r.warn("bad record\nsecond physical line")
        self._assert_last_frame_lines_drawn_matches(r, buf, before)

    def test_embedded_newline_in_done_rows_does_not_desync_lines_drawn(self):
        import io

        buf = io.StringIO()
        r = self._renderer(buf)
        r.start("lzg build", {})
        before = len(buf.getvalue())
        r.done("lzg build", {"output": "multi\nline\npath.lzg"})
        self._assert_last_frame_lines_drawn_matches(r, buf, before)

    def test_embedded_newline_in_error_lines_does_not_desync_lines_drawn(self):
        import io

        buf = io.StringIO()
        r = self._renderer(buf)
        r.start("lzg build", {})
        before = len(buf.getvalue())
        r.error("lzg build", ["bad\nheadline", "detail\rline"])
        self._assert_last_frame_lines_drawn_matches(r, buf, before)
