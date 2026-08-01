"""Tests for LZGraphs._term._rich: the live, redrawing TTY renderer.

This is the riskiest module in the ``_term`` layer (it owns terminal state,
not just terminal content), so this file weighs its tests accordingly:

* **The tests that matter most** drive a real pty (``isatty()`` genuinely
  true, never mocked) through a full session and assert the cursor is
  hidden exactly once at the start and shown exactly once at the end, with
  the show unconditionally the *last* cursor-visibility operation, both
  after a normal ``done()`` and after an uncaught ``KeyboardInterrupt``. One
  of the ``KeyboardInterrupt`` tests goes further and runs the whole thing
  in a real child process (``subprocess.Popen`` attached to the pty), so the
  ``atexit`` fallback and Python's own real "uncaught KeyboardInterrupt at
  the top level" handling are both exercised for real, not simulated.
* **Redraw arithmetic** (cursor-up count matching the previously drawn
  height, and no orphaned lines when a frame shrinks) is pinned precisely
  against ``_draw`` and the escape sequences it emits, since that is the
  exact mechanism the task calls out as easy to get subtly wrong.
* **Throttling** is driven with an injected, manually-advanced fake clock
  (never a real sleep), matching the task's own instruction that a
  timing-dependent test would be flaky in exactly the environment this
  ships to.
* **The ``SIGWINCH`` thread guard** is tested by monkeypatching
  ``signal.signal`` itself and asserting it is never called when a
  ``RichRenderer`` session is started from a background thread.

Tests that do not depend on genuine terminal behaviour (throttling, the
thread guard, the ``SIGWINCH``-reflow mechanism itself) construct a
``Capabilities`` by hand and write to a plain ``io.StringIO``, matching how
``test_term_widgets.py``/``test_term_ansi.py`` test pure logic without a
terminal; only the tests actually about cursor-control *sequences reaching
a real tty* open a pty, matching ``test_term_caps.py``'s own convention of
gating pty-only tests with ``@pytest.mark.skipif(not _HAVE_PTY, ...)``
individually rather than skipping the whole module (most of this file's
tests need no pty at all and should still run on a platform without one).
"""
from __future__ import annotations

import atexit
import dataclasses
import io
import os
import re
import select
import signal
import struct
import subprocess
import sys
import threading
import time

import pytest

from LZGraphs._term._caps import Capabilities, detect
from LZGraphs._term._rich import _MIN_REDRAW_INTERVAL, RichRenderer

try:
    import fcntl
    import pty
    import termios

    _HAVE_PTY = True
except ImportError:  # pragma: no cover - exercised only on Windows
    _HAVE_PTY = False

_NO_PTY_REASON = "pty/fcntl/termios are POSIX-only and unavailable on this platform"

_HIDE = b"\x1b[?25l"
_SHOW = b"\x1b[?25h"
_CURSOR_UP_RE = re.compile(r"\x1b\[(\d+)A")


class _PtyDrainer:
    """Continuously drains a pty master on a background thread.

    A pty's kernel buffer is small and finite, and a writer whose buffer
    fills blocks until someone reads. Every test in this module writes
    redraw frames from the same thread that later reads them, so draining
    has to run *concurrently with* the writes rather than after them.

    Draining afterwards worked on Linux, whose buffer is roomy enough for
    these frames, and deadlocked every macOS run: the renderer blocked
    forever inside ``_safe_write`` during ``done()``, taking the whole job
    to its timeout with no indication of which test was stuck. Buffer size
    is a platform detail no test should be pinned to, so this drains from
    the moment the pty is created.
    """

    def __init__(self, master_fd):
        self._fd = master_fd
        self._chunks = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.05)
            except (OSError, ValueError):
                return  # fd closed underneath us by the test's finally block
            if not ready:
                continue
            try:
                data = os.read(self._fd, 65536)
            except OSError:
                return
            if not data:
                return
            self._chunks.append(data)

    def collect(self, idle=0.2, timeout=5.0):
        """Stop once nothing new has arrived for ``idle`` seconds.

        Same quiet-period contract the old synchronous ``_drain`` had, so
        callers are unchanged; only the moment reading starts has moved.
        """
        deadline = time.monotonic() + timeout
        seen = len(self._chunks)
        quiet_since = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.02)
            if len(self._chunks) != seen:
                seen = len(self._chunks)
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= idle:
                break
        self._stop.set()
        self._thread.join(timeout=1.0)
        return b"".join(self._chunks)


_DRAINERS = {}


def _open_sized_pty(columns, lines=24):
    """Open a pty pair, size it, and wrap the slave side as a real stream.

    Returns ``(master_fd, stream)``; the caller closes both. Mirrors
    ``test_term_caps.py``'s helper of the same name/shape exactly, since
    both need the identical real, sized pty for the same reason: a stream
    whose ``isatty()`` is genuinely ``True``.

    Reading of the master side starts immediately, on a background thread,
    so a caller writing more than the pty buffer holds never blocks. See
    :class:`_PtyDrainer`.
    """
    master_fd, slave_fd = pty.openpty()
    winsize = struct.pack("HHHH", lines, columns, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
    _DRAINERS[master_fd] = _PtyDrainer(master_fd)
    return master_fd, os.fdopen(slave_fd, "w")


def _caps_for(stream, **overrides):
    """``Capabilities`` for a real pty ``stream``, with a pinned empty env.

    ``env={}`` avoids the actual test-runner's environment (``NO_COLOR``,
    ``TERM``, a CI runner's own vars) leaking into what should be a
    deterministic capability snapshot; only ``isatty()`` and the pty's own
    size come from the real stream.
    """
    caps = detect(stream=stream, env={})
    return dataclasses.replace(caps, **overrides) if overrides else caps


def _drain(master_fd, idle=0.2, timeout=5.0):
    """Return everything written to ``master_fd``, stopping once nothing new
    has arrived for ``idle`` seconds (bounded overall by ``timeout``).

    The reading itself already happened, on the background thread
    :func:`_open_sized_pty` started when it created the pty; this only waits
    for the quiet period and hands back what was collected, in order. The
    concurrency is what makes the writer safe at any frame size, on any
    platform's pty buffer.
    """
    drainer = _DRAINERS.pop(master_fd, None)
    if drainer is not None:
        return drainer.collect(idle=idle, timeout=timeout)

    # A pty this module did not open (so nothing is draining it): read it
    # here. Only safe when the writer has already finished.
    chunks = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master_fd], [], [], idle)
        if not ready:
            break
        try:
            data = os.read(master_fd, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def _plain_caps(**overrides) -> Capabilities:
    """A hand-built ``Capabilities`` for tests that do not need a real tty
    (throttling, SIGWINCH mechanics, thread-guard): pure logic against the
    renderer's internal state, not against genuine terminal behaviour."""
    base = Capabilities(
        is_tty=True,
        colours=0,
        width=60,
        real_width=60,
        height=24,
        unicode=True,
        interactive=True,
        supports_cursor_control=True,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# ─────────────────────────────────────────────────────────────────────────
# The tests that matter most: terminal state is always restored.
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_writing_far_more_than_a_pty_buffer_never_blocks():
    """Pins the concurrent drain: a renderer must be able to write more
    than the pty's kernel buffer holds without deadlocking.

    Every other pty test here writes only a few frames, so on Linux they
    stayed under the buffer and passed even when nothing read until the
    end. macOS's buffer is a fraction of the size and they hung there
    instead, blocked forever inside ``_safe_write``. This writes hundreds
    of kilobytes, far past any platform's buffer, so the guarantee is
    pinned by volume rather than by the host's happening to be roomy. It
    deadlocks against a drain that only starts after the writes.

    Volume comes from warnings rather than progress calls: progress is
    redraw-throttled to 15fps by design, so thousands of calls collapse
    into a single frame, whereas a warning must never be dropped and so is
    always written.
    """
    master_fd, stream = _open_sized_pty(72)
    try:
        caps = _caps_for(stream)
        r = RichRenderer(caps, stream=stream)
        r.start("lzg build", {"SOURCE": "big.tsv"})
        for i in range(400):
            r.warn(f"record {i} dropped: malformed junction_aa field")
        r.done("lzg build", {"nodes": "71,181"})
        stream.flush()
        out = _drain(master_fd)
    finally:
        stream.close()
        os.close(master_fd)

    assert len(out) > 100_000, (
        f"expected to push well past any pty buffer, only wrote {len(out)} bytes"
    )
    assert out.count(_SHOW) == 1
    assert out.rfind(_SHOW) > out.rfind(_HIDE)


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_cursor_hidden_on_start_and_shown_exactly_once_after_normal_done():
    master_fd, stream = _open_sized_pty(72)
    try:
        caps = _caps_for(stream)
        r = RichRenderer(caps, stream=stream)
        r.start("lzg build", {"SOURCE": "bcr.tsv.gz"})
        r.progress("ingest", 0.5)
        r.done("lzg build", {"nodes": "71,181"})
        stream.flush()
        out = _drain(master_fd)
    finally:
        stream.close()
        os.close(master_fd)

    assert out.count(_HIDE) == 1
    assert out.count(_SHOW) == 1
    assert out.rfind(_SHOW) > out.rfind(_HIDE), "show_cursor must be the last cursor op"


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_cursor_restored_after_keyboard_interrupt_via_context_manager():
    """The test that matters most: an interrupted build must never leave
    the cursor hidden. Uses ``with RichRenderer(...) as r:`` so restoration
    is guaranteed by Python's own try/finally semantics (``__exit__``),
    exactly the mechanism the task asks this module to provide."""
    master_fd, stream = _open_sized_pty(72)
    try:
        caps = _caps_for(stream)
        with pytest.raises(KeyboardInterrupt):
            with RichRenderer(caps, stream=stream) as r:
                r.start("lzg build", {"SOURCE": "bcr.tsv.gz"})
                r.progress("ingest", 0.3)
                raise KeyboardInterrupt
        stream.flush()
        out = _drain(master_fd)
    finally:
        stream.close()
        os.close(master_fd)

    assert out.count(_HIDE) == 1
    assert out.count(_SHOW) == 1
    assert out.rfind(_SHOW) > out.rfind(_HIDE)


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_context_manager_restores_cursor_for_any_exception_not_only_interrupt():
    master_fd, stream = _open_sized_pty(72)
    try:
        caps = _caps_for(stream)
        with pytest.raises(ValueError, match="boom"):
            with RichRenderer(caps, stream=stream) as r:
                r.start("lzg build", {})
                r.progress("ingest", 0.1)
                raise ValueError("boom")
        stream.flush()
        out = _drain(master_fd)
    finally:
        stream.close()
        os.close(master_fd)

    assert out.count(_HIDE) == 1
    assert out.count(_SHOW) == 1
    assert out.rfind(_SHOW) > out.rfind(_HIDE)


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_start_failure_itself_still_restores_cursor(monkeypatch):
    """The internal ``try/except/re-raise`` in ``start()``: if opening the
    session fails partway (here, forced by breaking the first frame's
    render), the cursor must not stay hidden for a session that never
    actually opened."""
    master_fd, stream = _open_sized_pty(72)
    try:
        caps = _caps_for(stream)
        r = RichRenderer(caps, stream=stream)
        monkeypatch.setattr(
            r, "_render_frame", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with pytest.raises(RuntimeError, match="boom"):
            r.start("lzg build", {})
        stream.flush()
        out = _drain(master_fd)
    finally:
        stream.close()
        os.close(master_fd)

    assert out.count(_HIDE) == 1
    assert out.count(_SHOW) == 1


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_atexit_hook_registered_by_start_restores_cursor_when_invoked(monkeypatch):
    """Simulates real interpreter shutdown calling the hook ``start()``
    registered, standing in for an uncaught exception that unwinds all the
    way past this call site without ever reaching ``done()``/``error()``
    or a ``with`` block's ``__exit__``."""
    master_fd, stream = _open_sized_pty(72)
    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn: registered.append(fn) or fn)
    try:
        caps = _caps_for(stream)
        r = RichRenderer(caps, stream=stream)
        r.start("lzg build", {})
        r.progress("ingest", 0.4)
        assert registered, "start() must register an atexit fallback"

        registered[-1]()  # simulate atexit firing at real process shutdown

        stream.flush()
        out = _drain(master_fd)
    finally:
        stream.close()
        os.close(master_fd)

    assert out.count(_SHOW) == 1
    assert out.rfind(_SHOW) > out.rfind(_HIDE)


_CHILD_SCRIPT = """
import sys
from LZGraphs._term._rich import RichRenderer
from LZGraphs._term._caps import detect

caps = detect(stream=sys.stderr, env={})
r = RichRenderer(caps, stream=sys.stderr)
r.start("lzg build", {"SOURCE": "bcr.tsv.gz"})
r.progress("ingest", 0.5)
raise KeyboardInterrupt
"""


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_subprocess_uncaught_keyboard_interrupt_restores_cursor_via_real_atexit():
    """End-to-end, no simulation: a real child process opens a session (no
    ``with``, so only the ``atexit`` fallback can save it), raises an
    uncaught ``KeyboardInterrupt``, and Python's own real top-level
    exception handling plus real ``atexit`` machinery run to completion.
    """
    master_fd, slave_fd = pty.openpty()
    closed_slave = False
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD_SCRIPT],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        closed_slave = True

        chunks = []
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if ready:
                try:
                    data = os.read(master_fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
            elif proc.poll() is not None:
                try:
                    data = os.read(master_fd, 65536)
                    if data:
                        chunks.append(data)
                except OSError:
                    pass
                break
        proc.wait(timeout=15)
        out = b"".join(chunks)
    finally:
        if not closed_slave:
            try:
                os.close(slave_fd)
            except OSError:
                pass
        os.close(master_fd)

    assert proc.returncode != 0, "the child's KeyboardInterrupt must remain uncaught"
    assert out.count(_HIDE) == 1
    assert out.count(_SHOW) == 1
    assert out.rfind(_SHOW) > out.rfind(_HIDE)


def test_stop_is_idempotent_even_if_invoked_twice():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r._stop()
    r._stop()
    out = buf.getvalue()
    assert out.count("\x1b[?25h") == 1


# ─────────────────────────────────────────────────────────────────────────
# Redraw arithmetic: cursor-up matches the previously drawn height, and a
# shrinking frame leaves nothing orphaned on screen.
# ─────────────────────────────────────────────────────────────────────────


def test_cursor_up_matches_previously_drawn_line_count_across_redraws():
    """Every redraw after the first must move up by exactly the height of
    whatever was on screen *before* it, never a guess. Uses a monotonically
    growing frame (a new, distinct progress() label each call) so every
    cursor_up in the captured output corresponds 1:1 with the previous
    frame's tracked height; see the shrink test below for the other half
    (a frame that gets *shorter*)."""
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])

    r.start("lzg build", {"SOURCE": "x"})
    heights = [r._lines_drawn]
    for i in range(1, 6):
        clock[0] += 1.0  # comfortably clear of the throttle interval
        r.progress(f"step{i}", i / 5)
        heights.append(r._lines_drawn)

    out = buf.getvalue()
    cursor_ups = [int(n) for n in _CURSOR_UP_RE.findall(out)]
    assert len(cursor_ups) == len(heights) - 1
    assert cursor_ups == heights[:-1]


def test_shrinking_frame_clears_every_orphaned_line_and_reascends():
    """Drives ``_draw`` directly with a longer then a shorter frame, the
    precise scenario the task calls out ("if the frame changes height
    between draws, handle it without leaving orphaned lines on screen")."""
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)

    r._draw(["a", "b", "c", "d", "e"])
    assert r._lines_drawn == 5
    first_len = len(buf.getvalue())

    r._draw(["x", "y"])
    assert r._lines_drawn == 2
    second = buf.getvalue()[first_len:]

    cursor_ups = [int(n) for n in _CURSOR_UP_RE.findall(second)]
    # one leading cursor_up(5) for the old frame's full height, and one
    # trailing cursor_up(3) re-ascending past the 3 now-blanked leftovers
    assert cursor_ups == [5, 3]
    # all 5 previously-drawn rows were cleared: 2 overwritten with new
    # content, 3 more explicitly blanked
    assert second.count("\x1b[K") == 5
    assert "x" in second and "y" in second
    assert "c" not in second and "d" not in second and "e" not in second


def test_growing_frame_emits_no_shrink_cleanup():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)

    r._draw(["a", "b"])
    first_len = len(buf.getvalue())
    r._draw(["a", "b", "c", "d"])
    second = buf.getvalue()[first_len:]

    cursor_ups = [int(n) for n in _CURSOR_UP_RE.findall(second)]
    assert cursor_ups == [2]  # only the leading move, no trailing re-ascend
    assert second.count("\x1b[K") == 4


def test_first_draw_emits_no_cursor_up():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r._draw(["only line"])
    assert "\x1b[" not in buf.getvalue().split("only line")[0]
    assert r._lines_drawn == 1


# ─────────────────────────────────────────────────────────────────────────
# Fix round 1, Finding 1: a frame taller than the terminal must never
# desync the redraw arithmetic. cursor_up can only move up to the physical
# top row; once a frame's line count exceeds caps.height, subsequent
# cursor_up(self._lines_drawn) calls no longer address the rows they think
# they do and the display corrupts progressively. Reproduced first against
# the pre-fix module (a 20-field frame on a height-6 terminal issued
# cursor_up amounts of 22/24/25, all far past the terminal's own height)
# before writing the fix; kept here as the regression test.
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("height", [1, 2, 3, 6, 24])
def test_frame_taller_than_terminal_never_exceeds_height_minus_one(height):
    caps = _plain_caps(height=height)
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])

    # 20 fields alone (no progress yet) already renders as a 22-line boxed
    # panel before any clamping (the exact shape of the reviewer's
    # reproduction), and every subsequent progress() call grows it further.
    r.start("lzg build", {f"FIELD{i}": f"value{i}" for i in range(20)})
    for i in range(3):
        clock[0] += 1.0
        r.progress(f"step{i}", i / 3)
    clock[0] += 1.0
    r.done("lzg build", {"nodes": "1"})

    out = buf.getvalue()
    cursor_ups = [int(n) for n in _CURSOR_UP_RE.findall(out)]
    budget = max(1, height - 1)
    assert cursor_ups, "expected at least one redraw after the first"
    assert max(cursor_ups) <= budget, (height, cursor_ups, budget)


def test_elision_marker_is_visible_when_rows_are_dropped():
    """Drives the oversized-frame scenario via 20 distinct ``progress()``
    labels rather than ``start()``'s own ``fields`` (fix round 2, D1
    curates *which* fields the live panel ever paints, from a fixed,
    small set, so an arbitrary field name is no longer guaranteed to
    render at all); ``progress()`` bars are not curated the same way (any
    label renders its own row), so they are what still reliably forces an
    oversized frame here, exercising the exact same ``_clamp_rows``
    elision mechanism this test is actually about.
    """
    caps = _plain_caps(height=6)
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})
    for i in range(19):
        clock[0] += 1.0
        r.progress(f"FIELD{i}", i / 20)
    # Only the *final*, fully-populated frame is under test below (the one
    # that actually needs eliding); clearing what accumulated on the way
    # there keeps the assertions about that one frame, not every partial
    # frame drawn while labels were still trickling in one at a time.
    buf.seek(0)
    buf.truncate(0)
    clock[0] += 1.0
    r.progress("FIELD19", 19 / 20)
    out = buf.getvalue()
    assert "hidden" in out, "dropping rows must say so, not truncate silently"
    assert "18" in out  # exactly how many of the 20 rows were hidden


def test_elision_keeps_the_tail_not_the_head():
    """Deliberate choice: drop the oldest/static rows first (the front),
    keep the tail (the live progress/most recent warnings), since that is
    what a user watching a build actually needs to see change. See
    ``test_elision_marker_is_visible_when_rows_are_dropped`` above for why
    this uses 20 ``progress()`` labels rather than ``start()`` fields."""
    caps = _plain_caps(height=6)
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})
    for i in range(19):
        clock[0] += 1.0
        r.progress(f"FIELD{i}", i / 20)
    # See the sibling test above: only the final, fully-populated frame
    # (the one that actually needs eliding) is under test, not every
    # partial frame drawn while labels were still trickling in.
    buf.seek(0)
    buf.truncate(0)
    clock[0] += 1.0
    r.progress("FIELD19", 19 / 20)
    out = buf.getvalue()
    assert "FIELD19" in out and "FIELD18" in out
    assert "FIELD0 " not in out and "FIELD1 " not in out


@pytest.mark.parametrize("width,height", [(60, 6), (60, 3), (40, 6)])
def test_clamped_live_frame_never_needs_the_fit_to_terminal_safety_net(width, height):
    """``_clamp_rows`` (keyed off ``_panel_overhead``) and
    ``_fit_to_terminal`` (a fixed, independently-derived budget) must agree
    on how tall a live frame is allowed to be. ``_clamp_rows`` deliberately
    elides rows so the assembled panel already fits; ``_fit_to_terminal``
    is only meant to be the *belt-and-suspenders* net for frames it never
    pre-clamped (the final ``done()``/``error()`` card). If
    ``_panel_overhead`` ever under-counts panel()'s own border overhead
    (boxed mode is title-plus-closing-border, two lines, not one),
    ``_clamp_rows`` keeps one row too many, ``panel()`` then emits a frame
    exactly one line taller than the budget, and ``_fit_to_terminal``
    silently chops the last surviving line -- ordinarily the panel's own
    closing border -- with no elision marker of its own, corrupting the
    frame instead of eliding it on purpose. Reproduced directly against a
    mutated ``_panel_overhead`` that always returns 1 (even in boxed mode)
    while writing this test: the live frame came out one line taller than
    ``max(1, height - 1)``, and this assertion caught it.
    """
    caps = _plain_caps(width=width, height=height)
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})
    for i in range(19):  # enough rows that _clamp_rows must actually elide
        clock[0] += 1.0
        r.progress(f"FIELD{i}", i / 20)

    frame = r._render_frame()
    budget = max(1, height - 1)
    assert len(frame) <= budget, (
        f"the live frame ({len(frame)} lines) already exceeds the row "
        f"budget ({budget}) before _fit_to_terminal's safety net even "
        "runs; _clamp_rows and _panel_overhead have disagreed about how "
        "much border overhead panel() adds"
    )
    # The safety net, applied on top, must therefore be a true no-op for
    # this frame: nothing left for it to cut.
    assert r._fit_to_terminal(frame) == frame


@pytest.mark.parametrize("height", [0, 1, 2])
def test_degenerate_heights_never_crash_and_stay_within_budget(height):
    """Some CI environments report a terminal height of 0, 1, or 2; this
    must never raise and must never draw more than one line (the
    max(1, height - 1) floor)."""
    caps = _plain_caps(height=height)
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {"A": "1", "B": "2"})
    assert r._lines_drawn == 1
    r.progress("ingest", 0.5)
    assert r._lines_drawn == 1
    r.done("lzg build", {"nodes": "1"})
    assert r._lines_drawn == 1


def test_error_card_keeps_its_headline_when_squeezed_by_height():
    """_fit_to_terminal's blunt safety net keeps the *top* of a card, which
    for error() is exactly the cross-marked headline: the single most
    important line, never the part sacrificed when a terminal is tiny."""
    caps = _plain_caps(height=2)
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.error("lzg build", ["column 'junction_aa' not found", "", "try: --column cdr3_aa"])
    out = buf.getvalue()
    assert "lzg build" in out  # the title line survives
    assert r._lines_drawn <= max(1, 2 - 1)


# ─────────────────────────────────────────────────────────────────────────
# Throttling: at most 15 redraws/second, proven with a fake clock only.
# ─────────────────────────────────────────────────────────────────────────


def test_progress_throttle_bounded_over_10000_calls_with_fake_clock():
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})

    # 10,000 calls spread across 2.0 simulated seconds: a realistic, not
    # even pathological, reporting cadence. No real time.sleep anywhere.
    dt = 2.0 / 10_000
    for i in range(10_000):
        clock[0] += dt
        r.progress("ingest", i / 10_000)

    # start() draws once; at most ~2.0 / (1/15) more redraws bound the rest,
    # independent of how fast this test machine actually runs.
    expected_ceiling = 1 + int(2.0 / _MIN_REDRAW_INTERVAL) + 2
    assert 1 < r.redraw_count <= expected_ceiling
    assert r.redraw_count < 100  # nowhere near "one redraw per call"


def test_progress_throttle_blocks_everything_when_clock_never_advances():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    for i in range(10_000):
        r.progress("ingest", i / 10_000)
    # every call landed at the same instant as start()'s own draw: nothing
    # after it clears the throttle interval, so only the initial draw ran.
    assert r.redraw_count == 1


def test_progress_redraws_again_once_the_interval_has_elapsed():
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})
    assert r.redraw_count == 1

    r.progress("ingest", 0.1)  # same instant, throttled away
    assert r.redraw_count == 1

    clock[0] += _MIN_REDRAW_INTERVAL + 0.001
    r.progress("ingest", 0.2)
    assert r.redraw_count == 2


def test_done_and_error_always_force_a_draw_regardless_of_throttle():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)  # clock frozen
    r.start("lzg build", {})
    r.progress("ingest", 0.5)  # throttled away, same instant
    assert r.redraw_count == 1
    r.done("lzg build", {"nodes": "1"})
    assert r.redraw_count == 2  # done() forces its own frame regardless


def test_warn_forces_a_draw_regardless_of_throttle():
    """Pins the exact frame count :meth:`RichRenderer.warn`'s own docstring
    measures directly ("a start()-warn()-update()-done() sequence with no
    artificial delay between calls painted only 2 frames (start, done)
    before this fix ... forcing here makes it 3, with the warning visible
    in the frame between them"): with the clock frozen (every call landing
    at the same instant, so nothing but a forced draw ever repaints),
    start() draws once, warn() must force a second draw, the immediately
    following update() lands in the same throttle window and is
    (correctly) skipped, and done() forces the third and final draw.

    Before this test, the only thing that caught ``warn()``'s
    ``self._redraw(force=True)`` regressing to an unforced
    ``self._redraw()`` was a real-pty subprocess test in
    ``test_cli_ui_integration.py``
    (``test_rich_mode_warning_survives_at_log_level_warn``), which is
    skipped outright on a platform without a pty and takes on the order of
    ten seconds to run. This is the fast, pty-free equivalent: no real
    sleep, no subprocess, same fake-clock pattern as every other test in
    this "Throttling" section.
    """
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)  # clock frozen
    r.start("lzg build", {})
    assert r.redraw_count == 1

    r.warn("alphabet mismatch: nucleotide data fed to an amino-acid engine")
    assert r.redraw_count == 2  # warn() must force its own redraw

    r.update(nodes=1)  # same instant as warn(): throttled away
    assert r.redraw_count == 2

    r.done("lzg build", {"nodes": "1"})
    assert r.redraw_count == 3  # done() forces its own frame regardless


# ─────────────────────────────────────────────────────────────────────────
# Nothing reaches stdout; writes go only to the given stream.
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_nothing_reaches_stdout(capsys):
    master_fd, stream = _open_sized_pty(72)
    try:
        caps = _caps_for(stream)
        r = RichRenderer(caps, stream=stream)
        r.start("lzg build", {"SOURCE": "x"})
        r.progress("ingest", 0.4)
        r.warn("412 non-productive skipped")
        r.done("lzg build", {"nodes": "1"})
        stream.flush()
        _drain(master_fd)
    finally:
        stream.close()
        os.close(master_fd)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_default_stream_is_stderr_not_stdout(capsys):
    caps = _plain_caps()
    r = RichRenderer(caps)  # no stream given: must default to sys.stderr
    r.start("lzg build", {})
    r.done("lzg build", {})
    captured = capsys.readouterr()
    assert captured.out == ""
    assert _HIDE.decode() in captured.err
    assert _SHOW.decode() in captured.err


# ─────────────────────────────────────────────────────────────────────────
# SIGWINCH: guarded registration, reflow, and restoring the prior handler.
# ─────────────────────────────────────────────────────────────────────────


def test_sigwinch_registration_skipped_off_main_thread_without_raising(monkeypatch):
    calls = []
    monkeypatch.setattr(signal, "signal", lambda *a, **kw: calls.append((a, kw)))
    caps = _plain_caps()
    buf = io.StringIO()
    errors = []

    def run():
        try:
            r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
            r.start("lzg build", {})
            r.done("lzg build", {})
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), "background thread did not finish"
    assert not errors, f"start()/done() must not raise off the main thread: {errors}"
    assert calls == [], "signal.signal must never be called off the main thread"


@pytest.mark.skipif(not hasattr(signal, "SIGWINCH"), reason="platform has no SIGWINCH")
def test_sigwinch_registration_skipped_when_platform_lacks_it(monkeypatch):
    monkeypatch.delattr(signal, "SIGWINCH", raising=False)
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    assert r._sigwinch_installed is False
    r.done("lzg build", {})


@pytest.mark.skipif(not hasattr(signal, "SIGWINCH"), reason="platform has no SIGWINCH")
def test_previous_sigwinch_handler_is_restored_on_stop_not_clobbered():
    def sentinel(signum, frame):
        return None

    old = signal.signal(signal.SIGWINCH, sentinel)
    try:
        caps = _plain_caps()
        buf = io.StringIO()
        r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
        r.start("lzg build", {})
        assert signal.getsignal(signal.SIGWINCH) == r._on_sigwinch
        r.done("lzg build", {})
        assert signal.getsignal(signal.SIGWINCH) is sentinel
    finally:
        signal.signal(signal.SIGWINCH, old)


@pytest.mark.skipif(not hasattr(signal, "SIGWINCH"), reason="platform has no SIGWINCH")
def test_sigwinch_handler_reflows_at_the_new_width(monkeypatch):
    caps = _plain_caps(width=60, real_width=60)
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {"SOURCE": "x"})

    wider = dataclasses.replace(caps, width=90, real_width=90)
    monkeypatch.setattr("LZGraphs._term._rich.detect", lambda stream=None, env=None: wider)

    before = r.redraw_count
    r._on_sigwinch(signal.SIGWINCH, None)

    assert r._caps.width == 90
    assert r.redraw_count == before + 1  # forced: a resize is never throttled away
    r.done("lzg build", {})


# ── Fix round 1, Finding 2: idempotent SIGWINCH installation ────────


@pytest.mark.skipif(not hasattr(signal, "SIGWINCH"), reason="platform has no SIGWINCH")
def test_double_start_then_done_restores_the_true_original_handler():
    """Reviewer-confirmed bug: start(), start(), done() left the renderer's
    OWN handler installed forever, because the second start() re-registered
    and captured the (already-installed) first handler as "the previous
    one", losing the true original for the life of the process."""

    def sentinel(signum, frame):
        return None

    old = signal.signal(signal.SIGWINCH, sentinel)
    try:
        caps = _plain_caps()
        buf = io.StringIO()
        r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)

        r.start("lzg build", {})
        assert signal.getsignal(signal.SIGWINCH) == r._on_sigwinch

        r.start("lzg build again", {})  # double start, no stop() in between
        assert signal.getsignal(signal.SIGWINCH) == r._on_sigwinch

        r.done("lzg build", {})
        assert signal.getsignal(signal.SIGWINCH) is sentinel
    finally:
        signal.signal(signal.SIGWINCH, old)


@pytest.mark.skipif(not hasattr(signal, "SIGWINCH"), reason="platform has no SIGWINCH")
def test_two_renderers_stopped_out_of_order_do_not_clobber_each_other():
    """Reviewer-confirmed bug: A starts, B starts (capturing A's handler as
    its own "previous"), then A stops first. A must not blindly restore
    its saved original over B's still-active handler."""
    original = signal.getsignal(signal.SIGWINCH)
    try:
        caps = _plain_caps()
        buf_a, buf_b = io.StringIO(), io.StringIO()
        a = RichRenderer(caps, stream=buf_a, clock=lambda: 0.0)
        b = RichRenderer(caps, stream=buf_b, clock=lambda: 0.0)

        a.start("lzg build", {})
        assert signal.getsignal(signal.SIGWINCH) == a._on_sigwinch

        b.start("lzg build", {})
        assert signal.getsignal(signal.SIGWINCH) == b._on_sigwinch

        a.done("lzg build", {})  # out of order: A stops while B is active
        assert signal.getsignal(signal.SIGWINCH) == b._on_sigwinch, (
            "A must not clobber B's still-active handler"
        )

        b.done("lzg build", {})  # restores to what was active when B started
        assert signal.getsignal(signal.SIGWINCH) == a._on_sigwinch
    finally:
        signal.signal(signal.SIGWINCH, original)


@pytest.mark.skipif(not hasattr(signal, "SIGWINCH"), reason="platform has no SIGWINCH")
def test_sigwinch_resize_changes_geometry_only_not_colour_or_unicode(monkeypatch):
    """Minor finding (fix round 1): a resize must change width/height only.
    Simulates a redetection that would ALSO report a different colour
    depth and unicode support (as if the environment had changed too), and
    proves that divergence never leaks into the live Capabilities."""
    caps = _plain_caps(width=60, real_width=60, height=24, colours=8, unicode=True)
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {"SOURCE": "x"})

    redetected = dataclasses.replace(
        caps, width=90, real_width=90, height=30, colours=0, unicode=False
    )
    monkeypatch.setattr("LZGraphs._term._rich.detect", lambda stream=None, env=None: redetected)

    r._on_sigwinch(signal.SIGWINCH, None)

    assert r._caps.width == 90
    assert r._caps.real_width == 90
    assert r._caps.height == 30
    assert r._caps.colours == 8, "colour depth must be carried forward, not re-detected"
    assert r._caps.unicode is True, "unicode support must be carried forward, not re-detected"
    r.done("lzg build", {})


# ─────────────────────────────────────────────────────────────────────────
# Protocol behaviour: card composition, clamping, post-session no-ops.
# ─────────────────────────────────────────────────────────────────────────


def test_done_card_title_matches_the_approved_mockup_composition():
    """Pins the _plain.py-matching interpretation of ``title``: command
    identity, reused by card() as ``f"{title} {dot} {word}"``. Produces
    the approved mockup's card title exactly; see the module docstring's
    "title in done()/error(), matching _plain.py" section."""
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.done("lzg build", {"nodes": "71,181"})
    assert "lzg build · done" in buf.getvalue()


def test_error_card_title_matches_the_approved_mockup_composition():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.error("lzg build", ["column 'junction_aa' not found"])
    assert "lzg build · error" in buf.getvalue()


def test_error_headline_is_the_first_line_marked_with_a_cross():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.error("lzg build", ["column 'junction_aa' not found", "", "try: --column cdr3_aa"])
    out = buf.getvalue()
    assert "✗ column 'junction_aa' not found" in out
    assert "try: --column cdr3_aa" in out


def test_error_and_done_tolerate_no_supporting_detail():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.error("lzg build", None)  # no lines at all
    out = buf.getvalue()
    assert "error" in out

    buf2 = io.StringIO()
    r2 = RichRenderer(caps, stream=buf2, clock=lambda: 0.0)
    r2.start("lzg build", {})
    r2.done("lzg build", None)  # no rows at all
    assert "done" in buf2.getvalue()


def test_progress_fraction_is_clamped_to_0_1():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.progress("ingest", 5.0)
    assert r._progress["ingest"][0] == 1.0
    r.progress("ingest", -3.0)
    assert r._progress["ingest"][0] == 0.0


def test_warnings_are_capped_to_the_most_recent_few():
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})
    for i in range(10):
        clock[0] += 1.0
        r.warn(f"warning number {i}")
    assert len(r._warnings) <= 4
    assert r._warnings[-1] == "warning number 9"


def test_calls_after_done_are_harmless_no_ops():
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.done("lzg build", {"nodes": "1"})
    out_before = buf.getvalue()

    r.progress("ingest", 0.9)
    r.update(extra="value")
    r.warn("late warning")

    assert buf.getvalue() == out_before, "post-session calls must not write anything further"


def test_update_merges_fields_without_reordering_existing_keys():
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {"SOURCE": "a"})
    clock[0] += 1.0
    r.update(SOURCE="a", FORMAT="AIRR-TSV")
    assert list(r._fields.keys()) == ["SOURCE", "FORMAT"]
    r.done("lzg build", {})


# ─────────────────────────────────────────────────────────────────────────
# Fix round 2, C2: a warning shown live must still be visible in the final
# done()/error() card, not wiped by it.
# ─────────────────────────────────────────────────────────────────────────


def test_warning_remains_visible_in_the_final_done_card():
    """Before this fix, done() built its card purely from the caller's own
    ``rows``, so a warning shown for a fraction of a second during the
    build vanished the instant done() drew over it: a user who looked away
    saw a clean success. Isolates exactly the bytes done() itself writes
    (everything after the warning was already on screen) and checks the
    warning survived into that final frame, not just some earlier one."""
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})
    clock[0] += 1.0
    r.warn("412 non-productive skipped")
    before_done = len(buf.getvalue())
    clock[0] += 1.0
    r.done("lzg build", {"nodes": "1"})
    final_frame = buf.getvalue()[before_done:]
    assert "412 non-productive skipped" in final_frame
    assert "done" in final_frame


def test_warning_remains_visible_in_the_final_error_card():
    """Same guarantee, on the error() path."""
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})
    clock[0] += 1.0
    r.warn("mixed input format recovered mid-file")
    before_error = len(buf.getvalue())
    clock[0] += 1.0
    r.error("lzg build", ["nothing left to build a graph from"])
    final_frame = buf.getvalue()[before_error:]
    assert "mixed input format recovered mid-file" in final_frame
    assert "nothing left to build a graph from" in final_frame


def test_many_warnings_are_capped_in_the_final_card_with_an_elided_count():
    """More warnings than the live cap (:data:`_MAX_WARNINGS`, 4) still
    only shows the most recent few in the final card, but must say how
    many were elided rather than just silently capping."""
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})
    for i in range(6):
        clock[0] += 1.0
        r.warn(f"warning number {i}")
    before_done = len(buf.getvalue())
    clock[0] += 1.0
    r.done("lzg build", {"nodes": "1"})
    final_frame = buf.getvalue()[before_done:]
    assert "warning number 5" in final_frame  # most recent survives
    assert "warning number 0" not in final_frame  # oldest was capped
    assert "2 more warning" in final_frame  # 6 total, 4 shown, 2 elided


def test_no_warnings_leaves_the_done_card_exactly_as_before():
    """A clean build (no warnings at all) must not gain a stray blank
    section: _append_warning_rows is a true no-op when there is nothing to
    append."""
    caps = _plain_caps()
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.done("lzg build", {"nodes": "1"})
    assert "warning" not in buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────
# Fix round 2, I1: a done()/error() card squeezed by a small terminal must
# say rows were hidden, not silently drop them (mirrors _clamp_rows, which
# already did this for the *live* frame; _fit_to_terminal, the safety net
# for the final card, used to just slice the excess off with no marker).
# ─────────────────────────────────────────────────────────────────────────


def test_done_card_shows_an_elision_marker_at_a_small_terminal():
    caps = _plain_caps(width=30, height=4)
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.done("lzg build", {
        "output": "foundation.lzg",
        "nodes": "71,181",
        "edges": "11,714,847",
        "kept": "2,341,102 of 2,341,517",
        "size": "1.2 GB",
    })
    out = buf.getvalue()
    assert "hidden" in out, "a squeezed card must say rows were hidden, not silently drop them"
    assert r._lines_drawn <= max(1, 4 - 1)


def test_done_card_at_a_small_terminal_never_exceeds_the_budget():
    caps = _plain_caps(width=30, height=4)
    buf = io.StringIO()
    r = RichRenderer(caps, stream=buf, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.done("lzg build", {
        "output": "foundation.lzg",
        "nodes": "71,181",
        "edges": "11,714,847",
        "kept": "2,341,102 of 2,341,517",
        "size": "1.2 GB",
    })
    assert r._lines_drawn == max(1, 4 - 1)


# ─────────────────────────────────────────────────────────────────────────
# Fix round 2, I2: a broken stream (full disk, closed pipe) degrades
# rather than raising, and does not kill the build it is narrating.
# ─────────────────────────────────────────────────────────────────────────


class _RaisingStream:
    """Every write/flush raises ``OSError``, standing in for a full disk or
    a downstream reader that closed its end of a pipe. Before fix round 2,
    ``_safe_write`` caught only ``UnicodeEncodeError``, so this reached
    ``stream.write(...)``'s own ``OSError`` straight out of ``start()``/
    ``progress()``/``update()``/``warn()``/``done()``/``error()`` and
    killed the build being narrated.
    """

    encoding = "utf-8"

    def write(self, text):
        raise OSError("broken pipe")

    def flush(self):
        raise OSError("broken pipe")


def test_oserror_on_write_does_not_raise_and_build_can_continue():
    caps = _plain_caps()
    stream = _RaisingStream()
    r = RichRenderer(caps, stream=stream, clock=lambda: 0.0)
    r.start("lzg build", {"source": "in.tsv"})  # must not raise
    r.progress("ingest", 0.5)
    r.update(nodes=5)
    r.warn("something recoverable happened")
    r.done("lzg build", {"nodes": "5"})  # reaching here is the proof


def test_oserror_on_write_does_not_raise_on_the_error_path():
    caps = _plain_caps()
    stream = _RaisingStream()
    r = RichRenderer(caps, stream=stream, clock=lambda: 0.0)
    r.start("lzg build", {})
    r.error("lzg build", ["nothing left to build a graph from"])  # must not raise


def test_oserror_during_the_unicode_fallback_rewrite_also_degrades():
    """The fallback rewrite after a ``UnicodeEncodeError`` is itself a
    second write and must be guarded too: a stream that is both
    ASCII-only *and* broken (an ``ascii``-pinned pipe that is also closed)
    must not raise on the retry either. Exercises the *other* half of
    ``_safe_write``'s guard than ``_RaisingStream`` above, which never
    reaches the fallback branch at all (it raises ``OSError`` on the very
    first write, before any encoding is even attempted)."""

    class _AsciiThenBrokenStream:
        encoding = "ascii"

        def write(self, text):
            text.encode("ascii", errors="strict")  # raises UnicodeEncodeError first
            raise OSError("broken pipe")  # only reached once text is ASCII-safe

        def flush(self):
            raise OSError("broken pipe")

    caps = _plain_caps()
    r = RichRenderer(caps, stream=_AsciiThenBrokenStream(), clock=lambda: 0.0)
    r.start("lzg build", {"source": "café.tsv"})  # non-ASCII triggers the fallback path
    r.done("lzg build", {"nodes": "1"})  # reaching here is the proof


# ─────────────────────────────────────────────────────────────────────────
# Fix round 2, D1/D2: the live panel shows only a curated field set, with
# designed numeric formatting, rather than echoing every internal key
# update() has ever been called with.
# ─────────────────────────────────────────────────────────────────────────


def test_live_panel_shows_only_the_curated_fields_in_a_stable_order():
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {
        "source": "bcr_repertoire.tsv.gz",
        "format": "AIRR-TSV (gzip)",
        "engine": "aap",
    })
    clock[0] += 1.0
    r.update(
        phase="construct", stage="start", mode="in_memory",
        nodes=15089, edges=46073, size_kb="46073.7", nonproductive=3,
    )
    out = buf.getvalue()
    # Curated fields render...
    assert "source" in out
    assert "format" in out
    assert "engine" in out
    assert "nodes" in out
    assert "edges" in out
    # ...but internal bookkeeping fields cmd_build also passes to update()
    # never leak into the live panel (they still reach the plain renderer's
    # full machine-readable log; this is display curation only).
    assert "phase" not in out
    assert "stage" not in out
    assert "size_kb" not in out
    assert "nonproductive" not in out
    # Curated fields keep cmd_build's own key names in a fixed order
    # (source, format, engine, nodes, edges), not update()'s call order.
    assert out.index("source") < out.index("format") < out.index("engine")
    assert out.index("engine") < out.index("nodes") < out.index("edges")


def test_live_panel_formats_node_and_edge_counts_like_the_done_card():
    """D2: the live panel used to show raw values (``nodes 15089``) while
    done()'s own card already ran the same values through counter()
    (``nodes 15,089``). The live row must look designed too."""
    caps = _plain_caps()
    buf = io.StringIO()
    clock = [0.0]
    r = RichRenderer(caps, stream=buf, clock=lambda: clock[0])
    r.start("lzg build", {})
    clock[0] += 1.0
    r.update(nodes=15089, edges=46073)
    out = buf.getvalue()
    assert "15,089" in out
    assert "46,073" in out
    assert "15089" not in out
    assert "46073" not in out
