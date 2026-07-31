"""The live, redrawing renderer for a real TTY.

This is the riskiest module in the ``_term`` layer, because it is the only
one that owns terminal *state* rather than just terminal *content*: it hides
the cursor while it redraws in place, and it is the one place in the whole
CLI where getting teardown wrong leaves a user staring at an invisible
cursor after ``lzg build`` was interrupted. Everything below is organised
around that one failure mode first, the redraw arithmetic second, and the
throttle/reflow niceties last.

## Terminal-state restoration, in depth

The invariant this module exists to protect: **the cursor is hidden between
``start()`` and whichever of ``done()``/``error()`` closes the session, and
visible at every other time**, no matter how the session ends. Four
independent mechanisms cooperate, deliberately redundant with each other
rather than relying on exactly one:

1. **The normal path.** ``done()`` and ``error()`` both draw their final
   card and then call the idempotent ``_stop()``, which writes
   :func:`~LZGraphs._term._ansi.show_cursor` and tears down the signal
   handler.
2. **``try``/``except``/re-raise around startup itself.** ``start()``'s own
   body (write ``hide_cursor``, draw the first frame) is wrapped so that if
   *anything* goes wrong while opening the session, ``_stop()`` still runs
   before the exception propagates: a half-opened session is never worse
   than a fully torn-down one.
3. **Context-manager support (``with``), the literal try/finally
   guarantee.** ``RichRenderer.__exit__`` calls ``_stop()`` unconditionally,
   exception or not, which is exactly what ``try/finally`` gives you for
   free; a caller that wraps a whole command in ``with ui:`` gets restoration
   guaranteed by the language, not by this module remembering to do the
   right thing.
4. **``atexit``, the fallback for callers who never use ``with``.**
   ``start()`` registers ``self._stop`` with :mod:`atexit`, so if an
   exception (a bare ``KeyboardInterrupt`` included) unwinds all the way to
   interpreter shutdown without ever passing through this module's own
   ``except``/``__exit__`` handling, the cursor is still restored just
   before the process actually exits. ``_stop()`` unregisters itself on the
   normal path so a long-lived process that opens many sessions does not
   accumulate exit hooks.

``_stop()`` itself is guarded by a ``self._stopped`` flag, so calling it
twice (``done()`` then the ``atexit`` hook firing anyway, or ``__exit__``
after ``done()`` already ran) is always a safe no-op; and its own writes are
wrapped in ``try/except Exception`` (not ``BaseException``) so a broken
stream during cleanup can never itself raise and mask whatever real
exception (a ``KeyboardInterrupt`` in particular) is already propagating.

## Redraw arithmetic

``_draw`` is handed the *complete* next frame (a list of already-``caps.width``-
wide lines from :mod:`._widgets`) and does three things, tracked by exactly
one piece of state, ``self._lines_drawn`` (the height of whatever is
currently on screen, never guessed):

1. Move up by ``self._lines_drawn`` (the *previous* frame's height), via
   :func:`~LZGraphs._term._ansi.cursor_up`.
2. Write every new line followed by
   :func:`~LZGraphs._term._ansi.clear_line`, so any leftover tail from a
   longer old line on that row disappears even though the new line is
   shorter.
3. If the new frame is *shorter* than the old one, the rows below the new
   content still hold stale text and are not overwritten by step 2 at all.
   Each of those leftover rows gets its own blank ``clear_line()`` pass, and
   the cursor is then walked back up by exactly that many rows, so it rests
   immediately after the new content, exactly where the *next* redraw's
   ``cursor_up(self._lines_drawn)`` assumes it is. Skipping this step is
   the one-line bug that leaves orphaned debris on screen the moment a
   build's row count ever shrinks (fields collapsing, a shorter final
   card than the in-flight frame that preceded it).

``self._lines_drawn`` is updated to the new frame's length only after both
of the above, so it always reflects what is actually on screen, never what
was merely requested.

## Throttling

At most :data:`_MAX_REDRAWS_PER_SECOND` (15) actual redraws happen per
second, regardless of how often ``progress()``/``update()``/``warn()`` are
called; a build reporting progress once per record must update internal
state instantly but only *paint* on a fixed cadence. The clock is
``time.monotonic`` by default but is an injectable constructor parameter
(``clock``), specifically so a test can drive 10,000 calls against a fake,
manually-advanced clock and get an exact, non-flaky bound on the number of
real draws, rather than asserting anything that depends on how fast the
test machine actually runs. :data:`_MIN_REDRAW_INTERVAL` (``1/15``) is a
module-level constant for the same reason: a test can import and reason
about it directly instead of hard-coding ``0.0667``. ``start()``, ``done()``,
and ``error()`` always draw (``force=True``): they are session boundaries
that happen at most once each, never a hot loop, and the caller must always
see the opening and closing frame regardless of timing.

## The frame never exceeds the terminal (fix round 1, Finding 1)

Nothing about the redraw arithmetic above bounds *how tall* a frame is
allowed to be, and ``cursor_up`` has no way to move above the physical top
row: once a frame's line count exceeds ``caps.height``, the terminal itself
starts scrolling underneath the redraw loop, every subsequent
``cursor_up(self._lines_drawn)`` addresses rows that are no longer where
this module thinks they are, and the display corrupts progressively. This
is reachable in practice: a ``lzg build`` panel with several fields and a
progress bar is easily ten-plus lines, and short tmux panes/split terminals
routinely report a height well under that.

Two cooperating bounds, at two different levels, both keyed off
``max(1, caps.height - 1)`` (one row is always reserved for the shell
prompt so the live frame never sits flush against it; the ``max(1, ...)``
guards the degenerate ``caps.height`` of 0, 1, or 2 some CI environments
report, so the budget is never zero or negative):

1. **Deliberate content elision** (``_clamp_rows``, used only by
   ``_render_frame``, the live in-progress frame). Applied *before*
   :func:`~LZGraphs._term._widgets.panel` builds the box, while row
   identity is still known: when the assembled rows (fields, then progress
   bars, then warnings, in that order) would not fit, rows are dropped from
   the **front** (the static fields), and the **tail** is kept, because
   the tail is the progress bars and the most recent warnings: the part of
   the panel that is actually changing and that a user watching a live
   build cares about moment to moment. A static ``SOURCE``/``FORMAT``
   field is comparatively safe to hide, and nothing is destroyed: every
   redraw rebuilds the frame from the same tracked state, so making the
   terminal taller again brings the elided fields straight back. Elision
   is never silent: dropped rows are replaced by exactly one row saying
   how many were hidden and why, so a user cannot mistake a clipped panel
   for the complete picture.
2. **A blunt safety net** (``_fit_to_terminal``, called by ``_draw`` for
   *every* frame, including the final ``done()``/``error()`` card). Row
   identity is gone by this point (``panel()``/``card()`` have already
   interleaved borders), so this simply keeps the top ``budget`` lines and
   drops the rest. This exists for two reasons: the final card is built
   from caller-supplied rows ``_render_frame`` never had a chance to
   pre-clamp, and at truly degenerate heights even *zero* content rows can
   leave ``panel()``'s own title-plus-border overhead (fixed at 1 line in
   narrow mode, 2 in boxed mode; see ``_panel_overhead``) taller than the
   budget, since this module does not reimplement ``panel()``'s
   box-drawing to shrink further. Keeping the top is deliberate here too:
   for ``error()`` specifically, the top is exactly the headline
   (``lines[0]``, marked with the cross), the single most important line
   in the card.

Together these guarantee the hard invariant this fix exists for: no
``cursor_up`` this renderer ever issues is larger than
``max(1, caps.height - 1)``, unconditionally, including at heights of 0, 1,
and 2.

## Reflow on ``SIGWINCH``

Where the platform has it, ``SIGWINCH`` fires whenever the terminal is
resized; the handler re-detects the terminal's *geometry only* (width and
height), via :func:`~LZGraphs._term._caps.detect` (the one function in the
whole layer allowed to read the environment or call ``isatty``, so this
module still never touches either directly), and forces an immediate
redraw at the new size. ``_refresh_capabilities`` deliberately does not
adopt the freshly detected ``colours``/``unicode``/anything else: a resize
event means the terminal changed *size*, nothing else, so every other field
of the current ``Capabilities`` is carried forward unchanged via
``dataclasses.replace`` (fix round 1, Minor finding: re-detecting the whole
``Capabilities`` let an unrelated environment change silently flip colour
depth or unicode support mid-render, which a mere resize must never do).

Registration is guarded twice, in the order the task cares about most:

* ``hasattr(signal, "SIGWINCH")`` is checked first, since the signal simply
  does not exist on Windows; a platform without it just never gets reflow,
  silently.
* ``threading.current_thread() is threading.main_thread()`` is checked
  *before* ever calling ``signal.signal``, because Python raises
  ``ValueError: signal only works in main thread`` if you don't. This is a
  proactive skip, not a try/except around the raise: a caller that
  constructs a ``RichRenderer`` on a worker thread (unusual, but not this
  module's place to forbid) gets a renderer with no reflow rather than a
  crash.

The previous handler (whatever it was, including ``SIG_DFL``) is captured
at registration time and restored by ``_stop()``, so this module is a good
citizen even if something else in the process also cares about
``SIGWINCH``. Two refinements landed in fix round 1, both reviewer-found:

* **Installation is idempotent** (``_install_sigwinch`` returns immediately
  if ``self._sigwinch_installed`` is already ``True``). Without this, a
  second ``start()`` call on the same instance without an intervening
  ``stop()`` (``start()``, ``start()``, ``done()``) would re-register and
  capture *this renderer's own handler* as "the previous one", permanently
  losing the real original for the life of the process.
* **Restoration only clobbers what is still ours** (``_restore_sigwinch``
  checks ``signal.getsignal(signal.SIGWINCH) == self._on_sigwinch`` before
  restoring; ``==`` deliberately, not ``is``, since bound methods compare
  equal but are never identical across separate attribute accesses).
  Without this, two renderers stopped out of order (``A`` starts, ``B``
  starts, capturing ``A``'s handler as its "previous", then ``A`` stops
  first) would have ``A`` blindly restore its own saved original over
  ``B``'s still-active handler, clobbering it.

**Documented limitation, not fixed further, on purpose (per review: make
the common cases safe, do not build a full handler stack):** in that same
out-of-order scenario, ``A``'s stop correctly declines to restore (``B`` is
active, not ``A``), but this leaves ``A``'s true original handler
un-restored until ``B`` itself later stops (at which point ``B`` restores
*its* saved previous, which is ``A``'s handler, not the true original).
The true original only comes back once every renderer that was ever active
during the overlap has stopped, in an order that eventually unwinds back to
it. A single active renderer at a time (the normal case: one ``RichRenderer``
per ``lzg`` invocation) and repeated ``start()``/``stop()`` cycles on one
instance are both fully correct; only genuinely overlapping renderers have
this residual limitation, and it never crashes or clobbers a *still-active*
handler, it only delays restoring the very first one.

## Streams

Every write goes to whatever ``stream`` this renderer was constructed with
(``sys.stderr`` by default, matching :func:`~LZGraphs._term._caps.detect`'s
own default and :class:`~LZGraphs._term._plain.PlainRenderer`'s). Nothing in
this module ever names ``sys.stdout``.

## ``title`` in ``done()``/``error()``, matching ``_plain.py``

``_plain.py`` (Task 4) settled on ``title`` meaning *command identity*
(``"lzg build"``), reused as the log tag, with the failure headline living
in ``lines[0]``; see its module docstring for the two reasons that reading
won. This module follows the identical interpretation, and it is exactly
what makes the approved mockup's card titles reproducible:
:func:`~LZGraphs._term._widgets.card` composes ``f"{title} {dot} {word}"``,
so ``card("lzg build", rows, caps, status="error")`` renders precisely
``"lzg build · error"``, matching ``demo_ui.py``'s ``error_screen()`` title
byte for byte. Had ``title`` instead meant the failure headline itself, this
composition would not have been reachable without a second, redundant
"command name" parameter nowhere in the ``Ui`` protocol; the shared reading
is not just consistent with ``_plain.py``, it is the only one that produces
the approved design.

## What this module does *not* try to be pixel-perfect about

The exact fields/labels a real ``lzg build`` shows (``SOURCE``, ``FORMAT``,
``ENGINE``, a rate sparkline, ...) are Task 6's decision, made by whatever
it passes as ``fields``/``label``/``rows``: this renderer turns *whatever
generic facts it is given* into a panel of ``kv`` rows plus one ``bar`` row
per distinct progress ``label``, using exactly the widgets Task 3 built for
this, never hand-building a box. ``demo_ui.py`` is, per the plan, "a paper
prototype ... not a specification of internals"; the redraw engine here is
built to render *whatever* generic session Task 6 describes correctly and
safely, not to hard-code the specific field names in the mockup screenshot.
"""
from __future__ import annotations

import atexit
import dataclasses
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, TextIO

from ._ansi import clear_line, colour, cursor_up, hide_cursor, show_cursor
from ._caps import detect
from ._widgets import bar, card, kv, panel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ._caps import Capabilities

#: Hard cap on actual screen redraws per second. See the module docstring's
#: "Throttling" section.
_MAX_REDRAWS_PER_SECOND = 15

#: ``1 / _MAX_REDRAWS_PER_SECOND``, exposed so tests can reason about the
#: exact threshold instead of hard-coding it.
_MIN_REDRAW_INTERVAL = 1.0 / _MAX_REDRAWS_PER_SECOND

#: Most recent warn() messages kept in the live frame; older ones scroll off
#: rather than growing the panel without bound over a long build.
_MAX_WARNINGS = 4

#: Progress bar width bounds, independent of caps.width so the bar itself
#: never collapses to nothing on a narrow terminal or stretches absurdly
#: wide on an ultra-wide one; see _bar_width.
_MIN_BAR_WIDTH = 10
_MAX_BAR_WIDTH = 30
#: How much of caps.width is reserved for the label, percentage, and detail
#: text alongside the bar itself.
_BAR_WIDTH_MARGIN = 30

#: One row of the terminal is always left for the shell prompt: the live
#: frame must never claim the entire height. See the module docstring's
#: "The frame never exceeds the terminal" section.
_PROMPT_RESERVE_ROWS = 1

#: Duplicated from ``_widgets._NARROW_WIDTH`` deliberately, not imported:
#: this module only needs the threshold to know how many of the lines
#: panel()/card() return are their own title/border furniture versus
#: content this module actually supplied, without depending on a sibling
#: module's private name or reimplementing its box-drawing.
_NARROW_WIDTH = 50


def _bar_width(caps: Capabilities) -> int:
    """Progress-bar width for the current panel width, clamped sensibly."""
    return max(_MIN_BAR_WIDTH, min(_MAX_BAR_WIDTH, caps.width - _BAR_WIDTH_MARGIN))


def _cross(caps: Capabilities) -> str:
    """The error-headline marker: a unicode cross, or ``"x"`` under ASCII."""
    return "✗" if caps.unicode else "x"


def _panel_overhead(caps: Capabilities) -> int:
    """Lines :func:`~LZGraphs._term._widgets.panel` adds around content rows:
    the title line plus the closing border in boxed mode
    (``caps.width >= _NARROW_WIDTH``), or just the title line in narrow
    mode. Neither ``_render_frame`` nor ``done()``/``error()`` ever pass a
    ``footer`` (that would add one more divider-plus-row pair), so this is
    the complete overhead for every frame this module draws.
    """
    return 1 if caps.width < _NARROW_WIDTH else 2


def _visible_row_budget(caps: Capabilities) -> int:
    """How many *content* rows fit alongside one reserved prompt row and
    panel()'s own border overhead, for the given ``caps``.

    Never negative, even at a degenerate ``caps.height`` of 0, 1, or 2
    (some CI environments report these): the outer ``max(1, ...)`` floors
    the visible height itself, and the ``max(0, ...)`` floors the content
    budget once the (fixed) border overhead is subtracted from it.
    """
    visible = max(1, caps.height - _PROMPT_RESERVE_ROWS)
    return max(0, visible - _panel_overhead(caps))


class RichRenderer:
    """The live-redrawing renderer for a real, cursor-capable TTY.

    Satisfies the :class:`~LZGraphs._term.Ui` protocol. Constructed only
    when :func:`~LZGraphs._term._caps.resolve_mode` has already decided the
    stream supports cursor control (``caps.supports_cursor_control``); this
    module trusts that decision rather than re-checking ``caps.is_tty``
    itself; see :mod:`._ansi`'s module docstring for why the cursor-motion
    primitives it wraps deliberately do not take ``caps`` either.

    Usable two ways, both restoring terminal state correctly:

    * Directly: ``r = RichRenderer(caps); r.start(...); ...; r.done(...)``.
      ``start()`` registers an ``atexit`` fallback, so even an uncaught
      exception that unwinds all the way past this call site still restores
      the cursor before the process exits.
    * As a context manager: ``with RichRenderer(caps) as r: r.start(...);
      ...``. ``__exit__`` calls the same idempotent teardown unconditionally,
      which is the strongest, most direct guarantee (Python's own
      try/finally semantics), and is the recommended pattern for a caller
      that can adopt it.

    ``clock`` is an injectable callable returning a monotonic timestamp
    (``time.monotonic`` by default), so the redraw throttle can be tested
    against a fake, manually-advanced clock rather than real sleeps.
    """

    def __init__(
        self,
        caps: Capabilities,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._caps = caps
        self._stream = stream if stream is not None else sys.stderr
        self._clock = clock

        self._lines_drawn = 0
        self._last_draw_time: float | None = None
        self._redraw_count = 0

        self._title = ""
        self._fields: dict[str, Any] = {}
        self._progress: dict[str, tuple[float, str | None]] = {}
        self._warnings: list[str] = []
        self._t0: float | None = None

        # True until start() runs, and again as soon as the session is torn
        # down (done()/error()/atexit/__exit__); guards _stop() idempotency
        # and turns post-session progress()/update()/warn() calls into
        # no-ops instead of writing to a stream nobody is watching redraw
        # any more.
        self._stopped = True

        self._sigwinch_installed = False
        self._prev_sigwinch_handler: Any = None

    @property
    def redraw_count(self) -> int:
        """Number of frames actually painted so far (throttle-observable)."""
        return self._redraw_count

    def __enter__(self) -> RichRenderer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop()
        # Never suppress: a KeyboardInterrupt (or anything else) that was
        # propagating through the `with` block keeps propagating after
        # terminal state is restored.
        return None

    # ── Ui protocol ──────────────────────────────────────────────────

    def start(self, title: str, fields: Mapping[str, Any] | None = None) -> None:
        self._title = title
        self._fields = dict(fields) if fields else {}
        self._progress = {}
        self._warnings = []
        self._lines_drawn = 0
        self._redraw_count = 0
        self._last_draw_time = None
        self._t0 = self._clock()
        self._stopped = False

        atexit.register(self._stop)
        self._install_sigwinch()
        try:
            self._stream.write(hide_cursor())
            self._stream.flush()
            self._redraw(force=True)
        except BaseException:
            # Startup itself failed (a broken stream, an unexpected error
            # from a helper) or was interrupted; never leave the cursor
            # hidden for a session that did not actually open.
            self._stop()
            raise

    def progress(self, label: str, fraction: float, detail: str | None = None) -> None:
        if self._stopped:
            return
        fraction = max(0.0, min(1.0, fraction))
        self._progress[label] = (fraction, detail)
        self._redraw()

    def update(self, **fields: Any) -> None:
        if self._stopped:
            return
        self._fields.update(fields)
        self._redraw()

    def warn(self, message: str) -> None:
        if self._stopped:
            return
        self._warnings.append(message)
        if len(self._warnings) > _MAX_WARNINGS:
            del self._warnings[: len(self._warnings) - _MAX_WARNINGS]
        self._redraw()

    def error(self, title: str, lines: Sequence[str] | None = None) -> None:
        rows: list[str] = []
        for i, line in enumerate(lines or ()):
            if not line:
                rows.append("")
                continue
            if i == 0:
                mark = colour("error", _cross(self._caps), self._caps)
                rows.append(f" {mark} {line}")
            else:
                rows.append(f" {line}")
        self._finish(card(title, rows, self._caps, status="error"))

    def done(self, title: str, rows: Mapping[str, Any] | None = None) -> None:
        kv_rows = [kv(str(key), str(value), self._caps) for key, value in (rows or {}).items()]
        self._finish(card(title, kv_rows, self._caps, status="ok"))

    # ── redraw engine ────────────────────────────────────────────────

    def _render_frame(self) -> list[str]:
        """Build the current frame's lines from tracked session state.

        Field rows first, then (separated by a blank row, if both are
        present) one bar row per distinct ``progress()`` label in the order
        it was first seen, then up to :data:`_MAX_WARNINGS` recent warnings.
        """
        rows: list[str] = [
            kv(key, str(value), self._caps) for key, value in self._fields.items()
        ]

        if self._progress:
            if rows:
                rows.append("")
            width = _bar_width(self._caps)
            for label, (fraction, detail) in self._progress.items():
                pct = f"{int(round(fraction * 100)):3d}%"
                text = f"{bar(fraction, width, self._caps)}  {pct}"
                if detail:
                    text += "  " + colour("muted", detail, self._caps)
                rows.append(kv(label, text, self._caps))

        for message in self._warnings:
            rows.append(kv("", colour("warn", message, self._caps), self._caps))

        rows = self._clamp_rows(rows)
        return panel(self._title, rows, self._caps)

    def _clamp_rows(self, rows: list[str]) -> list[str]:
        """Drop rows from the front when the assembled frame would not fit
        the terminal, keeping the tail and marking the elision explicitly.

        See the module docstring's "The frame never exceeds the terminal"
        section for the full reasoning: the tail (progress bars, then the
        most recent warnings) is the part of the panel actually changing
        moment to moment, so it is kept; the front (static fields) is
        dropped first, and replaced by exactly one row stating how many
        were hidden, so a clipped panel is never mistaken for the complete
        one.
        """
        budget = _visible_row_budget(self._caps)
        if len(rows) <= budget:
            return rows
        if budget <= 0:
            # Not even room for a lone elision marker; _fit_to_terminal is
            # the last-resort safety net for this degenerate case.
            return []
        keep_n = max(0, budget - 1)
        hidden = len(rows) - keep_n
        marker = colour(
            "muted",
            f"... {hidden} more row{'s' if hidden != 1 else ''} hidden "
            f"({self._caps.height}-row terminal)",
            self._caps,
        )
        tail = rows[-keep_n:] if keep_n > 0 else []
        return [marker, *tail]

    def _redraw(self, force: bool = False) -> bool:
        """Paint the current frame if the throttle allows it (or ``force``).

        Returns whether a redraw actually happened, mostly so callers other
        than tests never need to; nothing in this class currently branches
        on it.
        """
        if self._stopped:
            return False
        now = self._clock()
        if not force and self._last_draw_time is not None:
            if (now - self._last_draw_time) < _MIN_REDRAW_INTERVAL:
                return False
        self._last_draw_time = now
        self._draw(self._render_frame())
        return True

    def _fit_to_terminal(self, lines: list[str]) -> list[str]:
        """Absolute safety net: no frame this renderer draws may be taller
        than the terminal itself, regardless of where its rows came from.

        ``_render_frame`` already performs deliberate, row-aware elision
        (``_clamp_rows``), so this is a no-op for the live build frame in
        every ordinary case. It exists for two belt-and-suspenders reasons,
        detailed in the module docstring's "The frame never exceeds the
        terminal" section: the final ``done()``/``error()`` card is built
        from caller-supplied rows this module never pre-clamps, and at
        truly degenerate heights (0, 1, 2 rows) even zero content rows can
        leave panel()'s own border overhead taller than the budget. Keeps
        the *top* ``budget`` lines and drops the rest: for ``error()`` in
        particular, the top is exactly the headline, the single most
        important line in the card.
        """
        budget = max(1, self._caps.height - _PROMPT_RESERVE_ROWS)
        if len(lines) <= budget:
            return lines
        return lines[:budget]

    def _draw(self, lines: list[str]) -> None:
        """Redraw the live area to exactly ``lines`` (clamped to fit the
        terminal first; see :meth:`_fit_to_terminal`).

        See the module docstring's "Redraw arithmetic" section for the full
        reasoning; in short: move up by the previous frame's tracked height,
        overwrite each row (clearing its old tail), and if the new frame is
        shorter, blank out and re-ascend past whatever rows are left over
        so no stale content survives a shrinking frame.
        """
        lines = self._fit_to_terminal(lines)
        parts: list[str] = []
        if self._lines_drawn:
            parts.append(cursor_up(self._lines_drawn))
        for line in lines:
            parts.append(line)
            parts.append(clear_line())
            parts.append("\n")

        shrink = self._lines_drawn - len(lines)
        if shrink > 0:
            for _ in range(shrink):
                parts.append(clear_line())
                parts.append("\n")
            parts.append(cursor_up(shrink))

        self._stream.write("".join(parts))
        self._stream.flush()
        self._lines_drawn = len(lines)
        self._redraw_count += 1

    def _finish(self, panel_lines: list[str]) -> None:
        """Draw a final ``done()``/``error()`` card and close the session."""
        self._draw(panel_lines)
        self._stop()

    # ── teardown ─────────────────────────────────────────────────────

    def _stop(self) -> None:
        """Idempotent teardown: restore the cursor, the signal handler, and
        unregister the ``atexit`` fallback.

        Safe to call any number of times (only the first call after a
        ``start()`` does anything) and from any of the four call sites
        described in the module docstring's "Terminal-state restoration"
        section: the normal ``done()``/``error()`` path, ``start()``'s own
        failure handling, ``__exit__``, and the ``atexit`` hook itself.
        """
        if self._stopped:
            return
        self._stopped = True
        self._restore_sigwinch()
        try:
            atexit.unregister(self._stop)
        except Exception:
            pass
        try:
            self._stream.write(show_cursor())
            self._stream.flush()
        except Exception:
            # A broken stream during cleanup must never raise here: doing
            # so could mask whatever exception (a KeyboardInterrupt in
            # particular) is already propagating through the caller.
            pass

    # ── SIGWINCH ─────────────────────────────────────────────────────

    def _install_sigwinch(self) -> None:
        """Register a ``SIGWINCH`` handler, or silently do nothing where
        that is unsafe, unsupported, or already done.

        Three independent guards, checked in this order because the second
        prevents a real crash and must never be skipped, and the third
        (fix round 1, Finding 2) is what makes this idempotent:

        1. The platform has no ``SIGWINCH`` at all (Windows): nothing to
           register, reflow just never happens there.
        2. This is not the main thread: ``signal.signal`` raises
           ``ValueError`` off the main thread, so this is checked and
           skipped *before* calling it, not discovered via the exception.
        3. This instance has already installed its handler
           (``self._sigwinch_installed``). Re-registering here would call
           ``signal.signal`` again and get back *this renderer's own
           handler* as the "previous" one (since it is currently active),
           permanently losing the real original the moment a second
           ``start()`` happens without an intervening ``stop()``
           (``start()``, ``start()``, ``done()``). Skipping the second
           install leaves ``self._prev_sigwinch_handler`` holding whatever
           was truly there before the *first* install, which is what
           ``_restore_sigwinch`` must put back.
        """
        if not hasattr(signal, "SIGWINCH"):
            return
        if threading.current_thread() is not threading.main_thread():
            return
        if self._sigwinch_installed:
            return
        try:
            self._prev_sigwinch_handler = signal.signal(signal.SIGWINCH, self._on_sigwinch)
            self._sigwinch_installed = True
        except (ValueError, OSError):
            self._sigwinch_installed = False

    def _restore_sigwinch(self) -> None:
        """Put back whatever ``SIGWINCH`` handler was registered before this
        renderer installed its own, rather than clobbering it permanently.

        Only restores if the *currently active* handler is still this
        renderer's own (fix round 1, Finding 2): if some other code
        (typically a sibling ``RichRenderer`` that started after this one
        and has not yet stopped) has since become the active handler,
        restoring here would clobber *their* handler instead of ours. In
        that case this renderer's saved ``_prev_sigwinch_handler`` is
        simply dropped; see the module docstring's "SIGWINCH" section for
        the documented, deliberately-not-fixed-further limitation this
        leaves for genuinely overlapping renderers.
        """
        if not self._sigwinch_installed:
            return
        self._sigwinch_installed = False
        try:
            # `==`, not `is`: bound methods are equal-but-not-identical on
            # every separate attribute access (`self._on_sigwinch is
            # self._on_sigwinch` is False in plain Python), so `is` here
            # would never match the handler this same instance installed
            # and every restore would be silently skipped.
            if signal.getsignal(signal.SIGWINCH) == self._on_sigwinch:
                signal.signal(signal.SIGWINCH, self._prev_sigwinch_handler)
        except (ValueError, OSError):
            pass

    def _on_sigwinch(self, signum: int, frame: Any) -> None:
        self._refresh_capabilities()
        self._redraw(force=True)

    def _refresh_capabilities(self) -> None:
        """Re-detect only the terminal's geometry (width/height) on resize.

        A ``SIGWINCH`` means the terminal changed *size*, nothing else: it
        must never change colour depth, unicode support, or any other
        capability decided once at construction time (fix round 1, Minor
        finding: re-detecting the whole ``Capabilities`` let an unrelated
        environment change silently flip colour depth or unicode support
        mid-render). Delegates to :func:`~LZGraphs._term._caps.detect` (the
        one function allowed to read the environment or call ``isatty``)
        purely to reuse its fd-based sizing and clamping logic, then keeps
        every other field of the current ``Capabilities`` exactly as it
        was, via ``dataclasses.replace``. Swallows any error from the
        redetection itself: a resize event must never crash a build that
        is otherwise progressing fine.
        """
        try:
            redetected = detect(stream=self._stream)
        except Exception:
            return
        self._caps = dataclasses.replace(
            self._caps,
            width=redetected.width,
            real_width=redetected.real_width,
            height=redetected.height,
        )


__all__ = ["RichRenderer"]
