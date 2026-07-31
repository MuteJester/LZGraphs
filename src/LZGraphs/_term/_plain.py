"""The scrolling, greppable renderer for CI logs and piped output.

**This is deliberately not "the rich renderer with colour turned off."** The
approved mockup (``.private/cli_redesign/demo_ui.py``) falls into exactly
that trap in its own ``--plain`` mode: it draws the same bordered box, just
in ``+-|`` instead of ``╭─│``. In a CI log or a file that box is noise, not
signal. Nobody greps a box, and nobody can diff two runs of one against each
other. ``PlainRenderer`` renders a *different layout*: one stable, tagged,
``key=value`` fact per line, appended to a scrolling log, never redrawn,
never boxed. Modelled directly on the tagged-line style
``src/LZGraphs/cli.py`` already emits today (``[build] phase=... key=value
...``), refined into one small, reusable class instead of one-off
``f"[build] ..."`` strings scattered through every command.

## Line shape

Every line this renderer writes has the same shape, no exceptions::

    [tag] status=<word> key=value key=value ...

``tag`` is derived once, from whatever ``title`` is passed to
:meth:`~PlainRenderer.start`, :meth:`~PlainRenderer.done`, or
:meth:`~PlainRenderer.error` (see :func:`_tag_for`): a leading ``"lzg "`` is
stripped so ``"lzg build"`` becomes ``[build]``, matching the tag the CLI
already emits for that command today; a title with no such prefix is used
as-is (``"posterior"`` -> ``[posterior]``, matching ``[posterior]``).
:meth:`~PlainRenderer.progress`, :meth:`~PlainRenderer.update`, and
:meth:`~PlainRenderer.warn` reuse whatever tag the most recent ``start``/
``done``/``error`` call established (``[lzg]`` if none has run yet, so an
out-of-order call cannot raise).

``status`` is always present and is one of ``start``, ``progress``,
``info`` (from :meth:`update`), ``warn``, ``error``, or ``done``: a script
watching the log, or a human skimming it, can ``grep 'status=error'`` and
find every failure across every command, or ``grep '^\\[build\\]'`` and see
only that command's lines, regardless of which method produced them.

The remaining ``key=value`` pairs come straight from the caller's
``fields``/``rows`` mapping (or ``update``'s keyword arguments), in the
order given, rendered with :func:`str` and with any embedded newline or
carriage return replaced by a space so one *call* always produces exactly
one physical *line* (see :func:`_sanitize`). A value is **not** quoted or
escaped otherwise, including a value containing spaces: a value is read as
"everything up to the next ``word=`` looking token, or the end of the
line", exactly how the existing ``cli.py`` lines already read (e.g.
``sequences=1234 (12 V genes, 34 J genes) total=...``, where the
parenthetical is part of ``sequences``'s value). This is a deliberate,
minimal convention rather than full quoting: it stays exactly as simple as
what the CLI already prints, and it is parseable with one regex
(``r"(\\w+)="`` to find field boundaries, the value being the text between
one match and the next). **Documented limitation, verified rather than
merely asserted** (see ``test_documented_limitation_value_containing_word_
equals_mis_splits`` in the test file): a value must not itself contain a
bare ``word=`` substring, or the boundary parser reads that as the start of
a new field and silently mis-splits it. This renderer's *own*
internally-generated tokens (the tag, ``status=``, ``label=``, ``pct=``,
``detail=``, and the auto-appended ``elapsed=``) never do; a caller-supplied
field *value* (a filesystem path with an embedded ``=``, for instance) can
still contain one, and when it does, it is mis-split exactly as described.
This is a known, accepted limitation of the convention, not a guarantee
about arbitrary caller data.

## Progress throttling

A build that calls :meth:`~PlainRenderer.progress` 100,000 times (once per
record) must not emit 100,000 lines; a CI log that scrolls for a thousand
lines is as useless as no log. The throttle is **per label**: each distinct
``label`` a caller reports under has its own independent percentage/time
state, so a build that alternates between an ``"ingest"`` bar and a
``"save"`` bar (or rotates through any number of named steps) throttles each
one on its own terms instead of one step's updates silently resetting
another's window. Concretely, per label: emit at most once every **5
percentage points** (``_PROGRESS_PCT_STEP``) **or every 1.0 second of wall
time** (``_PROGRESS_TIME_STEP``), whichever comes first, with three
unconditional exceptions that always force a line regardless of both
thresholds:

1. The first call seen for *this* label (nothing has been reported yet for
   it, including a label seen before but since evicted from the bounded
   state below).
2. ``fraction`` reaches (or exceeds) ``1.0`` for the first time for this
   label (the completion of a step is never dropped, even if the last
   emitted line for it was a fraction of a second and a fraction of a
   percent ago).

This bounds a single label's monotonically-increasing 0.0 -> 1.0 sweep to at
most ``100 // 5 + 1 == 21`` lines regardless of how many times ``progress``
is called for it, plus at most one extra line per second of wall-clock time
the sweep actually takes; N labels interleaved bound to at most ``N`` times
that, not to one shared window that a second label can silently reset.

**Bounded state.** Per-label tracking is kept in an
:class:`~collections.OrderedDict` capped at
``_MAX_TRACKED_PROGRESS_LABELS`` (32) entries with least-recently-updated
eviction, so a caller that reports under an unbounded number of distinct
labels (a bug, or a label built from something high-cardinality like a
per-record id) cannot grow this renderer's memory without limit. The one
externally visible consequence of eviction: a label that falls out of the
tracked set and later reports again is treated as if it were being seen for
the first time (exception 1 above applies), which can produce one extra
line for that label; this is judged an acceptable trade for a hard memory
bound, and 32 is comfortably above the small, fixed number of phase labels
any real command reports (a handful, e.g. ``"ingest"``, ``"save"``).

**Global backstop (fix round 2).** The per-label rule above is *conditional
on label cardinality*: with more distinct labels in flight than
``_MAX_TRACKED_PROGRESS_LABELS``, every call can evict the very entry it
needed, so every call looks like exception 1 ("first call for this label")
and the per-label rule never engages at all (measured: 200 labels rotating
through 10,000 calls with no real time elapsing emitted all 10,000 lines,
the exact CI-log flood this whole mechanism exists to prevent). Raising the
per-label cap only moves this threshold, it does not remove it, so a second,
*unconditional* limit sits on top of the per-label one: regardless of
label, and regardless of *why* a line would otherwise be emitted (a fresh
label, a label change, a 100% completion), at most
``_MAX_PROGRESS_LINES_PER_SECOND`` (100) progress lines are ever written in
any trailing ``_GLOBAL_RATE_WINDOW_SECONDS`` (1.0 second) window. This is
checked as the last step before a line that the per-label rule already
approved is actually written, so none of the per-label exceptions (first
call, label change, 100% completion) are exempt from it; exempting them is
exactly how the per-label cap above gets defeated one level up, since an
unbounded number of distinct labels is also an unbounded number of "first
call for this label" events. 100 was chosen to sit far above every
realistic pattern this module is tested against (a single label's own
throttle already limits it to at most ~21 lines total and roughly 1/sec
sustained; even three concurrently rotating phases top out at 63 lines
total) while still being a real, finite, enforced ceiling: a pathological
caller rotating through hundreds of distinct labels with no real elapsed
time between calls is now bounded to at most 100 lines *per second of
actual wall-clock time*, not unbounded. Tracked as a fixed-size
:class:`~collections.deque` of the timestamps of the last
``_MAX_PROGRESS_LINES_PER_SECOND`` emitted progress lines (a standard
sliding-window-log rate limiter): a new line is allowed once fewer than
that many are on record, or once the oldest recorded timestamp has aged out
of the window. Only a line that is *about to be written* advances this
window; a line already denied by the per-label rule was never a candidate
and does not consume any of the global budget. One accepted consequence,
symmetric with the per-label eviction trade-off above: under sustained,
truly pathological load, a label's own 100%-completion line can itself be
the one denied by the global backstop, since "forced" here only ever meant
"forced past the per-label rule", never "exempt from every rule". In any
realistic session (a handful of labels, throttled well under 100/sec on
their own) this never triggers at all; it exists purely as the net under
the pathological case.

**NaN.** ``fraction = max(0.0, min(1.0, fraction))`` would silently produce
``1.0`` for a NaN input, *not* raise and *not* leave it unclamped: every
comparison against NaN is ``False``, so ``min(1.0, nan)`` returns ``1.0``
and the outer ``max(0.0, 1.0)`` then also returns ``1.0``. Reporting a NaN
progress value as "100% complete" is a worse failure than under-reporting
it, so :meth:`~PlainRenderer.progress` detects NaN explicitly (``fraction
!= fraction`` is the portable NaN test) and treats it as ``0.0`` *before*
that clamp runs, rather than relying on comparison semantics to do
something reasonable by accident.

## Encoding safety

A field value is whatever the caller hands in; nothing here assumes it is
representable in the destination stream's encoding, since a non-ASCII
sequence identifier from a real AIRR file, or a ``LANG=C``/some-CI-runners
stderr pinned to strict ASCII, are both realistic. Writing is funnelled
through :meth:`PlainRenderer._write`, which attempts the normal
:func:`print` and, on the *specific* failure that means "this string cannot
be encoded here" (``UnicodeEncodeError``, caught narrowly so an unrelated
failure such as a broken pipe is never swallowed), re-encodes the line
against the stream's own reported encoding with ``errors="replace"`` and
writes that instead. A rendering call degrading a value's presentation
(``café.tsv`` becoming ``caf?.tsv``) is acceptable; a rendering call taking
down the command it was merely reporting on is not, per the :class:`~
LZGraphs._term.Ui` protocol's own invariant.

## No boxes, no cursor control, colour is cosmetic only

This module never imports ``bar``, ``sparkline``, ``panel``, or ``card``
from :mod:`._widgets`: those are the rich renderer's vocabulary. It imports
only :func:`~LZGraphs._term._widgets.duration`, to format the elapsed time
:meth:`~PlainRenderer.done` appends automatically, since that is the one
value this renderer computes and owns itself (every other field's
formatting, e.g. thousands separators via ``counter`` or a size via
``bytes_human``, is the caller's job when it builds the ``fields``/``rows``
mapping it hands in; this renderer does not guess a field's semantic type
from its key name).

It never imports ``cursor_up``/``clear_line``/``hide_cursor``/
``show_cursor`` from :mod:`._ansi` either: there is no redraw here, so
there is nothing to move the cursor for. The only ``_ansi`` function used
is :func:`~LZGraphs._term._ansi.colour`, applied *only* to the ``status``
word itself (``warn`` in the ``warn`` palette colour, ``error`` in
``error``, ``done`` in ``ok``; ``start``/``progress``/``info`` stay
uncoloured, since they are the high-frequency lines and colouring them
would just be noise) and it is identity at ``caps.colours == 0`` by
construction, so it can never affect the layout and can never leak an
escape sequence into a piped log: "Colour is permitted when depth allows
it... but the layout must not depend on colour" holds because the exact
same tokens appear in the exact same positions whether or not colour is on;
only the bytes inside the ``status=`` value change.

## Streams

Every write goes to whatever ``stream`` this renderer was constructed with
(``sys.stderr`` by default, matching :func:`~LZGraphs._term._caps.detect`'s
own default), via a plain :func:`print` immediately flushed so a CI log
tailing the process sees each line as it happens. Nothing here ever touches
``sys.stdout``.
"""
from __future__ import annotations

import sys
import time
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any, TextIO

from ._ansi import colour
from ._widgets import duration

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ._caps import Capabilities

#: Emit at most once per this many percentage points of progress, per label.
_PROGRESS_PCT_STEP = 5

#: ...or at most once per this many seconds of wall time, per label,
#: whichever threshold is reached first. See the module docstring's
#: "Progress throttling" section for the full rule, including the two
#: unconditional exceptions and the bounded per-label state.
_PROGRESS_TIME_STEP = 1.0

#: Maximum number of distinct progress() labels tracked at once, with
#: least-recently-updated eviction beyond this. Bounds this renderer's
#: memory against an unbounded/high-cardinality label; see the module
#: docstring's "Bounded state" paragraph for the one visible consequence.
_MAX_TRACKED_PROGRESS_LABELS = 32

#: Unconditional global cap: at most this many progress() lines are ever
#: written in any trailing _GLOBAL_RATE_WINDOW_SECONDS window, regardless
#: of label and regardless of which per-label exception would otherwise
#: force a line through. See the module docstring's "Global backstop"
#: paragraph for why the per-label cap alone is not enough on its own.
_MAX_PROGRESS_LINES_PER_SECOND = 100

#: The window _MAX_PROGRESS_LINES_PER_SECOND is measured over. A separate
#: constant from _PROGRESS_TIME_STEP on purpose, even though both happen to
#: be 1.0 today: one paces a single label, the other is a hard safety net
#: over all labels combined, and they should be free to diverge later
#: without silently changing each other.
_GLOBAL_RATE_WINDOW_SECONDS = 1.0

#: Stripped from the front of a title before bracketing it into a tag, so
#: "lzg build" -> "[build]" matches the tag cli.py already emits for that
#: command. A title with no such prefix is bracketed as-is.
_LZG_PREFIX = "lzg "

#: The tag used before any of start()/done()/error() has ever been called,
#: so an out-of-order progress()/update()/warn() call cannot raise.
_DEFAULT_TAG = "[lzg]"

#: status word -> palette name, for the small set of lines worth colouring.
#: start/progress/info are deliberately left out: they are the highest
#: frequency lines and colouring them would be noise, not signal.
_STATUS_COLOUR = {
    "warn": "warn",
    "error": "error",
    "done": "ok",
}


def _tag_for(title: str) -> str:
    """Derive a stable ``[tag]`` prefix from a session ``title``.

    Strips a leading ``"lzg "`` (case-sensitive, matching how every command
    name is spelled in this codebase) so ``"lzg build"`` becomes
    ``"[build]"``. A blank or whitespace-only title falls back to
    ``"lzg"`` rather than producing an empty, confusing ``"[]"``.
    """
    body = title[len(_LZG_PREFIX):] if title.startswith(_LZG_PREFIX) else title
    body = body.strip() or "lzg"
    return f"[{body}]"


def _sanitize(value: Any) -> str:
    """Render ``value`` as a single physical line's worth of text.

    Embedded newlines/carriage returns are replaced with a space so one
    ``key=value`` field can never split what was meant to be one emitted
    line into what a naive line-based reader would see as two or more.
    """
    text = str(value)
    if "\n" in text or "\r" in text:
        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return text


def _format_fields(fields: Mapping[str, Any]) -> str:
    """Render a fields mapping as space-separated ``key=value`` tokens."""
    return " ".join(f"{key}={_sanitize(value)}" for key, value in fields.items())


class PlainRenderer:
    """The scrolling ``[tag] status=word key=value ...`` renderer.

    Satisfies the :class:`~LZGraphs._term.Ui` protocol. See the module
    docstring for the full line-shape and throttling contract.

    ``caps`` is required (this renderer never reads the environment or
    calls ``isatty`` itself; :mod:`._caps` already decided everything it
    needs to know). ``stream`` defaults to ``sys.stderr`` when omitted,
    matching :func:`~LZGraphs._term._caps.detect`'s own default; it is
    never ``sys.stdout``, by construction and by every test in
    ``tests/test_term_plain.py``.
    """

    def __init__(self, caps: Capabilities, stream: TextIO | None = None) -> None:
        self._caps = caps
        self._stream = stream if stream is not None else sys.stderr
        self._tag = _DEFAULT_TAG
        self._t0: float | None = None
        # label -> (pct, monotonic_time) of the last emitted progress line
        # for that label; an OrderedDict so eviction (see progress()) can
        # drop the least-recently-updated label once the cap is exceeded.
        # A label with no entry yet (never seen, or evicted) is treated as
        # its first-ever progress() call.
        self._last_progress: OrderedDict[str, tuple[int, float]] = OrderedDict()
        # Timestamps of the last _MAX_PROGRESS_LINES_PER_SECOND *emitted*
        # progress lines, oldest first, regardless of label: the global
        # backstop's sliding window. A fixed maxlen deque so it never grows
        # past the cap on its own.
        self._global_progress_times: deque[float] = deque(maxlen=_MAX_PROGRESS_LINES_PER_SECOND)

    def _global_rate_allows(self, now: float) -> bool:
        """True if one more progress line may be written right now.

        See the module docstring's "Global backstop" paragraph. A standard
        sliding-window-log check: allowed while fewer than the cap have
        been recorded, or once the oldest of the last-cap timestamps has
        aged out of the window.
        """
        times = self._global_progress_times
        if len(times) < _MAX_PROGRESS_LINES_PER_SECOND:
            return True
        return (now - times[0]) >= _GLOBAL_RATE_WINDOW_SECONDS

    def _write(self, line: str) -> None:
        """Write one line, degrading gracefully if it cannot be encoded.

        A caller-supplied value is not guaranteed representable in the
        destination stream's encoding (a non-ASCII sequence id on a
        ``LANG=C`` stderr, for instance). The normal :func:`print` path is
        tried first; only ``UnicodeEncodeError`` specifically is caught
        (never a broader exception, so an unrelated failure such as a
        broken pipe still propagates), and on that failure the line is
        re-encoded against the stream's own reported encoding with
        ``errors="replace"`` and written again. A rendering call must
        degrade a value's presentation rather than take down the command it
        is merely reporting on.
        """
        try:
            print(line, file=self._stream, flush=True)
        except UnicodeEncodeError:
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe_line, file=self._stream, flush=True)

    def _status_token(self, status: str) -> str:
        name = _STATUS_COLOUR.get(status)
        word = colour(name, status, self._caps) if name else status
        return f"status={word}"

    def _emit(self, status: str, fields: Mapping[str, Any] | None = None) -> None:
        parts = [self._tag, self._status_token(status)]
        if fields:
            parts.append(_format_fields(fields))
        self._write(" ".join(parts))

    def _emit_detail(self, status: str, value: Any) -> None:
        """One indented ``detail=`` continuation line under ``error()``.

        Indentation is real: extra spaces between ``status=word`` and the
        ``detail=`` token, so a human scanning the scrolling log sees the
        detail lines visually nested under the header line they belong to,
        satisfying "use structure and indentation, not just a red tint"
        without needing colour to convey it. Still starts with the exact
        same stable ``[tag] status=word`` prefix as every other line, so a
        script can find every line of this error (header and details alike)
        with one ``grep``.
        """
        self._write(f"{self._tag} {self._status_token(status)}     detail={_sanitize(value)}")

    def start(self, title: str, fields: Mapping[str, Any] | None = None) -> None:
        self._tag = _tag_for(title)
        self._t0 = time.monotonic()
        self._last_progress = OrderedDict()
        self._global_progress_times = deque(maxlen=_MAX_PROGRESS_LINES_PER_SECOND)
        self._emit("start", fields)

    def progress(self, label: str, fraction: float, detail: str | None = None) -> None:
        if fraction != fraction:  # NaN: see the module docstring's "NaN"
            # paragraph. Every comparison against NaN is False, so the
            # clamp below would otherwise silently pick 1.0 (100%,
            # falsely claiming completion) rather than clamping at all;
            # intercepted here before that can happen.
            fraction = 0.0
        fraction = max(0.0, min(1.0, fraction))
        pct = round(fraction * 100)
        now = time.monotonic()

        last = self._last_progress.get(label)
        if last is not None:
            last_pct, last_time = last
            pct_delta = pct - last_pct
            time_delta = now - last_time
            reached_completion = pct >= 100 and last_pct < 100
            if (
                pct_delta < _PROGRESS_PCT_STEP
                and time_delta < _PROGRESS_TIME_STEP
                and not reached_completion
            ):
                return  # per-label throttle: nothing new to report yet

        if not self._global_rate_allows(now):
            # The global backstop denies this line even though the
            # per-label rule (including any of its "forced" exceptions)
            # just approved it. Deliberately does NOT update
            # _last_progress for this label: a denied line was never
            # actually reported, so the next real attempt for this label
            # should still be judged against the last line that truly went
            # out, not against this suppressed one.
            return

        self._global_progress_times.append(now)
        self._last_progress[label] = (pct, now)
        self._last_progress.move_to_end(label)
        if len(self._last_progress) > _MAX_TRACKED_PROGRESS_LABELS:
            self._last_progress.popitem(last=False)  # evict least-recently-updated

        fields: dict[str, Any] = {"label": label, "pct": pct}
        if detail is not None:
            fields["detail"] = detail
        self._emit("progress", fields)

    def update(self, **fields: Any) -> None:
        self._emit("info", fields or None)

    def warn(self, message: str) -> None:
        self._emit("warn", {"message": message})

    def error(self, title: str, lines: Sequence[str] | None = None) -> None:
        """Report a fatal error under whatever tag ``title`` establishes.

        ``title`` plays exactly the role it plays in :meth:`start` and
        :meth:`done`: it identifies *which command* this report belongs to
        (``"lzg build"``), and is (re)used to derive the ``[tag]`` prefix,
        not emitted as a field itself (that would just repeat the tag).
        The error's actual content, headline included, is ``lines``: the
        header line this emits announces that an error occurred for this
        tag; each entry in ``lines`` becomes its own indented ``detail=``
        continuation line, in order, so the headline a caller puts first
        (e.g. ``"column 'junction_aa' not found"``) is simply the first
        detail line in the scrolling log, exactly where a human reading
        top-to-bottom expects it.
        """
        if title:
            self._tag = _tag_for(title)
        self._emit("error")
        for line in lines or ():
            self._emit_detail("error", line)

    def done(self, title: str, rows: Mapping[str, Any] | None = None) -> None:
        self._tag = _tag_for(title) if title else self._tag
        merged: dict[str, Any] = dict(rows) if rows else {}
        if "elapsed" not in merged and self._t0 is not None:
            merged["elapsed"] = duration(time.monotonic() - self._t0)
        self._emit("done", merged)


__all__ = ["PlainRenderer"]
