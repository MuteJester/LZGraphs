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
one match and the next). Its one documented limitation: a value must not
itself contain a bare ``word=`` substring, or a naive boundary parser will
read that as the start of a new field. Nothing this renderer emits does.

## Progress throttling

A build that calls :meth:`~PlainRenderer.progress` 100,000 times (once per
record) must not emit 100,000 lines; a CI log that scrolls for a thousand
lines is as useless as no log. The rule, per ``(label,)`` stream: emit at
most once every **5 percentage points** (``_PROGRESS_PCT_STEP``) **or every
1.0 second of wall time** (``_PROGRESS_TIME_STEP``), whichever comes first,
with three unconditional exceptions that always force a line regardless of
both thresholds:

1. The very first call (nothing has been reported yet for this renderer).
2. ``label`` differs from the previous call's label (a new phase has
   started; its first data point is never silently swallowed by the old
   phase's throttle state).
3. ``fraction`` reaches (or exceeds) ``1.0`` for the first time (the
   completion of a step is never dropped, even if the last emitted line was
   a fraction of a second and a fraction of a percent ago).

This bounds a monotonically-increasing 0.0 -> 1.0 sweep to at most
``100 // 5 + 1 == 21`` lines regardless of how many times ``progress`` is
called, plus at most one extra line per second of wall-clock time the sweep
actually takes.

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
#: "Progress throttling" section for the full rule, including the three
#: unconditional exceptions.
_PROGRESS_TIME_STEP = 1.0

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
        # (label, pct, monotonic_time) of the last emitted progress line,
        # or None before the first progress() call.
        self._last_progress: tuple[str, int, float] | None = None

    def _status_token(self, status: str) -> str:
        name = _STATUS_COLOUR.get(status)
        word = colour(name, status, self._caps) if name else status
        return f"status={word}"

    def _emit(self, status: str, fields: Mapping[str, Any] | None = None) -> None:
        parts = [self._tag, self._status_token(status)]
        if fields:
            parts.append(_format_fields(fields))
        print(" ".join(parts), file=self._stream, flush=True)

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
        line = f"{self._tag} {self._status_token(status)}     detail={_sanitize(value)}"
        print(line, file=self._stream, flush=True)

    def start(self, title: str, fields: Mapping[str, Any] | None = None) -> None:
        self._tag = _tag_for(title)
        self._t0 = time.monotonic()
        self._last_progress = None
        self._emit("start", fields)

    def progress(self, label: str, fraction: float, detail: str | None = None) -> None:
        fraction = max(0.0, min(1.0, fraction))
        pct = round(fraction * 100)
        now = time.monotonic()

        last = self._last_progress
        if last is not None and label == last[0]:
            pct_delta = pct - last[1]
            time_delta = now - last[2]
            reached_completion = pct >= 100 and last[1] < 100
            if (
                pct_delta < _PROGRESS_PCT_STEP
                and time_delta < _PROGRESS_TIME_STEP
                and not reached_completion
            ):
                return

        self._last_progress = (label, pct, now)
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
