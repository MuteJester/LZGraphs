"""Pure string builders for the ``_term`` layer: bars, sparklines, panels.

Every function here is a pure function from data (plus a
:class:`~LZGraphs._term._caps.Capabilities`) to a string or a list of
strings. Nothing in this module writes to a stream, reads the environment,
or keeps state between calls; the renderers in later tasks own all I/O and
simply print what these functions return. That split is what lets the whole
layer be tested without a terminal: every property below is checked by
measuring the returned strings, never by capturing real output.

## The alignment invariant

:func:`panel` and :func:`card` promise that **every line they return has
``visible_len`` exactly equal to ``caps.width``**, whether or not that line
contains colour escapes, whether the box is drawn or dropped, and whether a
row had to be truncated to fit. That promise is built out of two small
helpers, :func:`_fit` (pad-or-truncate a possibly-coloured string to an
exact width) and :func:`_truncate_visible` (walk a string copying escape
sequences untouched while counting only visible characters), so every call
site gets the invariant for free instead of re-deriving it.

## Degradation

Two independent axes gate what a widget may emit:

* ``caps.colours == 0``: achieved automatically by using ``colour``/``bold``/
  ``dim``/``reset`` from :mod:`._ansi` for every escape this module emits.
  This module never calls ``sgr`` or ``fg`` directly, so there is nothing to
  get wrong here; the depth-0 guarantee lives entirely in ``_ansi``.
* ``caps.unicode == False``: this module's own responsibility. Every
  non-ASCII character it might emit (box-drawing, block, sparkline, and the
  middle-dot separator in :func:`card`) has an ASCII fallback selected by
  ``caps.unicode``, so that at ``unicode=False`` this module never emits a
  code point outside ASCII. The one exception is the truncation marker,
  which is the ASCII ``"..."`` unconditionally (not just under
  ``unicode=False``): it needs no fallback because it was never anything
  else, which also sidesteps the fact that ``caps.unicode`` only promises
  the stream can encode the specific box/block probe in ``_caps``, not an
  ellipsis character.

## Narrow terminals

Below 50 columns, :func:`panel` (and, through it, :func:`card`) drops the
box entirely and renders plain ``key: value``-shaped lines with no border
characters, per the task's "reflow rather than wrap raggedly" requirement.
Those lines are still padded or truncated to exactly ``caps.width``: the
alignment invariant above applies unconditionally, not just to the boxed
layout, and a trailing run of spaces on a plain line is invisible, so
dropping the box costs nothing in the one property that has to hold.

## Row width vs. panel width

``bar``, ``sparkline``, ``counter``, ``duration``, and ``bytes_human`` are
plain values, not panel rows, so the alignment invariant does not apply to
them the same way:

* ``bar(fraction, width, caps)`` takes its width explicitly and *does*
  promise ``visible_len(bar(...)) == width`` for that same reason: it was
  given a width to fill.
* ``sparkline`` renders exactly ``len(values)`` characters (one per data
  point); it has no width parameter to honour.
* ``kv`` builds one row's *content* (a label, a value, an optional note) for
  a caller to hand to ``panel``; it has no fixed target width of its own,
  because ``panel`` is the one that knows the available width and performs
  the pad-or-truncate.
* ``counter``, ``duration``, and ``bytes_human`` are plain numeric
  formatters with no ``caps`` parameter at all: they emit no colour and no
  non-ASCII text, so neither degradation axis applies to them.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

from ._ansi import bold, colour, reset, visible_len

if TYPE_CHECKING:
    from ._caps import Capabilities

# Below this many columns, panel/card drop the box and render plain lines.
_NARROW_WIDTH = 50

# Fixed label width for `kv`'s key column, matching the approved mockup.
_KV_KEY_WIDTH = 9

# One CSI escape sequence: ESC "[" then parameter bytes then a final letter.
# Duplicated from `_ansi` deliberately rather than imported: this module only
# needs it to walk a string while truncating (`_truncate_visible`), and
# keeping its own copy avoids depending on a leading-underscore name from a
# sibling module's implementation.
_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# The truncation marker is plain ASCII unconditionally; see the module
# docstring's "Degradation" section for why it has no unicode variant.
_ELLIPSIS = "..."


class _BoxChars(NamedTuple):
    tl: str  # top-left corner
    tr: str  # top-right corner
    bl: str  # bottom-left corner
    br: str  # bottom-right corner
    h: str  # horizontal
    v: str  # vertical
    lt: str  # left tee (divider before a footer)
    rt: str  # right tee (divider before a footer)


_BOX_UNICODE = _BoxChars("╭", "╮", "╰", "╯", "─", "│", "├", "┤")
_BOX_ASCII = _BoxChars("+", "+", "+", "+", "-", "|", "+", "+")

_BAR_FULL_UNICODE, _BAR_EMPTY_UNICODE = "█", "░"
_BAR_FULL_ASCII, _BAR_EMPTY_ASCII = "#", "."

_SPARK_UNICODE = "▁▂▃▄▅▆▇█"
_SPARK_ASCII = "0123456789"

_DOT_UNICODE = "·"
_DOT_ASCII = "-"

_BYTE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")

_STATUSES = ("ok", "warn", "error")
_STATUS_WORDS = {"ok": "done", "warn": "warning", "error": "error"}


def _box(caps: Capabilities) -> _BoxChars:
    return _BOX_UNICODE if caps.unicode else _BOX_ASCII


def _dot(caps: Capabilities) -> str:
    return _DOT_UNICODE if caps.unicode else _DOT_ASCII


def _truncate_visible(s: str, keep: int) -> str:
    """Copy ``s`` up to ``keep`` visible characters, passing escapes through.

    Walks ``s`` left to right. Any CSI escape sequence encountered is copied
    in full and does not count against ``keep`` (matching how
    ``visible_len`` measures ``s``); every other character counts as one and
    stops the walk once ``keep`` have been copied. A partially-consumed
    colour span is expected here: :func:`_truncate` appends its own
    ``reset`` immediately after calling this, so a colour started before the
    cut point can never bleed into whatever follows.
    """
    if keep <= 0:
        return ""
    out: list[str] = []
    visible = 0
    i = 0
    n = len(s)
    while i < n and visible < keep:
        m = _CSI_RE.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(s[i])
        visible += 1
        i += 1
    return "".join(out)


def _truncate(text: str, width: int, caps: Capabilities) -> str:
    """Truncate ``text`` to at most ``width`` visible columns.

    Returns ``text`` unchanged if it already fits. Otherwise cuts it short
    and appends an ellipsis, closing any colour span opened before the cut
    with an explicit ``reset`` so no colour leaks into whatever the caller
    appends next (padding, a border character, and so on). The result
    always measures exactly ``width`` (or ``0`` when ``width <= 0``).
    """
    if visible_len(text) <= width:
        return text
    if width <= 0:
        return ""
    ellipsis = _ELLIPSIS if width >= len(_ELLIPSIS) else "." * width
    keep = max(0, width - len(ellipsis))
    result = _truncate_visible(text, keep) + reset(caps) + ellipsis
    result_len = visible_len(result)
    if result_len < width:
        result += " " * (width - result_len)
    elif result_len > width:
        result = _truncate_visible(result, width)
    return result


def truncate_path(path: str, width: int, caps: Capabilities) -> str:
    """Truncate a filesystem path to at most ``width`` visible columns,
    keeping the *start* of the path and its filename (the basename) rather
    than :func:`_truncate`'s generic head-keeping.

    A long path's basename is normally the single most useful part of it
    to keep on screen (it is what actually distinguishes one input/output
    file from another with the same directory prefix); plain head-keeping
    truncation drops exactly that, at exactly the point it becomes too
    long to fit. This cuts out of the *middle* instead: as much of the
    head as still fits, an ellipsis, then the whole basename.

    Falls back to :func:`_truncate` unchanged when ``path`` has no
    recognisable separator (nothing to call a "basename") or is already
    short enough that no truncation is needed at all; also falls back to
    truncating the bare basename when even *it* plus an ellipsis would not
    fit in ``width`` (an extremely narrow terminal), since the basename is
    still the more useful half to keep visible in that squeeze.
    """
    if visible_len(path) <= width:
        return path
    if width <= 0:
        return ""
    if "/" in path:
        sep = "/"
    elif "\\" in path:
        sep = "\\"
    else:
        return _truncate(path, width, caps)
    base = path.rsplit(sep, 1)[-1]
    ellipsis = _ELLIPSIS if width >= len(_ELLIPSIS) else "." * width
    reserve = len(base) + len(ellipsis)
    if reserve >= width:
        return _truncate(base, width, caps)
    head = path[: width - reserve]
    return head + ellipsis + base


def _pad(text: str, width: int) -> str:
    vlen = visible_len(text)
    if vlen >= width:
        return text
    return text + " " * (width - vlen)


def _fit(text: str, width: int, caps: Capabilities) -> str:
    """Pad or truncate ``text`` (which may contain colour escapes) to
    exactly ``width`` visible columns."""
    if width <= 0:
        return ""
    return _pad(_truncate(text, width, caps), width)


def bar(fraction: float, width: int, caps: Capabilities) -> str:
    """A filled/empty progress bar exactly ``width`` visible columns wide.

    ``fraction`` is clamped to ``[0.0, 1.0]`` rather than raising, so a
    caller passing a slightly-out-of-range value (a rate estimate that
    overshoots 100%, a negative delta) degrades to a full or empty bar
    instead of crashing or misaligning. Filled cells are coloured ``ok``;
    empty cells are coloured ``muted``, so at ``caps.colours == 0`` the two
    are visually distinguished only by character, matching the ascii
    fallback exactly.
    """
    fraction = max(0.0, min(1.0, fraction))
    width = max(0, width)
    filled = max(0, min(width, int(fraction * width)))
    empty = width - filled
    full_ch, empty_ch = (
        (_BAR_FULL_UNICODE, _BAR_EMPTY_UNICODE) if caps.unicode else (_BAR_FULL_ASCII, _BAR_EMPTY_ASCII)
    )
    return colour("ok", full_ch * filled, caps) + colour("muted", empty_ch * empty, caps)


def sparkline(values: list[float], caps: Capabilities) -> str:
    """A one-line trend indicator, one character per value in ``values``.

    Returns ``""`` for an empty sequence. When every value is equal
    (including all-zero), the scale reference ``hi`` would be zero; ``or 1``
    avoids the division and every value maps to the bottom (all-zero) or top
    (all equal and non-zero) of the ramp, a flat line rather than a crash.
    Under ``caps.unicode == False`` the 8-level block ramp becomes a 10-level
    ASCII digit ramp, still one character per value.
    """
    if not values:
        return ""
    ramp = _SPARK_UNICODE if caps.unicode else _SPARK_ASCII
    top = len(ramp) - 1
    hi = max(values) or 1
    chars = []
    for v in values:
        idx = int(v / hi * top)
        idx = max(0, min(top, idx))
        chars.append(ramp[idx])
    return colour("accent", "".join(chars), caps)


def kv(key: str, value: str, caps: Capabilities, note: str | None = None) -> str:
    """One ``key   value`` row for use inside a :func:`panel`.

    ``key`` is left-justified to a fixed label column (``_KV_KEY_WIDTH``) so
    consecutive rows line up under each other; an empty ``key`` produces a
    blank label column rather than a special case, since ``"".ljust(n)`` is
    already ``n`` spaces. ``note``, when given, is appended dimmed. This
    returns a row of whatever width it naturally comes out to; :func:`panel`
    is what pads or truncates it to the panel's own width.
    """
    label = colour("muted", key.ljust(_KV_KEY_WIDTH), caps)
    text = f" {label} {value}"
    if note:
        text += f"  {colour('muted', note, caps)}"
    return text


def counter(n: int) -> str:
    """Format an integer with thousands separators, e.g. ``1,234,567``."""
    return f"{n:,}"


def duration(seconds: float) -> str:
    """Format a duration in seconds as a short human-readable string.

    Sub-minute durations (including ``0``) render as seconds with one
    decimal place (``"12.4s"``, ``"0.0s"``); minute-scale durations render
    as ``M:SS``; hour-scale durations render as ``H:MM:SS``. Negative input
    is clamped to ``0`` rather than producing a negative or malformed
    string.
    """
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def bytes_human(n: int) -> str:
    """Format a byte count as a short human-readable string, e.g. ``"1.2 GB"``.

    Uses decimal (1000-based) unit steps. Negative input is clamped to
    ``0``, which renders as ``"0 B"``, an integer count with no decimal
    point; every larger unit renders with one decimal place.
    """
    value = float(max(0, n))
    idx = 0
    while value >= 1000 and idx < len(_BYTE_UNITS) - 1:
        value /= 1000
        idx += 1
    unit = _BYTE_UNITS[idx]
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _panel_boxed(title: str, rows: list[str], caps: Capabilities, footer: str | None) -> list[str]:
    width = caps.width
    inner = width - 2
    b = _box(caps)

    max_title_vis = max(0, inner - 3)
    title_text = title if visible_len(title) <= max_title_vis else _truncate(title, max_title_vis, caps)
    title_vis = visible_len(title_text)
    dashes = max(0, inner - 3 - title_vis)

    lines = [
        colour("muted", b.tl + b.h + " ", caps)
        + bold(title_text, caps)
        + colour("muted", " " + b.h * dashes + b.tr, caps)
    ]

    for row in rows:
        content = _fit(row, inner, caps)
        lines.append(colour("muted", b.v, caps) + content + colour("muted", b.v, caps))

    if footer is not None:
        lines.append(colour("muted", b.lt + b.h * inner + b.rt, caps))
        content = _fit(footer, inner, caps)
        lines.append(colour("muted", b.v, caps) + content + colour("muted", b.v, caps))

    lines.append(colour("muted", b.bl + b.h * inner + b.br, caps))
    return lines


def _panel_narrow(title: str, rows: list[str], caps: Capabilities, footer: str | None) -> list[str]:
    width = caps.width
    lines = [_fit(bold(title, caps), width, caps)]
    for row in rows:
        lines.append(_fit(row, width, caps))
    if footer is not None:
        lines.append(_fit(footer, width, caps))
    return lines


def panel(title: str, rows: list[str], caps: Capabilities, footer: str | None = None) -> list[str]:
    """Render a titled box of ``rows`` (plus an optional ``footer``) that is
    exactly ``caps.width`` visible columns wide on every line.

    At ``caps.width >= 50`` this draws a bordered box: a title line with the
    title embedded in the top border, one bordered line per row, a divider
    and one more bordered line for ``footer`` if given, and a closing
    border. Below that width the box is dropped entirely (see the module
    docstring's "Narrow terminals" section) and each line becomes ``title``,
    then each row, then ``footer``, all still padded or truncated to exactly
    ``caps.width``.

    A row (or the title, or the footer) longer than the available width is
    truncated with an ascii ellipsis rather than wrapped, and the resulting
    line still measures exactly ``caps.width``. An empty ``rows`` list is
    valid: the panel is then just a title line and a closing border (boxed
    mode) or just the title line (narrow mode).
    """
    if caps.width < _NARROW_WIDTH:
        return _panel_narrow(title, rows, caps, footer)
    return _panel_boxed(title, rows, caps, footer)


def card(title: str, rows: list[str], caps: Capabilities, status: str) -> list[str]:
    """A :func:`panel` whose title carries a coloured status word.

    ``status`` must be one of ``"ok"``, ``"warn"``, or ``"error"``; each maps
    directly to the identically-named palette entry, so the status word
    (``"done"``, ``"warning"``, ``"error"``) is coloured with
    ``colour(status, ...)``. The rest of the box (borders, rows) renders
    exactly as :func:`panel` would, so the alignment invariant holds for
    ``card`` for the same reason it holds for ``panel``: this *is* ``panel``,
    called with a composed title.
    """
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES!r}, got {status!r}")
    word = colour(status, _STATUS_WORDS[status], caps)
    full_title = f"{title} {_dot(caps)} {word}" if title else word
    return panel(full_title, rows, caps)
