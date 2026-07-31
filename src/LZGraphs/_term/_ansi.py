"""Escape sequences and named colour palette for the ``_term`` layer.

This module hands the rest of ``_term`` (widgets in the next task, then the
plain and rich renderers) a small set of pure string builders. Nothing here
writes to a stream; every function returns a string and the caller decides
where it goes. ``_caps`` remains the only place that reads ``os.environ`` or
calls ``isatty``; this module only ever asks a :class:`~LZGraphs._term._caps.
Capabilities` instance what it is allowed to do.

One uniform rule governs every function here: **a function that accepts
``caps`` gates on it and can never leak an escape sequence at
``caps.colours == 0``; a function that does not accept ``caps`` is a raw
primitive and always emits.** Concretely:

* ``colour``, ``fg``, ``bold``, ``dim``, ``reset`` all take ``caps`` and all
  gate on ``caps.colours``. At ``caps.colours == 0``: ``colour``/``bold``/
  ``dim`` return their text argument unchanged, and ``fg``/``reset`` return
  the empty string. A caller never has to write ``if caps.colours: ...``
  before calling any of the five, and no composition of them can leak an
  escape code when colour is disabled.
* ``sgr``, ``cursor_up``, ``clear_line``, ``hide_cursor``, ``show_cursor``
  take no ``caps`` argument and always emit their raw sequence. This is safe
  for two different reasons, not one, so read carefully:

  - ``sgr`` is the assembly-level primitive every gated function above is
    built from; it is only ever called internally, after the caller has
    already decided (via ``caps``) that emitting is correct. It is not part
    of the public colour API; prefer ``colour``/``bold``/``dim``/``fg``.
  - The cursor-movement quartet answers a different question entirely
    (can this stream interpret cursor motion at all?), governed by
    ``caps.supports_cursor_control``, not ``caps.colours``. They are only
    ever invoked by the rich renderer (Task 5), which by construction only
    runs when ``caps.supports_cursor_control`` is true, so the gate already
    happened once, upstream, at mode-resolution time; individually gating
    each call would duplicate a decision that is not theirs to make and
    would conflate two independent capability axes (colour depth vs. cursor
    support) if it mistakenly gated on ``caps.colours`` instead.

**Prefer :func:`colour` over hand-composing ``fg`` and ``reset``.** An
earlier revision of this module gave ``reset()`` no ``caps`` parameter,
which meant the natural-looking ``fg(name, caps) + text + reset()``
leaked a real ``\\x1b[0m`` at ``caps.colours == 0`` even though the coloured
prefix was empty, exactly the kind of escape leak the plan's "stdout carries
data" constraint forbids. ``reset`` now takes ``caps`` and gates like
everything else in its family, so that specific composition is safe again,
but :func:`colour` remains the recommended API: one call, nothing to
remember, impossible to leak by construction.

## The named palette

Colour must read on a terminal whose background theme is unknown and cannot
be queried (light or dark, unpredictably). That rules out pure black, pure
white, and bright/saturated yellow on a light background, per the task's own
constraint. Five semantic names are defined, each mapped to a 256-colour
value and an 8-colour (classic ANSI) value:

=========  ===========================  ===================================
name       256 value (approx. hex)      8-colour value
=========  ===========================  ===================================
ok         71  (``#5faf5f``)             32 (green)
warn       136 (``#af8700``)             33 (yellow)
error      167 (``#d75f5f``)             31 (red)
accent     37  (``#00afaf``)             36 (cyan)
muted      244 (``#808080``)             2  (SGR faint, see below)
=========  ===========================  ===================================

Reasoning for each, at 256-colour depth:

* ``ok`` (71, ``#5faf5f``): a desaturated mid green rather than the near-black
  standard "green" (ANSI 2/32 rendered as ``#00cd00`` or darker in many
  themes) or a lime/neon green. Dark enough to stay off a white background,
  light enough to stay off black.
* ``warn`` (136, ``#af8700``): a dark gold/amber, deliberately **not** yellow.
  Plain yellow is the one hue the task explicitly rules out on a light
  background (it disappears against white and vibrates against black at full
  saturation); amber keeps the "caution" association while staying legible
  on both.
* ``error`` (167, ``#d75f5f``): a muted brick red rather than saturated
  ``#ff0000``, which reads as harsh on light backgrounds and often clips to
  an even harsher red in dark-theme palettes that remap pure red.
* ``accent`` (37, ``#00afaf``): a mid teal/cyan, cool enough not to compete
  with ``ok``/``warn``/``error`` and legible at moderate brightness on either
  background; matches the informational cyan already used for format/engine
  labels in the approved mockup.
* ``muted`` (244, ``#808080``): true middle grey (equal RGB channels), the
  standard "secondary text" tone with roughly symmetric contrast against
  both black and white.

At 8-colour depth the terminal defines the exact RGB behind each of the
classic codes, so these are the closest available primitives rather than
exact colour matches:

* ``ok``/``warn``/``error``/``accent`` use the obvious classic codes (green
  32, yellow 33, red 31, cyan 36). Documented limitation: classic ANSI
  "yellow" renders close to a bright yellow in some terminal colour schemes,
  which is exactly the tone this module avoids at 256-colour depth; there is
  no ANSI-8 "amber". This is the best available primitive at that depth, not
  a claim that it fully solves the light-background problem: the 256-colour
  path (used by the large majority of terminals in practice, including every
  modern terminal emulator) carries the actual amber value.
* ``muted`` uses SGR *faint* (code 2) instead of a colour code. The base
  16-colour ANSI set has no dedicated grey among codes 30-37 (the "bright
  black" some terminals expose as 90 is an aixterm extension outside the
  strict 8-colour set this depth represents), and hard-coding, say, white
  (37) would be invisible on a light background. Faint dims *relative to*
  whatever foreground/background the terminal's own theme defines, so it
  degrades sensibly on both instead of picking an absolute shade that could
  clash with either.

## ``visible_len``

Strips CSI escape sequences (``\\x1b[`` followed by parameter bytes
``[0-9;?]`` and a single final letter, e.g. ``\\x1b[38;5;71m``,
``\\x1b[2K``, ``\\x1b[?25l``) via one non-overlapping regex substitution, so
adjacent or nested sequences with nothing but more escapes between them are
all removed in one pass rather than needing hand-rolled state. Every
remaining character then counts as exactly one column, including the
box-drawing and block characters this layer draws with (``█ ░ ▁▂▃▄▅▆▇ ╭╮╰╯│─
├┤``): those are single Unicode code points outside any escape sequence, so
the "count what is left" rule already gives them width 1 with no special
case needed.

Documented limits, deliberately not handled:

* **East Asian wide characters and emoji** render as two terminal columns in
  most terminals but count as one code point here, so a padded row containing
  one would fall one column short of the target width. This layer only ever
  renders ASCII labels, numbers, and the fixed box-drawing/block set above,
  so the mismatch does not arise in practice; it is pinned by a test so a
  future change that starts feeding wide characters through this function
  fails loudly (a wrong-but-stable count) rather than being silently assumed
  correct.
* **Combining marks** (e.g. a decomposed ``e`` + combining acute accent) count
  as one column *per code point*, i.e. two columns for that pair, rather than
  being collapsed onto their base character. Same rationale: not part of this
  layer's character set, not silently "handled", pinned by a test.
* A malformed or truncated escape sequence (an ``ESC`` with no matching final
  byte) is not recognised by the regex and its bytes are counted as ordinary
  characters. This layer never emits such a thing, so it is out of scope
  rather than silently patched over.

None of these raise; they produce a deterministic, documented count.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._caps import Capabilities

#: The semantic colour names every caller may use with :func:`fg`.
PALETTE_NAMES: tuple[str, ...] = ("ok", "warn", "error", "accent", "muted")

# 256-colour xterm palette indices. See the module docstring for the chosen
# values and the light/dark legibility reasoning behind each.
_PALETTE_256 = {
    "ok": 71,
    "warn": 136,
    "error": 167,
    "accent": 37,
    "muted": 244,
}

# Classic 8-colour (ANSI) SGR codes. "muted" uses 2 (SGR faint) rather than a
# base colour code; see the module docstring for why.
_PALETTE_8 = {
    "ok": 32,
    "warn": 33,
    "error": 31,
    "accent": 36,
    "muted": 2,
}

# Matches one CSI (Control Sequence Introducer) escape: ESC "[" then zero or
# more parameter bytes (digits, ";", "?") then a single final letter. Covers
# every sequence this module emits (SGR colour/style, cursor movement, clear
# line, cursor show/hide).
_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def sgr(code: int | str) -> str:
    """Raw SGR escape for ``code`` (e.g. ``sgr(1)`` for bold, ``sgr("38;5;71")``).

    Takes no ``caps``: it is the low-level primitive every colour/style
    function in this module is built from, always emitting its sequence
    regardless of terminal capability. Callers wanting graceful degradation
    use :func:`fg`, :func:`bold`, or :func:`dim` instead.
    """
    return f"\x1b[{code}m"


def fg(name: str, caps: Capabilities) -> str:
    """Foreground colour prefix for the named palette entry.

    Returns the empty string at ``caps.colours == 0`` (there is nothing to
    set), a classic ANSI colour code at ``caps.colours == 8``, and a
    ``38;5;N`` 256-colour code at ``caps.colours == 256``. This is a bare
    prefix, not a wrap: normal callers want :func:`colour`, which wraps text
    and resets in one call with no way to leak an escape at depth 0. Use
    ``fg`` directly only when a caller genuinely needs the raw prefix (for
    example to splice into a larger hand-built SGR sequence); if you pair it
    with a reset yourself, pass ``caps`` to :func:`reset` too, or the
    composition can leak exactly as described in the module docstring.
    """
    if name not in _PALETTE_256:
        raise ValueError(f"unknown palette name {name!r}; choose one of {PALETTE_NAMES!r}")
    if caps.colours == 256:
        return sgr(f"38;5;{_PALETTE_256[name]}")
    if caps.colours == 8:
        return sgr(_PALETTE_8[name])
    return ""


def colour(name: str, text: str, caps: Capabilities) -> str:
    """Wrap ``text`` in the named palette colour, resetting afterwards.

    This is the API callers should reach for to colour a span of text: one
    call, no composition invariant to remember, and it cannot leak an
    escape sequence at ``caps.colours == 0`` because it returns ``text``
    unchanged there. Equivalent to ``fg(name, caps) + text + reset(caps)``
    but safe by construction rather than by caller discipline.
    """
    prefix = fg(name, caps)
    if not prefix:
        return text
    return f"{prefix}{text}{reset(caps)}"


def bold(s: str, caps: Capabilities) -> str:
    """Wrap ``s`` in bold, or return it unchanged at ``caps.colours == 0``."""
    if caps.colours == 0:
        return s
    return f"{sgr(1)}{s}{reset(caps)}"


def dim(s: str, caps: Capabilities) -> str:
    """Wrap ``s`` in faint/dim, or return it unchanged at ``caps.colours == 0``."""
    if caps.colours == 0:
        return s
    return f"{sgr(2)}{s}{reset(caps)}"


def reset(caps: Capabilities) -> str:
    """Raw SGR reset sequence, or the empty string at ``caps.colours == 0``.

    Takes ``caps`` specifically so that hand-composed sequences such as
    ``fg(name, caps) + text + reset(caps)`` cannot leak an escape code when
    colour is disabled; see the module docstring. Prefer :func:`colour`,
    which does this for you.
    """
    if caps.colours == 0:
        return ""
    return sgr(0)


def cursor_up(n: int) -> str:
    """Move the cursor up ``n`` lines, or the empty string when ``n <= 0``.

    ANSI substitutes a missing/zero parameter with 1, so a bare ``\\x1b[0A``
    would still move the cursor up one line. That would be a real bug for
    the rich renderer's redraw loop, which computes ``n`` from a tracked
    line count and must be able to pass 0 (nothing drawn yet) and mean it.
    """
    if n <= 0:
        return ""
    return f"\x1b[{n}A"


def clear_line() -> str:
    """Clear from the cursor to the end of the current line."""
    return "\x1b[K"


def hide_cursor() -> str:
    """Hide the terminal cursor."""
    return "\x1b[?25l"


def show_cursor() -> str:
    """Show the terminal cursor."""
    return "\x1b[?25h"


def visible_len(s: str) -> int:
    """Display width of ``s`` ignoring CSI escape sequences.

    See the module docstring's "``visible_len``" section for exactly what
    counts as one column and the documented limits (East Asian wide
    characters, emoji, and combining marks are not measured correctly; a
    malformed escape sequence is not recognised as one).
    """
    return len(_CSI_RE.sub("", s))
