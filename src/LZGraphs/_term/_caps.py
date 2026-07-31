"""Terminal capability detection.

This is the single place in the ``_term`` layer that reads ``os.environ`` or
calls ``isatty``/``shutil.get_terminal_size``. Every other module in the
layer (widgets, the plain renderer, the rich renderer) asks a
``Capabilities`` instance instead of touching the environment itself, so the
rules below live in exactly one place.

Precedence, in the order applied here:

1. An explicit ``requested`` mode wins (``resolve_mode``), except that
   ``"rich"`` downgrades to ``"plain"`` silently on a non-TTY stream *or*
   under ``TERM=dumb``: cursor redraw sequences written into a pipe, a file,
   or a terminal that has declared it cannot process cursor motion all
   produce garbage, and there is no reasonable way to honour "rich" there.
2. ``NO_COLOR`` presence (any value, including ``""``) forces ``colours =
   0``. This is the published convention at https://no-color.org: presence
   is what matters, not truthiness, so ``NO_COLOR=""`` must still disable
   colour even though ``bool("")`` is ``False``.
3. ``TERM=dumb`` forces ``colours = 0`` and, in the *auto* mode (no explicit
   request), forces the resolved mode to ``"plain"``.
4. Presence of ``CI``, ``GITHUB_ACTIONS`` or ``GITLAB_CI`` forces the *auto*
   mode to ``"plain"``.
5. ``FORCE_COLOR`` presence forces a non-zero colour depth even off a TTY,
   but does not by itself force ``"rich"`` mode.
6. Otherwise: a TTY auto-resolves to ``"rich"``, a non-TTY to ``"plain"``.

Rules 3 and 4 are folded into two ``Capabilities`` fields at detection time
(``supports_cursor_control`` and ``interactive``) rather than re-read from
the environment in ``resolve_mode``, which only ever sees ``caps`` and
``requested`` (keeping it a pure, easily-tested function).

**The principle governing rules 3 and 4, for whoever extends this table
next: a capability limit overrides explicit intent; a policy default does
not.**

- ``TERM=dumb`` is a *capability* statement: the terminal is declaring it
  cannot process cursor movement, so an explicit ``requested="rich"`` under
  ``TERM=dumb`` downgrades to ``"plain"`` exactly as it does on a non-TTY
  stream (rule 1's exception is "non-TTY *or* ``TERM=dumb``", both captured
  in ``Capabilities.supports_cursor_control``). Emitting cursor-up sequences
  into a terminal that says it cannot handle them produces garbage no matter
  what the user asked for.
- ``CI``/``GITHUB_ACTIONS``/``GITLAB_CI`` is a *policy* default: plenty of CI
  systems allocate a real, cursor-capable TTY, and a user who explicitly
  passes ``--ui rich`` there (a demo recording, an interactive debugging
  session on a CI runner) may genuinely want it. Explicit intent wins, so an
  explicit ``rich`` request is honoured even under ``CI=1`` as long as the
  stream is a real, non-dumb TTY. CI only steers the *automatic* choice (no
  ``--ui`` given), via ``Capabilities.interactive``.

Colour depth (``Capabilities.colours``): ``COLORTERM`` containing
``"truecolor"`` or ``"24bit"`` gives 256; else ``TERM`` containing
``"256color"`` gives 256; else a TTY (or ``FORCE_COLOR``) gives 8; else 0.
``NO_COLOR`` and ``TERM=dumb`` both override this to 0 unconditionally.

Width: ``Capabilities.real_width`` is whatever ``shutil.get_terminal_size()``
reports, unclamped, so a caller that genuinely needs the true size can still
see it. ``Capabilities.width`` is that value clamped to ``[40, 100]`` so
panels neither collapse on a narrow terminal nor stretch absurdly on an
ultra-wide monitor.

Unicode: ``Capabilities.unicode`` is True only if the stream's encoding can
represent the box-drawing and block characters the rich renderer draws with
(the probe string is a full block, a lower one-eighth block, and a rounded
box corner). A stream with no ``encoding`` attribute, or an ``encoding`` of
``None``, is treated as ASCII-only rather than raising.

Windows: enabling virtual terminal (ANSI) processing on the console requires
``ctypes.windll``, which does not exist off Windows. ``_enable_windows_vt``
checks ``sys.platform`` *before* importing anything Windows-specific, so on
Linux/macOS it is a plain ``return True`` with no ``ctypes`` access at all;
the WinAPI calls themselves are also wrapped in a ``try/except`` so a console
that refuses the mode change degrades to ``colours = 0`` / ``"plain"``
instead of raising.
"""
from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_MIN_WIDTH = 40
_MAX_WIDTH = 100

# Presence of any of these (value is irrelevant) marks a CI runner.
_CI_VARS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI")

# The characters the rich renderer draws boxes and bars with. If the stream's
# encoding cannot represent these, widgets must fall back to ASCII.
_UNICODE_PROBE = "█▁╭"  # "█▁╭"

_VALID_REQUESTS = (None, "rich", "plain", "quiet")


@dataclass(frozen=True)
class Capabilities:
    """What the target stream and environment can do.

    All fields are computed once by :func:`detect` and then treated as an
    immutable fact by the rest of the ``_term`` layer.
    """

    is_tty: bool
    colours: int  # 0, 8, or 256
    width: int  # real_width clamped to [40, 100]
    real_width: int  # unclamped shutil.get_terminal_size().columns
    height: int
    unicode: bool
    interactive: bool  # supports_cursor_control and not CI (auto-mode default)
    supports_cursor_control: bool  # is_tty, not TERM=dumb, and Windows VT ok


def _stream_is_tty(stream: Any) -> bool:
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def _probe_unicode(stream: Any) -> bool:
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        _UNICODE_PROBE.encode(encoding)
    except (LookupError, UnicodeEncodeError, TypeError):
        return False
    return True


def _terminal_size() -> tuple[int, int]:
    size = shutil.get_terminal_size()
    return size.columns, size.lines


def _colour_depth(env: Mapping[str, str], is_tty: bool) -> int:
    """Colour depth ignoring the Windows VT outcome (folded in by the caller)."""
    if "NO_COLOR" in env:
        return 0
    if env.get("TERM", "") == "dumb":
        return 0
    forced_on = "FORCE_COLOR" in env
    if not is_tty and not forced_on:
        return 0
    colorterm = env.get("COLORTERM", "")
    if "truecolor" in colorterm or "24bit" in colorterm:
        return 256
    if "256color" in env.get("TERM", ""):
        return 256
    return 8


def _supports_cursor_control(env: Mapping[str, str], is_tty: bool, vt_ok: bool) -> bool:
    """True if the stream can plausibly interpret cursor-movement escapes.

    This is a capability check only: non-TTY streams and ``TERM=dumb`` both
    make redraw sequences meaningless, and a failed Windows VT enable means
    the console will not interpret them either. It deliberately does not
    look at CI/GITHUB_ACTIONS/GITLAB_CI, which are a policy default, not a
    capability limit (see the module docstring).
    """
    if not is_tty or not vt_ok:
        return False
    if env.get("TERM", "") == "dumb":
        return False
    return True


def _interactive(env: Mapping[str, str], supports_cursor_control: bool) -> bool:
    """The *auto-mode* default: capability-limited, plus the CI policy default."""
    if not supports_cursor_control:
        return False
    if any(name in env for name in _CI_VARS):
        return False
    return True


def _enable_windows_vt() -> bool:
    """Best-effort enable of ANSI/VT100 processing on the Windows console.

    Returns True unconditionally on any non-Windows platform: the
    ``sys.platform`` check happens before anything Windows-only is touched,
    so importing or calling this on Linux/macOS never imports ``ctypes``'s
    ``windll`` attribute (which only exists on Windows) and never raises.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        STD_ERROR_HANDLE = -12
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(STD_ERROR_HANDLE)
        if not handle or handle == -1:
            return False
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if not kernel32.SetConsoleMode(handle, new_mode):
            return False
        return True
    except Exception:
        return False


def detect(stream: Any = None, env: Mapping[str, str] | None = None) -> Capabilities:
    """Inspect ``stream`` and ``env`` and return the resulting ``Capabilities``.

    Both arguments are injectable so tests never need to touch
    ``sys.stderr`` or ``os.environ``. When omitted they default to
    ``sys.stderr`` and ``os.environ`` respectively.
    """
    if stream is None:
        stream = sys.stderr
    if env is None:
        env = os.environ

    is_tty = _stream_is_tty(stream)
    vt_ok = _enable_windows_vt()

    colours = 0 if not vt_ok else _colour_depth(env, is_tty)

    real_width, height = _terminal_size()
    width = max(_MIN_WIDTH, min(_MAX_WIDTH, real_width))

    unicode_ok = _probe_unicode(stream)
    cursor_ok = _supports_cursor_control(env, is_tty, vt_ok)
    interactive = _interactive(env, cursor_ok)

    return Capabilities(
        is_tty=is_tty,
        colours=colours,
        width=width,
        real_width=real_width,
        height=height,
        unicode=unicode_ok,
        interactive=interactive,
        supports_cursor_control=cursor_ok,
    )


def resolve_mode(caps: Capabilities, requested: str | None = None) -> str:
    """Resolve the final render mode from ``caps`` and an explicit request.

    ``requested`` is one of ``"rich"``, ``"plain"``, ``"quiet"`` or ``None``
    (auto-detect). See the module docstring for the full precedence order.
    """
    if requested not in _VALID_REQUESTS:
        raise ValueError(
            f"requested must be one of {_VALID_REQUESTS!r}, got {requested!r}"
        )

    if requested == "quiet":
        return "quiet"
    if requested == "plain":
        return "plain"
    if requested == "rich":
        return "rich" if caps.supports_cursor_control else "plain"

    # requested is None: auto-detect from the environment-derived caps.
    return "rich" if caps.interactive else "plain"
