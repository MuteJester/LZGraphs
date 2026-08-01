"""Tests for LZGraphs._term._caps: capability detection and mode resolution.

Every test injects its own fake stream and env dict; none of these tests
touch ``sys.stderr`` or ``os.environ`` directly. Width is tested two ways:
against a real pty-backed stream sized to a known width, which exercises the
actual ``stream.fileno()`` path ``_terminal_size`` uses; and, for the
fallback branch itself (no usable fd), by monkeypatching
``shutil.get_terminal_size``, since that is genuinely what ``_terminal_size``
falls back to when the stream cannot answer for itself.
"""
from __future__ import annotations

import io
import os
import shutil
import struct
import sys

import pytest

from LZGraphs._term._caps import Capabilities, detect, resolve_mode

try:
    import fcntl
    import pty
    import termios

    _HAVE_PTY = True
except ImportError:  # pragma: no cover - exercised only on Windows
    _HAVE_PTY = False

_NO_PTY_REASON = "pty/fcntl/termios are POSIX-only and unavailable on this platform"


def _open_sized_pty(columns, lines=24):
    """Open a pty pair, size it, and wrap the slave side as a real stream.

    Returns ``(master_fd, stream)``. The caller must close both: ``stream``
    (a text file wrapping the slave fd) and ``master_fd`` directly.
    """
    master_fd, slave_fd = pty.openpty()
    winsize = struct.pack("HHHH", lines, columns, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
    return master_fd, os.fdopen(slave_fd, "w")


class FakeStream:
    """A minimal stand-in for a text stream, for injecting into detect()."""

    def __init__(self, is_tty=False, encoding="utf-8", isatty_raises=False):
        self._is_tty = is_tty
        self.encoding = encoding
        self._isatty_raises = isatty_raises

    def isatty(self):
        if self._isatty_raises:
            raise OSError("no controlling terminal")
        return self._is_tty


class NoEncodingStream:
    """A stream object with no ``encoding`` attribute at all."""

    def __init__(self, is_tty=False):
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


TTY = FakeStream(is_tty=True)
PIPE = FakeStream(is_tty=False)


def _patch_terminal_size(monkeypatch, columns, lines=24):
    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((columns, lines))
    )


# ── Capabilities.colours: rule 2, NO_COLOR ─────────────────


def test_no_color_empty_string_still_disables_colour():
    """NO_COLOR="" must disable colour: presence matters, not truthiness."""
    caps = detect(stream=TTY, env={"NO_COLOR": ""})
    assert caps.colours == 0


def test_no_color_non_empty_disables_colour():
    caps = detect(stream=TTY, env={"NO_COLOR": "1"})
    assert caps.colours == 0


def test_no_color_absent_tty_gets_colour():
    caps = detect(stream=TTY, env={})
    assert caps.colours == 8


def test_no_color_beats_force_color():
    """NO_COLOR (rule 2) outranks FORCE_COLOR (rule 5)."""
    caps = detect(stream=PIPE, env={"NO_COLOR": "", "FORCE_COLOR": "1"})
    assert caps.colours == 0


# ── Capabilities.colours / mode: rule 3, TERM=dumb ─────────


def test_term_dumb_forces_colours_zero():
    caps = detect(stream=TTY, env={"TERM": "dumb"})
    assert caps.colours == 0


def test_term_dumb_forces_auto_mode_plain():
    caps = detect(stream=TTY, env={"TERM": "dumb"})
    assert resolve_mode(caps, None) == "plain"


def test_term_dumb_beats_force_color():
    caps = detect(stream=TTY, env={"TERM": "dumb", "FORCE_COLOR": "1"})
    assert caps.colours == 0


# ── mode: rule 4, CI environments ──────────────────────────


@pytest.mark.parametrize("ci_var", ["CI", "GITHUB_ACTIONS", "GITLAB_CI"])
def test_ci_env_forces_auto_mode_plain(ci_var):
    caps = detect(stream=TTY, env={ci_var: "true"})
    assert resolve_mode(caps, None) == "plain"


def test_ci_env_value_irrelevant():
    """Presence forces plain even when the CI var is falsy-looking."""
    caps = detect(stream=TTY, env={"CI": "false"})
    assert resolve_mode(caps, None) == "plain"


# ── Capabilities.colours: rule 5, FORCE_COLOR ──────────────


def test_force_color_on_non_tty_gives_colour_but_stays_plain():
    caps = detect(stream=PIPE, env={"FORCE_COLOR": "1"})
    assert caps.colours != 0
    assert resolve_mode(caps, None) == "plain"


def test_force_color_respects_colorterm_depth():
    caps = detect(stream=PIPE, env={"FORCE_COLOR": "1", "COLORTERM": "truecolor"})
    assert caps.colours == 256


def test_without_force_color_non_tty_has_no_colour():
    caps = detect(stream=PIPE, env={})
    assert caps.colours == 0


# ── mode: rule 6, plain TTY vs non-TTY default ─────────────


def test_auto_tty_gives_rich():
    caps = detect(stream=TTY, env={})
    assert resolve_mode(caps, None) == "rich"


def test_auto_non_tty_gives_plain():
    caps = detect(stream=PIPE, env={})
    assert resolve_mode(caps, None) == "plain"


# ── mode: rule 1, explicit request precedence ──────────────


def test_explicit_plain_wins_even_on_tty_with_no_forcing_env():
    caps = detect(stream=TTY, env={})
    assert resolve_mode(caps, "plain") == "plain"


def test_explicit_quiet_always_wins():
    caps = detect(stream=TTY, env={"CI": "1", "NO_COLOR": ""})
    assert resolve_mode(caps, "quiet") == "quiet"


def test_explicit_rich_on_tty_stays_rich():
    caps = detect(stream=TTY, env={})
    assert resolve_mode(caps, "rich") == "rich"


def test_explicit_rich_on_non_tty_downgrades_to_plain():
    caps = detect(stream=PIPE, env={})
    assert resolve_mode(caps, "rich") == "plain"


def test_invalid_requested_raises():
    caps = detect(stream=TTY, env={})
    with pytest.raises(ValueError):
        resolve_mode(caps, "verbose")


# ── Colour depth ladder ─────────────────────────────────────


def test_colour_depth_256_from_colorterm_truecolor():
    caps = detect(stream=TTY, env={"COLORTERM": "truecolor"})
    assert caps.colours == 256


def test_colour_depth_256_from_colorterm_24bit():
    caps = detect(stream=TTY, env={"COLORTERM": "24bit"})
    assert caps.colours == 256


def test_colour_depth_256_from_term_256color():
    caps = detect(stream=TTY, env={"TERM": "xterm-256color"})
    assert caps.colours == 256


def test_colour_depth_8_plain_tty():
    caps = detect(stream=TTY, env={"TERM": "xterm"})
    assert caps.colours == 8


def test_colour_depth_0_non_tty_no_env():
    caps = detect(stream=PIPE, env={})
    assert caps.colours == 0


# ── Width: derived from the stream's own fd (real pty) ──────
#
# Converted from monkeypatching shutil.get_terminal_size against a
# fileno-less FakeStream (which never exercised the injected `stream` at
# all) to a real pty sized to the width under test, since _terminal_size now
# reads the stream's own fd first. Same assertions as before; the mechanism
# is what changed, per fix round 2.


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_width_derived_from_stream_fd_not_stdout(monkeypatch):
    """The headline bug (Important, round 2): stderr is a 137-column pty
    while stdout, which shutil.get_terminal_size() would consult, reports
    something else (simulated here as its 80-column fallback). real_width
    must reflect the stream that was actually passed to detect(), not
    stdout, or every panel renders at the wrong width whenever `lzg build`
    output is redirected while stderr stays attached to a real terminal.
    """
    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((80, 24))
    )
    master_fd, stream = _open_sized_pty(137)
    try:
        caps = detect(stream=stream, env={})
        assert caps.is_tty is True
        assert caps.real_width == 137
    finally:
        stream.close()
        os.close(master_fd)


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_width_clamped_at_minimum_via_stream_fd():
    master_fd, stream = _open_sized_pty(20)
    try:
        caps = detect(stream=stream, env={})
        assert caps.real_width == 20
        assert caps.width == 40
    finally:
        stream.close()
        os.close(master_fd)


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_width_clamped_at_maximum_via_stream_fd():
    master_fd, stream = _open_sized_pty(400)
    try:
        caps = detect(stream=stream, env={})
        assert caps.real_width == 400
        assert caps.width == 100
    finally:
        stream.close()
        os.close(master_fd)


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_width_passthrough_via_stream_fd():
    master_fd, stream = _open_sized_pty(72)
    try:
        caps = detect(stream=stream, env={})
        assert caps.real_width == 72
        assert caps.width == 72
    finally:
        stream.close()
        os.close(master_fd)


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_height_via_stream_fd():
    master_fd, stream = _open_sized_pty(80, lines=40)
    try:
        caps = detect(stream=stream, env={})
        assert caps.height == 40
    finally:
        stream.close()
        os.close(master_fd)


# ── Width: fallback when the stream cannot be queried directly ─────
#
# These three exercise _terminal_size's fallback branch on purpose: the
# stream genuinely has no queryable fd, so falling through to
# shutil.get_terminal_size() (monkeypatched here for determinism) is the
# documented, correct behaviour, not a bypass of the injection point.


def test_width_no_fileno_attribute_falls_back(monkeypatch):
    """(b) A stream with no fileno() at all (FakeStream defines none)."""
    _patch_terminal_size(monkeypatch, columns=91, lines=30)
    caps = detect(stream=TTY, env={})
    assert not hasattr(TTY, "fileno")
    assert caps.real_width == 91
    assert caps.height == 30


def test_width_fileno_raises_unsupported_operation_falls_back(monkeypatch):
    """(c) io.StringIO().fileno() raises io.UnsupportedOperation, exactly
    like pytest's own captured stdout/stderr streams do."""
    _patch_terminal_size(monkeypatch, columns=91, lines=30)
    stream = io.StringIO()
    with pytest.raises(io.UnsupportedOperation):
        stream.fileno()
    caps = detect(stream=stream, env={})
    assert caps.real_width == 91


@pytest.mark.skipif(not _HAVE_PTY, reason=_NO_PTY_REASON)
def test_width_zero_columns_falls_back(monkeypatch):
    """(d) Some CI environments allocate a pty but never configure its
    window size, so TIOCGWINSZ reports 0 columns without raising. That must
    be treated as unusable, not clamped into a nonsense 40-column width."""
    _patch_terminal_size(monkeypatch, columns=91, lines=30)
    master_fd, stream = _open_sized_pty(0, lines=0)
    try:
        caps = detect(stream=stream, env={})
        assert caps.real_width == 91
    finally:
        stream.close()
        os.close(master_fd)


# ── Unicode probe ───────────────────────────────────────────


def test_unicode_probe_true_for_utf8():
    caps = detect(stream=FakeStream(is_tty=True, encoding="utf-8"), env={})
    assert caps.unicode is True


def test_unicode_probe_false_for_ascii():
    caps = detect(stream=FakeStream(is_tty=True, encoding="ascii"), env={})
    assert caps.unicode is False


def test_unicode_probe_false_when_encoding_is_none():
    caps = detect(stream=FakeStream(is_tty=True, encoding=None), env={})
    assert caps.unicode is False


def test_unicode_probe_false_when_encoding_attr_missing():
    caps = detect(stream=NoEncodingStream(is_tty=True), env={})
    assert caps.unicode is False


def test_unicode_probe_false_for_bogus_encoding_name():
    caps = detect(stream=FakeStream(is_tty=True, encoding="not-a-real-encoding"), env={})
    assert caps.unicode is False


# ── isatty robustness ────────────────────────────────────────


def test_isatty_raising_is_treated_as_non_tty():
    caps = detect(stream=FakeStream(isatty_raises=True), env={})
    assert caps.is_tty is False


# ── Defaults: stream=None / env=None use real globals ───────


def test_detect_defaults_do_not_raise():
    caps = detect()
    assert isinstance(caps, Capabilities)
    assert caps.width >= 40


# ── Windows VT guard is inert off-Windows ───────────────────


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "asserts the non-Windows short-circuit; on Windows the no-argument "
        "form genuinely probes the process stderr handle, which is a pipe "
        "under CI and correctly reports False"
    ),
)
def test_enable_windows_vt_is_noop_off_windows():
    from LZGraphs._term._caps import _enable_windows_vt

    # On this platform (Linux/macOS CI), sys.platform != "win32", so the
    # function must short-circuit to True without importing ctypes.windll.
    assert _enable_windows_vt() is True


def test_enable_windows_vt_true_for_a_stream_with_no_descriptor():
    """A stream with no OS descriptor has no console to configure, so the
    probe must not veto it. This is what keeps ``detect(stream=FakeStream)``
    identical on every platform; without it, Windows CI zeroed the colour
    depth of every injected fake terminal.
    """
    from LZGraphs._term._caps import _enable_windows_vt

    assert _enable_windows_vt(TTY) is True
    assert _enable_windows_vt(PIPE) is True


def test_interactive_false_when_not_tty():
    caps = detect(stream=PIPE, env={})
    assert caps.interactive is False


def test_interactive_true_on_plain_tty():
    caps = detect(stream=TTY, env={})
    assert caps.interactive is True


def test_interactive_false_under_ci_even_on_tty():
    caps = detect(stream=TTY, env={"CI": "1"})
    assert caps.interactive is False


def test_capabilities_is_frozen():
    caps = detect(stream=TTY, env={})
    with pytest.raises(Exception):
        caps.colours = 256


# ── Fix round 1: TERM=dumb is a capability limit, CI is a policy default ──
# (see module docstring's "capability vs policy" principle)


def test_term_dumb_downgrades_explicit_rich_to_plain():
    """(a) TERM=dumb is a capability statement: the terminal cannot process
    cursor movement, so --ui rich must downgrade exactly as non-TTY does."""
    caps = detect(stream=TTY, env={"TERM": "dumb"})
    assert resolve_mode(caps, "rich") == "plain"
    assert caps.colours == 0


def test_term_dumb_auto_mode_still_plain():
    """(b) Existing behaviour, pinned: TERM=dumb with no explicit request
    (auto mode) already resolved to plain before this fix; must still."""
    caps = detect(stream=TTY, env={"TERM": "dumb"})
    assert resolve_mode(caps, None) == "plain"


def test_ci_with_explicit_rich_on_tty_stays_rich():
    """(c) Deliberate asymmetry with (a): CI is a *policy* default, not a
    capability limit. A CI runner can allocate a real, cursor-capable TTY
    (e.g. for a demo recording or an interactive debugging session), and a
    user who explicitly asks for --ui rich there means it. Only the
    *automatic* choice (no --ui given) is steered away from rich by CI;
    an explicit request overrides that policy, unlike TERM=dumb which is a
    hard capability limit no request can override."""
    caps = detect(stream=TTY, env={"CI": "1"})
    assert resolve_mode(caps, "rich") == "rich"
    # Sanity: the asymmetry is real, not an oversight. Auto mode under the
    # same env still goes plain; only the explicit request differs from (a).
    assert resolve_mode(caps, None) == "plain"


def test_quiet_honoured_regardless_of_term_dumb():
    """(d) --ui quiet asks for less than plain, not more than rich, so a
    capability limit that only concerns redraw sequences has nothing to
    downgrade it from."""
    caps = detect(stream=TTY, env={"TERM": "dumb"})
    assert resolve_mode(caps, "quiet") == "quiet"


def test_supports_cursor_control_false_under_term_dumb():
    caps = detect(stream=TTY, env={"TERM": "dumb"})
    assert caps.supports_cursor_control is False


def test_supports_cursor_control_true_under_ci_on_tty():
    """CI does not affect supports_cursor_control, only `interactive` does."""
    caps = detect(stream=TTY, env={"CI": "1"})
    assert caps.supports_cursor_control is True
    assert caps.interactive is False


def test_supports_cursor_control_false_on_non_tty():
    caps = detect(stream=PIPE, env={})
    assert caps.supports_cursor_control is False
