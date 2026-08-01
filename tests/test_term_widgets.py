"""Tests for LZGraphs._term._widgets: pure string builders.

Every widget here is a pure function of data plus ``Capabilities``, so no
test touches a real terminal, ``os.environ``, or a stream; ``Capabilities``
instances are built directly via ``_caps()`` below at whichever width,
colour depth, and unicode setting a test needs.

The centrepiece is ``test_panel_lines_measure_exactly_the_requested_width``
and its ``card`` counterpart: a table-driven property, across widths (40,
60, 80, 100) x colour depths (0, 8, 256) x unicode (True, False), asserting
that *every* line a panel or card emits has ``visible_len`` exactly equal to
``caps.width``. That single property is worth more than any number of
example-based golden strings, per the task brief, and every other alignment
test in this file is really a special case of it (degenerate rows, over-long
rows, narrow mode, embedded colour).
"""
from __future__ import annotations

import itertools
import re

import pytest

from LZGraphs._term._ansi import colour as raw_colour
from LZGraphs._term._ansi import visible_len
from LZGraphs._term._caps import Capabilities
from LZGraphs._term._widgets import (
    bar,
    bytes_human,
    card,
    counter,
    duration,
    kv,
    panel,
    sparkline,
)

_ESCAPE_RE = re.compile(r"\x1b\[")
_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_WIDTHS = (40, 60, 80, 100)
_COLOURS = (0, 8, 256)
_UNICODE = (True, False)

_UNICODE_BOX_CHARS = "╭╮╰╯│─├┤"
_ASCII_BOX_CHARS = "+-|"


def _caps(width: int = 80, colours: int = 256, unicode: bool = True) -> Capabilities:
    """A ``Capabilities`` instance with only the fields ``_widgets`` reads
    set meaningfully; ``is_tty``/``interactive``/``supports_cursor_control``
    are fixed at plausible defaults since nothing in ``_widgets`` reads
    them."""
    return Capabilities(
        is_tty=True,
        colours=colours,
        width=width,
        real_width=width,
        height=24,
        unicode=unicode,
        interactive=True,
        supports_cursor_control=True,
    )


def _strip(s: str) -> str:
    """Remove CSI escapes, for tests that need to inspect plain content."""
    return _CSI_RE.sub("", s)


def _sample_rows(caps: Capabilities) -> list:
    """Rows exercising kv, an embedded bar, an embedded sparkline, a blank
    separator, and two deliberately over-long rows (one plain, one already
    colour-wrapped so truncation has to cut through an open colour span and
    close it itself). One fixture used everywhere so the width x colour x
    unicode matrix below exercises every code path in every combination."""
    return [
        kv("SOURCE", "bcr_repertoire.tsv.gz", caps),
        kv("FORMAT", "AIRR-TSV", caps, note="detected"),
        "",
        kv("ingest", bar(0.71, 20, caps) + "  71%", caps),
        kv("trend", sparkline([3, 5, 6, 8, 7, 6, 8], caps), caps),
        "x" * 500,
        raw_colour("ok", "y" * 500, caps),
    ]


_MATRIX = list(itertools.product(_WIDTHS, _COLOURS, _UNICODE))
_MATRIX_IDS = [f"w{w}-c{c}-u{u}" for w, c, u in _MATRIX]


# ── The core property: panel/card lines measure exactly caps.width ──


@pytest.mark.parametrize("width,colours,unicode", _MATRIX, ids=_MATRIX_IDS)
def test_panel_lines_measure_exactly_the_requested_width(width, colours, unicode):
    caps = _caps(width, colours, unicode)
    rows = _sample_rows(caps)
    footer = kv("note", "footer text goes here", caps)
    lines = panel("lzg build", rows, caps, footer=footer)
    assert lines
    for line in lines:
        assert visible_len(line) == width, (width, colours, unicode, line)


@pytest.mark.parametrize("width,colours,unicode", _MATRIX, ids=_MATRIX_IDS)
@pytest.mark.parametrize("status", ["ok", "warn", "error"])
def test_card_lines_measure_exactly_the_requested_width(width, colours, unicode, status):
    caps = _caps(width, colours, unicode)
    rows = _sample_rows(caps)
    lines = card("lzg build", rows, caps, status)
    assert lines
    for line in lines:
        assert visible_len(line) == width, (width, colours, unicode, status, line)


@pytest.mark.parametrize("width,colours,unicode", _MATRIX, ids=_MATRIX_IDS)
@pytest.mark.parametrize("fraction", [0.0, 0.001, 0.25, 0.5, 0.999, 1.0, -3.0, 4.0])
def test_bar_measures_exactly_the_requested_width(width, colours, unicode, fraction):
    caps = _caps(width, colours, unicode)
    assert visible_len(bar(fraction, width, caps)) == width


# ── Depth 0: never leak an escape sequence ──


@pytest.mark.parametrize("width,unicode", itertools.product(_WIDTHS, _UNICODE))
def test_depth_0_never_leaks_escape_sequences(width, unicode):
    caps = _caps(width, colours=0, unicode=unicode)
    rows = _sample_rows(caps)
    outputs = [
        bar(0.5, width, caps),
        bar(0.0, width, caps),
        bar(1.0, width, caps),
        sparkline([1, 4, 2, 8, 0], caps),
        sparkline([], caps),
        kv("k", "v", caps, note="n"),
    ]
    outputs.extend(panel("t", rows, caps, footer="f"))
    outputs.extend(card("t", rows, caps, "error"))
    for out in outputs:
        assert _ESCAPE_RE.search(out) is None, out


# ── Unicode False: never emit a character outside ASCII ──


@pytest.mark.parametrize("width,colours", itertools.product(_WIDTHS, _COLOURS))
def test_unicode_false_output_is_pure_ascii(width, colours):
    caps = _caps(width, colours, unicode=False)
    rows = _sample_rows(caps)
    outputs = [
        bar(0.5, width, caps),
        sparkline([1, 4, 2, 8, 0], caps),
        kv("k", "v", caps, note="n"),
    ]
    outputs.extend(panel("t", rows, caps, footer="f"))
    outputs.extend(card("t", rows, caps, "ok"))
    for out in outputs:
        assert all(ord(ch) < 128 for ch in out), out


# ── Narrow terminals (below 50 columns): box dropped, still exact width ──


def test_panel_narrow_mode_drops_the_box_but_still_measures_exact_width():
    # Deliberately hyphen-free content: unlike `_sample_rows` (which
    # includes "AIRR-TSV", a legitimate hyphen in business data), this rows
    # list must contain nothing that could be mistaken for the ascii box
    # fallback ("+-|") so the "box is gone" assertion below tests only
    # decoration, not incidentally-similar content.
    caps = _caps(width=40)
    rows = [
        kv("SOURCE", "bcr_repertoire.tsv.gz", caps),
        kv("ingest", bar(0.71, 20, caps) + "  71%", caps),
        kv("trend", sparkline([3, 5, 6, 8, 7, 6, 8], caps), caps),
        "",
    ]
    lines = panel("lzg build", rows, caps, footer="f")
    combined = "".join(_strip(line) for line in lines)
    assert not any(ch in combined for ch in _UNICODE_BOX_CHARS)
    assert not any(ch in combined for ch in _ASCII_BOX_CHARS)
    for line in lines:
        assert visible_len(line) == 40


@pytest.mark.parametrize("width", [60, 80, 100])
def test_panel_boxed_mode_draws_the_box_at_or_above_50_columns(width):
    caps = _caps(width=width)
    lines = panel("lzg build", [kv("a", "b", caps)], caps)
    assert _UNICODE_BOX_CHARS[0] in lines[0]  # top-left corner present
    assert _UNICODE_BOX_CHARS[2] in lines[-1]  # bottom-left corner present


def test_panel_ascii_box_uses_plus_dash_pipe_when_unicode_false():
    caps = _caps(width=60, unicode=False)
    lines = panel("t", [kv("a", "b", caps)], caps)
    combined = "".join(_strip(line) for line in lines)
    assert not any(ch in combined for ch in _UNICODE_BOX_CHARS)
    assert "+" in combined and "-" in combined and "|" in combined


# ── Degenerate inputs, pinned ──


def test_sparkline_empty_is_empty_string():
    assert sparkline([], _caps()) == ""


def test_sparkline_all_zero_does_not_crash_and_is_flat():
    result = _strip(sparkline([0, 0, 0], _caps()))
    assert len(result) == 3
    assert len(set(result)) == 1


def test_sparkline_all_equal_nonzero_does_not_crash_and_is_flat():
    result = _strip(sparkline([5, 5, 5], _caps()))
    assert len(result) == 3
    assert len(set(result)) == 1


def test_sparkline_ascii_fallback_is_digit_ramp():
    result = _strip(sparkline([0, 4, 8], _caps(unicode=False)))
    assert result.isdigit()


def test_bar_zero_fraction_is_all_empty_cells():
    assert _strip(bar(0.0, 10, _caps())) == "░" * 10


def test_bar_full_fraction_is_all_filled_cells():
    assert _strip(bar(1.0, 10, _caps())) == "█" * 10


@pytest.mark.parametrize("fraction", [-5.0, -0.001])
def test_bar_below_zero_clamps_to_empty(fraction):
    caps = _caps()
    result = bar(fraction, 10, caps)
    assert visible_len(result) == 10
    assert _strip(result) == "░" * 10


@pytest.mark.parametrize("fraction", [1.001, 2.0, 50.0])
def test_bar_above_one_clamps_to_full(fraction):
    caps = _caps()
    result = bar(fraction, 10, caps)
    assert visible_len(result) == 10
    assert _strip(result) == "█" * 10


def test_bar_positive_infinity_clamps_to_full_without_raising():
    # `int(fraction * width)` overflows on a non-finite fraction, so this
    # only passes if `fraction` is clamped to [0.0, 1.0] *before* that
    # multiplication; clamping only the resulting cell count (which is
    # already in range for any finite fraction) would not save this case,
    # so this pins the clamp as load-bearing rather than redundant.
    caps = _caps()
    result = bar(float("inf"), 10, caps)
    assert visible_len(result) == 10
    assert _strip(result) == "█" * 10


def test_bar_negative_infinity_clamps_to_empty_without_raising():
    caps = _caps()
    result = bar(float("-inf"), 10, caps)
    assert visible_len(result) == 10
    assert _strip(result) == "░" * 10


def test_bar_nan_clamps_without_raising():
    # `int(float("nan"))` raises ValueError, so this too only passes if the
    # clamp runs before the multiplication.
    caps = _caps()
    result = bar(float("nan"), 10, caps)
    assert visible_len(result) == 10


def test_bar_ascii_fallback_uses_hash_and_dot():
    caps = _caps(unicode=False)
    assert _strip(bar(0.5, 10, caps)) == "#" * 5 + "." * 5


def test_panel_with_no_rows_does_not_crash():
    caps = _caps(width=80)
    lines = panel("empty panel", [], caps)
    assert len(lines) == 2  # just the top and bottom border
    for line in lines:
        assert visible_len(line) == 80


def test_panel_row_longer_than_width_is_truncated_with_ellipsis():
    caps = _caps(width=80)
    lines = panel("t", ["x" * 500], caps)
    body = lines[1]
    assert visible_len(body) == 80
    assert "..." in body


def test_panel_title_longer_than_width_is_truncated_with_ellipsis():
    caps = _caps(width=60)
    lines = panel("this title is far too long to fit inside a sixty column box", [], caps)
    assert visible_len(lines[0]) == 60
    assert "..." in _strip(lines[0])


def test_counter_zero():
    assert counter(0) == "0"


def test_counter_thousands_separators():
    assert counter(1234567) == "1,234,567"


def test_counter_negative():
    assert counter(-42) == "-42"


def test_duration_zero():
    assert duration(0) == "0.0s"


def test_duration_subminute_has_one_decimal():
    assert duration(12.4) == "12.4s"


def test_duration_minute_scale_is_m_ss():
    assert duration(62) == "1:02"


def test_duration_hour_scale_is_h_mm_ss():
    assert duration(3661) == "1:01:01"


def test_duration_negative_clamps_to_zero():
    assert duration(-5) == "0.0s"


def test_bytes_human_zero():
    assert bytes_human(0) == "0 B"


def test_bytes_human_sub_kilobyte_has_no_decimal():
    assert bytes_human(500) == "500 B"


def test_bytes_human_scales_up_through_units():
    assert bytes_human(1500) == "1.5 KB"
    assert bytes_human(1_200_000_000) == "1.2 GB"


def test_bytes_human_negative_clamps_to_zero():
    assert bytes_human(-10) == "0 B"


def test_kv_pads_key_to_fixed_column_and_appends_note():
    caps = _caps(colours=0)
    result = kv("nodes", "71,181", caps, note="extra")
    assert result == " " + "nodes".ljust(9) + " 71,181" + "  extra"


def test_kv_without_note_has_no_note_text():
    caps = _caps(colours=0)
    result = kv("nodes", "71,181", caps)
    assert result == " " + "nodes".ljust(9) + " 71,181"


def test_kv_empty_key_gives_blank_label_column():
    caps = _caps(colours=0)
    result = kv("", "value only", caps)
    assert result == " " + " " * 9 + " value only"


def test_card_invalid_status_raises():
    caps = _caps()
    with pytest.raises(ValueError):
        card("t", [], caps, "nope")


def test_card_title_includes_status_word():
    caps = _caps(colours=0)
    lines = card("lzg build", [], caps, "ok")
    assert "done" in _strip(lines[0])
    lines = card("lzg build", [], caps, "error")
    assert "error" in _strip(lines[0])
    lines = card("lzg build", [], caps, "warn")
    assert "warning" in _strip(lines[0])
