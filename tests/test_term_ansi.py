"""Tests for LZGraphs._term._ansi: escape sequences and the named palette.

No test touches a real terminal or stream; ``Capabilities`` instances are
built directly with :func:`_caps` below, one per colour depth (0, 8, 256).
"""
from __future__ import annotations

import itertools

import pytest

from LZGraphs._term._ansi import (
    PALETTE_NAMES,
    bold,
    clear_line,
    cursor_up,
    dim,
    fg,
    hide_cursor,
    reset,
    sgr,
    show_cursor,
    visible_len,
)
from LZGraphs._term._caps import Capabilities


def _caps(colours: int) -> Capabilities:
    """A minimal Capabilities instance at a given colour depth.

    The other fields are irrelevant to this module (only ``colours`` is ever
    read by anything in ``_ansi``), so they are fixed at plausible defaults.
    """
    return Capabilities(
        is_tty=True,
        colours=colours,
        width=80,
        real_width=80,
        height=24,
        unicode=True,
        interactive=True,
        supports_cursor_control=True,
    )


CAPS0 = _caps(0)
CAPS8 = _caps(8)
CAPS256 = _caps(256)

BOX_AND_BLOCK_CHARS = "█░▁▂▃▄▅▆▇╭╮╰╯│─├┤"


# ── sgr(): the raw primitive ─────────────────────────────────


def test_sgr_wraps_int_code():
    assert sgr(1) == "\x1b[1m"


def test_sgr_wraps_str_code():
    assert sgr("38;5;71") == "\x1b[38;5;71m"


# ── fg(): identity at depth 0, distinct sequences at 8 and 256 ──


@pytest.mark.parametrize("name", PALETTE_NAMES)
def test_fg_is_empty_string_at_depth_0(name):
    """fg's "input" is a colour name, not text; the meaningful identity
    property is that fg(name, caps) + text collapses to plain text on its
    own, which requires fg to return "" here (see module docstring)."""
    assert fg(name, CAPS0) == ""


@pytest.mark.parametrize("name", PALETTE_NAMES)
def test_fg_depth_8_is_a_single_basic_sgr_code(name):
    code = fg(name, CAPS8)
    assert code.startswith("\x1b[")
    assert code.endswith("m")
    # Exactly one SGR parameter, no "38;5;" 256-colour prefix.
    assert ";" not in code


@pytest.mark.parametrize("name", PALETTE_NAMES)
def test_fg_depth_256_uses_38_5_n_form(name):
    code = fg(name, CAPS256)
    assert code.startswith("\x1b[38;5;")
    assert code.endswith("m")


@pytest.mark.parametrize("name", PALETTE_NAMES)
def test_fg_depth_8_and_256_differ(name):
    assert fg(name, CAPS8) != fg(name, CAPS256)


def test_fg_all_five_names_distinct_at_depth_8():
    codes = [fg(name, CAPS8) for name in PALETTE_NAMES]
    assert len(set(codes)) == len(PALETTE_NAMES)


def test_fg_all_five_names_distinct_at_depth_256():
    codes = [fg(name, CAPS256) for name in PALETTE_NAMES]
    assert len(set(codes)) == len(PALETTE_NAMES)


def test_fg_unknown_name_raises():
    with pytest.raises(ValueError):
        fg("not-a-real-colour", CAPS256)


def test_palette_names_are_the_five_specified():
    assert set(PALETTE_NAMES) == {"ok", "warn", "error", "accent", "muted"}


# ── bold()/dim(): identity at depth 0, wrap-and-reset otherwise ──


def test_bold_is_identity_at_depth_0():
    assert bold("hello", CAPS0) == "hello"


def test_dim_is_identity_at_depth_0():
    assert dim("hello", CAPS0) == "hello"


def test_bold_wraps_at_depth_8():
    out = bold("hello", CAPS8)
    assert out == "\x1b[1mhello\x1b[0m"


def test_dim_wraps_at_depth_256():
    out = dim("hello", CAPS256)
    assert out == "\x1b[2mhello\x1b[0m"


def test_bold_and_dim_use_different_codes():
    assert bold("x", CAPS8) != dim("x", CAPS8)


def test_bold_empty_string_at_depth_0_is_still_empty():
    assert bold("", CAPS0) == ""


# ── reset() / cursor movement: raw primitives, no caps ───────


def test_reset_is_raw_sgr_zero():
    assert reset() == "\x1b[0m"


def test_cursor_up_positive():
    assert cursor_up(5) == "\x1b[5A"


def test_cursor_up_zero_is_empty():
    """0 tracked lines must move nothing: ANSI treats a missing/zero
    parameter as 1, so a bare "\\x1b[0A" would still move the cursor."""
    assert cursor_up(0) == ""


def test_cursor_up_negative_is_empty():
    assert cursor_up(-3) == ""


def test_clear_line_sequence():
    assert clear_line() == "\x1b[K"


def test_hide_and_show_cursor_sequences():
    assert hide_cursor() == "\x1b[?25l"
    assert show_cursor() == "\x1b[?25h"


# ── visible_len: escapes are invisible, everything else is width 1 ──


def test_visible_len_plain_ascii():
    assert visible_len("hello") == 5


def test_visible_len_empty_string():
    assert visible_len("") == 0


def test_visible_len_ignores_sgr_escape():
    coloured = fg("ok", CAPS256) + "hello" + reset()
    assert visible_len(coloured) == visible_len("hello") == 5


def test_visible_len_ignores_basic_sgr_escape():
    coloured = fg("warn", CAPS8) + "hi" + reset()
    assert visible_len(coloured) == 2


def test_visible_len_ignores_bold_wrap():
    assert visible_len(bold("abc", CAPS8)) == visible_len("abc") == 3


def test_visible_len_ignores_cursor_and_clear_sequences():
    s = cursor_up(3) + "line" + clear_line()
    assert visible_len(s) == 4


def test_visible_len_ignores_hide_show_cursor():
    s = hide_cursor() + "x" + show_cursor()
    assert visible_len(s) == 1


@pytest.mark.parametrize("ch", list(BOX_AND_BLOCK_CHARS))
def test_visible_len_box_and_block_chars_are_width_1_each(ch):
    assert visible_len(ch) == 1


def test_visible_len_box_and_block_string_counts_each_as_1():
    assert visible_len(BOX_AND_BLOCK_CHARS) == len(BOX_AND_BLOCK_CHARS)


def test_visible_len_mixed_box_chars_and_colour():
    s = fg("accent", CAPS256) + "╭─" + reset() + dim("│", CAPS256) + "╰╯"
    assert visible_len(s) == 5


# ── Nested / repeated / adjacent escape sequences ────────────


def test_visible_len_adjacent_escapes_with_nothing_between():
    s = fg("ok", CAPS256) + bold("", CAPS256) + "x" + reset() + reset()
    assert visible_len(s) == 1


def test_visible_len_nested_bold_inside_fg():
    s = fg("warn", CAPS256) + bold("X", CAPS256) + reset()
    assert visible_len(s) == 1


def test_visible_len_repeated_reset_calls():
    s = "abc" + reset() + reset() + reset()
    assert visible_len(s) == 3


def test_visible_len_all_five_colours_concatenated():
    s = "".join(fg(name, CAPS256) + name[0] + reset() for name in PALETTE_NAMES)
    assert visible_len(s) == len(PALETTE_NAMES)


# ── The padding property the next task depends on ────────────

_PAD_WIDTHS = (10, 20, 40, 60, 80, 100)

_SAMPLE_ROWS = [
    ("plain ascii row", CAPS0),
    ("row with fg ok", CAPS256),
    ("row with fg warn", CAPS8),
    ("row with bold", CAPS256),
    ("row with dim", CAPS8),
    ("", CAPS256),
    ("╭─ box drawing ╮", CAPS256),
    ("█░ blocks ▁▂▃", CAPS8),
]


def _build_row(label: str, caps: Capabilities) -> str:
    """Build a representative row: some plain, some coloured, some not."""
    if not label:
        return ""
    if "fg ok" in label:
        return fg("ok", caps) + label + reset()
    if "fg warn" in label:
        return fg("warn", caps) + label + reset()
    if "bold" in label:
        return bold(label, caps)
    if "dim" in label:
        return dim(label, caps)
    return label


@pytest.mark.parametrize(
    "label,caps,width",
    [
        (label, caps, width)
        for (label, caps), width in itertools.product(_SAMPLE_ROWS, _PAD_WIDTHS)
        if width >= visible_len(_build_row(label, caps))
    ],
)
def test_padding_reaches_exact_target_width(label, caps, width):
    row = _build_row(label, caps)
    padded = row + " " * (width - visible_len(row))
    assert visible_len(padded) == width


def test_padding_property_holds_for_uncoupled_plain_string():
    row = "no colour codes here"
    width = 60
    padded = row + " " * (width - visible_len(row))
    assert visible_len(padded) == width


def test_padding_property_holds_for_fully_coloured_row():
    row = (
        fg("ok", CAPS256)
        + "SOURCE"
        + reset()
        + "  "
        + dim("bcr_repertoire.tsv.gz", CAPS256)
    )
    width = 80
    padded = row + " " * (width - visible_len(row))
    assert visible_len(padded) == width


# ── Wide characters and combining marks: documented, pinned limits ──


def test_visible_len_emoji_counts_as_one_not_two():
    """Documented limitation: this layer never renders emoji, so a wide
    glyph is counted as 1 column (its code point count), not the 2 display
    columns a real terminal would use. Pinned so a change in behaviour here
    is caught rather than silently assumed to still be "fine"."""
    assert visible_len("\U0001f600") == 1  # "\U0001f600" == the grinning-face emoji


def test_visible_len_cjk_wide_char_counts_as_one_not_two():
    assert visible_len("中") == 1  # a single wide CJK ideograph


def test_visible_len_combining_mark_counts_per_code_point():
    """Documented limitation: a decomposed base + combining mark counts as
    2 code points / columns here, not the 1 display column a terminal
    renders. "e" + COMBINING ACUTE ACCENT (U+0301)."""
    decomposed = "é"
    assert len(decomposed) == 2
    assert visible_len(decomposed) == 2
