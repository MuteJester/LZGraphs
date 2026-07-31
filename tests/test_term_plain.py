"""Tests for LZGraphs._term._plain: the scrolling CI/pipe renderer.

No test touches a real terminal. ``Capabilities`` instances are built
directly via ``_caps()`` below, and every renderer under test is constructed
against an injected ``io.StringIO`` (or, for the two tests that care about
the *default* stream specifically, a monkeypatched ``sys.stderr``) rather
than a real file descriptor.

The centrepiece properties, each required by the task brief and each tested
by measuring captured output rather than by trusting the implementation:

* zero escape sequences at ``caps.colours == 0``, over *every* protocol
  method (``test_depth_0_*``);
* nothing ever reaches ``sys.stdout`` (``test_nothing_written_to_stdout*``);
* ``progress()`` is throttled to a bound derived from the documented rule,
  not merely "fewer than 10,000" (``test_progress_throttle_*``);
* every emitted line is parseable by ``_parse_line`` below, a small
  boundary-regex splitter standing in for "a script" (``test_*_parseable``
  and ``test_all_driven_lines_are_parseable``);
* no line contains a box-drawing or block character
  (``test_no_box_drawing_characters_anywhere``), pinning the trap shut;
* ``error()`` is comprehensible without colour via structure/indentation,
  not a red tint (``test_error_*``).
"""
from __future__ import annotations

import io
import re

import pytest

from LZGraphs._term import _plain
from LZGraphs._term._caps import Capabilities
from LZGraphs._term._plain import _PROGRESS_PCT_STEP, _PROGRESS_TIME_STEP, PlainRenderer

# Same set used by test_term_ansi.py / test_term_widgets.py: every
# box-drawing and block character the rich renderer's widgets might emit.
# This module must never produce any of them.
_BOX_AND_BLOCK_CHARS = "█░▁▂▃▄▅▆▇╭╮╰╯│─├┤"

_ESCAPE_RE = re.compile(r"\x1b")
_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# The "simple splitter" the greppable claim is tested against: find every
# `word=` boundary in a line's content (after its `[tag]` prefix) and take
# each field's value as the text up to the next boundary or end of line.
# This is exactly the convention documented in _plain's module docstring,
# and it is what already makes today's cli.py lines parseable (e.g.
# "sequences=1234 (12 V genes, 34 J genes) total=...": no further `word=`
# occurs inside the parenthetical, so it is correctly kept as part of
# `sequences`'s value rather than mis-split).
_TAG_RE = re.compile(r"^\[(?P<tag>[^\]]*)\]\s*(?P<rest>.*)$")
_FIELD_BOUNDARY_RE = re.compile(r"(\w+)=")


def _strip_escapes(s: str) -> str:
    return _CSI_RE.sub("", s)


def _parse_line(line: str) -> tuple[str, dict]:
    """Split one emitted line into its ``[tag]`` and a ``{key: value}`` dict.

    Strips colour escapes first (harmless at depth 0, necessary at depth 8/
    256) so the same parser works regardless of colour depth.
    """
    plain = _strip_escapes(line)
    m = _TAG_RE.match(plain)
    assert m is not None, f"line has no [tag] prefix: {line!r}"
    tag, rest = m.group("tag"), m.group("rest")
    matches = list(_FIELD_BOUNDARY_RE.finditer(rest))
    fields = {}
    for i, fm in enumerate(matches):
        key = fm.group(1)
        start = fm.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(rest)
        fields[key] = rest[start:end].strip()
    return tag, fields


def _caps(colours: int = 0) -> Capabilities:
    """A minimal Capabilities instance at a given colour depth.

    Only ``colours`` is ever read by ``_plain`` (it never draws unicode
    widgets or reflows on width), so the rest are fixed at plausible
    defaults.
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


def _drive_all_methods(renderer: PlainRenderer) -> None:
    """Exercise every protocol method once, with realistic-looking data."""
    renderer.start("lzg build", {"source": "bcr_repertoire.tsv.gz", "format": "AIRR-TSV"})
    renderer.progress("ingest", 0.0, detail="0 seq")
    renderer.progress("ingest", 0.5, detail="500 seq")
    renderer.progress("ingest", 1.0, detail="1000 seq")
    renderer.update(nodes=71181, edges=11714847)
    renderer.warn("412 non-productive records skipped")
    renderer.error(
        "lzg build",
        ["column 'junction_aa' not found", "available columns: sequence_id, v_call, cdr3_aa"],
    )
    renderer.done("lzg build", {"output": "foundation.lzg", "nodes": 71181})


# ── Zero escape sequences at depth 0, over every protocol method ──


def test_depth_0_start_has_no_escapes():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).start("lzg build", {"source": "x.tsv"})
    assert _ESCAPE_RE.search(stream.getvalue()) is None


def test_depth_0_progress_has_no_escapes():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).progress("ingest", 0.5, detail="500 seq")
    assert _ESCAPE_RE.search(stream.getvalue()) is None


def test_depth_0_update_has_no_escapes():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).update(nodes=123, edges=456)
    assert _ESCAPE_RE.search(stream.getvalue()) is None


def test_depth_0_warn_has_no_escapes():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).warn("something looked odd")
    assert _ESCAPE_RE.search(stream.getvalue()) is None


def test_depth_0_error_has_no_escapes():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).error("lzg build", ["headline", "detail one", "detail two"])
    assert _ESCAPE_RE.search(stream.getvalue()) is None


def test_depth_0_done_has_no_escapes():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).done("lzg build", {"nodes": 123})
    assert _ESCAPE_RE.search(stream.getvalue()) is None


def test_depth_0_full_session_has_no_escapes():
    """All six methods, driven together, still produce zero escapes."""
    stream = io.StringIO()
    _drive_all_methods(PlainRenderer(CAPS0, stream))
    assert _ESCAPE_RE.search(stream.getvalue()) is None


@pytest.mark.parametrize("colours", [8, 256])
def test_warn_error_done_do_carry_colour_above_depth_0(colours):
    """Sanity check on the fixtures above: colour is not simply "never
    implemented" (which would make the depth-0 tests vacuous). warn/error/
    done colour their status word once caps allows it; the depth-0 tests
    prove that colouring is gated, not absent."""
    stream = io.StringIO()
    r = PlainRenderer(_caps(colours), stream)
    r.warn("x")
    r.error("lzg build", ["y"])
    r.done("lzg build", {"z": 1})
    assert _ESCAPE_RE.search(stream.getvalue()) is not None


# ── Nothing is ever written to stdout ──


def test_nothing_written_to_stdout_with_explicit_stream(capsys):
    stream = io.StringIO()
    _drive_all_methods(PlainRenderer(CAPS0, stream))
    captured = capsys.readouterr()
    assert captured.out == ""
    # And the content did go somewhere: the explicitly injected stream.
    assert stream.getvalue() != ""


def test_nothing_written_to_stdout_with_default_stream(monkeypatch, capsys):
    """The default stream (no `stream=` given) is sys.stderr, never stdout."""
    fake_stderr = io.StringIO()
    monkeypatch.setattr(_plain.sys, "stderr", fake_stderr)
    _drive_all_methods(PlainRenderer(CAPS0))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert fake_stderr.getvalue() != ""


# ── Progress throttling ──


def test_progress_throttle_bounded_over_10000_calls(monkeypatch):
    """Driving progress() 10,000 times with a monotonically increasing
    fraction must emit far fewer than 10,000 lines: the documented rule
    (at most every _PROGRESS_PCT_STEP percentage points, or every
    _PROGRESS_TIME_STEP seconds, per label) bounds a single 0->1 sweep to
    `100 // _PROGRESS_PCT_STEP + 1` lines. Wall-clock time is frozen so this
    assertion is exact and not a flaky function of how fast the test
    machine happens to run, isolating the percentage-based half of the
    rule.
    """
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    monkeypatch.setattr(_plain.time, "monotonic", lambda: 1000.0)

    for i in range(10001):
        renderer.progress("ingest", i / 10000, detail=f"{i} seq")

    lines = [line for line in stream.getvalue().splitlines() if line]
    expected = 100 // _PROGRESS_PCT_STEP + 1
    assert len(lines) == expected
    assert len(lines) < 100  # bounded by the rule, not merely "< 10000"


def test_progress_throttle_time_based_branch(monkeypatch):
    """Same pct twice in a row: no emit until _PROGRESS_TIME_STEP elapses."""
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    clock = [0.0]
    monkeypatch.setattr(_plain.time, "monotonic", lambda: clock[0])

    renderer.progress("ingest", 0.5)  # first call: always emits
    clock[0] += _PROGRESS_TIME_STEP / 2
    renderer.progress("ingest", 0.5)  # same pct, half the time step: skip
    clock[0] += _PROGRESS_TIME_STEP
    renderer.progress("ingest", 0.5)  # same pct, past the time step: emit

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 2


def test_progress_throttle_label_change_forces_emit(monkeypatch):
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    monkeypatch.setattr(_plain.time, "monotonic", lambda: 0.0)

    renderer.progress("ingest", 0.5)
    renderer.progress("save", 0.5)  # different label, same pct/time: emit

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 2
    _, fields1 = _parse_line(lines[0])
    _, fields2 = _parse_line(lines[1])
    assert fields1["label"] == "ingest"
    assert fields2["label"] == "save"


def test_progress_throttle_completion_forces_emit(monkeypatch):
    """Reaching 100% always emits, even mid-throttle-window."""
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    monkeypatch.setattr(_plain.time, "monotonic", lambda: 0.0)

    renderer.progress("ingest", 0.97)  # first call: emits, pct=97
    renderer.progress("ingest", 1.0)  # +3 pct, no time elapsed: still forced

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 2
    _, fields = _parse_line(lines[1])
    assert fields["pct"] == "100"


# ── Every emitted line is parseable ──


def test_start_line_is_parseable():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).start("lzg build", {"source": "x.tsv", "engine": "flashback"})
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    tag, fields = _parse_line(line)
    assert tag == "build"
    assert fields["status"] == "start"
    assert fields["source"] == "x.tsv"
    assert fields["engine"] == "flashback"


def test_progress_line_is_parseable():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).progress("ingest", 0.71, detail="1,662,183 seq")
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    tag, fields = _parse_line(line)
    assert tag == "lzg"  # no start() called: falls back to the default tag
    assert fields["status"] == "progress"
    assert fields["label"] == "ingest"
    assert fields["pct"] == "71"
    assert fields["detail"] == "1,662,183 seq"


def test_update_line_is_parseable():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).update(nodes=71181, edges=11714847)
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    tag, fields = _parse_line(line)
    assert fields["status"] == "info"
    assert fields["nodes"] == "71181"
    assert fields["edges"] == "11714847"


def test_warn_line_is_parseable():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).warn("412 non-productive records skipped")
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    tag, fields = _parse_line(line)
    assert fields["status"] == "warn"
    assert fields["message"] == "412 non-productive records skipped"


def test_done_line_is_parseable_and_auto_elapsed():
    stream = io.StringIO()
    r = PlainRenderer(CAPS0, stream)
    r.start("lzg build", {})
    r.done("lzg build", {"output": "foundation.lzg", "nodes": 71181})
    lines = [ln for ln in stream.getvalue().splitlines() if ln]
    tag, fields = _parse_line(lines[-1])
    assert tag == "build"
    assert fields["status"] == "done"
    assert fields["output"] == "foundation.lzg"
    assert fields["nodes"] == "71181"
    assert "elapsed" in fields  # auto-appended since the caller didn't supply one


def test_done_caller_supplied_elapsed_is_not_overwritten():
    stream = io.StringIO()
    r = PlainRenderer(CAPS0, stream)
    r.start("lzg build", {})
    r.done("lzg build", {"elapsed": "99.9s"})
    (line,) = [ln for ln in stream.getvalue().splitlines() if "status=done" in ln]
    _, fields = _parse_line(line)
    assert fields["elapsed"] == "99.9s"


def test_all_driven_lines_are_parseable_and_carry_a_status():
    """The generic property behind every method-specific test above: no
    matter which of the six methods produced it, every line has a [tag]
    prefix and a `status=` field, at both colour depths."""
    for caps in (CAPS0, CAPS8, CAPS256):
        stream = io.StringIO()
        _drive_all_methods(PlainRenderer(caps, stream))
        lines = [ln for ln in stream.getvalue().splitlines() if ln]
        assert len(lines) >= 6
        for line in lines:
            tag, fields = _parse_line(line)
            assert tag  # non-empty
            assert "status" in fields
            assert fields["status"] in {"start", "progress", "info", "warn", "error", "done"}


# ── No box-drawing or block character anywhere ──


@pytest.mark.parametrize("colours", [0, 8, 256])
def test_no_box_drawing_characters_anywhere(colours):
    stream = io.StringIO()
    _drive_all_methods(PlainRenderer(_caps(colours), stream))
    output = stream.getvalue()
    for ch in _BOX_AND_BLOCK_CHARS:
        assert ch not in output, f"box/block character {ch!r} leaked into plain output"


# ── error(): readable without colour via structure/indentation ──


def test_error_header_line_has_no_extra_fields():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).error("lzg build", ["headline", "detail two"])
    lines = [ln for ln in stream.getvalue().splitlines() if ln]
    tag, fields = _parse_line(lines[0])
    assert tag == "build"
    assert fields == {"status": "error"}


def test_error_detail_lines_are_indented_under_the_header():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).error(
        "lzg build", ["column 'junction_aa' not found", "try: lzg build bcr.tsv -o rep.lzg"]
    )
    lines = [ln for ln in stream.getvalue().splitlines() if ln]
    assert len(lines) == 3  # header + 2 detail lines
    header, detail1, detail2 = lines

    # Structural marker #1: every line shares the same stable [tag] status=
    # prefix, so a script (or a human's eye) can group them with one grep.
    assert header.startswith("[build] status=error")
    assert detail1.startswith("[build] status=error")
    assert detail2.startswith("[build] status=error")

    # Structural marker #2: detail lines carry extra indentation between
    # "status=error" and their "detail=" field, visually nesting them under
    # the header even with colour entirely disabled.
    assert re.search(r"status=error {2,}detail=", detail1)
    assert re.search(r"status=error {2,}detail=", detail2)
    # ...and the header line itself has no such indentation/field, so the
    # two kinds of line are visually distinguishable without colour.
    assert "detail=" not in header

    _, fields1 = _parse_line(detail1)
    _, fields2 = _parse_line(detail2)
    assert fields1["detail"] == "column 'junction_aa' not found"
    assert fields2["detail"] == "try: lzg build bcr.tsv -o rep.lzg"


def test_error_with_no_lines_still_emits_one_readable_line():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).error("lzg build")
    lines = [ln for ln in stream.getvalue().splitlines() if ln]
    assert lines == ["[build] status=error"]


# ── Tag derivation ──


def test_tag_strips_lzg_prefix():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).start("lzg build", {})
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    assert line.startswith("[build]")


def test_tag_keeps_multi_word_command_name():
    """Matches the existing cli.py convention: "lzg flashback build" style
    commands already emit a "[flashback build]" tag today."""
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).start("lzg flashback build", {})
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    assert line.startswith("[flashback build]")


def test_tag_without_lzg_prefix_used_as_is():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).start("posterior", {})
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    assert line.startswith("[posterior]")


def test_progress_before_start_uses_default_tag_without_raising():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).progress("ingest", 0.5)
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    assert line.startswith("[lzg]")


# ── Embedded newlines never split one call into more than one line ──


def test_warn_message_with_embedded_newline_stays_one_line():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).warn("line one\nline two")
    lines = [ln for ln in stream.getvalue().splitlines() if ln]
    assert len(lines) == 1
    assert "line one line two" in lines[0]


# ── Values with embedded spaces still parse (the documented convention) ──


def test_field_value_with_spaces_is_kept_whole_when_it_is_the_last_field():
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).update(gene_info="12 V genes, 34 J genes")
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    _, fields = _parse_line(line)
    assert fields["gene_info"] == "12 V genes, 34 J genes"
