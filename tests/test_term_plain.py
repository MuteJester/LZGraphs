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

import inspect
import io
import re

import pytest

from LZGraphs._term import Ui, _plain
from LZGraphs._term._caps import Capabilities
from LZGraphs._term._plain import (
    _MAX_PROGRESS_LINES_PER_SECOND,
    _MAX_TRACKED_PROGRESS_LABELS,
    _PROGRESS_PCT_STEP,
    _PROGRESS_TIME_STEP,
    PlainRenderer,
)
from LZGraphs._term._rich import RichRenderer

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


# ── Documented limitation, verified rather than merely asserted ──


def test_documented_limitation_value_containing_word_equals_mis_splits():
    """Fix round 1, minor item: the module docstring used to claim (with no
    test behind it) that "nothing this renderer emits" contains a bare
    ``word=`` substring. That claim was never checked, and a caller-supplied
    value like a path containing ``=`` violates it. This test verifies, and
    therefore pins, what the boundary parser actually does in that case
    (mis-split the value), rather than leaving the docstring's claim
    unverified. This is a documented, accepted limitation of the convention,
    not a bug this renderer needs to fix.
    """
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).update(path="a=b/c=d.tsv")
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    _, fields = _parse_line(line)
    # The true value ("a=b/c=d.tsv") is NOT recovered: the parser reads the
    # embedded "a=" and "c=" as new field boundaries.
    assert fields["path"] != "a=b/c=d.tsv"
    assert fields["path"] == ""
    assert fields["a"] == "b/"
    assert fields["c"] == "d.tsv"


# ── Finding 1 (fix round 1): the throttle is per-label, with bounded state ──


def test_progress_throttle_alternating_labels_bounded(monkeypatch):
    """Reproduces the coordinator's Finding 1 directly: alternating between
    two labels, each independently sweeping 0.0 -> 1.0, must bound to
    roughly 21 lines PER label (the same rule
    test_progress_throttle_bounded_over_10000_calls pins for a single
    label), not to one shared window where the second label's calls reset
    the first label's throttle state (the pre-fix bug: a single global
    ``(label, pct, time)`` tuple meant switching labels every call defeated
    the throttle completely, emitting one line per call).
    """
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    monkeypatch.setattr(_plain.time, "monotonic", lambda: 1000.0)

    steps_per_label = 5000
    for i in range(2 * steps_per_label):
        label = "ingest" if i % 2 == 0 else "save"
        step = i // 2
        renderer.progress(label, step / (steps_per_label - 1))

    lines = [line for line in stream.getvalue().splitlines() if line]
    per_label_expected = 100 // _PROGRESS_PCT_STEP + 1
    assert len(lines) == 2 * per_label_expected
    assert len(lines) < 100  # bounded, nowhere near the 10,000 calls made

    by_label: dict[str, int] = {"ingest": 0, "save": 0}
    for line in lines:
        _, fields = _parse_line(line)
        by_label[fields["label"]] += 1
    assert by_label["ingest"] == per_label_expected
    assert by_label["save"] == per_label_expected


def test_progress_throttle_three_rotating_labels_bounded(monkeypatch):
    """Same property, three labels rotating instead of two alternating,
    matching the coordinator's second reproduction exactly."""
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    monkeypatch.setattr(_plain.time, "monotonic", lambda: 1000.0)

    labels = ["ingest", "save", "index"]
    steps_per_label = 3334
    for i in range(len(labels) * steps_per_label):
        label = labels[i % len(labels)]
        step = i // len(labels)
        renderer.progress(label, step / (steps_per_label - 1))

    lines = [line for line in stream.getvalue().splitlines() if line]
    per_label_expected = 100 // _PROGRESS_PCT_STEP + 1
    assert len(lines) == len(labels) * per_label_expected
    assert len(lines) < 150


def test_progress_label_state_bounded_for_many_distinct_labels():
    """Finding 1's REQUIRED bound: a caller reporting under an unbounded
    number of distinct labels (a bug, or a high-cardinality label) must not
    grow this renderer's tracked state without limit. White-box check on
    the internal cache directly, since the bound is a memory property with
    no externally observable line-count consequence of its own."""
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    for i in range(5000):
        renderer.progress(f"label-{i}", 0.5)
    assert len(renderer._last_progress) <= _MAX_TRACKED_PROGRESS_LABELS


# ── Fix round 2: the per-label LRU cap can itself be defeated by enough ──
# distinct labels (every call evicts the entry it needs, so every call
# looks like a fresh label and the per-label rule never engages at all).
# An unconditional global rate backstop closes that hole.


def test_progress_throttle_200_rotating_labels_bounded_by_global_rate(monkeypatch):
    """Reproduces the coordinator's residual finding directly: with more
    distinct labels (200) than _MAX_TRACKED_PROGRESS_LABELS (32), the
    per-label cap evicts the entry every call needs, so the per-label rule
    alone no longer bounds anything (confirmed separately, against the
    pre-backstop module, to emit all 10,000 lines). The global backstop
    catches this regardless: with the clock frozen (the pathological case
    - no real time ever passes), at most _MAX_PROGRESS_LINES_PER_SECOND
    lines can ever be written, since the sliding window can never see any
    of them "age out".
    """
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    monkeypatch.setattr(_plain.time, "monotonic", lambda: 1000.0)

    n_labels = 200
    n_calls = 10000
    for i in range(n_calls):
        renderer.progress(f"label-{i % n_labels}", 0.5)

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == _MAX_PROGRESS_LINES_PER_SECOND
    assert len(lines) < 1000  # bounded by the rate, nowhere near the 10,000 calls made


def test_progress_throttle_single_label_cadence_unaffected_by_backstop_with_advancing_clock(
    monkeypatch,
):
    """Test (c): the global backstop must not throttle legitimate output
    when the clock genuinely advances, as opposed to the pathological
    "no time ever passes" case the previous test targets. Re-runs the
    single-label 0->1 sweep (normally 21 lines, pinned with a frozen clock
    by test_progress_throttle_bounded_over_10000_calls) with a clock that
    advances a little on every call instead of staying literally frozen;
    the increment is kept small enough that the whole sweep still
    completes well inside one second of (simulated) wall time, so the
    per-label rule's *own* time-based branch (already covered by
    test_progress_throttle_time_based_branch) is not what is under test
    here - only whether the *new* global backstop introduces any
    additional suppression once the clock is merely "not frozen". It must
    not: 21 emissions inside well under a second is nowhere near the
    100/sec cap.
    """
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    clock = [1000.0]
    monkeypatch.setattr(_plain.time, "monotonic", lambda: clock[0])

    for i in range(10001):
        clock[0] += 0.00005  # genuinely advancing, but the whole sweep
        # still finishes in ~0.5 simulated seconds: nowhere near this
        # label's own 1.0s per-label time step, so only the pct-based half
        # of the per-label rule is exercised, exactly as in the frozen-
        # clock version this test's expected count is compared against.
        renderer.progress("ingest", i / 10000)

    lines = [line for line in stream.getvalue().splitlines() if line]
    expected = 100 // _PROGRESS_PCT_STEP + 1
    assert len(lines) == expected


def test_progress_throttle_many_labels_paced_over_real_time_all_get_through(monkeypatch):
    """Test (c), the more targeted version: many distinct labels (more than
    both the per-label LRU cap and the global rate cap) reporting exactly
    once each, but paced so that no more than a small fraction of the
    global rate is used in any real 1-second window. None of these should
    be denied: the backstop targets *rate*, not label cardinality by
    itself, so a genuinely-paced burst of variety must pass through
    untouched even though a comparable *instantaneous* burst (the test
    above) is exactly what gets capped.
    """
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream)
    clock = [1000.0]
    monkeypatch.setattr(_plain.time, "monotonic", lambda: clock[0])

    n_labels = 150  # > _MAX_TRACKED_PROGRESS_LABELS and > the global rate
    for i in range(n_labels):
        clock[0] += 0.02  # ~50 calls/sec: comfortably under the 100/sec cap
        renderer.progress(f"label-{i}", 1.0)

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == n_labels


def test_progress_nan_fraction_clamps_to_zero_not_hundred():
    """Fix round 1, minor item: max(0.0, min(1.0, nan)) silently returns 1.0
    because every comparison against NaN is False (min/max never "sees" the
    NaN as smaller or larger than anything). progress() now intercepts NaN
    explicitly and treats it as 0%, since falsely reporting 100% complete is
    a worse failure than under-reporting."""
    stream = io.StringIO()
    PlainRenderer(CAPS0, stream).progress("ingest", float("nan"))
    (line,) = [ln for ln in stream.getvalue().splitlines() if ln]
    _, fields = _parse_line(line)
    assert fields["pct"] == "0"


# ── Finding 2 (fix round 1): a non-encodable value must not crash the command ──


def test_non_ascii_value_does_not_raise_on_ascii_stream():
    """A real encoding-constrained stream (a TextIOWrapper over BytesIO
    pinned to strict ASCII), matching a LANG=C stderr or some CI runners.
    Sequence identifiers from real AIRR files are not guaranteed ASCII,
    so a value like "café.tsv" must degrade the rendering, not raise
    UnicodeEncodeError and take down the command reporting it."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    renderer = PlainRenderer(CAPS0, stream)

    renderer.update(source="café.tsv")  # must not raise

    stream.flush()
    output = raw.getvalue().decode("ascii")
    assert "source=" in output
    assert "caf" in output
    assert "é" not in output  # the non-ASCII byte was not smuggled through


def test_non_ascii_value_degrades_with_a_replacement_marker():
    """More specific than the test above: pins *what* the degradation looks
    like (Python's "replace" error handler substitutes "?" per character
    on encode), not just that it avoids raising."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    PlainRenderer(CAPS0, stream).update(source="café.tsv")
    stream.flush()
    output = raw.getvalue().decode("ascii")
    assert "source=caf?.tsv" in output


def test_error_detail_with_non_ascii_value_does_not_raise():
    """The other write path (_emit_detail, used by error()'s detail lines)
    gets the same guard, not just _emit's."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    PlainRenderer(CAPS0, stream).error("lzg build", ["naïve Bayes column missing"])
    stream.flush()
    output = raw.getvalue().decode("ascii")
    assert "na" in output
    assert "ï" not in output


# ── Finding 3 (fix round 1): Ui is an enforceable, checked contract ──


def test_plain_renderer_satisfies_ui_protocol():
    assert isinstance(PlainRenderer(CAPS0, io.StringIO()), Ui)


def test_rich_renderer_satisfies_ui_protocol():
    """Coordinated with _rich.py rather than duplicating its expectations:
    imports the real RichRenderer and constructs it with a plain
    Capabilities instance and an injected stream. Construction alone does
    no I/O (no start() call), so this needs no real terminal."""
    assert isinstance(RichRenderer(_caps(colours=8), io.StringIO()), Ui)


_UI_METHOD_NAMES = sorted(
    name
    for name, obj in vars(Ui).items()
    if not name.startswith("_") and inspect.isfunction(obj)
)


def test_ui_protocol_exposes_the_six_documented_methods():
    """Sanity check on the introspection below: if this list ever drifted,
    the "covers every protocol method automatically" guarantee of the
    signature-comparison test would silently stop covering part of it."""
    assert set(_UI_METHOD_NAMES) == {"start", "progress", "update", "warn", "error", "done"}


def _param_shape(func) -> list[tuple[str, object, bool]]:
    """(name, kind, is_required) for each parameter, deliberately ignoring
    annotations: every module here uses ``from __future__ import
    annotations``, so comparing raw annotation strings across three
    separately-authored modules would be fragile to cosmetic spelling
    differences without catching any more real drift than name/kind/
    required-ness already does. A renamed, reordered, added, removed, or
    required-vs-optional-flipped parameter changes this shape; that is
    exactly the drift ``runtime_checkable`` alone (Finding 3) cannot catch,
    since it only checks that a same-named attribute exists.
    """
    return [
        (p.name, p.kind, p.default is inspect.Parameter.empty)
        for p in inspect.signature(func).parameters.values()
    ]


@pytest.mark.parametrize("method_name", _UI_METHOD_NAMES)
def test_renderer_signatures_match_the_ui_protocol(method_name):
    expected = _param_shape(getattr(Ui, method_name))
    assert _param_shape(getattr(PlainRenderer, method_name)) == expected
    assert _param_shape(getattr(RichRenderer, method_name)) == expected


# ── Reserved keys and field-name collisions (Task 6 fix round 1) ──
#
# Finding 1: cli.py's original cmd_build wiring passed a caller field
# literally named "status" alongside `phase`, colliding with this
# renderer's own unconditional `status=<word>` token. The formatted line
# then carried two `status=` tokens at different values (e.g. "[build]
# status=info phase=save status=start output=..."), which a key=value
# parser cannot recover both values from: whichever it reads, the other is
# silently discarded. Greppability is this whole layout's reason to exist,
# so this is a real defect, fixed in the renderer itself
# (`_disambiguate_fields`) rather than by trusting every future caller to
# avoid the word "status".


def _field_keys(line: str) -> list[str]:
    """Every ``key`` at a ``key=`` boundary in ``line``'s content (after its
    ``[tag]`` prefix), in order, *keeping duplicates* rather than collapsing
    them into a dict.

    Deliberately not ``_parse_line``: building a ``{key: value}`` dict from
    the raw matches (as ``_parse_line`` does, for every other test in this
    file) silently keeps only the *last* value for a repeated key, which is
    exactly the failure mode this section exists to catch, not paper over.
    This is "a real parser" in the same sense ``_parse_line`` already is
    (the same ``_FIELD_BOUNDARY_RE`` boundary regex the module docstring
    documents as the convention a script would use), just stopping one step
    earlier, before the lossy dict conversion.
    """
    plain = _strip_escapes(line)
    m = _TAG_RE.match(plain)
    assert m is not None, f"line has no [tag] prefix: {line!r}"
    return [fm.group(1) for fm in _FIELD_BOUNDARY_RE.finditer(m.group("rest"))]


def test_no_emitted_line_ever_contains_a_duplicate_key():
    """Property over the whole line set a realistic session produces,
    checked with the real field-boundary parser, not a check for the
    literal string ``"status"``: any future reserved-key collision this
    renderer's own line shape introduces would be caught the same way.
    """
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream=stream)
    _drive_all_methods(renderer)
    # The exact shape that used to collide: a caller field literally named
    # "status", passed alongside "phase", on both an update() and a done().
    renderer.update(phase="save", status="start", output="a.lzg")
    renderer.update(phase="save", status="done", output="a.lzg", size_kb=1.5)
    renderer.done("lzg build", {"status": "surprise", "output": "a.lzg"})

    lines = stream.getvalue().splitlines()
    assert lines  # sanity: _drive_all_methods really wrote something
    for line in lines:
        keys = _field_keys(line)
        assert len(keys) == len(set(keys)), f"duplicate key in line: {line!r}"


def test_caller_field_named_status_is_disambiguated_not_dropped():
    """The caller's colliding value is not merely deduplicated away: it
    survives, under a deterministically renamed key, alongside the
    renderer's own ``status=`` token.
    """
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream=stream)
    renderer.update(phase="save", status="start", output="a.lzg")

    tag, fields = _parse_line(stream.getvalue().splitlines()[0])
    assert fields["status"] == "info"  # update()'s own generic status word
    assert fields["status_"] == "start"  # the caller's field, disambiguated
    assert fields["phase"] == "save"
    assert fields["output"] == "a.lzg"


def test_two_independently_colliding_fields_in_one_call_are_both_kept():
    """A caller passing *both* ``status`` and (already) ``status_`` in the
    same call must not have the second clobber the first: each gets its own
    disambiguated name.
    """
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream=stream)
    renderer.update(status="first", status_="second")

    tag, fields = _parse_line(stream.getvalue().splitlines()[0])
    assert fields["status"] == "info"
    assert fields["status_"] == "first"
    assert fields["status__"] == "second"


def test_non_colliding_field_names_pass_through_unchanged():
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream=stream)
    renderer.update(phase="read", nodes=5)

    tag, fields = _parse_line(stream.getvalue().splitlines()[0])
    assert fields["phase"] == "read"
    assert fields["nodes"] == "5"


def test_elapsed_is_not_forced_through_disambiguation_on_a_plain_update():
    """``"elapsed"`` is deliberately not a reserved key (see ``_plain.py``'s
    module docstring): a plain ``update()`` never emits its own ``elapsed``
    token, so a caller's own honest ``elapsed=...`` field (``cli.py``
    reports one on nearly every phase) must keep reading exactly
    ``elapsed=...``, not get needlessly renamed to ``elapsed_=...``.
    """
    stream = io.StringIO()
    renderer = PlainRenderer(CAPS0, stream=stream)
    renderer.update(phase="read", elapsed="0.12s")

    tag, fields = _parse_line(stream.getvalue().splitlines()[0])
    assert fields["elapsed"] == "0.12s"
