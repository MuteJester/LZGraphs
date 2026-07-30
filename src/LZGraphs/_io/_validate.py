"""Validate an input file with the same content-sniffing pipeline ``build`` uses.

``validate_input`` used to run its own first-line-only classifier, which
routinely disagreed with the sniffing pipeline that ``read_sequences``/
``LZGraph.from_file`` actually build against: a FASTA file, for instance, was
reported as ``plain`` with the ``>`` header lines counted as records. This
module reimplements the same report contract on top of ``detect_format`` and
the shared readers (``iter_fasta``, ``iter_fastq``, ``iter_tabular_rows``,
``_is_wellformed``, ``_first_line_kind``, ``_resolve_column_override``,
``_materialize_stdin``) so a validated file is described exactly the way it
would be read.
"""
from __future__ import annotations

import os
from dataclasses import replace

from ._compress import open_text
from ._public import (
    _first_line_kind,
    _is_wellformed,
    _materialize_stdin,
    _resolve_column_override,
)
from ._readers import _is_count, iter_fasta, iter_fastq, iter_tabular_rows
from ._sniff import detect_format
from ._spec import FormatError

_VALID_EXPECT_FORMATS = {"plain", "plain_seqcount", "tabular"}
_PLAIN_FAMILY = {"plain", "plain_seqcount"}
_PRODUCTIVE_TRUE = {"t", "true", "1", "yes"}


def _family(kind: str) -> str:
    """Group a detected/declared kind into the shape ``expect_format`` asserts.

    ``plain`` and ``plain_seqcount`` are one family (their difference is a
    per-record detail handled as ``mode='mixed'``, not a format assertion
    failure); ``tabular``, ``fasta``, and ``fastq`` are each their own family.
    """
    if kind in _PLAIN_FAMILY:
        return "plain"
    return kind


def _new_report(path, detected_kind, *, variant, strict_input, expect_format):
    return {
        "path": path,
        "variant": variant,
        "detected_kind": detected_kind,
        "expect_format": expect_format,
        "strict_input": bool(strict_input),
        "ok": True,
        "mode": None,
        "total_lines": 0,
        "records": 0,
        "blank_lines": 0,
        "plain_records": 0,
        "seqcount_records": 0,
        "tabular_rows": 0,
        "warning_count": 0,
        "error_count": 0,
        "issues": [],
        "seq_column": None,
        "v_column": None,
        "j_column": None,
        "abundance_column": None,
        "has_header": False,
    }


def _add_issue(report, level, message, *, line=None):
    item = {"level": level, "message": message}
    if line is not None:
        item["line"] = int(line)
    report["issues"].append(item)
    if level == "error":
        report["error_count"] += 1
        report["ok"] = False
    else:
        report["warning_count"] += 1


def _summarize(report):
    status = "ok" if report["ok"] else "error"
    parts = [
        f"status={status}",
        f"kind={report['detected_kind']}",
        f"mode={report['mode']}",
        f"lines={report['total_lines']}",
        f"records={report['records']}",
        f"errors={report['error_count']}",
        f"warnings={report['warning_count']}",
    ]
    if report["expect_format"]:
        parts.append(f"expect={report['expect_format']}")
    return " ".join(parts)


class _CountingStream:
    """Wrap a text stream, tallying ``total_lines``/``blank_lines`` as they pass.

    Transparent to both iteration (``iter_fasta``, ``iter_plain``,
    ``csv.DictReader``) and ``.readline()`` (``iter_fastq``), so it can sit in
    front of any of the shared readers without changing their behaviour.
    """

    def __init__(self, stream, report):
        self._stream = stream
        self._report = report

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._stream)
        self._count(line)
        return line

    def readline(self, *args, **kwargs):
        line = self._stream.readline(*args, **kwargs)
        if line:
            self._count(line)
        return line

    def _count(self, line):
        self._report["total_lines"] += 1
        if not line.strip():
            self._report["blank_lines"] += 1


def _dispatch(path, report, fn, *args, **kwargs):
    """Open ``path``, run ``fn(counting_stream, report, *args, **kwargs)``, then close it.

    Every ``_validate_*`` handler below takes ``(stream, report, ...)`` in
    that order, so this can wrap any of them uniformly: it opens a fresh
    stream, wraps it in ``_CountingStream`` so ``total_lines``/``blank_lines``
    are tallied regardless of which handler runs, and always closes the real
    underlying stream afterward.
    """
    stream, _codec = open_text(path)
    try:
        fn(_CountingStream(stream, report), report, *args, **kwargs)
    finally:
        stream.close()


def _validate_tabular(stream, report, spec, *, strict_input, no_genes):
    report["mode"] = "tabular"
    report["has_header"] = bool(spec.header)
    report["seq_column"] = spec.seq_column
    report["v_column"] = None if no_genes else spec.v_column
    report["j_column"] = None if no_genes else spec.j_column
    report["abundance_column"] = spec.abundance_column

    try:
        for sequence, _abundance, _v_call, _j_call, productive in iter_tabular_rows(
            stream, spec
        ):
            # `stream` is the `_CountingStream` `_dispatch` wrapped this call in, so
            # by the time `iter_tabular_rows`'s internal `csv.DictReader` has yielded
            # a row, `total_lines` already reflects that row's true file line (the
            # header consumed first counts as line 1, matching every other reader
            # here that starts counting from the top of the file). Verified directly
            # against a file with a known bad row rather than assumed: header=1,
            # first data row=2, and so on -- this also stays correct for a row whose
            # field embeds a quoted newline, since csv.reader calls next() on the
            # underlying iterable exactly as many times as physical lines that row
            # spans, and each such call increments this same counter.
            line_no = report["total_lines"]
            if productive is not None and productive.strip().lower() not in _PRODUCTIVE_TRUE:
                continue
            if not _is_wellformed(sequence):
                level = "error" if strict_input else "warning"
                _add_issue(report, level, f"not a usable sequence: {sequence!r}", line=line_no)
                continue
            report["tabular_rows"] += 1
            report["records"] += 1
    except FormatError as exc:
        _add_issue(report, "error", str(exc))

    if report["records"] == 0 and report["error_count"] == 0:
        _add_issue(report, "error", "tabular input contains no sequence records")


def _validate_plain_family(stream, report, *, strict_input, first_mode):
    report["mode"] = first_mode
    warned_mixed = False

    for raw in stream:
        # `stream` is the `_CountingStream` wrapping this call, so `total_lines`
        # has already been incremented for `raw` by the time this loop body runs
        # (its `__next__` counts, then returns) -- this is the true 1-based file
        # line, counting blank lines too, not a record ordinal.
        line_no = report["total_lines"]
        line = raw.strip()
        if not line:
            continue

        parts = line.split("\t")
        sequence = parts[0].strip()
        has_count_field = len(parts) > 1
        kind = "plain_seqcount" if (has_count_field and sequence) else "plain"

        malformed = None
        if not sequence:
            malformed = "empty sequence"
        elif not _is_wellformed(sequence):
            malformed = f"not a usable sequence: {sequence!r}"
        elif has_count_field:
            count_text = parts[1].strip()
            if count_text and not _is_count(count_text):
                malformed = f"invalid abundance {count_text!r}"

        if malformed is not None:
            level = "error" if strict_input else "warning"
            _add_issue(report, level, malformed, line=line_no)
            continue

        if kind != first_mode:
            report["mode"] = "mixed"
            if strict_input:
                _add_issue(
                    report, "error", f"mixed {first_mode!r} and {kind!r} records",
                    line=line_no,
                )
                continue
            if not warned_mixed:
                _add_issue(
                    report, "warning", f"mixed {first_mode!r} and {kind!r} records",
                    line=line_no,
                )
                warned_mixed = True

        if kind == "plain":
            report["plain_records"] += 1
        else:
            report["seqcount_records"] += 1
        report["records"] += 1

    if report["records"] == 0 and report["error_count"] == 0:
        _add_issue(report, "error", "input contains no sequence records")


def _validate_reader(stream, report, *, strict_input, reader_fn, kind_label):
    """Validate FASTA/FASTQ, whose records span several physical lines.

    Unlike the line-oriented plain-family/tabular handlers, a record here has
    no single line number worth reporting: `iter_fasta`/`iter_fastq` only
    yield the assembled sequence once a record is complete, and recovering
    the physical line its header started on would mean re-tracking each
    reader's own line consumption from the outside (duplicating logic this
    module deliberately reuses rather than reimplements). So `line` is left
    unset here, and the record's 1-based position in the file is folded into
    the message text instead -- a real number, just not one labelled `line`.
    """
    report["mode"] = kind_label
    record_index = 0
    try:
        for sequence in reader_fn(stream):
            record_index += 1
            if not _is_wellformed(sequence):
                level = "error" if strict_input else "warning"
                _add_issue(
                    report, level,
                    f"record {record_index}: not a usable sequence: {sequence!r}",
                )
                continue
            report["records"] += 1
    except FormatError as exc:
        _add_issue(report, "error", str(exc))

    if report["records"] == 0 and report["error_count"] == 0:
        _add_issue(report, "error", f"{kind_label} input contains no sequence records")


def _apply_column_overrides(spec, v_column, j_column, abundance_column):
    if not spec.header:
        return spec
    return replace(
        spec,
        v_column=_resolve_column_override(spec.header, v_column) or spec.v_column,
        j_column=_resolve_column_override(spec.header, j_column) or spec.j_column,
        abundance_column=(
            _resolve_column_override(spec.header, abundance_column) or spec.abundance_column
        ),
    )


def _validate_real_path(path, *, seq_column, v_column, j_column, abundance_column,
                        variant, no_genes, strict_input, expect_format):
    """The actual validation, always against a real (reopenable) filesystem path."""
    spec = None
    sniff_error = None
    try:
        spec = detect_format(path, variant=variant, seq_column=seq_column)
        detected_kind = spec.format
    except FormatError as exc:
        sniff_error = exc
        try:
            detected_kind = _first_line_kind(path)
        except UnicodeDecodeError:
            detected_kind = "binary"

    report = _new_report(
        path, detected_kind, variant=variant,
        strict_input=strict_input, expect_format=expect_format,
    )

    # An empty file, binary content, or a tabular file whose sequence column
    # detect_format could not resolve (e.g. a bad --seq-column, or headers it
    # cannot recognise) all end the same way: one clear error, no records.
    # There is nothing left to read in any of these cases -- for the tabular
    # one specifically, `spec` is None so there is no resolved delimiter/
    # column to read with, and retrying would just raise the same error again.
    if detected_kind in ("empty", "binary") or (spec is None and detected_kind == "tabular"):
        _add_issue(report, "error", str(sniff_error) if sniff_error else "empty input")
        report["summary"] = _summarize(report)
        return report

    if expect_format is not None and _family(detected_kind) != _family(expect_format):
        _add_issue(
            report, "error",
            f"expected format '{expect_format}', saw '{detected_kind}'",
        )
        report["summary"] = _summarize(report)
        return report

    if detected_kind == "tabular":
        spec = _apply_column_overrides(spec, v_column, j_column, abundance_column)
        _dispatch(
            path, report, _validate_tabular, spec,
            strict_input=strict_input, no_genes=no_genes,
        )
    elif detected_kind in _PLAIN_FAMILY:
        first_mode = expect_format if expect_format in _PLAIN_FAMILY else detected_kind
        _dispatch(
            path, report, _validate_plain_family,
            strict_input=strict_input, first_mode=first_mode,
        )
    elif detected_kind == "fastq":
        _dispatch(
            path, report, _validate_reader,
            strict_input=strict_input, reader_fn=iter_fastq, kind_label="fastq",
        )
    else:  # fasta
        _dispatch(
            path, report, _validate_reader,
            strict_input=strict_input, reader_fn=iter_fasta, kind_label="fasta",
        )

    report["summary"] = _summarize(report)
    return report


def validate_input(path, *, seq_column=None, v_column=None, j_column=None,
                   abundance_column=None, variant='aap', no_genes=False,
                   strict_input=False, expect_format=None):
    """Validate an input file without building a graph.

    Built on the same ``detect_format`` sniffing pipeline as ``read_sequences``
    and ``LZGraph.from_file``, so ``detected_kind`` and ``records`` describe
    what a build would actually see instead of a first-line-only guess.

    ``expect_format`` acts as an assertion: if the file's sniffed format does
    not agree with it (comparing ``tabular`` vs. the ``plain``/
    ``plain_seqcount`` family vs. ``fasta``/``fastq``), that mismatch is
    reported as an error rather than silently coerced.
    """
    if expect_format is not None and expect_format not in _VALID_EXPECT_FORMATS:
        raise ValueError(
            f"expect_format must be one of {sorted(_VALID_EXPECT_FORMATS)}, "
            f"got '{expect_format}'"
        )

    tmp_path = _materialize_stdin() if path == "-" else None
    real_path = tmp_path if tmp_path is not None else path
    try:
        report = _validate_real_path(
            real_path, seq_column=seq_column, v_column=v_column, j_column=j_column,
            abundance_column=abundance_column, variant=variant, no_genes=no_genes,
            strict_input=strict_input, expect_format=expect_format,
        )
    finally:
        if tmp_path is not None:
            os.remove(tmp_path)
    report["path"] = path
    return report
