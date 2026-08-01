"""Public reading API, built on the sniffing pipeline.

Wires the historical ``read_sequences`` / ``read_sequences_simple`` /
``detect_input_kind`` surface (unchanged signatures, consumed by ``cli.py``
and ``_graph.py``) onto the content-sniffing pipeline from ``_sniff.py`` and
the streaming readers in ``_readers.py``.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import replace

from ._compress import open_text
from ._readers import (
    _is_count,
    _parse_count,
    iter_fasta,
    iter_fastq,
    iter_plain,
    iter_seqcount,
    iter_tabular_rows,
)
from ._sniff import _SNIFF_LINES, detect_format
from ._spec import VALID_FORMATS, FormatError, InputSpec, RecordStats, format_family

_PRODUCTIVE_TRUE = {"t", "true", "1", "yes"}

#: How often (in records read) ``_read_sequences_from_path`` calls a given
#: ``progress_cb`` (see ``read_sequences``). Cheap enough per call (one
#: modulo check) that this could be 1, but there is no reporting value in
#: calling back more often than a UI could ever throttle down to anyway;
#: every real ``Ui.progress()`` implementation already paces its own
#: output (see ``LZGraphs._term``), so this just avoids the pure overhead
#: of an unconditional per-record callback on a multi-hundred-thousand-
#: record file.
_PROGRESS_RECORD_STEP = 2000

# Real AIRR amino-acid data carries '*' for a stop codon and '-' or '.' for
# an alignment gap. Rejecting those silently discards exactly the
# non-productive rows that keep_nonproductive=True exists to preserve.
# Everything else stays rejected, which is what keeps FASTA headers ('>'),
# delimited rows (',' and tab), FASTQ markers ('@', '+') and leaked numeric
# counts out of graphs.
_EXTRA_RESIDUE_CHARACTERS = frozenset("*-.")


def _is_wellformed(sequence: str) -> bool:
    """A usable sequence has at least one residue, and no foreign characters.

    The residue marks are permitted because real AIRR amino-acid data carries
    '*' for a stop codon and '-' or '.' for an alignment gap, and rejecting
    those discards exactly the non-productive rows that keep_nonproductive
    exists to preserve. Requiring at least one letter is what stops the marks
    alone from qualifying: '-' and '.' are the usual missing-value sentinels
    in delimited exports and are not sequences.
    """
    if not sequence:
        return False
    if not any(character.isalpha() for character in sequence):
        return False
    return all(
        character.isalpha() or character in _EXTRA_RESIDUE_CHARACTERS
        for character in sequence
    )


_STREAM_PREFIX_BYTES = 8192
_UTF8_BOM = b"\xef\xbb\xbf"


def raw_prefix_is_streamable(path: str, fmt: str) -> bool:
    """True when ``path``'s raw bytes are safe for the C streaming fast path.

    ``cli.py``'s ``cmd_build`` has a fast path (``LZGraph.from_file``) that
    streams a file's bytes straight into the C builder for ``plain``/
    ``plain_seqcount`` input, bypassing everything ``read_sequences`` does to
    normalise a file first: ``utf-8-sig`` BOM stripping, universal-newline
    handling, and ``_is_wellformed`` filtering. That is fine for a genuinely
    clean file, but silently wrong for one that needs any of that
    normalisation, since the C reader has none of it. This inspects a prefix
    of the file's real bytes (not through ``open_text``, which would
    transparently strip a BOM and translate newlines before this could see
    either) and declines whenever the fast path would disagree with
    ``read_sequences``:

    - a leading UTF-8 BOM: ``open_text``'s ``utf-8-sig`` strips it; the C
      reader does not, so it ends up glued onto the first sequence,
    - a lone carriage return (a ``\\r`` not part of a ``\\r\\n`` pair):
      universal-newline handling turns this into a line break; the C reader
      treats a bare ``\\r`` as an ordinary character, collapsing every
      "line" it should have separated into one sequence,
    - any meaningful line within the same bounded prefix ``detect_format``
      already sniffs (``_SNIFF_LINES``, currently 32) that is not itself a
      usable sequence (e.g. a column-header name sitting alone on the first
      line, or a malformed record a few lines further down):
      ``read_sequences`` drops each such record via ``_is_wellformed``; the
      C reader has no such check and ingests every one of them as real data.

    For ``plain_seqcount``, only the sequence field (before the tab) of each
    checked line is tested, since a genuine ``sequence<TAB>count`` line
    always contains a non-alphabetic tab and count that would otherwise
    always fail the check on its own.

    This validates only the sniff prefix, not the whole file: a malformed
    record beyond line ``_SNIFF_LINES`` (e.g. line 10000 of a
    foundation-scale file) still reaches the fast path and is ingested
    rather than dropped, because catching it here would mean reading the
    whole file up front and defeat the constant-memory streaming this path
    exists to provide in the first place. That residual gap is deliberate,
    not an oversight: compressed input (which never reaches this gate at
    all, see ``can_stream_plain`` in ``cli.py``) and ``--strict-input``
    (which runs ``validate_input`` over the entire file before ``cmd_build``
    ever streams) both already give a user who needs full per-record
    validation a way to get it.

    Cheap by construction: opens ``path`` once and reads at most
    ``_STREAM_PREFIX_BYTES`` bytes, checking only the first ``_SNIFF_LINES``
    of them, since this runs on every ``lzg build``. Callers must only pass
    this a real (uncompressed) filesystem path already known to be
    ``plain``/``plain_seqcount``; it is not a general-purpose format
    detector.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(_STREAM_PREFIX_BYTES)
    except OSError:
        return False

    if head.startswith(_UTF8_BOM):
        return False

    # Any '\r' surviving a '\r\n' -> '\n' collapse is a lone carriage return.
    if head.replace(b"\r\n", b"\n").count(b"\r"):
        return False

    lines = head.decode("utf-8", errors="replace").splitlines()[:_SNIFF_LINES]
    meaningful = [stripped for line in lines if (stripped := line.strip())]
    if not meaningful:
        return False

    if fmt == "plain_seqcount":
        meaningful = [line.split("\t", 1)[0].strip() for line in meaningful]

    return all(_is_wellformed(line) for line in meaningful)


def plan_streaming_read(path, *, variant="aap", seq_column=None, expect_format=None):
    """Decide whether ``path`` may take the constant-memory C streaming path.

    This is the single implementation of the fast-path routing decision
    shared by ``cli.py``'s ``cmd_build``, ``LZGraph.from_file``, and
    ``FlashBackGraph.from_file``. All three used to make this decision
    independently (``cmd_build`` correctly; the two ``from_file`` methods
    not at all, streaming raw bytes through the C reader unconditionally),
    which is how a user calling ``LZGraph.from_file`` directly instead of
    going through ``lzg build`` could silently get a different, wrong graph
    for the same file. The rule itself is unchanged from ``cmd_build``'s
    original gate: stream only when the file is uncompressed, detected as
    ``plain``/``plain_seqcount``, and ``raw_prefix_is_streamable`` confirms
    its sniffed prefix has no leading BOM, no lone carriage return, and no
    line that is not itself a well-formed sequence. Everything else must be
    read through ``detect_format`` plus the readers (``read_sequences``),
    which is what a ``False`` return tells the caller to do.

    ``expect_format``, when given, is passed straight through as
    ``detect_format``'s ``override``: a coercion, not an assertion, exactly
    as ``cmd_build`` originally did it. A user-declared format still gets a
    chance at the fast path even when content sniffing alone would be
    ambiguous, because ``raw_prefix_is_streamable`` independently confirms
    the coerced interpretation actually produces well-formed lines before
    this can return ``True``. This function never *asserts* that
    ``expect_format`` matches the real content; a caller that also wants
    that (``LZGraph.from_file``'s ``strict_input``/``expect_format``
    contract) gets it from ``validate_input``/``read_sequences`` instead,
    exactly as ``cmd_build`` already runs ``validate_input`` alongside this
    for that purpose.

    Returns ``(can_stream, spec)``. ``spec`` is the resolved
    :class:`InputSpec` whenever detection succeeds, even when ``can_stream``
    is ``False`` (e.g. the format is ``tabular``), so a caller that also
    wants ``spec.alphabet`` for a warning never needs a second
    ``detect_format`` call. ``spec`` is ``None``, and ``can_stream`` is
    always ``False``, when ``path`` is ``"-"`` (stdin is a single
    non-seekable pipe, never safe for the raw-byte fast path, which reopens
    its input from byte zero) or when detection itself fails (empty,
    binary, or otherwise unclassifiable content); in either case the real
    error, if any, is left to surface from the safe ``read_sequences``
    branch instead of raised here.
    """
    if path == "-":
        return False, None
    try:
        spec = detect_format(
            path, variant=variant, seq_column=seq_column, override=expect_format,
        )
    except FormatError:
        return False, None
    can_stream = (
        spec.compression == "none"
        and spec.format in ("plain", "plain_seqcount")
        and raw_prefix_is_streamable(path, spec.format)
    )
    return can_stream, spec


def empty_read_error(path, stats: RecordStats) -> ValueError:
    """The explanatory error raised when a read kept zero records.

    Shared by ``cmd_build``, ``LZGraph.from_file``, and
    ``FlashBackGraph.from_file`` so a file that drops every record (e.g. an
    AIRR file with a blank ``productive`` column) fails the same, specific
    way through every entry point, instead of falling through to
    ``LZGraph.__init__``'s bare ``sequences must be a non-empty list``,
    which names neither how many records were read nor why none survived.
    """
    return ValueError(
        f"{path}: read {stats.total} record(s), 0 kept "
        f"(malformed={stats.malformed}, nonproductive={stats.nonproductive}); "
        "nothing left to build a graph from"
    )


def _iter_plain_strict(stream):
    """Yield ``(sequence, count)`` from plain/plain_seqcount lines, raising if
    the file mixes both record shapes.

    ``detect_format`` classifies a whole file up front (unlike the original
    per-line approach), so plain 5-tuple reading never sees a "mixed" file to
    reject. This narrow helper is the one piece of strict-mode behaviour
    ``read_sequences`` still has to perform itself: without it,
    ``tests/test_io.py::test_read_sequences_strict_plain_seqcount_fails_on_mixed``
    regresses, because forcing the format via ``override`` (see
    ``read_sequences`` below) makes ``iter_seqcount`` read straight through a
    line that has no count field, silently defaulting it to 1 instead of
    raising. This is deliberately NOT the full malformed-record accounting
    Task 12 restores (see the module docstring / task report for why).
    """
    first_kind = None
    for raw in stream:
        line = raw.rstrip()
        if not line.strip():
            continue
        parts = line.split("\t")
        sequence = parts[0].strip() if parts else ""
        if not sequence:
            continue
        if len(parts) > 1:
            kind = "plain_seqcount"
            count = _parse_count(parts[1])
        else:
            kind = "plain"
            count = 1
        if first_kind is None:
            first_kind = kind
        elif kind != first_kind:
            raise FormatError(f"mixed {first_kind!r} and {kind!r} records")
        yield sequence, count


def _iter_records(stream, spec: InputSpec, *, strict_input: bool = False):
    """Normalise any detected format to a common 5-tuple shape.

    Yields ``(sequence, abundance, v_call, j_call, productive)`` so callers
    never have to branch on ``spec.format`` themselves. ``productive`` is
    only ever populated for tabular input (``iter_tabular_rows`` already
    returns five elements); every other format yields ``None`` for it.
    """
    if spec.format == "tabular":
        yield from iter_tabular_rows(stream, spec)
        return

    if spec.format in ("plain", "plain_seqcount"):
        if strict_input:
            pairs = _iter_plain_strict(stream)
        elif spec.format == "plain_seqcount":
            pairs = iter_seqcount(stream)
        else:
            pairs = ((sequence, 1) for sequence in iter_plain(stream))
        for sequence, abundance in pairs:
            yield sequence, abundance, None, None, None
        return

    source = iter_fastq(stream) if spec.format == "fastq" else iter_fasta(stream)
    for sequence in source:
        yield sequence, 1, None, None, None


def _materialize_stdin() -> str:
    """Buffer stdin to a temp file so it can be sniffed and then re-read.

    ``detect_format`` peeks at up to 32 lines and then closes its stream,
    trusting that ``path`` can be reopened from byte zero afterwards
    (``read_sequences`` does exactly that, via its own later ``open_text``
    call). That assumption holds for any real file on disk (reopening it
    just rereads the same bytes), but not for stdin, which is a single
    non-seekable pipe: whatever the sniff peek consumes is gone for good, so
    a naive "detect_format('-') then open_text('-') again" sequence silently
    drops every line the peek read (verified: piping 2 short lines through
    read_sequences('-') without this fix yields zero sequences). Buffering
    to a temp file gives stdin the same two-pass behaviour every other path
    already gets, without touching _compress.py or _sniff.py.
    """
    fd, path = tempfile.mkstemp(prefix="lzgraphs_stdin_")
    try:
        with os.fdopen(fd, "wb") as tmp:
            shutil.copyfileobj(sys.stdin.buffer, tmp)
    except BaseException:
        os.remove(path)
        raise
    return path


def _resolve_column_override(header: tuple, requested: str | None) -> str | None:
    """Case-insensitively look up an explicit column name in ``header``.

    Returns ``None`` (never raises) when ``requested`` is absent from
    ``header``: mirrors the previous fail-soft handling of v/j/abundance column
    overrides, where an explicit column that doesn't exist in this particular
    file just means "no such data here", unlike ``seq_column`` (resolved by
    ``detect_format`` itself), which raises when named explicitly but missing.
    """
    if requested is None:
        return None
    lowered = {h.lower().strip(): h for h in header}
    return lowered.get(requested.strip().lower())


def _apply_column_overrides(spec: InputSpec, v_column, j_column, abundance_column) -> InputSpec:
    """Apply explicit v/j/abundance column overrides onto a resolved ``spec``.

    Explicit v/j/abundance column overrides only apply to tabular input, and
    only take effect when the named column actually exists in this file's
    header; otherwise the auto-detected column (if any) stands. Shared by
    ``read_sequences`` (via ``_read_sequences_from_path``) and
    ``validate_input`` (via ``_validate.py``'s ``_validate_real_path``) so
    build and validate can never resolve a different override for the same
    file: this used to be implemented twice, once per module.
    """
    if spec.format != "tabular" or not spec.header:
        return spec
    return replace(
        spec,
        v_column=_resolve_column_override(spec.header, v_column) or spec.v_column,
        j_column=_resolve_column_override(spec.header, j_column) or spec.j_column,
        abundance_column=(
            _resolve_column_override(spec.header, abundance_column)
            or spec.abundance_column
        ),
    )


def read_sequences(path, seq_column=None, v_column=None, j_column=None,
                   abundance_column=None, variant='aap', no_genes=False,
                   strict_input=False, expect_format=None, *,
                   keep_nonproductive=False, progress_cb=None):
    """Read sequences from a file, auto-detecting format.

    Returns a dict with ``sequences``, ``abundances``, ``v_genes``,
    ``j_genes``, and ``stats`` (a :class:`~LZGraphs._io._spec.RecordStats`).

    ``strict_input=True`` rejects mixed plain/plain_seqcount records (see
    ``_iter_plain_strict``) and also raises :class:`FormatError` on the
    first malformed record instead of skipping it. AIRR rows whose
    ``productive`` column is not truthy are dropped and counted unless
    ``keep_nonproductive=True``.

    ``expect_format``, like ``validate_input``'s parameter of the same name,
    is an assertion: the file's real content is auto-detected first, and a
    mismatch between what was declared and what was actually found raises
    :class:`FormatError` naming both, instead of forcing the declared reader
    onto content it does not describe (see ``_detect_format_for_read``). To
    force a classification onto content regardless of whether it matches
    (coercion), call ``detect_format(path, override=...)`` directly instead.

    ``progress_cb``, when given, is called periodically (every
    :data:`_PROGRESS_RECORD_STEP` records, plus once more at ``1.0`` when
    the read finishes) with a fraction in ``[0.0, 1.0]`` estimating how
    much of ``path`` has been consumed so far, so a caller (``cli.py``'s
    ``cmd_build``) can drive a real progress bar for this in-memory read
    path, not just the C library's own streaming ingest. See
    ``_read_sequences_from_path``'s docstring for exactly what this
    estimate is based on and the one case (compressed input) it does not
    cover. Never raises on the caller's behalf: any exception the callback
    itself raises is swallowed, since a reporting failure must not take
    down the read it is merely narrating.
    """
    tmp_path = _materialize_stdin() if path == "-" else None
    real_path = tmp_path if tmp_path is not None else path
    try:
        return _read_sequences_from_path(
            real_path, seq_column=seq_column, v_column=v_column, j_column=j_column,
            abundance_column=abundance_column, variant=variant, no_genes=no_genes,
            strict_input=strict_input, expect_format=expect_format,
            keep_nonproductive=keep_nonproductive, progress_cb=progress_cb,
        )
    finally:
        if tmp_path is not None:
            os.remove(tmp_path)


def _sniff_with_fallback(path, *, variant, seq_column):
    """Classify ``path``, falling back to the legacy first-line classifier.

    Shared by ``read_sequences``'s ``_detect_format_for_read`` and
    ``validate_input``'s ``_validate_real_path`` (see ``_validate.py``): both
    need to attempt ``detect_format`` and, if it raises, fall back to the
    same best-effort ``_first_line_kind`` classification before deciding
    what to do next. That fallback dance used to be duplicated between the
    two, byte for byte.

    Returns ``(spec, detected_kind, sniff_error, terminal)``:

    - ``spec`` is the resolved :class:`InputSpec`, or ``None`` when
      ``detect_format`` raised.
    - ``detected_kind`` is always populated: ``spec.format`` on success, or
      the ``_first_line_kind`` classification (or ``'binary'`` if even that
      cannot decode the file) on failure.
    - ``sniff_error`` is the :class:`FormatError` ``detect_format`` raised,
      or ``None`` when it succeeded.
    - ``terminal`` is ``True`` for input neither caller can do anything more
      with: empty, binary, or a tabular file whose sequence column could not
      be resolved at all (``spec is None`` for a ``tabular`` classification).
      Both callers stop identically on ``terminal``; they diverge only in
      how they surface that (raise vs. report an issue), which is left to
      each, along with the mismatch check and everything after, since a
      declared-vs-detected mismatch is handled differently by each
      (an assertion failure vs. a report entry) and only ``read_sequences``
      ever needs a *usable* spec back (retrying with ``override=`` when the
      sniff itself failed but the fallback classification still agreed with
      what was declared).
    """
    try:
        spec = detect_format(path, variant=variant, seq_column=seq_column)
        detected_kind = spec.format
        sniff_error = None
    except FormatError as exc:
        spec = None
        sniff_error = exc
        try:
            detected_kind = _first_line_kind(path)
        except UnicodeDecodeError:
            detected_kind = "binary"

    terminal = detected_kind in ("empty", "binary") or (
        spec is None and detected_kind == "tabular"
    )
    return spec, detected_kind, sniff_error, terminal


def _detect_format_for_read(path, *, variant, seq_column, expect_format):
    """Resolve an ``InputSpec`` for ``path``, asserting ``expect_format`` if given.

    ``expect_format`` used to be threaded straight into ``detect_format``'s
    ``override``, which *coerces* the classification unconditionally, even
    onto content that plainly is not that format: a tabular file declared
    ``expect_format="plain"`` was read line-for-line as plain text, so its
    header and every row failed ``_is_wellformed`` (delimiters and gene
    calls are not bare sequences) and ``read_sequences`` silently returned an
    empty list, no error, no warning. This instead auto-detects the file's
    real content first and raises :class:`FormatError` naming both the
    declared and detected format on a genuine mismatch, matching
    ``validate_input``'s existing contract for the same argument (see its
    ``_validate_real_path``, which this mirrors).

    When the auto-detector cannot resolve a spec on its own (``detect_format``
    raises, e.g. for a plain/plain_seqcount file whose first line happens to
    look like an unresolvable single-column tabular header),
    ``_first_line_kind`` supplies the same best-effort classification
    ``validate_input`` uses for its own assertion check. Only once that
    check has passed is ``detect_format`` retried with
    ``override=expect_format`` to obtain a spec actually usable for reading:
    the one place this still uses coercion, and only after the content has
    already been shown to belong to the declared family.

    Fundamentally broken input (empty, binary, or a tabular file whose
    sequence column could not be resolved at all) reports its own specific
    error rather than a generic mismatch message, again mirroring
    ``_validate_real_path``: the two must agree here, or the same file could
    pass one assertion path and fail the other with a misleading message.
    """
    if expect_format is None:
        return detect_format(path, variant=variant, seq_column=seq_column)

    if expect_format not in VALID_FORMATS:
        raise ValueError(
            f"expect_format must be one of {sorted(VALID_FORMATS)}, "
            f"got '{expect_format}'"
        )

    spec, detected_kind, sniff_error, terminal = _sniff_with_fallback(
        path, variant=variant, seq_column=seq_column
    )

    if terminal:
        raise sniff_error if sniff_error is not None else FormatError(f"{path} is empty")

    if format_family(detected_kind) != format_family(expect_format):
        raise FormatError(
            f"{path}: declared format {expect_format!r} does not match "
            f"detected format {detected_kind!r}"
        )

    if spec is not None:
        return spec
    return detect_format(
        path, variant=variant, seq_column=seq_column, override=expect_format
    )


def _read_sequences_from_path(path, *, seq_column, v_column, j_column,
                              abundance_column, variant, no_genes,
                              strict_input, expect_format,
                              keep_nonproductive=False, progress_cb=None):
    """The actual read, always against a real (reopenable) filesystem path.

    ``progress_cb`` (D5, fix round 2): called every
    :data:`_PROGRESS_RECORD_STEP` records with an estimated completion
    fraction, plus once more at exactly ``1.0`` when the read finishes (in
    the ``finally`` below, so it fires even on an early raise). The
    estimate is ``stream.tell() / os.path.getsize(path)``: accurate for
    genuinely uncompressed content, where the text stream is a thin wrapper
    directly over the file's own bytes, but *not* attempted at all for
    compressed input (``codec != "none"``), where the decompressed text
    stream's position bears no reliable relationship to the compressed
    file's size on disk (it can already exceed it well before the read is
    anywhere near done), so reporting a bytes-based fraction there would be
    actively misleading rather than merely absent; this simply never
    calls back in that case. A caller wanting a bar for the compressed case
    would need ``open_text``'s raw (pre-decompression) handle threaded
    through here too, a larger change than this pass makes; see the Task 6
    fix-round-2 report for the reviewer note on exactly this limitation.
    """
    spec = _detect_format_for_read(
        path, variant=variant, seq_column=seq_column, expect_format=expect_format,
    )

    spec = _apply_column_overrides(spec, v_column, j_column, abundance_column)

    # Gene data is structural, not per-row: only tabular input ever has a
    # v_column/j_column to read from. Building v_genes/j_genes as lists here
    # only when that column actually exists (and no_genes isn't set) keeps
    # them either None (no gene data, matching _graph.py's "None = no gene
    # data" contract) or exactly len(sequences) long, appending '' for a
    # row with no value rather than skipping it, so the list never desyncs
    # from sequences. An empty-but-non-None list would silently break
    # LZGraph construction: the C extension only checks v_genes/j_genes for
    # NULL, not for length, so a shorter-than-sequences list reads past its
    # own end.
    want_v = spec.format == "tabular" and spec.v_column is not None and not no_genes
    want_j = spec.format == "tabular" and spec.j_column is not None and not no_genes

    sequences: list = []
    abundances: list = []
    v_genes: list | None = [] if want_v else None
    j_genes: list | None = [] if want_j else None
    total = malformed = nonproductive = 0

    stream, codec = open_text(path)
    total_bytes = 0
    if progress_cb is not None and codec == "none":
        try:
            total_bytes = os.path.getsize(path)
        except OSError:
            total_bytes = 0
    try:
        for sequence, abundance, v_call, j_call, productive in _iter_records(
            stream, spec, strict_input=strict_input
        ):
            total += 1
            if total_bytes and total % _PROGRESS_RECORD_STEP == 0:
                try:
                    progress_cb(min(1.0, stream.tell() / total_bytes))
                except Exception:
                    pass
            if (
                productive is not None
                and not keep_nonproductive
                and productive.strip().lower() not in _PRODUCTIVE_TRUE
            ):
                nonproductive += 1
                continue
            if not _is_wellformed(sequence):
                if strict_input:
                    raise FormatError(
                        f"{path}: record {total} is not a usable sequence: "
                        f"{sequence!r}\n"
                        "  drop --strict to skip records like this instead"
                    )
                malformed += 1
                continue
            sequences.append(sequence)
            abundances.append(abundance)
            if want_v:
                v_genes.append(v_call or '')
            if want_j:
                j_genes.append(j_call or '')
    finally:
        if total_bytes:
            try:
                progress_cb(1.0)
            except Exception:
                pass
        # open_text documents its stream as always safe to close, including
        # for stdin (wrapped with closefd=False), so this never special-cases
        # path == "-".
        stream.close()

    return {
        'sequences': sequences,
        'abundances': abundances,
        'v_genes': v_genes,
        'j_genes': j_genes,
        'stats': RecordStats(
            total=total, kept=len(sequences),
            malformed=malformed, nonproductive=nonproductive,
        ),
    }


def read_sequences_simple(path, seq_column=None, variant='aap'):
    """Read just sequences (no genes/abundances). For score, decompose, etc."""
    return read_sequences(
        path, seq_column=seq_column, variant=variant, no_genes=True
    )['sequences']


def _first_line_kind(path: str) -> str:
    """Legacy-style classification from the first line alone.

    Used only as a fallback (see ``detect_input_kind``) when the richer
    sniffer raises. Mirrors the legacy first-line detection approach exactly,
    including reusing ``_is_count`` so this can never drift from what
    ``looks_like_seqcount``/``_parse_count`` consider a genuine count.
    """
    stream, _codec = open_text(path)
    try:
        first_line = stream.readline()
    finally:
        stream.close()
    stripped = first_line.strip()
    if not stripped:
        return 'empty'
    parts = stripped.split('\t')
    if len(parts) == 2 and parts[0].strip() and _is_count(parts[1]):
        return 'plain_seqcount'
    if '\t' in first_line or ',' in first_line:
        return 'tabular'
    return 'plain'


def detect_input_kind(path, variant='aap'):
    """Classify a sequence input file from its content.

    ``detect_format`` classifies more thoroughly than the legacy first-line
    sniff (it can, e.g., look far enough ahead to notice a tabular file has
    no resolvable sequence column, or that a file is empty/binary/not valid
    UTF-8), and raises ``FormatError`` when it does. On that error, this
    falls back to ``_first_line_kind``, which covers most such cases (e.g.
    empty files, files that are actually tabular) using the legacy first-line
    approach. That fallback reopens and rereads the file itself rather than
    reusing ``detect_format``'s result, so it does not catch every error the
    same way: content that is not valid UTF-8, for instance, makes
    ``detect_format`` raise ``FormatError`` (which this function catches),
    but ``_first_line_kind`` then raises its own uncaught ``UnicodeDecodeError``
    trying to read the same bytes. So this function usually returns a label but
    is not guaranteed to; callers that must never see an exception (e.g. a
    fast-path decision) need to guard the call themselves rather than rely on
    this docstring's older claim that it always returns.
    """
    try:
        return detect_format(path, variant=variant).format
    except FormatError:
        return _first_line_kind(path)
