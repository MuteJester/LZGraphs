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
from ._sniff import detect_format
from ._spec import FormatError, InputSpec, RecordStats

_PRODUCTIVE_TRUE = {"t", "true", "1", "yes"}

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
    call). That assumption holds for any real file on disk -- reopening it
    just rereads the same bytes -- but not for stdin, which is a single
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


def read_sequences(path, seq_column=None, v_column=None, j_column=None,
                   abundance_column=None, variant='aap', no_genes=False,
                   strict_input=False, expect_format=None, *,
                   keep_nonproductive=False):
    """Read sequences from a file, auto-detecting format.

    Returns a dict with ``sequences``, ``abundances``, ``v_genes``,
    ``j_genes``, and ``stats`` (a :class:`~LZGraphs._io._spec.RecordStats`).

    ``strict_input=True`` rejects mixed plain/plain_seqcount records (see
    ``_iter_plain_strict``) and also raises :class:`FormatError` on the
    first malformed record instead of skipping it. AIRR rows whose
    ``productive`` column is not truthy are dropped and counted unless
    ``keep_nonproductive=True``.
    """
    tmp_path = _materialize_stdin() if path == "-" else None
    real_path = tmp_path if tmp_path is not None else path
    try:
        return _read_sequences_from_path(
            real_path, seq_column=seq_column, v_column=v_column, j_column=j_column,
            abundance_column=abundance_column, variant=variant, no_genes=no_genes,
            strict_input=strict_input, expect_format=expect_format,
            keep_nonproductive=keep_nonproductive,
        )
    finally:
        if tmp_path is not None:
            os.remove(tmp_path)


def _read_sequences_from_path(path, *, seq_column, v_column, j_column,
                              abundance_column, variant, no_genes,
                              strict_input, expect_format,
                              keep_nonproductive=False):
    """The actual read, always against a real (reopenable) filesystem path."""
    spec = detect_format(
        path, variant=variant, seq_column=seq_column, override=expect_format
    )

    # Explicit v/j/abundance column overrides only apply to tabular input,
    # and only take effect when the named column actually exists in this
    # file's header; otherwise the auto-detected column (if any) stands.
    if spec.format == "tabular" and spec.header:
        spec = replace(
            spec,
            v_column=_resolve_column_override(spec.header, v_column) or spec.v_column,
            j_column=_resolve_column_override(spec.header, j_column) or spec.j_column,
            abundance_column=(
                _resolve_column_override(spec.header, abundance_column)
                or spec.abundance_column
            ),
        )

    # Gene data is structural, not per-row: only tabular input ever has a
    # v_column/j_column to read from. Building v_genes/j_genes as lists here
    # only when that column actually exists (and no_genes isn't set) keeps
    # them either None (no gene data, matching _graph.py's "None = no gene
    # data" contract) or exactly len(sequences) long -- appending '' for a
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

    stream, _codec = open_text(path)
    try:
        for sequence, abundance, v_call, j_call, productive in _iter_records(
            stream, spec, strict_input=strict_input
        ):
            total += 1
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
    UTF-8) -- and raises ``FormatError`` when it does. On that error, this
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
