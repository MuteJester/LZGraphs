"""Streaming record readers, one per detected input format."""
from __future__ import annotations

import csv
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation

from ._spec import FormatError, normalize_header_field


def _is_count(raw: str) -> bool:
    """True when ``raw`` is a genuine count: an integer or an integral float.

    Shared with ``_sniff.looks_like_seqcount`` so the detector and this
    reader can never drift apart on what a count field looks like again.
    Calling ``_parse_count`` and checking its result would not work:
    ``_parse_count`` returns 1 both for a genuine count of 1 and for
    unparseable input, so the two cases are indistinguishable after the
    fact.
    """
    text = raw.strip()
    if not text:
        return False
    try:
        int(text)
        return True
    except ValueError:
        pass
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return False
    return parsed.is_finite() and parsed == parsed.to_integral_value()


def _parse_count(raw: str) -> int:
    """Parse an abundance, tolerating the float forms pandas and R emit.

    ``int()`` first because it is exact and cheap. ``Decimal`` for the
    fallback because ``float()`` silently loses precision above 2**53.
    Anything unparseable or non-integral becomes 1.
    """
    if not _is_count(raw):
        return 1
    text = raw.strip()
    try:
        value = int(text)
    except ValueError:
        value = int(Decimal(text))
    return value if value >= 1 else 1


def iter_fasta(stream) -> Iterator[str]:
    """Yield sequences from FASTA, joining wrapped lines.

    Header lines are structurally incapable of being yielded, which is what
    keeps them out of built graphs.
    """
    parts: list[str] = []
    for raw in stream:
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if parts:
                yield "".join(parts)
                parts = []
            continue
        parts.append(line)
    if parts:
        yield "".join(parts)


def iter_fastq(stream) -> Iterator[str]:
    """Yield the sequence line of each 4-line FASTQ record."""
    while True:
        header = stream.readline()
        if not header:
            return
        if not header.strip():
            continue
        sequence = stream.readline().strip()
        separator = stream.readline()
        # Missing quality line at EOF is accepted because sequence is already complete.
        stream.readline()  # quality line, discarded
        if not separator.startswith("+"):
            raise FormatError(
                "malformed FASTQ record: expected '+' on the third line, "
                f"found {separator.strip()!r}"
            )
        # Skip empty sequences, matching iter_fasta behavior.
        if sequence:
            yield sequence


def iter_tabular_rows(stream, spec) -> Iterator[tuple]:
    """Yield ``(sequence, abundance, v_call, j_call, productive)`` from a table.

    Only the resolved sequence column is ever emitted as a sequence, which is
    what keeps whole delimited rows out of built graphs.
    """
    reader = csv.DictReader(stream, delimiter=spec.delimiter or "\t")
    # Normalise the DictReader's own fieldnames with the exact same function
    # detect_format applies to the header it resolves spec.seq_column/
    # abundance_column/v_column/j_column against (normalize_header_field,
    # from _spec.py). Without this, a padded real-world header like
    # "junction_aa, duplicate_count" resolves 'duplicate_count' on the
    # detect_format side but the row dict here is keyed by the unstripped
    # ' duplicate_count', so row.get(spec.abundance_column) silently misses
    # and every abundance defaults to 1. Accessing .fieldnames triggers the
    # lazy read of the header row exactly once, before any data row is read.
    if reader.fieldnames:
        reader.fieldnames = [normalize_header_field(name) for name in reader.fieldnames]
    productive_key = None
    for row in reader:
        if productive_key is None:
            productive_key = next(
                (k for k in row if k and k.strip().lower() == "productive"), ""
            )
        sequence = (row.get(spec.seq_column) or "").strip()
        if not sequence:
            continue
        abundance = 1
        if spec.abundance_column:
            raw_count = row.get(spec.abundance_column) or ""
            abundance = _parse_count(raw_count)
        v_call = (row.get(spec.v_column) or None) if spec.v_column else None
        j_call = (row.get(spec.j_column) or None) if spec.j_column else None
        productive = row.get(productive_key) if productive_key else None
        yield sequence, abundance, v_call, j_call, productive


def iter_plain(stream) -> Iterator[str]:
    """Yield one sequence per non-blank line."""
    for raw in stream:
        line = raw.strip()
        if line:
            yield line


def iter_seqcount(stream) -> Iterator[tuple[str, int]]:
    """Yield ``(sequence, abundance)`` from ``seq<TAB>count`` lines.

    Skips records with an empty sequence field, matching iter_fasta/iter_fastq.
    """
    for raw in stream:
        line = raw.rstrip()
        if not line.strip():
            continue
        parts = line.split("\t")
        sequence = parts[0].strip() if parts else ""
        if not sequence:
            # Skip records with empty sequence, consistent with other readers.
            continue
        count = _parse_count(parts[1]) if len(parts) > 1 else 1
        yield sequence, count
