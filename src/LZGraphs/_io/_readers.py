"""Streaming record readers, one per detected input format."""
from __future__ import annotations

import csv
from collections.abc import Iterator

from ._spec import FormatError


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
    productive_key = None
    for row in reader:
        if productive_key is None:
            productive_key = next(
                (k for k in row if k and k.lower() == "productive"), ""
            )
        sequence = (row.get(spec.seq_column) or "").strip()
        if not sequence:
            continue
        abundance = 1
        if spec.abundance_column:
            try:
                abundance = int((row.get(spec.abundance_column) or "1").strip())
            except ValueError:
                abundance = 1
            if abundance < 1:
                abundance = 1
        v_call = (row.get(spec.v_column) or None) if spec.v_column else None
        j_call = (row.get(spec.j_column) or None) if spec.j_column else None
        productive = row.get(productive_key) if productive_key else None
        yield sequence, abundance, v_call, j_call, productive
