"""Streaming record readers, one per detected input format."""
from __future__ import annotations

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
