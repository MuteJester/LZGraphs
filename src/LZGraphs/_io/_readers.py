"""Streaming record readers, one per detected input format."""
from __future__ import annotations

from collections.abc import Iterator


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
