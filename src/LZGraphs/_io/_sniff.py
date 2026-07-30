"""Content-based detection of sequence file format and alphabet."""
from __future__ import annotations

from ._spec import FormatError

_NUCLEOTIDE = set("ACGTUN")
_IGNORED_RESIDUES = set("-.*_ \t")


def reject_binary(head: str, path: str) -> None:
    """Raise if ``head`` looks like binary rather than sequence text."""
    if "\x00" not in head:
        return
    preview = " ".join(f"{ord(c):02x}" for c in head[:16])
    raise FormatError(
        f"{path} looks like a binary file, not sequence data\n"
        f"  first bytes: {preview}"
    )


def infer_alphabet(samples: list[str]) -> str:
    """Classify residues as nucleotide, amino acid, or ambiguous."""
    letters = {c for s in samples for c in s.upper()} - _IGNORED_RESIDUES
    if not letters:
        return "ambiguous"
    if letters <= _NUCLEOTIDE:
        return "nucleotide"
    return "amino_acid"
