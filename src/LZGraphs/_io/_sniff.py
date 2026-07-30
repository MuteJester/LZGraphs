"""Content-based detection of sequence file format and alphabet."""
from __future__ import annotations

from ._spec import FormatError

_CORE_NUCLEOTIDE = set("ACGTUN")
_IUPAC_NUCLEOTIDE = set("ACGTUNRYSWKMBDHV")
_IGNORED_RESIDUES = set("-.*_ \t")
_MIN_CORE_FRACTION = 0.65


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
    """Classify residues as nucleotide, amino acid, or ambiguous.

    Every IUPAC ambiguity code is also a valid amino-acid letter, so
    membership alone cannot separate the two alphabets. Real nucleotide data
    is overwhelmingly core bases even when it carries ambiguous calls, so a
    core-base majority is required as well.
    """
    residues = [
        character
        for sample in samples
        for character in sample.upper()
        if character not in _IGNORED_RESIDUES
    ]
    if not residues:
        return "ambiguous"
    if not set(residues) <= _IUPAC_NUCLEOTIDE:
        return "amino_acid"
    core = sum(character in _CORE_NUCLEOTIDE for character in residues)
    if core / len(residues) >= _MIN_CORE_FRACTION:
        return "nucleotide"
    return "amino_acid"


def _first_meaningful(lines: list[str]) -> str | None:
    """First line that is neither blank nor a FASTA-style comment.

    ``iter_fasta`` skips ``;`` comments, so the detector must skip them too
    or the two will disagree about the same file.
    """
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(";"):
            return stripped
    return None


def looks_like_fasta(lines: list[str]) -> bool:
    """True when the first meaningful line opens a FASTA record."""
    first = _first_meaningful(lines)
    return first is not None and first.startswith(">")


def looks_like_fastq(lines: list[str]) -> bool:
    """True when the prefix matches the 4-line FASTQ record shape."""
    meaningful = [line.strip() for line in lines if line.strip()]
    if len(meaningful) < 3:
        return False
    return meaningful[0].startswith("@") and meaningful[2].startswith("+")
