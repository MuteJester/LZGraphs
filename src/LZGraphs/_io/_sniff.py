"""Content-based detection of sequence file format and alphabet."""
from __future__ import annotations

from ._compress import open_text
from ._readers import _is_count
from ._spec import FormatError, InputSpec

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
    """True when the prefix matches the 4-line FASTQ record shape.

    Checks unfiltered line positions to detect blank lines inside records,
    ensuring the detector and reader stay in sync.
    """
    # Skip leading blank lines to find the first @ header.
    first_header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("@"):
            first_header_idx = i
            break

    if first_header_idx is None:
        return False

    # The separator should be exactly 2 positions after the header.
    separator_idx = first_header_idx + 2
    if separator_idx >= len(lines):
        return False

    return lines[separator_idx].strip().startswith("+")


_SEQ_COLUMNS = {
    "aap": ["junction_aa", "cdr3_amino_acid", "cdr3_aa", "aminoacid"],
    "ndp": ["junction", "cdr3_rearrangement", "cdr3_nt", "nucleotide"],
    "naive": ["junction_aa", "cdr3_amino_acid", "junction", "cdr3_rearrangement"],
}
_SEQ_FALLBACK = ["sequence", "cdr3", "seq"]
_ABUNDANCE_COLUMNS = ["duplicate_count", "count", "abundance", "reads", "copies"]
_V_COLUMNS = ["v_call", "v_gene", "vgene"]
_J_COLUMNS = ["j_call", "j_gene", "jgene"]


def sniff_delimiter(header_line: str) -> str | None:
    """Pick the delimiter of a header row, preferring tab over comma."""
    if "\t" in header_line:
        return "\t"
    if "," in header_line:
        return ","
    return None


def _pick(header: list[str], candidates: list[str]) -> str | None:
    lowered = {name.lower().strip(): name for name in header}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def resolve_columns(
    header: list[str], seq_column: str | None, variant: str
) -> tuple[str, str | None, str | None, str | None]:
    """Resolve the sequence, abundance, and gene column names."""
    if seq_column is not None:
        match = _pick(header, [seq_column.strip().lower()])
        if match is None:
            available = "  ".join(header)
            raise FormatError(
                f"column {seq_column!r} not found\n"
                f"  available columns:\n    {available}"
            )
        resolved = match
    else:
        candidates = _SEQ_COLUMNS.get(variant, _SEQ_COLUMNS["aap"]) + _SEQ_FALLBACK
        resolved = _pick(header, candidates)
        if resolved is None:
            available = "  ".join(header)
            raise FormatError(
                "could not identify a sequence column\n"
                f"  available columns:\n    {available}\n"
                "  name it explicitly with --column"
            )
    return (
        resolved,
        _pick(header, _ABUNDANCE_COLUMNS),
        _pick(header, _V_COLUMNS),
        _pick(header, _J_COLUMNS),
    )


def looks_like_seqcount(lines: list[str]) -> bool:
    """True when every sampled line is ``token<TAB>count``.

    ``count`` is judged with the same ``_is_count`` predicate that
    ``_readers._parse_count`` uses, so this detector and that reader agree
    on borderline forms such as the ``"3.0"`` pandas and R emit. A plain
    ``.isdigit()`` check here would reject those, sending them to
    ``iter_plain`` instead, which yields the whole ``"seq\\t3.0"`` line as a
    sequence.
    """
    sampled = [line.strip() for line in lines if line.strip()]
    if not sampled:
        return False
    for line in sampled:
        parts = line.split("\t")
        if len(parts) != 2:
            return False
        if not _is_count(parts[1]):
            return False
    return True


_SNIFF_LINES = 32
_VALID_OVERRIDES = ("fasta", "fastq", "tabular", "plain", "plain_seqcount")


def _read_prefix(path: str) -> tuple[list[str], str]:
    """Read up to ``_SNIFF_LINES`` lines of ``path`` for content sniffing.

    ``open_text`` documents its returned stream as always safe to close,
    including when ``path`` is ``"-"``: stdin is wrapped with
    ``closefd=False``, so closing the wrapper never closes the real
    descriptor. That means this can close unconditionally instead of
    special-casing stdin.
    """
    stream, codec = open_text(path)
    try:
        lines = []
        for _ in range(_SNIFF_LINES):
            line = stream.readline()
            if not line:
                break
            lines.append(line)
    except UnicodeDecodeError as exc:
        raise FormatError(
            f"{path} is not valid UTF-8 text, so it cannot be sequence data\n"
            f"  {exc}"
        ) from None
    finally:
        stream.close()
    return lines, codec


def _prefix_samples(lines: list[str], fmt: str) -> list[str]:
    """Pull up to 8 genuine residue samples out of a detected-format prefix.

    Only the residue-bearing field of each format's record shape is kept.
    This matters most for FASTQ: filtering out lines that merely *start*
    with ``@`` or ``+`` still leaves the quality string in the sample set,
    and Phred-quality characters can include letters (``I``, for one) that
    are valid amino acids but not valid IUPAC nucleotide codes, which
    corrupts ``infer_alphabet`` into reporting "amino_acid" for ordinary
    nucleotide reads.
    """
    if fmt == "fastq":
        samples: list[str] = []
        i = 0
        while i < len(lines) and len(samples) < 8:
            if lines[i].strip().startswith("@"):
                if i + 1 < len(lines):
                    sequence = lines[i + 1].strip()
                    if sequence:
                        samples.append(sequence)
                i += 4
            else:
                i += 1
        return samples

    samples = [line.strip() for line in lines[:8] if line.strip()]
    if fmt == "fasta":
        return [s for s in samples if not s.startswith((">", ";"))]
    if fmt == "plain_seqcount":
        return [s.split("\t", 1)[0] for s in samples]
    return samples


def detect_format(
    path: str,
    *,
    variant: str = "aap",
    seq_column: str | None = None,
    override: str | None = None,
) -> InputSpec:
    """Classify ``path`` by content and resolve its columns."""
    if override is not None and override not in _VALID_OVERRIDES:
        raise FormatError(
            f"unknown --format {override!r}\n"
            f"  choose one of: {', '.join(_VALID_OVERRIDES)}"
        )

    lines, codec = _read_prefix(path)
    reject_binary("".join(lines), path)
    if not any(line.strip() for line in lines):
        raise FormatError(f"{path} is empty")

    fmt = override
    if fmt is None:
        if looks_like_fasta(lines):
            fmt = "fasta"
        elif looks_like_fastq(lines):
            fmt = "fastq"
        elif sniff_delimiter(lines[0]) and not looks_like_seqcount(lines):
            fmt = "tabular"
        elif looks_like_seqcount(lines):
            fmt = "plain_seqcount"
        else:
            fmt = "plain"

    if fmt != "tabular":
        samples = _prefix_samples(lines, fmt)
        return InputSpec(
            path=path,
            format=fmt,
            compression=codec,
            alphabet=infer_alphabet(samples),
        )

    delimiter = sniff_delimiter(lines[0]) or "\t"
    header = [h.strip() for h in lines[0].rstrip("\n").split(delimiter)]
    seq, abundance, v_col, j_col = resolve_columns(header, seq_column, variant)
    index = header.index(seq)
    samples = [
        row.rstrip("\n").split(delimiter)[index]
        for row in lines[1:9]
        if row.strip() and len(row.rstrip("\n").split(delimiter)) > index
    ]
    return InputSpec(
        path=path,
        format="tabular",
        compression=codec,
        delimiter=delimiter,
        seq_column=seq,
        abundance_column=abundance,
        v_column=v_col,
        j_column=j_col,
        alphabet=infer_alphabet(samples),
        header=tuple(header),
    )
