"""Content-based detection of sequence file format and alphabet."""
from __future__ import annotations

from ._compress import open_text
from ._readers import _is_count
from ._spec import VALID_FORMATS, FormatError, InputSpec

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

# Every known sequence-column name, across every variant, plus the
# variant-agnostic fallback names. Used only to recognise a lone header line
# in an otherwise undelimited file: a real residue string is never spelled
# "junction" or "aminoAcid", so a case-insensitive match here is unambiguous.
_ALL_SEQ_COLUMN_NAMES = frozenset(
    name for names in _SEQ_COLUMNS.values() for name in names
) | frozenset(_SEQ_FALLBACK)


def _is_lone_seq_column_header(line: str) -> bool:
    """True when ``line``, taken alone, names a known sequence column.

    Guards a single-column file whose only header is e.g. ``junction`` or
    ``aminoAcid``: without this, such a file has no delimiter and no digits
    for ``_is_wellformed``'s incidental protections to reject, so the header
    row would otherwise be sniffed as ``plain`` and ingested as a sequence.
    """
    return line.strip().lower() in _ALL_SEQ_COLUMN_NAMES


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


def _duplicate_header_names(header: list[str]) -> list[tuple[str, int]]:
    """Names that appear more than once in ``header``, matched case-insensitively.

    Column resolution throughout this module (``_pick``) is case-insensitive,
    so ``junction_aa`` and ``Junction_AA`` address the same column and must be
    treated as one duplicated name here too.

    Empty cells are excluded on purpose: a trailing delimiter producing a
    lone ``""`` column (e.g. ``"junction_aa\\t"``) is common in real exports
    and is not the ambiguity this guards against, since no candidate list in
    this module and no ``--column`` value ever resolves to an empty name.
    Flagging it would be a false positive on ordinary files.
    """
    counts: dict[str, int] = {}
    first_seen: dict[str, str] = {}
    for name in header:
        stripped = name.strip()
        if not stripped:
            continue
        key = stripped.lower()
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, stripped)
    return [(first_seen[key], count) for key, count in counts.items() if count > 1]


def _check_no_duplicate_columns(path: str, header: list[str]) -> None:
    """Refuse a header with any duplicated column name rather than guess.

    ``csv.DictReader`` (``iter_tabular_rows``) keeps only the *last*
    occurrence of a repeated header key, so a duplicated name makes the
    resolved column silently address different data than the user believes:
    total silent substitution, not a dropped record. This is guarded once,
    here, for the whole header, not only the resolved sequence column: a
    duplicated ``duplicate_count`` corrupts abundance and a duplicated
    ``v_call`` corrupts gene data by the exact same mechanism.

    ``--column`` cannot resolve this even when it names the affected column:
    it is resolved by the same case-insensitive name lookup (``_pick``) that
    produced the ambiguity, so it would match both duplicated columns
    equally. The message says so plainly instead of pointing at a flag that
    cannot help.
    """
    duplicates = _duplicate_header_names(header)
    if not duplicates:
        return
    detail = "\n".join(
        f"    {name!r} appears {count} times" for name, count in duplicates
    )
    raise FormatError(
        f"{path} has duplicate column names in its header:\n"
        f"{detail}\n"
        "  csv.DictReader keeps only the last occurrence of a repeated "
        "name, so the column that gets read may silently be a different "
        "one than you expect\n"
        "  --column cannot disambiguate this: it resolves by the same name "
        "that is duplicated, so it would match every copy equally\n"
        "  rename or remove the duplicate column(s) in the source file and "
        "try again"
    )


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
    if override is not None and override not in VALID_FORMATS:
        raise FormatError(
            f"unknown --format {override!r}\n"
            f"  choose one of: {', '.join(VALID_FORMATS)}"
        )

    lines, codec = _read_prefix(path)
    reject_binary("".join(lines), path)
    if not any(line.strip() for line in lines):
        raise FormatError(f"{path} is empty")

    fmt = override
    reclassified_lone_header = False
    if fmt is None:
        if looks_like_fasta(lines):
            fmt = "fasta"
        elif looks_like_fastq(lines):
            fmt = "fastq"
        elif sniff_delimiter(lines[0]) and not looks_like_seqcount(lines):
            fmt = "tabular"
        elif looks_like_seqcount(lines):
            fmt = "plain_seqcount"
        elif _is_lone_seq_column_header(lines[0]):
            # Reaching here already implies lines[0] has no delimiter and
            # is not seqcount-shaped (see the elif chain above), so this is
            # a single undelimited line. If it also spells a known
            # sequence-column name, it is a header, not a sequence.
            fmt = "tabular"
            reclassified_lone_header = True
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

    sniffed_delimiter = sniff_delimiter(lines[0])
    split_delimiter = sniffed_delimiter or "\t"
    header = [h.strip() for h in lines[0].rstrip("\n").split(split_delimiter)]
    # Guard the whole header before anything resolves against it: a
    # single-column file can never trigger this (nothing to duplicate), but
    # every other tabular header must be checked before either branch below
    # picks a column out of it.
    _check_no_duplicate_columns(path, header)
    if reclassified_lone_header:
        # The file has exactly one column, so it is unambiguously the
        # sequence column when seq_column is not given. resolve_columns's
        # variant-specific candidate list does not apply to that auto-detect
        # case (e.g. a bare "junction" header is only in the ndp/naive
        # lists, not aap's), so bypass it rather than risk a spurious "could
        # not identify a sequence column" error for a file that has no other
        # column it could possibly be. An explicit seq_column, however, is
        # still resolved through resolve_columns: its seq_column-not-None
        # branch never consults the variant candidate list either (it only
        # looks up the one requested name), so it is safe to reuse here, and
        # doing so is what makes a typo'd --seq-column raise on this branch
        # instead of being silently ignored (a lone real column always
        # satisfies the request trivially, so this never rejects a correct
        # explicit name).
        if seq_column is not None:
            seq, abundance, v_col, j_col = resolve_columns(header, seq_column, variant)
        else:
            seq, abundance, v_col, j_col = header[0], None, None, None
        delimiter = None
    else:
        seq, abundance, v_col, j_col = resolve_columns(header, seq_column, variant)
        delimiter = split_delimiter
    index = header.index(seq)
    samples = [
        row.rstrip("\n").split(split_delimiter)[index]
        for row in lines[1:9]
        if row.strip() and len(row.rstrip("\n").split(split_delimiter)) > index
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
