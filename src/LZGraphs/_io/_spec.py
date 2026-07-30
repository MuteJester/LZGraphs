"""Value types describing a detected input file."""
from __future__ import annotations

from dataclasses import dataclass, field


class FormatError(ValueError):
    """The input file cannot be interpreted as sequence data."""


@dataclass(frozen=True)
class InputSpec:
    """What the sniffer concluded about one input file."""

    path: str
    format: str
    compression: str
    delimiter: str | None = None
    seq_column: str | None = None
    abundance_column: str | None = None
    v_column: str | None = None
    j_column: str | None = None
    alphabet: str = "ambiguous"
    header: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RecordStats:
    """How many input records ``read_sequences`` read, kept, and dropped.

    ``total`` is the number of records that reached ``_iter_records``'s yield
    point, not the number of records physically present in the file: the
    per-format readers (``iter_fasta``, ``iter_fastq``, ``iter_seqcount``,
    ``iter_tabular_rows``, ``iter_plain``) silently drop blank lines and
    records whose sequence field is empty (e.g. a FASTA header immediately
    followed by another header, or a tabular row with nothing in the
    sequence column) before that point, with no signal left behind. Those
    records are invisible to this accounting: they inflate neither ``total``
    nor ``malformed``. ``malformed`` and ``nonproductive`` are therefore
    exact counts of what was rejected among the records the readers actually
    surfaced, while ``total`` (and by extension this whole struct) can
    undercount the true number of malformed records in the source file. Do
    not present ``malformed`` as a complete defect count without this
    caveat.

    Attributes:
        total: Records seen by the record-normalisation layer (see caveat
            above).
        kept: Records that ended up in ``sequences``.
        malformed: Records rejected for not being a usable sequence (empty
            or containing non-alphabetic characters).
        nonproductive: AIRR rows dropped because their ``productive`` column
            was not truthy (only ever nonzero for tabular input, and only
            when ``keep_nonproductive`` is ``False``).
    """

    total: int = 0
    kept: int = 0
    malformed: int = 0
    nonproductive: int = 0
