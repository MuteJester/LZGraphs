"""Value types describing a detected input file."""
from __future__ import annotations

from dataclasses import dataclass, field


class FormatError(ValueError):
    """The input file cannot be interpreted as sequence data."""


# The single vocabulary of formats accepted anywhere a caller names a format
# explicitly: `detect_format`'s `override` parameter, `read_sequences`'s and
# `validate_input`'s `expect_format` parameter, and the CLI's
# `--expect-format` flag. Previously each of those four sites kept its own
# tuple, and they had drifted (the CLI and `validate_input` were missing
# `fasta`/`fastq`, which `_sniff.py` already supported), so a request that
# `read_sequences` handled fine was refused at the argparse layer or reported
# as a validation error. Defined once, here, so the four sites cannot drift
# apart again: `_sniff.py` and `_validate.py` import it directly, and
# `cli.py` gets it via `LZGraphs._io`'s re-export.
VALID_FORMATS: tuple[str, ...] = ("fasta", "fastq", "plain", "plain_seqcount", "tabular")

# `plain` and `plain_seqcount` are grouped as one family for `expect_format`
# assertion purposes: their difference is a per-record detail (see
# `_validate.py`'s `mode='mixed'` handling and `_public.py`'s
# `_iter_plain_strict`), not something a file-level format assertion should
# fail on. `tabular`, `fasta`, and `fastq` are each their own family. Shared
# by `_validate.py` (which has always compared families this way) and
# `_public.py`'s `read_sequences`, so the two "expect_format is an assertion"
# call sites can never disagree about what counts as a genuine mismatch.
PLAIN_FAMILY = frozenset({"plain", "plain_seqcount"})


def format_family(kind: str) -> str:
    """Group a detected/declared format into the shape ``expect_format`` asserts."""
    return "plain" if kind in PLAIN_FAMILY else kind


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
