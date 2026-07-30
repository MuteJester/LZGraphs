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
