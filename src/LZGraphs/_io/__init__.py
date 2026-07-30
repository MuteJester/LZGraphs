"""Input detection and reading for sequence repertoire files."""
from __future__ import annotations

from ._legacy import (
    detect_input_kind,
    read_sequences,
    read_sequences_simple,
    validate_input,
)

__all__ = [
    "detect_input_kind",
    "read_sequences",
    "read_sequences_simple",
    "validate_input",
]
