"""Input detection and reading for sequence repertoire files."""
from __future__ import annotations

from ._public import (
    detect_input_kind,
    raw_prefix_is_streamable,
    read_sequences,
    read_sequences_simple,
)
from ._sniff import detect_format
from ._spec import VALID_FORMATS, FormatError, InputSpec, RecordStats
from ._validate import validate_input

__all__ = [
    "detect_format",
    "detect_input_kind",
    "raw_prefix_is_streamable",
    "read_sequences",
    "read_sequences_simple",
    "validate_input",
    "FormatError",
    "InputSpec",
    "RecordStats",
    "VALID_FORMATS",
]
