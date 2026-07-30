"""Value types describing a detected input file."""
from __future__ import annotations


class FormatError(ValueError):
    """The input file cannot be interpreted as sequence data."""
