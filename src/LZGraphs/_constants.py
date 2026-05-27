"""Shared numeric constants for the Python layer.

Mirrors selected values from ``include/lzgraph/common.h`` so Python code
can refer to them by name instead of scattering magic numbers.
"""
from __future__ import annotations


# Exact floor returned by the C library for any unscored / zero-probability
# sequence. Mirrors ``LZG_LOG_EPS`` in ``include/lzgraph/common.h``.
# Value: log(1e-300).
LOG_EPS: float = -690.7755278982137

# Comparison threshold for "this log_pgen represents a real, scored sequence"
# (i.e. is NOT at the floor). Anything strictly greater than this is treated
# as in-repertoire / scorable. Used in ``__contains__`` and in tests asserting
# that a sequence has a valid (non-floor) log probability.
LOG_EPS_THRESHOLD: float = -690.0

# Comparison threshold for "this log_pgen IS at (or very near) the floor".
# Used in tests that expect a sequence to be unscored / out-of-repertoire.
# Note: LOG_EPS itself ≈ -690.78, so any "below LOG_EPS_AT_FLOOR" check is a
# safe upper bound on the floor with some slack.
LOG_EPS_AT_FLOOR: float = -689.0
