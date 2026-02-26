"""Module-level numeric constants shared across LZGraphs internals."""
import numpy as np

# Machine epsilon — cached once at module level (avoids repeated np.finfo calls)
_EPS = np.finfo(np.float64).eps
_LOG_EPS = np.log(_EPS)
