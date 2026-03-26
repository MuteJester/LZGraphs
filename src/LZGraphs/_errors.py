"""LZGraph exception hierarchy.

Design: custom exceptions only where users need DISTINCT handling logic.
Everything else uses stdlib ValueError/RuntimeError/OSError with descriptive messages.

    LZGraphError (base — catch-all for any library error)
    ├── NoGeneDataError      — gene op on graph without gene data
    ├── ConvergenceError     — numerical method did not converge
    └── CorruptFileError     — LZG file is corrupt (bad magic, CRC mismatch, etc.)
"""


class LZGraphError(Exception):
    """Base exception for all LZGraph operations.

    Catch this to handle any library error without catching unrelated
    ValueError/RuntimeError from other code::

        try:
            result = graph.simulate(1000)
        except LZGraphError as e:
            print(f"LZGraph failed: {e}")
    """


class NoGeneDataError(LZGraphError):
    """Graph has no V/J gene annotations.

    Raised when gene-dependent operations (gene_simulate, v_genes, etc.)
    are called on a graph built without v_genes/j_genes.
    """


class ConvergenceError(LZGraphError):
    """Numerical method did not converge.

    Raised by predicted_richness() or other analytical computations
    when the iterative solver fails to reach the requested tolerance.
    Consider using a smaller depth or falling back to Monte Carlo.
    """


class CorruptFileError(LZGraphError, OSError):
    """LZG file is corrupt or not a valid LZG file.

    Inherits from both LZGraphError and OSError, so it can be caught
    by either. Raised on bad magic number, CRC mismatch, unsupported
    format version, or truncated data.
    """
