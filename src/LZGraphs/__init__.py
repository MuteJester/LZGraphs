"""LZGraphs — LZ76 compression graphs for sequence repertoire analysis.

High-performance C backend with full LZ76 dictionary constraint enforcement.
"""

__version__ = "3.0.1"

from ._graph import LZGraph
from ._pgen_dist import PgenDistribution
from ._simulation_result import SimulationResult
from ._errors import LZGraphError, NoGeneDataError, ConvergenceError, CorruptFileError
from . import _clzgraph as _c


def jensen_shannon_divergence(a, b):
    """Jensen-Shannon divergence between two LZGraphs."""
    return _c.jensen_shannon_divergence(a._cap, b._cap)


def k_diversity(sequences, k, *, variant='aap', draws=100, seed=None):
    """K-diversity: subsample k sequences, count nodes, repeat.

    Returns {'mean': float, 'std': float, 'ci_low': float, 'ci_high': float}
    """
    return _c.k_diversity(list(sequences), k, variant, draws,
                          seed if seed is not None else -1)


def saturation_curve(sequences, *, variant='aap', log_every=100):
    """Node/edge saturation as sequences are added.

    Returns list of {'n_sequences': int, 'n_nodes': int, 'n_edges': int}
    """
    return _c.saturation_curve(list(sequences), variant, log_every)


def lz76_decompose(sequence):
    """LZ76 decomposition into subpatterns.

    Example: lz76_decompose('CASSLGIRRT') -> ['C','A','S','SL','G','I','R','RT']
    """
    return _c.lz76_decompose(sequence)


def set_log_level(level='info'):
    """Enable logging to stderr at the given level.

    Levels: 'none', 'error', 'warn', 'info', 'debug', 'trace'.
    'none' disables logging. Default is 'info'.

    Example:
        LZGraphs.set_log_level('info')   # see build progress and timing
        LZGraphs.set_log_level('debug')  # see algorithm decisions
        LZGraphs.set_log_level('none')   # silence all output
    """
    _c.set_log_level(level)


def set_log_callback(callback, level='info'):
    """Set a custom log callback.

    Args:
        callback: A callable(level: int, message: str), or None to disable.
                  Level values: 1=error, 2=warn, 3=info, 4=debug, 5=trace.
        level: Maximum level to emit.

    Example:
        import logging
        logger = logging.getLogger('lzgraphs')
        LEVEL_MAP = {1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO,
                     4: logging.DEBUG, 5: logging.DEBUG}
        LZGraphs.set_log_callback(
            lambda lvl, msg: logger.log(LEVEL_MAP.get(lvl, logging.DEBUG), msg),
            level='info'
        )
    """
    _c.set_log_callback(callback, level)


__all__ = [
    'LZGraph',
    'PgenDistribution',
    'SimulationResult',
    'LZGraphError',
    'NoGeneDataError',
    'ConvergenceError',
    'CorruptFileError',
    'jensen_shannon_divergence',
    'k_diversity',
    'saturation_curve',
    'lz76_decompose',
    'set_log_level',
    'set_log_callback',
]
