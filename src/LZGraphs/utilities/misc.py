
"""Provide support for other library functionality.

This module support various other operations in this library,
any function with no specific scope appears here.

The module contains the following functions:

- `choice(options,probs)` - choose a random element from a list given a probability distribution over the elements.
- `window(iterable, size)` - Return a sliding window generator of size "size".
- `chunkify(L, n)` - Yield successive n-sized chunks from L.
"""

from itertools import tee

import numpy as np

from ..exceptions import EmptyDataError, InvalidProbabilityError

__all__ = ["choice", "window", "chunkify", "_is_v_gene", "_is_j_gene"]


_GENE_META_KEYS = frozenset({'Vsum', 'Jsum', 'weight'})

# IMGT V/J gene substring markers covering all chain types:
# TCR: TRAV, TRBV, TRGV, TRDV  (contain AV, BV, GV, DV)
# BCR: IGHV, IGKV, IGLV         (contain HV, KV, LV)
_V_GENE_MARKERS = ('AV', 'BV', 'GV', 'DV', 'HV', 'KV', 'LV')
_J_GENE_MARKERS = ('AJ', 'BJ', 'GJ', 'DJ', 'HJ', 'KJ', 'LJ')


def _is_v_gene(key):
    """Check if an edge data key represents a V gene.

    Handles simple names ('V30'), TCR IMGT ('TRBV5-1*01'),
    and BCR IMGT ('IGHV3-23*01', 'IGKV1-39*01', 'IGLV2-14*01').
    """
    if key in _GENE_META_KEYS:
        return False
    key_upper = key.upper()
    if key_upper.startswith('V'):
        return True
    return any(marker in key_upper for marker in _V_GENE_MARKERS)


def _is_j_gene(key):
    """Check if an edge data key represents a J gene.

    Handles simple names ('J2'), TCR IMGT ('TRBJ2-1*01'),
    and BCR IMGT ('IGHJ4*02', 'IGKJ2*01', 'IGLJ3*01').
    """
    if key in _GENE_META_KEYS:
        return False
    key_upper = key.upper()
    if key_upper.startswith('J'):
        return True
    return any(marker in key_upper for marker in _J_GENE_MARKERS)


def choice(options, probs):
    """Choose a single random variable from a list given a probability distribution.

    Args:
        options (list): The list of values from which a single random one should be chosen.
        probs (list): Probability distribution (must sum to ~1.0).

    Returns:
        element: A random variable from list "options" with probability p in probs.

    Raises:
        ValueError: If options is empty, lengths don't match, or probabilities invalid.
    """
    n = len(options)
    if n == 0:
        raise EmptyDataError("options cannot be empty")

    if n != len(probs):
        raise InvalidProbabilityError(
            message=f"Length mismatch: options has {n} elements, "
            f"probs has {len(probs)} elements"
        )

    # Validate probabilities sum to approximately 1
    prob_sum = sum(probs)
    if not (0.99 <= prob_sum <= 1.01):
        raise InvalidProbabilityError(prob_sum=prob_sum)

    # Fast path for single option
    if n == 1:
        return options[0]

    x = np.random.rand()

    # For large neighbor lists, use numpy searchsorted (O(log n) vs O(n))
    if n > 8:
        cum = np.cumsum(probs)
        idx = np.searchsorted(cum, x)
        return options[min(idx, n - 1)]

    # For small lists, linear scan is faster (no numpy overhead)
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]
def window(iterable, size):
    """Return a sliding window generator of size "size".

          Args:
              iterable (iterable): An iterable of elements
              size (int): The size of the sliding window.

          Returns:
              zip: a zip of all windows of size "size"
          """
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)

def chunkify(L, n):
    """ Yield successive n-sized chunks from L.

          Args:
              L (iterable): An iterable of elements to partition into n-sized chunks
              n (int) the size of each chunck.
          Returns:
              generator: a generator that will return the next chunck each time its called until all of L is returned.
    """
    for i in range(0, len(L), n):
        yield L[i:i+n]
