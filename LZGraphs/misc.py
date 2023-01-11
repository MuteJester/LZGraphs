
"""Provide support for other library functionality.

This module support various other operations in this library,
any function with no specific scope appears here.

The module contains the following functions:

- `choice(options,probs)` - choose a random element from a list given a probability distribution over the elements.
- `window(iterable, size)` - Return a sliding window generator of size "size".
- `get_dictionary_subkeys(target)` - Returns a list of all sub dictionary keys.
- `chunkify(L, n)` - Yield successive n-sized chunks from L.
"""

from itertools import tee
import numpy as np

def choice(options,probs):
    """Choose a single random variable from a list given a probability distribution

      Args:
          options (list): The list of values from which a single random one should be chosen
          probs (float): Probability distribution.

      Returns:
          element: a random variable from list "options" with probability p in probas.
      """
    x = np.random.rand()
    cum = 0
    i = None
    for i,p in enumerate(probs):
        cum += p
        if x < cum:
            break
    return options[i]
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
def get_dictionary_subkeys(target):
    """Returns a list of all sub dictionary keys.

          Args:
              target (dict): a dictionary of dictionaries.
          Returns:
              list: a list of all the keys contained in the sub dictionaries.
          """
    subkeys = []
    for key in target:
        subkeys +=[*target[key]]
    return subkeys
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
