from itertools import product

import numpy as np


__all__ = [
    'saturation_function',
    'weight_function',
    'generate_kmer_dictionary',
]


def saturation_function(x, h, k):
    """
          a version of the hill saturation function used in the "random_walk_ber_shortest" random walk method
          where based on the parameters the function controls the probability of choosing the shortest path action
          at each step

                  Parameters:
                          x (float): the length of the input at time t divided by the target length
                          h (float): the saturation constant
                          k (int): the saturation factor degree

                  Returns:
                          float : value between 0 - 1 (used as probability for bernoulli trail)
   """
    if x == 0:
        return 0.0
    return 1 / (1 + ((h / x) ** k))

def weight_function(x, y, z):
    return 1 - z['data'].weight

def generate_kmer_dictionary(max_length):
    """
    This function generates all unique K-Mers for k starting at 1 up to max_length.
    It is a helper function used to derive the node dictionary for the naive LZ-Graph,
    where the length distribution of nucleotide sub-patterns is generally maxed at about 6.

    Parameters:
        max_length (int): The length of the maximal K-Mer family.

    Returns:
        list: A list of all unique K-Mers for K = 1 to K = max_length.
    """
    kmer_list = []
    for k in range(1, max_length + 1):
        kmer_list += [''.join(kmer) for kmer in product(['A', 'T', 'G', 'C'], repeat=k)]

    return kmer_list
