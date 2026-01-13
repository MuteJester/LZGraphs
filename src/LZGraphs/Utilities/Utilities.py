from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



def restore_gene_counts(column):
    """ This function is used during the graph union operation, it converts the gene probability distribution at each
        edge back to a count vector.
                        Args:
                            column (pandas Series): An LZGraph
                        Returns:
                            pandas Series: padnas series of v and j counts instead of probabilites
      """
    vgs, jgs = [], []
    # total number of observed V genes/alleles
    vsum = column['Vsum']
    # total number of observed J genes/alleles
    jsum = column['Jsum']
    # extract v and j columns
    for col in column.index:
        if 'BV' in col:
            vgs.append(col)
        elif 'BJ' in col:
            jgs.append(col)

    column[vgs] *= vsum
    column[jgs] *= jsum

    return column

def renormalize_edge_genes(column):
    """ This function is used during the graph union operation, it normalizes the gene counts by the total number
    of observed v / j genes/alleles.
                    Args:
                        column (pandas Series): An LZGraph
                    Returns:
                        pandas Series: padnas series of v and j counts instead of probabilites
              """
    vgs, jgs = [], []
    # total number of observed V genes/alleles
    vsum = column['Vsum']
    # total number of observed J genes/alleles
    jsum = column['Jsum']
    # extract v and j columns
    for col in column.index:
        if 'BV' in col:
            vgs.append(col)
        elif 'BJ' in col:
            jgs.append(col)

    column[vgs] /= vsum
    column[jgs] /= jsum

    return column


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
    return 1 - z['weight']

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



