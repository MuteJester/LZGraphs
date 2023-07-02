import numpy as np
from .NodeEdgeSaturationProbe import NodeEdgeSaturationProbe


def LZCentrality(lzgraph,sequence):
    """
    Calculates the LZCentrality of a given CDR3 sequence in a repertoire represented by an LZGraph.

    Args:
        lzgraph (LZGraph): The LZGraph representing the repertoire.
        sequence (str): The CDR3 sequence for which LZCentrality needs to be calculated.

    Returns:
        float: The LZCentrality value for the given sequence.

    Calculates the out degree at each node of the given sequence using the `sequence_variation_curve` method
    of the lzgraph object. Missing nodes are penalized by assigning a value of -1. The average of the out degrees
    is then computed using `np.mean` and returned.

    Example:
       >>> graph = NDPLZGraph(Repertoire)
       >>> sequence = "ACCGACAGGATTTACGT"
       >>> lzcentrality = LZCentrality(graph, sequence)
       >>> print(lzcentrality)
       """
    # calculate out degree at each node of the sequence
    svc= lzgraph.sequence_variation_curve(sequence)[1]
    # penalize for missing nodes
    svc = [-1 if type(i) != int else i for i in svc ]
    return np.mean(svc)



def K1000_Diversity(list_of_sequences,lzgraph_encoding_function,draws=25):
    """
      Calculates the K1000 Diversity index of a list of CDR3 sequences based on the provided LZGraph encoding function.

      Args:
          list_of_sequences (list): A list of CDR3 sequences.
          lzgraph_encoding_function (function): The LZGraph encoding function to be used. (e.g., AAPLZGraph.encode_sequence)
          draws (int, optional): The number of draws for the resampling test. Defaults to 25.

      Returns:
          float: The average K1000 Diversity index calculated from the resampling tests.

      The K1000 Diversity index is computed by resampling the list of sequences and building LZGraphs based on the provided
      encoding function. The resampling is performed `draws` number of times, with each resampled set containing 1000 unique
      sequences. For each resampled set, the K1000 Diversity index is calculated using the 'nodes' value of the last item in
      the result dictionary obtained from the NodeEdgeSaturationProbe's resampling test. The average K1000 Diversity index
      from all the resampling tests is then returned using `np.mean`.

      Example:
          >>> sequences = ["ACGT", "CGTA", "GTAC"]
          >>> encoding_function = NDPLZGraph.encode_sequence
          >>> diversity = K1000_diversity(sequences, encoding_function, draws=30)
          >>> print(diversity)
      """
    # sample 1000 unique sequences
    NESP = NodeEdgeSaturationProbe(node_function=lzgraph_encoding_function)
    result = NESP.resampling_test(list(set(list_of_sequences)),n_tests=draws,sample_size=1000)
    K_tests = [list(i.values())[-1]['nodes'] for i in result]
    return np.mean(K_tests)
