import random
import numpy as np
from .AminoAcidPositional import derive_lz_and_position
from .decomposition import lempel_ziv_decomposition
from .misc import window
from .NucleotideDoublePositional import derive_lz_reading_frame_position
from tqdm.auto import tqdm
import itertools


class NodeEdgeSaturationProbe:
    """

      The class supplies methods used to emulate the creation process of an LZGraph without actually running the full
      creation procedure, rather just accumulate a counter for the number of nodes and edges based on the provided
      number of sequences.

      Args:
          node_function (str): the selected node extraction method to use 'naive' - emulate Naive LZGraph extraction
           / 'ndp'- emulate Nucleotide Double Positional LZGraph / 'aap' - Amino Acid Positional LZGraph.

      Attributes:

          log_memory (dict): a dictionary containing the results of a single test run


      """

    def __init__(self, node_function='naive', log_level=1, verbose=False):
        self.nodes = set()
        self.edges = set()
        self.log_memory = dict()
        self.verbose = verbose
        self.log_level = log_level
        self.node_function = None
        if node_function == 'naive':
            self.node_function = self.naive_node_extractor
        elif node_function == 'ndp':
            self.node_function = self.ndp_node_extractor
        elif node_function == 'aap':
            self.node_function = self.aap_node_extractor

    def log(self, args):
        if self.log_level == 1:
            self.log_memory[args] = {'nodes': len(self.nodes), 'edges': len(self.edges)}

    @staticmethod
    def naive_node_extractor(seq):
        """ This function implements the node extraction procedure used by the Naive LZGraph.
                             Args:
                                 seq (str): An sequence of nucleotides or amino acids
                             Returns:
                                 list: a list of nodes extract from the given sequence
           """
        return lempel_ziv_decomposition(seq)

    @staticmethod
    def ndp_node_extractor(seq):
        """ This function implements the node extraction procedure used by the Nucleotide Double Positional LZGraph.
                                 Args:
                                     seq (str): An sequence of nucleotides or amino acids
                                 Returns:
                                     list: a list of nodes extract from the given sequence
               """
        LZ, POS, locations = derive_lz_reading_frame_position(seq)
        nodes_local = list(map(lambda x, y, z: x + str(y) + '_' + str(z), LZ, POS, locations))
        return nodes_local

    @staticmethod
    def aap_node_extractor(seq):
        """ This function implements the node extraction procedure used by the Amino Acid Positional LZGraph.
                                   Args:
                                       seq (str): An sequence of nucleotides or amino acids
                                   Returns:
                                       list: a list of nodes extract from the given sequence
                 """
        LZ, locations = derive_lz_and_position(seq)
        nodes_local = list(map(lambda x, y: x + '_' + str(y), LZ, locations))
        return nodes_local

    def test_sequences(self, sequence_list, log_every=1000, iteration_number=None):
        """ Given a list of sequences this function will gradually aggregate the nodes that make up the respective
        LZGraph and log the node and edge counts every K sequences.

        The result will be saved in the log_memory attribute of the class.


                Args:
                    sequence_list (str): A list of nucleotide  or amino acid sequences
                    log_every (int): after how many sequences to log the number of nodes and edges
                Returns:
                    None:
        """

        slen = len(sequence_list)
        itr = None

        if self.verbose:
            itr = tqdm(enumerate(sequence_list, start=1), leave=False, position=0, total=slen)
        else:
            itr = enumerate(sequence_list, start=1)

        for ax, seq in itr:
            nodes_local = self.node_function(seq)
            self.nodes.update(nodes_local)
            self.edges.update((window(nodes_local, 2)))

            if ax % log_every == 0 or ax >= slen:
                self.log(ax)

    def _reset(self):
        self.nodes = set()
        self.edges = set()
        self.log_memory = dict()

    def resampling_test(self, sequence_list, n_tests, log_every=1000, sample_size=0):
        """ Given a list of sequences this function will gradually aggregate the nodes that make up the respective
        LZGraph and log the node and edge counts every K sequences.
        The above procedure will be carried out N times each time starting from X randomly sampled sequences from
        the given sequence list.

                Args:
                    sequence_list (str): A list of nucleotide  or amino acid sequences
                    log_every (int): after how many sequences to log the number of nodes and edges
                    n_tests (int): the number of realizations to perform
                    sample_size (int): the number of sequences that will be randomly sampled from sequence_list
                    at each realization
                Returns:
                    list: a list of logs for each realization given by the parameter n_tests
        """

        result = []
        if sample_size == 0:
            for n in range(n_tests):
                np.random.shuffle(sequence_list)
                self.test_sequences(sequence_list, log_every, n)
                # save logs
                # reset aux
                result.append(self.log_memory.copy())
                self._reset()
        else:
            for n in range(n_tests):
                np.random.shuffle(sequence_list)
                self.test_sequences(random.sample(sequence_list, sample_size), log_every, n)
                # save logs
                # reset aux
                result.append(self.log_memory.copy())
                self._reset()
        return result
