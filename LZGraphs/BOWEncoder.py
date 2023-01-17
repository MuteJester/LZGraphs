import numpy as np
from tqdm.auto import tqdm
from collections.abc import Iterable
from LZGraphs.decomposition import lempel_ziv_decomposition


class LZBOW:
    """

         This class supplies a full suite for the conversion of repertoires into a bag of words representation
         based on a given sub-pattern (graph node) deriving function.
         This class requires fitting on a set of sequences in order to derive the dictionary of unique sub patterns used
         to generate the bag of words representation.
         After the class been fitted on a source set of sequences each time a transformation is needed one can use
         the transform method to get the vector representation.

         Args:
             encoding_function (str): the selected node extraction method to use 'naive' - emulate Naive LZGraph extraction
              / 'ndp'- emulate Nucleotide Double Positional LZGraph / 'aap' - Amino Acid Positional LZGraph.

         Attributes:

             dictionary (set): a set of sub-patterns (graph nodes) representing the dictionary of the BOW vector
             dictionary_size (int): The size of the dictionary
             observed_sequences (int): The number of sequences used to derive the dictionary
             encoding_function (func): the function used to derive sub-patterns from a sequence (in the context of this
             library it is one of the 3: Naive, Nucleotide Double Positional, Amino Acid Positional
             dictionary_index_map (dict): a dictionary that maps the set of sub-patterns to numerical positions in
             the BOW vector.
             dictionary_index_inverse_map (dict): a dictionary that maps numerical positions to the sub-patterns from
             the dictionary set


         """
    def __init__(self, encoding_function=lempel_ziv_decomposition):
        self.dictionary = set()
        self.dictionary_size = 0
        self.observed_sequences = 0
        self.encoding_function = encoding_function

        self.dictionary_index_map = dict()
        self.dictionary_index_inverse_map = dict()

    def _derive_index_maps(self):
        self.dictionary_index_map = {pattern: idx for idx, pattern in enumerate(self.dictionary)}
        self.dictionary_index_inverse_map = {idx: pattern for idx, pattern in enumerate(self.dictionary)}
        self.dictionary_size = len(self.dictionary)

    def fit(self, data):
        if type(data) == str:
            encoded = self.encoding_function(data)
            self.dictionary = self.dictionary | set(encoded)
            self._derive_index_maps()

        elif isinstance(data, Iterable):
            for seq in tqdm(data, leave=False, position=0):
                encoded = self.encoding_function(seq)
                self.dictionary = self.dictionary | set(encoded)
                self.observed_sequences += 1
            self._derive_index_maps()

    def _seq_to_index(self, seq):
        encoded = self.encoding_function(seq)
        return [self.dictionary_index_map[i] for i in encoded if i in self.dictionary]

    def transform(self, data, normalize=False):
        if type(data) == str:
            result = np.zeros(self.dictionary_size)
            result[self._seq_to_index(data)] += 1
            return result
        elif isinstance(data, Iterable):
            result = np.zeros(self.dictionary_size)
            for seq in tqdm(data, leave=False, position=0):
                result[self._seq_to_index(seq)] += 1
            if normalize:
                return result / result.sum()
            else:
                return result

    def load_from(self, other):
        self.dictionary = other.dictionary
        self.dictionary_size = other.dictionary_size
        self.observed_sequences = other.observed_sequences
        self.encoding_function = other.encoding_function

        self.dictionary_index_map = other.dictionary_index_map
        self.dictionary_index_inverse_map = other.dictionary_index_inverse_map

    def __add__(self, other):
        if self.encoding_function != other.encoding_function:
            raise Exception('Encoding Function Mismatch Between BOW Objects')
        union = LZBOW(self.encoding_function)
        union.dictionary = self.dictionary | other.dictionary
        union.observed_sequences = self.observed_sequences + other.observed_sequences
        union.dictionary_index_map = {pattern: idx for idx, pattern in enumerate(union.dictionary)}
        union.dictionary_index_inverse_map = {idx: pattern for idx, pattern in enumerate(union.dictionary)}
        union.dictionary_size = len(self.dictionary)
        return union
