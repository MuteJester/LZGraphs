import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

from .decomposition import lempel_ziv_decomposition

# class BOWVectorizer:
#     """
#        A Bag of words vectorizer, can be fitted on a list of repertoires and used
#        to output LZ-BOW representation
#
#        ...
#
#        Methods
#        -------
#        fit(list_of_repertoires):
#            fits the vectorizer model to the dictionary derived from the repertoires given by the argument
#            "list_of_repertoires"
#
#        transform(list_of_repertoires):
#             given a list of repertoires the function will use the fitted BOW dictionary to return
#             the bag of words vectors for each repertoire in the list
#
#        """
#     def __init__(self):
#
#         self.vectorizer = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 1))
#
#
#     def fit(self,list_of_repertoires):
#
#         """
#            fits the BOW dictionary based on the repertoires given in "list_of_repertoires"
#
#                    Parameters:
#                            list_of_repertoires (list): A list of pandas DataFrame's that have a column named "cdr3_rearrangement"
#
#                    Returns:
#                            None
#         """
#
#         cmp = []
#         for d in tqdm(list_of_repertoires,leave=False):
#             for cdr3 in d['cdr3_rearrangement']:
#                 cmp.append(' '.join(lempel_ziv_decomposition(cdr3)))
#         self.vectorizer.fit(cmp)
#
#     def transform(self,list_of_repertoires):
#         """
#               transforms a list of repertoires into a list of bag of words vectors derived based on fitted repertoires
#
#                       Parameters:
#                               list_of_repertoires (list): A list of pandas DataFrame's that have a column named "cdr3_rearrangement"
#
#                       Returns:
#                               None
#        """
#         n_encoded = []
#         for d in tqdm(list_of_repertoires):
#             cmp = []
#             for cdr3 in d['cdr3_rearrangement']:
#                 cmp.append(' '.join(lempel_ziv_decomposition(cdr3)))
#             n_encoded.append(self.vectorizer.transform([' '.join(cmp)]).todense())
#         n_encoded = np.array(n_encoded)
#         n_encoded = n_encoded.squeeze()
#         return n_encoded
#
#
from collections.abc import Iterable
from lzgraphs.decomposition import lempel_ziv_decomposition

from collections.abc import Iterable
from lzgraphs.decomposition import lempel_ziv_decomposition


class LZBOW:
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
