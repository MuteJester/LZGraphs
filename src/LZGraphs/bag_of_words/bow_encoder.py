from collections.abc import Iterable

import numpy as np
from tqdm.auto import tqdm

from ..utilities.decomposition import lempel_ziv_decomposition
from ..exceptions import EncodingFunctionMismatchError

__all__ = ["LZBOW"]


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

    def __repr__(self):
        return (f"LZBOW(dictionary_size={self.dictionary_size}, "
                f"observed_sequences={self.observed_sequences})")

    def _derive_index_maps(self):
        self.dictionary_index_map = {pattern: idx for idx, pattern in enumerate(self.dictionary)}
        self.dictionary_index_inverse_map = {idx: pattern for idx, pattern in enumerate(self.dictionary)}
        self.dictionary_size = len(self.dictionary)

    def fit(self, data):
        if isinstance(data, str):
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

    def transform(self, data, normalize=False, per_sequence=False):
        if isinstance(data, str):
            result = np.zeros(self.dictionary_size)
            result[self._seq_to_index(data)] += 1
            return result
        elif isinstance(data, Iterable):
            if per_sequence:
                data_list = list(data)
                matrix = np.zeros((len(data_list), self.dictionary_size))
                for i, seq in enumerate(tqdm(data_list, leave=False, position=0)):
                    matrix[i, self._seq_to_index(seq)] += 1
                if normalize:
                    row_sums = matrix.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1  # avoid division by zero
                    return matrix / row_sums
                return matrix
            else:
                result = np.zeros(self.dictionary_size)
                for seq in tqdm(data, leave=False, position=0):
                    result[self._seq_to_index(seq)] += 1
                if normalize:
                    total = result.sum()
                    return result / total if total > 0 else result
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
            raise EncodingFunctionMismatchError(
                "Cannot combine BOW objects with different encoding functions. "
                "Both objects must use the same encoding function."
            )
        union = LZBOW(self.encoding_function)
        union.dictionary = self.dictionary | other.dictionary
        union.observed_sequences = self.observed_sequences + other.observed_sequences
        union.dictionary_index_map = {pattern: idx for idx, pattern in enumerate(union.dictionary)}
        union.dictionary_index_inverse_map = {idx: pattern for idx, pattern in enumerate(union.dictionary)}
        union.dictionary_size = len(union.dictionary)
        return union

    def fit_transform(self, data, normalize=False, per_sequence=False):
        """
        Fit the encoder on data and transform it in one step.

        Equivalent to calling fit(data) followed by transform(data), but
        avoids processing the data twice for fitting.

        Args:
            data: A string (single sequence) or iterable of strings.
            normalize (bool): If True, normalize the output vectors.
            per_sequence (bool): If True and data is iterable, return a
                2D matrix (n_sequences x dictionary_size).

        Returns:
            np.ndarray: BOW vector(s) for the input data.

        Example:
            >>> bow = LZBOW()
            >>> matrix = bow.fit_transform(sequences, per_sequence=True)
        """
        self.fit(data)
        return self.transform(data, normalize=normalize, per_sequence=per_sequence)

    def tfidf_transform(self, data):
        """
        Transform sequences into TF-IDF weighted bag-of-words vectors.

        TF-IDF (Term Frequency - Inverse Document Frequency) weights
        down-weight subpatterns that appear in many sequences and up-weight
        those that are more discriminative.

        The encoder must be fitted before calling this method.

        Args:
            data: An iterable of sequence strings.

        Returns:
            np.ndarray: 2D matrix (n_sequences x dictionary_size) with TF-IDF weights.

        Example:
            >>> bow = LZBOW()
            >>> bow.fit(train_sequences)
            >>> tfidf_matrix = bow.tfidf_transform(test_sequences)
        """
        # Get per-sequence term frequency matrix
        tf_matrix = self.transform(data, per_sequence=True)

        n_docs = tf_matrix.shape[0]
        if n_docs == 0:
            return tf_matrix

        # Compute document frequency: number of sequences containing each term
        doc_freq = np.count_nonzero(tf_matrix, axis=0).astype(np.float64)

        # IDF = log(1 + N / (1 + df)), smoothed variant that's always non-negative
        idf = np.log1p(n_docs / (1.0 + doc_freq))

        # Normalize TF per row (L1 normalization)
        row_sums = tf_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        tf_normalized = tf_matrix / row_sums

        return tf_normalized * idf
