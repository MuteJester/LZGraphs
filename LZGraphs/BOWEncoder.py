import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

from .decomposition import lempel_ziv_decomposition


class BOWVectorizer:
    """
       A Bag of words vectorizer, can be fitted on a list of repertoires and used
       to output LZ-BOW representation

       ...

       Methods
       -------
       fit(list_of_repertoires):
           fits the vectorizer model to the dictionary derived from the repertoires given by the argument
           "list_of_repertoires"

       transform(list_of_repertoires):
            given a list of repertoires the function will use the fitted BOW dictionary to return
            the bag of words vectors for each repertoire in the list

       """
    def __init__(self):

        self.vectorizer = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 1))


    def fit(self,list_of_repertoires):

        """
           fits the BOW dictionary based on the repertoires given in "list_of_repertoires"

                   Parameters:
                           list_of_repertoires (list): A list of pandas DataFrame's that have a column named "cdr3_rearrangement"

                   Returns:
                           None
        """

        cmp = []
        for d in tqdm(list_of_repertoires,leave=False):
            for cdr3 in d['cdr3_rearrangement']:
                cmp.append(' '.join(lempel_ziv_decomposition(cdr3)))
        self.vectorizer.fit(cmp)

    def transform(self,list_of_repertoires):
        """
              transforms a list of repertoires into a list of bag of words vectors derived based on fitted repertoires

                      Parameters:
                              list_of_repertoires (list): A list of pandas DataFrame's that have a column named "cdr3_rearrangement"

                      Returns:
                              None
       """
        n_encoded = []
        for d in tqdm(list_of_repertoires):
            cmp = []
            for cdr3 in d['cdr3_rearrangement']:
                cmp.append(' '.join(lempel_ziv_decomposition(cdr3)))
            n_encoded.append(self.vectorizer.transform([' '.join(cmp)]).todense())
        n_encoded = np.array(n_encoded)
        n_encoded = n_encoded.squeeze()
        return n_encoded


