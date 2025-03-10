import unittest
import pandas as pd
import numpy as np
from LZGraphs import generate_kmer_dictionary, NaiveLZGraph, NDPLZGraph, get_k1000_diversity, AAPLZGraph
from LZGraphs.Metircs.Metrics import LZCentrality


class TestLZGraphs(unittest.TestCase):

    def setUp(self):
        self.test_data_1 = pd.read_csv('./ExampleData1.csv',index_col=0)
        self.test_data_2 = pd.read_csv('./ExampleData2.csv',index_col=0)
        self.test_data_3 = pd.read_csv('./ExampleData3.csv',index_col=0)

    # Define a teardown method if you need to clean up after each test method (optional)
    def tearDown(self):
        # Teardown code here, e.g., closing files or connections
        pass

    def test_naive_lz_dictionary_creation(self):
        lz_dictionary = generate_kmer_dictionary(6)

        keys = lz_dictionary[15:25]
        prev =['GC', 'CA', 'CT', 'CG', 'CC', 'AAA', 'AAT', 'AAG', 'AAC', 'ATA']


        self.assertEqual(keys,prev)

    def test_creation_of_naive_lz_graph(self):
        lz_dictionary = generate_kmer_dictionary(6)
        my_naive_lzgraph = NaiveLZGraph(self.test_data_1['cdr3_rearrangement'], lz_dictionary)


        self.assertEqual(len(my_naive_lzgraph.edges),5137)
        self.assertEqual(my_naive_lzgraph.initial_states['T'],4994)
        self.assertEqual(my_naive_lzgraph.subpattern_individual_probability.loc['C','proba'],0.050210381498478625)
        self.assertEqual(my_naive_lzgraph.subpattern_individual_probability.loc['CA','proba'],0.048503228527530355)
        self.assertEqual(my_naive_lzgraph.per_node_observed_frequency['AGT'],406)
        self.assertCountEqual(my_naive_lzgraph.terminal_state_map['C'],
                         ['C', 'TTT', 'T', 'TT', 'TTC', 'TC', 'CTTT', 'CTTC', 'TTTT'])
        self.assertCountEqual(my_naive_lzgraph.terminal_state_map['TTTT'],
                         ['C', 'TTT', 'T', 'TT', 'TTC', 'TC', 'CTTT', 'CTTC', 'TTTT'])



        feature_vector = my_naive_lzgraph.eigenvector_centrality()
        self.assertEqual(np.round(feature_vector['AA'],5),0.12581)

        lzpgens = []
        # iterate over each sequence
        for sequence in self.test_data_1['cdr3_rearrangement'].iloc[:15]:
            # convert sequence to graph sub-patterns
            walk = NaiveLZGraph.encode_sequence(sequence)
            # calculate the lzpgen based on the fitted NaiveLZGraph
            lzpgen = my_naive_lzgraph.walk_probability(walk, verbose=False)
            lzpgens.append(lzpgen)
        self.assertEqual(np.round(np.log(lzpgens[0]),2),-84.92)

        self.assertIsNotNone(my_naive_lzgraph.unsupervised_random_walk())
        self.assertEqual(len(my_naive_lzgraph.random_walk(25)[0]),25)

    def test_creation_of_NDP_lz_graph(self):
        my_ndp_lzgraph = NDPLZGraph(self.test_data_2)



        self.assertEqual(len(my_ndp_lzgraph.edges),20587)
        self.assertEqual(my_ndp_lzgraph.initial_states['T0_1'],4991)
        self.assertEqual(my_ndp_lzgraph.subpattern_individual_probability.loc['A0_19','proba'],0.0005410062716652975)
        self.assertEqual(my_ndp_lzgraph.subpattern_individual_probability.loc['CA2_7','proba'],0.04535435910794077)
        self.assertEqual(my_ndp_lzgraph.per_node_observed_frequency['AC0_14'],69)
        self.assertCountEqual(my_ndp_lzgraph.terminal_state_map['TC1_51'],
                         ['C2_54', 'T2_57', 'TTT0_60', 'TTC0_54', 'TC1_51', 'C2_57', 'T2_54', 'T2_60', 'TT1_57',
                          'TT1_60', 'TTT0_57', 'TC1_60', 'C2_60', 'TTT0_63', 'TTC0_57', 'TTC0_60', 'T2_63']
                         )
        self.assertCountEqual(my_ndp_lzgraph.terminal_state_map['TC1_60'],
                         ['TC1_60', 'TTT0_63', 'T2_63']
                         )
        self.assertCountEqual(my_ndp_lzgraph.terminal_state_map['TT1_45'],
        ['TT1_45', 'TTT0_51', 'TTT0_48'])

        feature_vector = my_ndp_lzgraph.eigenvector_centrality()
        self.assertEqual(np.round(np.log(feature_vector['AT0_26']),5),-38.79923)

        lzpgens = []
        # iterate over each sequence
        for sequence in self.test_data_2['cdr3_rearrangement'].iloc[:15]:
            # convert sequence to graph sub-patterns
            walk = NDPLZGraph.encode_sequence(sequence)
            # calculate the lzpgen based on the fitted NaiveLZGraph
            lzpgen = my_ndp_lzgraph.walk_probability(walk, verbose=False)
            lzpgens.append(lzpgen)
        self.assertEqual(np.round(np.log(lzpgens[0]),2),-43.25)

        self.assertIsNotNone(my_ndp_lzgraph.unsupervised_random_walk())
        self.assertIsNotNone(my_ndp_lzgraph.gene_random_walk(seq_len='unsupervised'))
        self.assertEqual(my_ndp_lzgraph.random_walk(initial_state='T0_1')[0],'T0_1')

        self.assertEqual(my_ndp_lzgraph.marginal_vgenes['TRBV2-1*01'],0.0502)
        self.assertEqual(my_ndp_lzgraph.lengths[42],1143)
        self.assertEqual(my_ndp_lzgraph.lengths[27],1)
        self.assertEqual(my_ndp_lzgraph.terminal_states['C2_42'],316)

    def test_creation_of_AAP_lz_graph(self):
        my_aap_lzgraph = AAPLZGraph(self.test_data_3)



        self.assertEqual(len(my_aap_lzgraph.edges),9528)
        self.assertEqual(my_aap_lzgraph.initial_states['C_1'],4996)
        self.assertEqual(my_aap_lzgraph.subpattern_individual_probability.loc['Y_9','proba'],0.0031496933193346966)
        self.assertEqual(my_aap_lzgraph.subpattern_individual_probability.loc['Y_5','proba'],0.00014735407341916708)
        self.assertEqual(my_aap_lzgraph.per_node_observed_frequency['SG_5'],89)

        self.assertCountEqual(my_aap_lzgraph.terminal_state_map['F_19'],
                              ['F_19', 'F_21', 'F_20']
                              )

        self.assertCountEqual(my_aap_lzgraph.terminal_state_map['F_17'],
                              ['F_19', 'F_21', 'F_18', 'F_17', 'F_20']
                              )

        self.assertCountEqual(my_aap_lzgraph.terminal_state_map['TF_12'],
                              ['YF_18', 'TF_12', 'YF_17', 'F_16', 'FF_16', 'TF_17', 'TF_18', 'FF_15', 'TF_20', 'HF_19',
                               'FF_17', 'YF_16', 'F_17', 'F_22', 'TF_15', 'F_18', 'FF_18', 'HF_17', 'TF_21', 'F_19',
                               'F_21', 'YF_20', 'HF_20', 'YF_19', 'FF_19', 'TF_19', 'F_15', 'F_14', 'F_20']
                              )

        feature_vector = my_aap_lzgraph.eigenvector_centrality()
        self.assertEqual(np.round(np.log(feature_vector['Q_15']),5),-16.29151)

        lzpgens = []
        # iterate over each sequence
        for sequence in self.test_data_3['cdr3_amino_acid'].iloc[:15]:
            # convert sequence to graph sub-patterns
            walk = NDPLZGraph.encode_sequence(sequence)
            # calculate the lzpgen based on the fitted NaiveLZGraph
            lzpgen = my_aap_lzgraph.walk_probability(walk, verbose=False)
            lzpgens.append(lzpgen)
        self.assertEqual(np.round(np.log(lzpgens[0]),2),-72.09)

        self.assertIsNotNone(my_aap_lzgraph.unsupervised_random_walk())
        self.assertIsNotNone(my_aap_lzgraph.genomic_random_walk())
        self.assertEqual(my_aap_lzgraph.random_walk(initial_state='C_1')[0],'C_1')

        self.assertEqual(my_aap_lzgraph.marginal_vgenes['TRBV19-1*01'],0.0774)
        self.assertEqual(my_aap_lzgraph.lengths[14],1161)
        self.assertEqual(my_aap_lzgraph.lengths[21],5)
        self.assertEqual(my_aap_lzgraph.terminal_states['F_17'],308)

    def test_k1000_index(self):
        k1000 = get_k1000_diversity(list_of_sequences=self.test_data_2['cdr3_rearrangement'].to_list(),
                                    lzgraph_encoding_function='ndp',
                                    draws=50)
        self.assertLessEqual(k1000,2160)
        self.assertGreaterEqual(k1000,2130)

        k1000 = get_k1000_diversity(list_of_sequences=self.test_data_3['cdr3_amino_acid'].to_list(),
                                    lzgraph_encoding_function='aap',
                                    draws=50)
        self.assertLessEqual(k1000, 950)
        self.assertGreaterEqual(k1000, 910)

    def test_lz_centrality(self):
        my_ndp_lzgraph = NDPLZGraph(self.test_data_2)
        lzc = LZCentrality(my_ndp_lzgraph, 'TGTGCCTGCGTAACACAGGGGGTTTGGTATGGCTACACCTTC')

        self.assertEqual(lzc,14.105263157894736)

if __name__ == '__main__':
    unittest.main()
