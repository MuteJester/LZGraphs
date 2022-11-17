import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from lzgraphs.AAPLZGraph import derive_lz_and_position
from tqdm.auto import tqdm
import pandas as pd

from LZGraphs import graph_union
# sample_path = 'C:/Users/Tomas/Desktop/Immunobiology/HIV C1/'
# samples = os.listdir(sample_path)
# table_test = pd.read_table(sample_path+samples[0],low_memory=False)
#
# T  = table_test[table_test.cdr3_rearrangement.notna()][['cdr3_rearrangement','cdr3_amino_acid',
#        'chosen_v_family','chosen_j_family','chosen_j_gene','chosen_v_gene','chosen_j_allele','chosen_v_allele']].dropna()
#
# T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV0','TRBV'))
# T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV','TRBV'))
# T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ0','TRBJ'))
# T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ','TRBJ'))
#
# T['V'] = T['chosen_v_family']+'-'+T['chosen_v_gene'].astype(int).astype(str)+'*0'+T['chosen_v_allele'].astype(int).astype(str)
# T['J'] = T['chosen_j_family']+'-'+T['chosen_j_gene'].astype(int).astype(str)+'*0'+T['chosen_j_allele'].astype(int).astype(str)

#from LZGraphs.NDPLZGraph import NDPLZGraph, encode_sequence, get_lz_and_pos
from LZGraphs.AAPLZGraph import encode_sequence,AAPLZGraph

#lzg = NDPLZGraph(T,verbose=True)

# print('Nodes: ',len(lzg.nodes))
# print('Edges: ',len(lzg.edges))
# print('CASSLGIRRTNTEAFF Pgen: ',lzg.walk_probability('CASSLGIRRTNTEAFF'))
# print('Random Generate: ',lzg.unsupervised_random_walk())


# print('Nodes: ',len(lzg.nodes))
# print('Edges: ',len(lzg.edges))
# print('TGTGCCAGCAGCCAGCAGGGCCGGGATACGCAGTATTTT Pgen: ',lzg.walk_probability(encode_sequence('TGTGCCAGCAGCCAGCAGGGCCGGGATACGCAGTATTTT')))
# print('Random Generate: ',lzg.unsupervised_random_walk())
# print(lzg.sequence_variation_curve('TGTGCCAGCAGCCAGCAGGGCCGGGATACGCAGTATTTT'))

from tqdm.auto import tqdm
from multiprocessing import Pool
from lzgraphs.misc import window
from tqdm.auto import tqdm
import pickle
from LZGraphs.NDPLZGraph import get_lz_and_pos

class rarity_counter:
    def __init__(self,cdr3):
        self.nodes = list()
        self.edges  = list()
        for cdr3 in tqdm(cdr3, leave=False,position=0):
            LZ, POS, locations = get_lz_and_pos(cdr3)
            nodes_local = list(map(lambda x, y, z: x + str(y) + '_' + str(z), LZ, POS, locations))
            self.nodes +=nodes_local
            self.edges += list(window(nodes_local,2))
        self.nodes = set(self.nodes)
        self.edges = set(self.edges)

# sample_path = 'C:/Users/Tomas/Desktop/Immunobiology/HIV C1/'
#
# NRES = []
# N = 15
# for _ in range(N):
#     samples = os.listdir(sample_path)
#     import random
#     random.shuffle(samples)
#     table_test = pd.read_table(sample_path + samples[0], low_memory=False)
#     T = table_test[table_test.cdr3_rearrangement.notna()][['cdr3_rearrangement', 'cdr3_amino_acid',
#                                                            'chosen_v_family', 'chosen_j_family', 'chosen_j_gene',
#                                                            'chosen_v_gene', 'chosen_j_allele', 'chosen_v_allele']].dropna()
#
# with open('/Users/Tomas/Desktop/Immunobiology/LZGraphs Paper/Final Figures/Data For Figures/192_sample_hiv_ds1_grpah_new_nedge_nnodes_N.pkl','wb') as h:
#     pickle.dump(NRES,h)

import pickle

def proc1():
    with open('C:/Users/Tomas/Desktop/Immunobiology/Covid/aa_cdr3_list.pkl','rb') as h:
        covid_samples = pickle.load(h)
    NRES=[]
    tg = rarity_counter(covid_samples[0])
    n_nodes = [len(tg.nodes)]
    n_edges = [len(tg.edges)]
    n_seqs = [len(covid_samples[0])]

    itr = tqdm(covid_samples[1:],position=1)
    #prev = covid_samples[0]

    agg_nodes = set(tg.nodes)
    agg_edges = set(tg.nodes)
    agg_seq_len = len(covid_samples[0])

    for sample in itr:
        #prev = prev+sample
        tg = rarity_counter(sample)

        #new_added_nodes.append(list(set(tg.nodes) - last_nodes))
        #last_nodes = set(tg.nodes)

        #new_added_edges.append(list(set(tg.edges) - last_edges))
        #last_edges = set(tg.edges)

        agg_nodes = agg_nodes|set(tg.nodes)
        agg_edges = agg_edges|set(tg.edges)
        agg_seq_len  += len(sample)

        n_nodes.append(len(agg_nodes))
        n_edges.append(len(agg_edges))
        n_seqs.append(agg_seq_len)

        itr.set_postfix({'nodes ':n_nodes[-1],'edges ':n_edges[-1]})
    NRES.append({'n_seqs':n_seqs,'n_nodes':n_nodes,'n_edges':n_edges})
    with open('/Users/Tomas/Desktop/Immunobiology/LZGraphs Paper/Final Figures/Data For Figures/covid1400_grpah_new_nedge_nnodes_N.pkl','wb') as h:
        pickle.dump(NRES,h)


def proc2():
    with open('C:/Users/Tomas/Desktop/Immunobiology/Covid/aa_cdr3_list.pkl', 'rb') as h:
        covid_samples = pickle.load(h)
    NRES = []
    for _ in tqdm(range(10),position=0):
        random.shuffle(covid_samples)
        tg = rarity_counter(covid_samples[0])
        n_nodes = [len(tg.nodes)]
        n_edges = [len(tg.edges)]
        n_seqs = [len(covid_samples[0])]

        itr = tqdm(covid_samples[1:], position=1)
        # prev = covid_samples[0]

        agg_nodes = set(tg.nodes)
        agg_edges = set(tg.nodes)
        agg_seq_len = len(covid_samples[0])
        for sample in itr:
            # prev = prev+sample
            tg = rarity_counter(sample)

            # new_added_nodes.append(list(set(tg.nodes) - last_nodes))
            # last_nodes = set(tg.nodes)

            # new_added_edges.append(list(set(tg.edges) - last_edges))
            # last_edges = set(tg.edges)

            agg_nodes = agg_nodes | set(tg.nodes)
            agg_edges = agg_edges | set(tg.edges)
            agg_seq_len += len(sample)

            n_nodes.append(len(agg_nodes))
            n_edges.append(len(agg_edges))
            n_seqs.append(agg_seq_len)

            itr.set_postfix({'nodes ': n_nodes[-1], 'edges ': n_edges[-1]})
        NRES.append({'n_seqs': n_seqs, 'n_nodes': n_nodes, 'n_edges': n_edges})
    with open(
        '/Users/Tomas/Desktop/Immunobiology/LZGraphs Paper/Final Figures/Data For Figures/covid1400_10N_grpah_new_nedge_nnodes_N.pkl',
        'wb') as h:
        pickle.dump(NRES, h)
def proc3():
    with open('/Users/Tomas/Desktop/Immunobiology/LZGraphs Paper/Final Figures/Data For Figures/hiv_dataset1_all_cdr3_seqs.pkl','rb') as h:
        hiv1_s = pickle.load(h)
    with open(
        '/Users/Tomas/Desktop/Immunobiology/LZGraphs Paper/Final Figures/Data For Figures/hiv_dataset2_all_cdr3_seqs.pkl',
        'rb') as h:
        hiv2_s = pickle.load(h)

    covid_samples = hiv1_s+hiv2_s
    NRES = []
    for _ in tqdm(range(10),position=0):
        random.shuffle(covid_samples)
        tg = rarity_counter(covid_samples[0])
        n_nodes = [len(tg.nodes)]
        n_edges = [len(tg.edges)]
        n_seqs = [len(covid_samples[0])]

        itr = tqdm(covid_samples[1:], position=1)
        # prev = covid_samples[0]

        agg_nodes = set(tg.nodes)
        agg_edges = set(tg.nodes)
        agg_seq_len = len(covid_samples[0])
        for sample in itr:
            # prev = prev+sample
            tg = rarity_counter(sample)

            # new_added_nodes.append(list(set(tg.nodes) - last_nodes))
            # last_nodes = set(tg.nodes)

            # new_added_edges.append(list(set(tg.edges) - last_edges))
            # last_edges = set(tg.edges)

            agg_nodes = agg_nodes | set(tg.nodes)
            agg_edges = agg_edges | set(tg.edges)
            agg_seq_len += len(sample)

            n_nodes.append(len(agg_nodes))
            n_edges.append(len(agg_edges))
            n_seqs.append(agg_seq_len)

            itr.set_postfix({'nodes ': n_nodes[-1], 'edges ': n_edges[-1]})
        NRES.append({'n_seqs': n_seqs, 'n_nodes': n_nodes, 'n_edges': n_edges})
    with open(
        '/Users/Tomas/Desktop/Immunobiology/LZGraphs Paper/Final Figures/Data For Figures/hiv1+hiv2_datasets_10N_grpah_new_nedge_nnodes_N.pkl',
        'wb') as h:
        pickle.dump(NRES, h)

def proc4():
    with open(
        '/Users/Tomas/Desktop/Immunobiology/LZGraphs Paper/Final Figures/Data For Figures/emerson_cdr3_neuc_repertoires.pkl',
        'rb') as h:
        covid_samples = pickle.load(h)
    NRES = []
    for NN in (range(10)):
        random.shuffle(covid_samples)
        tg = rarity_counter(covid_samples[0])
        n_nodes = [len(tg.nodes)]
        n_edges = [len(tg.edges)]
        n_seqs = [len(covid_samples[0])]


        aux  = 0
        # prev = covid_samples[0]

        agg_nodes = set(tg.nodes)
        agg_edges = set(tg.nodes)
        agg_seq_len = len(covid_samples[0])
        for en,sample in enumerate(covid_samples[1:]):
            # prev = prev+sample
            tg = rarity_counter(set(sample))

            # new_added_nodes.append(list(set(tg.nodes) - last_nodes))
            # last_nodes = set(tg.nodes)

            # new_added_edges.append(list(set(tg.edges) - last_edges))
            # last_edges = set(tg.edges)

            agg_nodes = agg_nodes | set(tg.nodes)
            agg_edges = agg_edges | set(tg.edges)
            agg_seq_len += len(sample)

            n_nodes.append(len(agg_nodes))
            n_edges.append(len(agg_edges))
            n_seqs.append(agg_seq_len)

            aux+=1
            if aux % 2 == 0:
                aux=0
                print("Realization : ",NN,'rep: ' ,en,' nodes ', n_nodes[-1], ' edges ', n_edges[-1],' n_seqs ' , n_seqs[-1])

        NRES.append({'n_seqs': n_seqs, 'n_nodes': n_nodes, 'n_edges': n_edges})

    with open(
        '/Users/Tomas/Desktop/Immunobiology/LZGraphs Paper/Final Figures/Data For Figures/nuc_double_p_emerson_10N_grpah_new_nedge_nnodes_N.pkl',
        'wb') as h:
        pickle.dump(NRES, h)

def proc5():
    import sonia
    from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
    from sonia.plotting import Plotter
    from sonia.evaluate_model import EvaluateModel
    from sonia.sequence_generation import SequenceGeneration
    from LZGraphs.AAPLZGraph import encode_sequence, AAPLZGraph

    sample_path = 'C:/Users/Tomas/Desktop/Immunobiology/HIV C1/'
    samples = os.listdir(sample_path)

    paried_results = []
    from tqdm.notebook import tqdm
    for sample in tqdm(samples):
        table_test = pd.read_table(sample_path + sample, low_memory=False)

        T = table_test[table_test.cdr3_rearrangement.notna()][['cdr3_rearrangement', 'cdr3_amino_acid',
                                                               'chosen_v_family', 'chosen_j_family', 'chosen_j_gene',
                                                               'chosen_v_gene', 'chosen_j_allele',
                                                               'chosen_v_allele']].dropna()

        T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV0', 'TRBV'))
        T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV', 'TRBV'))
        T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ0', 'TRBJ'))
        T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ', 'TRBJ'))

        T['V'] = T['chosen_v_family'] + '-' + T['chosen_v_gene'].astype(int).astype(str) + '*0' + T[
            'chosen_v_allele'].astype(int).astype(str)
        T['J'] = T['chosen_j_family'] + '-' + T['chosen_j_gene'].astype(int).astype(str) + '*0' + T[
            'chosen_j_allele'].astype(int).astype(str)

        lzg = AAPLZGraph(T)

        samples_ = T[['cdr3_amino_acid', 'V', 'J']]
        qm = SoniaLeftposRightpos(chain_type='human_T_beta',
                                  # gen_seqs=list(Sample_f.values))#,
                                  data_seqs=list(samples_.values))

        qm.add_generated_seqs(int(2e5))

        # define and train model
        qm.infer_selection(epochs=30, verbose=1)

        # from sonia.plotting import Plotter
        # pl=Plotter(qm)
        # plot_sonia=Plotter(qm)
        # plot_sonia.plot_model_learning()

        # # generate Sequences
        # gn = SequenceGeneration(qm)
        # sonia_generated_seqs = gn.generate_sequences_post(len(samples_))
        ev = EvaluateModel(qm)

        table_test = pd.read_table(sample_path + sample)
        Sample_f = table_test[table_test.cdr3_rearrangement.notna()][['cdr3_amino_acid',
                                                                      'chosen_v_family', 'chosen_j_family']].copy()
        Sample_f = Sample_f.dropna()

        Q_data_original, pgen_data_original, ppost_data_original = ev.evaluate_seqs(Sample_f.values)

        XX = -np.log10(lzg.train_pgen)
        YY = -np.log10(ppost_data_original)

        paried_results.append((XX, YY))
        os.system('cls')
    with open('C:/Users/Tomas/Desktop/Immunobiology/LZGraphs Paper/Final Figures/hiv00_lzg_vs_soina_paired_resuls.pkl',
              'wb') as h:
        pickle.dump(paried_results, h)

def proc6():
    sample_path = 'C:/Users/Tomas/Desktop/Immunobiology/HIV C1/'
    samples = os.listdir(sample_path)
    table_test = pd.read_table(sample_path + samples[0], low_memory=False)

    T = table_test[table_test.cdr3_rearrangement.notna()][['cdr3_rearrangement', 'cdr3_amino_acid',
                                                           'chosen_v_family', 'chosen_j_family', 'chosen_j_gene',
                                                           'chosen_v_gene', 'chosen_j_allele',
                                                           'chosen_v_allele']].dropna()

    T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV0', 'TRBV'))
    T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV', 'TRBV'))
    T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ0', 'TRBJ'))
    T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ', 'TRBJ'))

    T['V'] = T['chosen_v_family'] + '-' + T['chosen_v_gene'].astype(int).astype(str) + '*0' + T[
        'chosen_v_allele'].astype(int).astype(str)
    T['J'] = T['chosen_j_family'] + '-' + T['chosen_j_gene'].astype(int).astype(str) + '*0' + T[
        'chosen_j_allele'].astype(int).astype(str)

    gt = AAPLZGraph
    full = gt(T)
    half1 = gt(T.iloc[:len(T) // 2, :])
    half2 = gt(T.iloc[(len(T)) - (len(T) // 2) - 1:, :])

    graph_union(half1,half2)

    #from .LZGraphs.Utilities import graph_union
    print('H1==H1  ', half1==half1)
    print('H2 == H1  ',half2==half1)

def proc7():
    from  LZGraphs.NDPLZGraph import NDPLZGraph
    sample_path = 'C:/Users/Tomas/Desktop/Immunobiology/HIV C1/'
    samples = os.listdir(sample_path)
    table_test = pd.read_table(sample_path + samples[0], low_memory=False)

    T = table_test[table_test.cdr3_rearrangement.notna()][['cdr3_rearrangement', 'cdr3_amino_acid',
                                                           'chosen_v_family', 'chosen_j_family', 'chosen_j_gene',
                                                           'chosen_v_gene', 'chosen_j_allele',
                                                           'chosen_v_allele']].dropna()

    T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV0', 'TRBV'))
    T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV', 'TRBV'))
    T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ0', 'TRBJ'))
    T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ', 'TRBJ'))

    T['V'] = T['chosen_v_family'] + '-' + T['chosen_v_gene'].astype(int).astype(str) + '*0' + T[
        'chosen_v_allele'].astype(int).astype(str)
    T['J'] = T['chosen_j_family'] + '-' + T['chosen_j_gene'].astype(int).astype(str) + '*0' + T[
        'chosen_j_allele'].astype(int).astype(str)

    gt = AAPLZGraph

    lzg = gt(T)

    print(lzg.vj_probabilities)

def AAPG_Test():
    from LZGraphs.AAPLZGraph import AAPLZGraph
    sample_path = 'C:/Users/Tomas/Desktop/Immunobiology/HIV C1/'
    samples = os.listdir(sample_path)
    table_test = pd.read_table(sample_path + samples[0], low_memory=False)

    T = table_test[table_test.cdr3_rearrangement.notna()][['cdr3_rearrangement', 'cdr3_amino_acid',
                                                           'chosen_v_family', 'chosen_j_family', 'chosen_j_gene',
                                                           'chosen_v_gene', 'chosen_j_allele',
                                                           'chosen_v_allele']].dropna()

    T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV0', 'TRBV'))
    T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV', 'TRBV'))
    T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ0', 'TRBJ'))
    T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ', 'TRBJ'))

    T['V'] = T['chosen_v_family'] + '-' + T['chosen_v_gene'].astype(int).astype(str) + '*0' + T[
        'chosen_v_allele'].astype(int).astype(str)
    T['J'] = T['chosen_j_family'] + '-' + T['chosen_j_gene'].astype(int).astype(str) + '*0' + T[
        'chosen_j_allele'].astype(int).astype(str)

    single_sample = T.copy()
    #
    # for sample in samples[:5]:
    #     table_test = pd.read_table(sample_path + sample, low_memory=False)
    #
    #     X = table_test[table_test.cdr3_rearrangement.notna()][['cdr3_rearrangement', 'cdr3_amino_acid',
    #                                                            'chosen_v_family', 'chosen_j_family', 'chosen_j_gene',
    #                                                            'chosen_v_gene', 'chosen_j_allele',
    #                                                            'chosen_v_allele']].dropna()
    #
    #     X['chosen_v_family'] = X['chosen_v_family'].apply(lambda x: x.replace('XCRBV0', 'XRBV'))
    #     X['chosen_v_family'] = X['chosen_v_family'].apply(lambda x: x.replace('XCRBV', 'XRBV'))
    #     X['chosen_j_family'] = X['chosen_j_family'].apply(lambda x: x.replace('XCRBJ0', 'XRBJ'))
    #     X['chosen_j_family'] = X['chosen_j_family'].apply(lambda x: x.replace('XCRBJ', 'XRBJ'))
    #
    #     X['V'] = X['chosen_v_family'] + '-' + X['chosen_v_gene'].astype(int).astype(str) + '*0' + X[
    #         'chosen_v_allele'].astype(int).astype(str)
    #     X['J'] = X['chosen_j_family'] + '-' + X['chosen_j_gene'].astype(int).astype(str) + '*0' + X[
    #         'chosen_j_allele'].astype(int).astype(str)
    #
    #     T = pd.concat([T,X])
    #


    #
    #
    #
    # size_test_graph = AAPLZGraph(T)

    print('============End Of Size Test================')
    print('\n\n')
    with open('hivd1_s0_AAPG_graph_test.pkl', 'rb') as h:
        lzg = pickle.load(h)
    new_graph = AAPLZGraph(single_sample)


    # with open('hivd1_s0_AAPG_graph_test.pkl', 'wb') as h:
    #     new_graph = AAPLZGraph(T)
    #     pickle.dump(new_graph,h)

    print('Test 1 : # Node: ', len(lzg.nodes) == len(new_graph.nodes))
    print('Test 2 : # Edges: ', len(lzg.edges) == len(new_graph.edges))

    t = 0
    bad_nodes = []
    for node in tqdm(lzg.nodes,leave=False):
        new = pd.DataFrame(dict(new_graph.graph[node])).replace(np.nan, 0).round(10)
        original = pd.DataFrame(dict(lzg.graph[node]))
        original = original.loc[new.index, new.columns].replace(np.nan, 0).round(10)
        if new.equals(original):
            t += 1
        else:
            bad_nodes.append(node)

    print('Test 3 : # Mismatched Nodes: ', len(bad_nodes) ,  '   T: ',(t / len(lzg.nodes))*100, ' %')
    print('Test 4 : Is Metadata Equal: ', lzg==new_graph)
    print('Test 5 : Gen Test: ', new_graph.gene_random_walk('unsupervised'))



AAPG_Test()
