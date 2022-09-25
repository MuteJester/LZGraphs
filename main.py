import os

import pandas as pd

sample_path = 'C:/Users/Tomas/Desktop/Immunobiology/HIV C1/'
samples = os.listdir(sample_path)
table_test = pd.read_table(sample_path+samples[0],low_memory=False)

T  = table_test[table_test.cdr3_rearrangement.notna()][['cdr3_rearrangement','cdr3_amino_acid',
       'chosen_v_family','chosen_j_family','chosen_j_gene','chosen_v_gene','chosen_j_allele','chosen_v_allele']].dropna()

T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV0','TRBV'))
T['chosen_v_family'] = T['chosen_v_family'].apply(lambda x: x.replace('TCRBV','TRBV'))
T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ0','TRBJ'))
T['chosen_j_family'] = T['chosen_j_family'].apply(lambda x: x.replace('TCRBJ','TRBJ'))

T['V'] = T['chosen_v_family']+'-'+T['chosen_v_gene'].astype(int).astype(str)+'*0'+T['chosen_v_allele'].astype(int).astype(str)
T['J'] = T['chosen_j_family']+'-'+T['chosen_j_gene'].astype(int).astype(str)+'*0'+T['chosen_j_allele'].astype(int).astype(str)



from src.LZGraph.AAPLZGraph import AAPLZGraph
from src.LZGraph.NDPLZGraph import NDPLZGraph, encode_sequence

lzg = NDPLZGraph(T,verbose=True)

# print('Nodes: ',len(lzg.nodes))
# print('Edges: ',len(lzg.edges))
# print('CASSLGIRRTNTEAFF Pgen: ',lzg.walk_probability('CASSLGIRRTNTEAFF'))
# print('Random Generate: ',lzg.unsupervised_random_walk())


print('Nodes: ',len(lzg.nodes))
print('Edges: ',len(lzg.edges))
print('TGTGCCAGCAGCCAGCAGGGCCGGGATACGCAGTATTTT Pgen: ',lzg.walk_probability(encode_sequence('TGTGCCAGCAGCCAGCAGGGCCGGGATACGCAGTATTTT')))
print('Random Generate: ',lzg.unsupervised_random_walk())
print(lzg.sequence_variation_curve('TGTGCCAGCAGCCAGCAGGGCCGGGATACGCAGTATTTT'))
