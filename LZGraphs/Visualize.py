


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pylab

import networkx as nx
from .decomposition import lempel_ziv_decomposition
mpl.rcParams['figure.figsize'] = (15,8)
sns.set_context('poster')

def sequence_genomic_edges_variability_plot(graph, cdr3_sample, threshold=None, figsize=None):
    """ Generate a Matplotlib plot that shows the distribution of V and J genes at each edge in a given sequence.

              Args:
                  graph (LZGraph): An LZGraph object that was embedded with gene annotations.
                  cdr3_sample (str) A sequence of nucleotides/amino-acids depending on the LZGraph passed.
                  threshold (float)  Default = None, if a value between 0-1 that controls the percentage of complete paths to show.
                  figsize (int) the size of the matplotlib figure.
              Returns:
                  None: Shows figure via plt.show()
        """
    vgene_table, jgene_table = graph.path_gene_table(cdr3_sample, threshold)
    plt.figure(figsize=(15, 8) if figsize is None else figsize)
    plt.subplot(1, 2, 1)
    ax = sns.heatmap(jgene_table.iloc[:, :-2],
                     xticklabels=[graph.clean_node(i.split('->')[0]) + '->' + graph.clean_node(i.split('->')[1]) for i in
                                  jgene_table.columns[:-2]],
                     cmap='coolwarm', linewidths=3)
    ax.set_facecolor('xkcd:black')

    label_col_vals = jgene_table.iloc[:, :-2].isna().any(axis=1)
    for i in ax.get_yticklabels():
        if not label_col_vals[i.get_text()]:
            i.set_color("red")

    plt.subplot(1, 2, 2)
    ax = sns.heatmap(vgene_table.iloc[:, :-2],
                     xticklabels=[graph.clean_node(i.split('->')[0]) + '->' + graph.clean_node(i.split('->')[1]) for i in
                                  jgene_table.columns[:-2]],
                     cmap='coolwarm', linewidths=3, yticklabels=vgene_table.index)

    label_col_vals = vgene_table.iloc[:, :-2].isna().any(axis=1)
    for i in ax.get_yticklabels():
        if not label_col_vals[i.get_text()]:
            i.set_color("red")

    ax.set_facecolor('xkcd:black')
    plt.gcf().suptitle(cdr3_sample, fontsize=26)

    plt.tight_layout()
    plt.show()


def sequence_genomic_node_variability_plot(graph, cdr3):
    """ Generate a Matplotlib plot that shows the number of unique genes/alleles at each node of a given sequence based on a given graph.

                 Args:
                     graph (LZGraph): An LZGraph object that was embedded with gene annotations.
                     cdr3 (str) A sequence of nucleotides/amino-acids depending on the LZGraph passed.
                 Returns:
                     None: Shows figure via plt.show()
           """
    j_df = graph.gene_variation(cdr3)
    sns.barplot(data=j_df, x='sp', y='genes', hue='type')
    plt.grid(lw=2, ls=':', axis='y')
    plt.xlabel('LZ Sub Patterns')
    plt.ylabel('Unique Gene/Allele Possibilities')
    plt.legend()
    plt.show()

def sequence_possible_paths_plot(graph,sequence):
    """ Generate a Matplotlib plot that shows the number of alternative paths a sequence can take at each node based on an LZGraph.

                 Args:
                     graph (LZGraph): An LZGraph object that was embedded with gene annotations.
                     sequence (str) A sequence of nucleotides/amino-acids depending on the LZGraph passed.
                 Returns:
                     None: Shows figure via plt.show()
           """
    curvec = [(graph.graph.out_degree(i)) for i in graph.encode_sequence(sequence)]
    sns.lineplot(x=np.arange(len(curvec)), y=(curvec), color='tab:blue')
    sns.scatterplot(x=np.arange(len(curvec)), y=(curvec), color='tab:blue')
    plt.xticks(np.arange(len(curvec)),
               labels=[i for i in lempel_ziv_decomposition(sequence)],
               rotation=0)
    plt.grid(lw=2,ls=':')
    plt.ylabel('# of Paths')
    plt.xlabel('LZ Sub Pattern')
    plt.show()

def ancestors_descendants_curves_plot(graph,sequence):
    """ Generate a Matplotlib plot that shows the number of ancestor and descendant nodes for each node in a given sequence.

                 Args:
                     graph (LZGraph): An LZGraph object that was embedded with gene annotations.
                     sequence (str) A sequence of nucleotides/amino-acids depending on the LZGraph passed.
                 Returns:
                     None: Shows figure via plt.show()
           """
    descendants_curve = []
    ancestors_curve = []

    for node in graph.encode_sequence(sequence):
        descendants = nx.descendants(graph.graph, node)
        descendants_curve.append(len(descendants))
        ancestors = nx.ancestors(graph.graph, node)
        ancestors_curve.append(len(ancestors))
    ancestors_curve = np.array(ancestors_curve)
    descendants_curve = np.array(descendants_curve)
    # out_values =np.array( [(lzg.graph.out_degree(i)) for i in encode_sequence(sequence)])

    plt.title(sequence)
    ax = sns.lineplot(x=np.arange(len(descendants_curve)), y=descendants_curve, label='Descendants Curve',
                      color='tab:orange')
    ax.set_ylabel('Number of Descendants')
    ax2 = ax.twinx()

    sns.lineplot(x=np.arange(len(descendants_curve)), y=ancestors_curve, label='Ancestors Curve', ax=ax2)
    ax2.set_ylabel('Number of Ancestors')

    plt.xticks(np.arange(len(descendants_curve)), labels=lempel_ziv_decomposition(sequence), rotation=45)
    plt.grid(lw=2, ls=':')
    plt.xlabel('LZ Sub-pattern')
    plt.show()

def draw_graph(graph,file_name='LZGraph.png'):
    """ Generate a plot of a given graph and save is in project folder.

                 Args:
                     graph (LZGraph): An LZGraph object that was embedded with gene annotations.
                     file_name (str) the name of the saved image.
                 Returns:
                     None: Shows figure via plt.show()
    """
    #initialze Figure
    plt.figure(num=None, figsize=(30, 30), dpi=300)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos,alpha=0.3,node_size=100)
    nx.draw_networkx_edges(graph,pos,alpha=0.3)
    #nx.draw_networkx_labels(graph,pos)

    plt.savefig(file_name,bbox_inches="tight")
    pylab.close()
    del fig
