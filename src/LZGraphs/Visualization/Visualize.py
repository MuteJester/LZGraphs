


import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pylab

from ..Utilities.decomposition import lempel_ziv_decomposition

mpl.rcParams['figure.figsize'] = (15,8)
sns.set_context('poster')

def sequence_genomic_edges_variability_plot(graph, cdr3_sample, threshold=None, figsize=None):
    """Generate a heatmap showing V and J gene distribution at each edge.

    This plot shows which V and J genes are associated with each edge transition
    in a given sequence. Genes appearing in all edges are colored red.

    Note:
        This function only works with NDPLZGraph and AAPLZGraph which have gene
        annotations. It will raise an AttributeError with NaiveLZGraph.

    Args:
        graph: An NDPLZGraph or AAPLZGraph object with gene annotations.
        cdr3_sample (str): A sequence of nucleotides/amino-acids.
        threshold (float): Value between 0-1 controlling percentage of paths to show.
            Default is None (show all).
        figsize (tuple): Figure size as (width, height). Default is (15, 8).

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
    """Generate a bar plot showing V and J gene counts at each node.

    This plot shows how many unique V and J genes/alleles are associated
    with each node (LZ subpattern) in a given sequence.

    Note:
        This function only works with NDPLZGraph and AAPLZGraph which have gene
        annotations. It will raise an AttributeError with NaiveLZGraph.

    Args:
        graph: An NDPLZGraph or AAPLZGraph object with gene annotations.
        cdr3 (str): A sequence of nucleotides/amino-acids.

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

def sequence_possible_paths_plot(graph, sequence):
    """Generate a Matplotlib plot showing the number of alternative paths at each node.

    This plot shows how many outgoing edges exist at each node in the sequence's
    path through the graph. Higher values indicate more alternative paths available.

    Args:
        graph (LZGraph): An LZGraph object (NDPLZGraph, AAPLZGraph, or NaiveLZGraph).
        sequence (str): A sequence of nucleotides/amino-acids depending on the graph type.

    Returns:
        None: Shows figure via plt.show()
    """
    encoded_nodes = graph.encode_sequence(sequence)
    curvec = [(graph.graph.out_degree(i)) for i in encoded_nodes]

    # Generate clean labels for x-axis
    if hasattr(graph, 'clean_node'):
        labels = [graph.clean_node(node) for node in encoded_nodes]
    else:
        # Fallback for NaiveLZGraph which uses raw LZ decomposition as nodes
        labels = encoded_nodes

    sns.lineplot(x=np.arange(len(curvec)), y=curvec, color='tab:blue')
    sns.scatterplot(x=np.arange(len(curvec)), y=curvec, color='tab:blue')
    plt.xticks(np.arange(len(curvec)), labels=labels, rotation=0)
    plt.grid(lw=2, ls=':')
    plt.ylabel('# of Paths')
    plt.xlabel('LZ Sub Pattern')
    plt.show()

def ancestors_descendants_curves_plot(graph, sequence):
    """Generate a Matplotlib plot showing ancestors and descendants at each node.

    This plot shows how many ancestor and descendant nodes exist for each node
    in the sequence's path through the graph. Useful for understanding sequence
    position in the graph structure.

    Args:
        graph (LZGraph): An LZGraph object (NDPLZGraph, AAPLZGraph, or NaiveLZGraph).
        sequence (str): A sequence of nucleotides/amino-acids depending on the graph type.

    Returns:
        None: Shows figure via plt.show()
    """
    encoded_nodes = graph.encode_sequence(sequence)
    descendants_curve = []
    ancestors_curve = []

    for node in encoded_nodes:
        descendants = nx.descendants(graph.graph, node)
        descendants_curve.append(len(descendants))
        ancestors = nx.ancestors(graph.graph, node)
        ancestors_curve.append(len(ancestors))

    ancestors_curve = np.array(ancestors_curve)
    descendants_curve = np.array(descendants_curve)

    # Generate clean labels for x-axis
    if hasattr(graph, 'clean_node'):
        labels = [graph.clean_node(node) for node in encoded_nodes]
    else:
        # Fallback for NaiveLZGraph which uses raw LZ decomposition as nodes
        labels = encoded_nodes

    plt.title(sequence)
    ax = sns.lineplot(x=np.arange(len(descendants_curve)), y=descendants_curve,
                      label='Descendants Curve', color='tab:orange')
    ax.set_ylabel('Number of Descendants')
    ax2 = ax.twinx()

    sns.lineplot(x=np.arange(len(descendants_curve)), y=ancestors_curve,
                 label='Ancestors Curve', ax=ax2)
    ax2.set_ylabel('Number of Ancestors')

    plt.xticks(np.arange(len(descendants_curve)), labels=labels, rotation=45)
    plt.grid(lw=2, ls=':')
    plt.xlabel('LZ Sub-pattern')
    plt.show()

def draw_graph(graph, file_name='LZGraph.png'):
    """Generate a plot of a given graph and save it to a file.

    Args:
        graph (LZGraph): An LZGraph object (NDPLZGraph, AAPLZGraph, or NaiveLZGraph).
        file_name (str): The name of the saved image file. Default is 'LZGraph.png'.

    Returns:
        None: Saves the figure to disk.
    """
    # Get the underlying NetworkX graph
    nx_graph = graph.graph

    # Initialize Figure
    plt.figure(num=None, figsize=(30, 30), dpi=300)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(nx_graph)
    nx.draw_networkx_nodes(nx_graph, pos, alpha=0.3, node_size=100)
    nx.draw_networkx_edges(nx_graph, pos, alpha=0.3)

    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    del fig
