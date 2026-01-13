# Tutorials for LZGraphs

Welcome to the tutorials page for LZGraphs, where you will find step-by-step guides on leveraging the power of LZGraphs for T-cell receptor beta chain (TCRB) repertoire analysis. Whether you are constructing graphs or exploring advanced features, these tutorials are designed to enhance your understanding and utilization of LZGraphs in your research endeavors.

## Getting Started with Graph Construction

LZGraphs offers various graph types to suit your analysis needs, including Amino Acid Positional, Nucleotide Double Positional, and Naive graphs. Each graph type is tailored to specific aspects of TCRB repertoire analysis, providing a versatile toolkit for your research.

### Amino Acid Positional and Nucleotide Double Positional Graphs

These graph types are pivotal for in-depth TCRB repertoire analysis. Follow the example below to construct an Amino Acid Positional graph. The process is analogous for Nucleotide Double Positional Graphs, with minor adjustments as noted.

#### Essential Points:

- For Nucleotide Double Positional Graphs, use "cdr3_rearrangement" as the column name for sequences.
- For Amino Acid Positional Graphs, the sequence column should be "cdr3_amino_acid".
- V and J genes/alleles are categorized under "V" and "J" columns, respectively. This genetic information enriches the graph and enhances sequence generation and genomic insights.
- Including V/J genes/alleles is optional but recommended for comprehensive analysis.

```python
from LZGraphs import AAPLZGraph
import pandas as pd

# Load your data
data = pd.read_csv("/path/to/data/file.csv")

# Construct the graph
lzgraph = AAPLZGraph(data, verbose=True)
```
Verbose:
```markdown
Gene Information Loaded.. |  0.01  Seconds
Graph Constructed.. |  0.94  Seconds
Graph Metadata Derived.. |  0.94  Seconds
Individual Subpattern Empirical Probability Derived.. |  0.98  Seconds
Graph Edge Weight Normalized.. |  1.0  Seconds
Graph Edge Gene Weights Normalized.. |  1.13  Seconds
Terminal State Map Derived.. |  1.2  Seconds
Individual Subpattern Empirical Probability Derived.. |  1.37  Seconds
Terminal State Map Derived.. |  1.37  Seconds
LZGraph Created Successfully.. |  1.37  Seconds
==============================
```
The data file is a pandas DataFrame that includes the
sequences under a column named "cdr3_amino_acid".
Optionally, it can also have "V" and "J" columns containing 
relevant genomic information such as Gene/Family/Allele.
Any one of these columns can be used to create a graph, 
with functionality constrained by the genetic information on the edges.

Example of the input DataFrame:

| cdr3_amino_acid  | V           | J          |
|------------------|-------------|------------|
| CASSLEPSGGTDTQYF | TRBV16-1*01 | TRBJ1-2*01 |
| CASSDTSGGTDTQYF  | TRBV1-1*01  | TRBJ1-5*01 |
| CASSLEPQTFTDTFFF | TRBV16-1*01 | TRBJ2-7*01 |

### Constructing a Naive LZGraph

Below is an example of constructing a Naive LZGraph.

```python 
from LZGraphs.Graphs.Naive import NaiveLZGraph, generate_kmer_dictionary

list_of_sequences = ['TGTGCCAGCAGGGCGGGATACGCAGTATTT',
                     'TGTGCCAGCAGCCAGCATCGTCGCAGTATTTT'
                     'TGTGCCAGCAGCCAGCAGGGCCGGGATACGCAGTATTTT'
                     ...]

# generate all possible nodes with kmers up to length 6                 
dictionary = generate_kmer_dictionary(6)
# construct a NaiveLZGraph from your list_of_sequences based on the generated dictionary
lzgraph = NaiveLZGraph(list_of_sequences, dictionary, verbose=True)
```
Note:

- The `generate_dictionary` function takes an integer `(X)` and generates all possible patterns that can be constructed from `(X)` nucleotides. For CDR3 sequences, we suggest using `X=6` as the maximum observed sub-pattern length in the CDR3 region is 6.
- The dictionary can be replaced with any number of sub-patterns if not all combinations are desired.
- The Naive LZGraph is most relevant in the context of nucleotides, as amino acid CDR3 sequences result in sparse graphs.

---

## Exploring Graph Attributes

After constructing an LZGraph, there are several attributes available to quickly infer repertoire statistical properties as well as explore the graph object itself, built above the networkx DiGraph class. Below are examples of how you can evaluate and explore those attributes.

### Amino Acid Positional and Nucleotide Double Positional Graphs

#### Exploring Graph Nodes and Edges

```python
# Retrieving the list of all graph nodes / edges
node_list = list(lzgraph.nodes)
edge_list = list(lzgraph.edges) 

print('First 10 Nodes: ',list(lzgraph.nodes)[:10])
print('First 10 Edges: ',list(lzgraph.edges)[:10])
```
Output:
```markdown
First 10 Nodes:  ['C_1', 'A_2', 'S_3', 'SQ_5', 'Q_6', 'G_7', 'R_8', 'D_9', 'T_10', 'QY_12']
First 10 Edges:  [('C_1', 'A_2'), ('C_1', 'T_2'), ('C_1', 'V_2'), ('C_1', 'S_2'), ('C_1', 'R_2'), ('C_1', 'G_2'), ('C_1', 'D_2'), ('C_1', 'F_2'), ('C_1', 'I_2'), ('A_2', 'S_3')]
```
#### Distribution of Observed Sequence Lengths:
After fitting an LZGraph to a repertoire you can quickly access the
distribution of different sequence lengths observed in the repertoire:

```python
print(lzgraph.lengths)
```
Output:
```python
{13: 2973, 15: 5075, 14: 4412, 16: 2862,
 12: 1147, 17: 1401, 19: 268, 22: 11, 20: 108,
 11: 508, 18: 655, 21: 38, 10: 72,
 24: 1, 9: 14, 7: 2, 4: 1,8: 1}
```
Note:
The key of the dictionary is an observed sequence length, the value corresponds to the
total number of sequences observed matching the key length.
#### Exploring Graph Initial and Terminal States:
 Initial state is defined as the first sub-pattern observed in a sequence,
 Terminal state is the last sub-pattern observed in a sequence, the lists and counts
 of all observed initial and final states are saved in every LZGraph and can be 
 accessed:
##### Getting all the initial states
```python
# Getting all the initial states
init_states = lzgraph.initial_states
print(init_states)
```
Output:
```python
C_1    19523
R_1       20
dtype: int64
```
##### Getting all the terminal states

```python
term_states = lzgraph.print(lzgraph.terminal_states)
print(term_states)
```
Output:
```python
F_13      2672
F_15      4490
F_14      3860
F_16      2502
F_12      1028
          ... 
G_9          1
AFF_12       1
FF_11        1
F_8          1
HF_22        1
Length: 86, dtype: int64
```
#### Graph V and J Marginal Distributions:
If V and J annotation were provided while constructing the graph,
the genetic data was embedded into each edge of the graph.
The marginal probability of each gene/allele can be observed by the following way:
```python
# print(lzgraph.marginal_vgenes)
print(lzgraph.marginal_jgenes)
```
Output:
```markdown
TRBJ1-2*01    0.159804
TRBJ1-1*01    0.143588
TRBJ2-7*01    0.141951
TRBJ2-1*01    0.090081
TRBJ2-3*01    0.088086
TRBJ1-5*01    0.076986
TRBJ2-5*01    0.075963
TRBJ2-2*01    0.057343
TRBJ1-4*01    0.050437
TRBJ1-6*01    0.047675
TRBJ1-3*01    0.043787
TRBJ2-6*01    0.015500
TRBJ2-4*01    0.008798
Name: J, dtype: float64
```

## Calculating the LZCentrality

The `LZGraph` library provides a function, `LZCentrality`, for calculating the LZCentrality of a given CDR3 sequence in a repertoire represented by an LZGraph. The LZCentrality is a measure of sequence centrality within a repertoire, as presented in our paper.

Here's how you can use the `LZCentrality` function:

```python
from LZGraphs import LZCentrality, NDPLZGraph

# Assume we have a list of CDR3 sequences and a sequence of interest
sequences = ["sequence1", "sequence2", "sequence3", ...]
sequence_of_interest = "sequence1"

# Create an LZGraph
graph = NDPLZGraph(sequences)

# Calculate the LZCentrality
lzcentrality = LZCentrality(graph, sequence_of_interest)
print(f"The LZCentrality of {sequence_of_interest} is {lzcentrality}")
```

In this example, `NDPLZGraph` is the LZGraph class used. You can replace it with any other LZGraph class, such as `AAPLZGraph`.

This function provides an efficient and scalable way to calculate the LZCentrality, allowing you to gain important insights into the centrality of a sequence within a TCRB repertoire.

## Calculating the K1000 Diversity Index

The `LZGraph` library provides a convenient function, `K1000_Diversity`, for calculating the K1000 Diversity index of a list of CDR3 sequences. This index, as presented in our paper, offers a new way to measure the diversity of a repertoire.

Here's how you can use the `K1000_Diversity` function:

```python
from LZGraphs.Metrics import K1000_Diversity
from LZGraphs import AAPLZGraph

# Assume we have a list of CDR3 sequences
sequences = ["sequence1", "sequence2", "sequence3", ...]

# Calculate the K1000 Diversity index
k1000 = K1000_Diversity(sequences, AAPLZGraph.encode_sequence, draws=30)
print(f"The K1000 Diversity index is {k1000}")
```

In this example, `AAPLZGraph.encode_sequence` is the LZGraph encoding function used. You can replace it with any other LZGraph encoding function, such as `NDPLZGraph.encode_sequence`.

The `draws` parameter specifies the number of draws for the resampling test. The higher the value, the more realizations will be generated for the K1000 index, and the returned mean value will be a better approximation.

The `K1000_Diversity` function works by resampling the list of sequences and building LZGraphs based on the provided encoding function. The resampling is performed `draws` number of times, with each resampled set containing 1000 unique sequences. For each resampled set, the K1000 Diversity index is calculated using the 'nodes' value of the last item in the result dictionary obtained from the NodeEdgeSaturationProbe's resampling test. The average K1000 Diversity index from all the resampling tests is then returned.

This function provides an efficient and scalable way to calculate the K1000 Diversity index, allowing you to gain important insights into the diversity of a TCRB repertoire.

## Deriving Sequence Generation Probability (Pgen)
After an LZGraph is derived for a given repertoire, one can assess the generation
probability (Pgen) of a new sequence with respect to that repertoire (LZGraph).
```python
new_sequence = "CASRGERGDNEQFF"
# encode the sequence into relevant graph format
encoded_sequence = AAPLZGraph.encode_sequence(new_sequence)
#output
# ['C_1', 'A_2', 'S_3', 'R_4', 'G_5', 'E_6', 'RG_8', 'D_9', 'N_10', 'EQ_12', 'F_13', 'F_14']

pgen = lzgraph.walk_probability(encoded_sequence)
print(f'{new_sequence} Pgen: {pgen}')
```
Output:
```markdown
CASRGERGDNEQFF Pgen: 6.692834748776949e-13
```

## Generating New Sequences
An LZGraph constructed from a source repertoire can generate new sequences following
the statistical properties observed in the source.
Note that that if V and J genes were provided the generation function will be testing
that the random sequence and their sub-pattern all are associated to initially random
V and J genes selected before generating the sequence.
```python
generated_walk,selected_v,selected_j = lzgraph.genomic_random_walk()
# replace genomic_random_walk with random_walk() if annotation was not provided
print(generated_walk,selected_v,selected_j)
```
Output:
```markdown
['C_1', 'S_2', 'A_3', 'T_4', 'G_5', 'GI_7', 'Q_8', 'Y_9', 'QE_11', 'TQ_13', 'YF_15'],
'TRBV29-1*01',
'TRBJ2-5*01'
```
You can transform the resulting walk back into a sequence by cleaning each node and
concatenating them into a single string.
Each LZGraph has a "clean_node" method that accompanies the relevant "encode_sequence"
method.
```python
clean_sequence = ''.join([AAPLZGraph.clean_node(node) for node in generated_walk])
# Note that AAPLZGraph needs to be replaced in case you used the NDPLZGraph
print(clean_sequence)
```
Output:
```markdown
CASSPGTGGTGELFF
```

## Feature Vector From Graph (Eigen Centrality)
One can derive a vector representation of an LZGraph in order to perform any
modeling task. The resulting vector in a sense capture the structural dynamics of 
a repertoire.
Note that the number of features in the resulting vector equal the number of nodes
in the graph, and the value in each node correspond to each node, so it is recommended
using the Naive LZGraph, this allows to create LZGraphs to different repertoires while
maintaining the same dictionary of sub-patterns (nodes).

```python
from LZGraphs.Graphs.Naive import NaiveLZGraph, generate_kmer_dictionary

# Create a dictionary that will be shared between all repertoires
dictionary = generate_kmer_dictionary(6)
# create a graph, this can be done in a loop over many repertoires
lzgraph = NaiveLZGraph(list_of_sequences, dictionary, verbose=True)
# derive the feature vector
feature_vector = lzgraph.eigenvector_centrality()
print(pd.Series(feature_vector))
```
Output:
```markdown
A         3.009520e-01
T         1.183398e-01
G         1.186366e-01
C         2.461758e-01
AA        1.252643e-01
              ...     
CCCCGC    5.252576e-10
CCCCCA    5.252576e-10
CCCCCT    5.252576e-10
CCCCCG    5.252576e-10
CCCCCC    5.252576e-10
Length: 5460, dtype: float64
```

Note:
The `generate_dictionary` function given the value 6 return `5460` sub-patterns,
many of which may not exist in your graph, nonetheless this promises that all
the repertoires encoded will produce the same feature vector dimension but the
value across the different sub-pattern will be different.

## Plotting Graph Related Features
The library provides multiple plots that can be useful when analyzing sequence
and comparing between sequences.
### Ancestors Descendants Curves Plot
In this chart you can examine at each sub-pattern of a given sequence
the amount descendants nodes reachable at each node contained in your sequence,
as well as the amount of ancestors at each node.
Sequence with different attribute differ both by the slope and the convergence
rate of these curve as well as by the intersection point between the curves.

```python
from LZGraphs.Visualization.Visualize import ancestors_descendants_curves_plot

sequence = 'CASTPGTASGYTF'
ancestors_descendants_curves_plot(lzgraph, sequence)
```
Output:
![alt text](images/ad_curve_example.png)
### Sequence Possible Paths Plot
In this chart we look at a reduced and immediate version of Descendants curve.
For each sub-pattern derived from a given sequence and based on an LZGraph,
We can examine the number of alternatives there are at each node, this indicates
the rarity of a sequence and is correlated with the difference from the 
mean Levenshtein distance of the repertoire as shown in the LZGraphs paper.

```python
from LZGraphs.Visualization.Visualize import sequence_possible_paths_plot

sequence = 'CASTPGTASGYTF'
sequence_possible_paths_plot(lzgraph, sequence)
```
Output:
![alt text](images/sequence_path_number_example.png)

### Node Genomic Variability Plot
In this chart we look at the number of V and J genes/alleles per node in a given sequence with respect
to a given repertoire.
Not only can one infer the sub-patterns in a sequence that have the exceptional number of V and J alternatives
but also when comparing between the same sequence in different repertoires (different LZGraphs) one can infer
the amount difference at each sub-pattern between the two repertoires.

```python
from LZGraphs.Visualization.Visualize import sequence_genomic_node_variability_plot

sequence = 'CASTPGTASGYTF'
sequence_genomic_node_variability_plot(lzgraph, sequence)
```
Output:
![alt text](images/number_of_vj_at_nodes_example.png)


### Edge Genomic Variability Plot
In this chart we look at the number of V and J genes/alleles per edge in a given sequence with respect
to a given repertoire.

* Allele/gene names colored in red signify that the allele/gene appeared in all the edges in the given sequence.
* Black cells signify that this spesific allele/gene wasnt observed at that edge.
* The color gradient at each cell represents the probability of choosing that edge under the constraint of having
 that specific V/J.

```python
from LZGraphs.Visualization.Visualize import sequence_genomic_edges_variability_plot

sequence = 'CASTPGTASGYTF'
sequence_genomic_edges_variability_plot(lzgraph, sequence)
```
Output:
![alt text](images/number_of_vj_at_edges_example.png)



## BOW Vectorizer
The library also provides a convenient class implementing the BOW logic presented in the paper.
Note that each instance of the BOW wrapper class has to fitted on a list of sequences (repertoire) before it
can transform new lists of sequences into BOW vectors.

```python
from LZGraphs import LZBOW
from LZGraphs import NDPLZGraph

sequence_list = data.cdr3_rearrangement.to_list()

# create vectorizer and choose the Nucleotide Double Positional (ndp) encdoing function (default is Naive)
vectorizer = LZBOW(encoding_function=NDPLZGraph.encode_sequence)

# fit on sequence list
vectorizer.fit(sequence_list)

# BOW dictionary
vectorizer.dictionary

bow_vector = vectorizer.transform(new_list_of_sequences)
```

