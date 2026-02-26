__version__ = "2.2.0"

# =============================================================================
# Graph classes
# =============================================================================
from .graphs.amino_acid_positional import AAPLZGraph
from .graphs.nucleotide_double_positional import NDPLZGraph
from .graphs.naive import NaiveLZGraph

# =============================================================================
# Graph operations
# =============================================================================
from .graphs.graph_operations import graph_union

# =============================================================================
# Bag of Words
# =============================================================================
from .bag_of_words.bow_encoder import LZBOW

# =============================================================================
# Metrics - Diversity
# =============================================================================
from .metrics.diversity import (
    lz_centrality,
    k_diversity,
    k100_diversity,
    k500_diversity,
    k1000_diversity,
    k5000_diversity,
    adaptive_k_diversity,
)

# =============================================================================
# Metrics - Entropy / Information Theory
# =============================================================================
from .metrics.entropy import (
    node_entropy,
    edge_entropy,
    graph_entropy,
    normalized_graph_entropy,
    sequence_perplexity,
    repertoire_perplexity,
    jensen_shannon_divergence,
    cross_entropy,
    kl_divergence,
    mutual_information_genes,
    transition_predictability,
    graph_compression_ratio,
    repertoire_compressibility_index,
    transition_kl_divergence,
    transition_jsd,
    transition_mutual_information_profile,
    path_entropy_rate,
)

# =============================================================================
# Metrics - Saturation
# =============================================================================
from .metrics.saturation import NodeEdgeSaturationProbe

# =============================================================================
# Metrics - Convenience
# =============================================================================
from .metrics.convenience import compare_repertoires

# =============================================================================
# Metrics - PGen Distribution
# =============================================================================
from .metrics.pgen_distribution import LZPgenDistribution, compare_lzpgen_distributions

# =============================================================================
# Utilities
# =============================================================================
from .utilities.helpers import generate_kmer_dictionary
from .utilities.decomposition import lempel_ziv_decomposition

# =============================================================================
# Visualization (optional dependency)
# =============================================================================
try:
    from .visualization.visualize import (
        plot_gene_edge_variability,
        plot_gene_node_variability,
        plot_possible_paths,
        plot_ancestor_descendant_curves,
        plot_graph,
    )
except ImportError:
    pass  # Visualization features not available without matplotlib/seaborn

# =============================================================================
# Exceptions
# =============================================================================
from .exceptions import (
    # Base
    LZGraphError,
    # Input validation
    InputValidationError,
    EmptyDataError,
    MissingColumnError,
    InvalidSequenceError,
    InvalidProbabilityError,
    # Graph construction
    GraphConstructionError,
    EncodingError,
    # Gene data
    GeneDataError,
    NoGeneDataError,
    GeneAnnotationError,
    # Walk/probability
    WalkError,
    NoValidPathError,
    MissingNodeError,
    MissingEdgeError,
    # Serialization
    SerializationError,
    UnsupportedFormatError,
    CorruptedFileError,
    # BOW
    BOWError,
    EncodingFunctionMismatchError,
    UnfittedBOWError,
    # Graph operations
    GraphOperationError,
    IncompatibleGraphsError,
    # Metrics
    MetricsError,
    InsufficientDataError,
)


__all__ = [
    # Graph classes
    'AAPLZGraph',
    'NDPLZGraph',
    'NaiveLZGraph',
    # Graph operations
    'graph_union',
    # Bag of Words
    'LZBOW',
    # Diversity metrics
    'lz_centrality',
    'k_diversity',
    'k100_diversity',
    'k500_diversity',
    'k1000_diversity',
    'k5000_diversity',
    'adaptive_k_diversity',
    # Entropy metrics
    'node_entropy',
    'edge_entropy',
    'graph_entropy',
    'normalized_graph_entropy',
    'sequence_perplexity',
    'repertoire_perplexity',
    'jensen_shannon_divergence',
    'cross_entropy',
    'kl_divergence',
    'mutual_information_genes',
    'transition_predictability',
    'graph_compression_ratio',
    'repertoire_compressibility_index',
    'transition_kl_divergence',
    'transition_jsd',
    'transition_mutual_information_profile',
    'path_entropy_rate',
    # Saturation
    'NodeEdgeSaturationProbe',
    # Convenience
    'compare_repertoires',
    # PGen distribution
    'LZPgenDistribution',
    'compare_lzpgen_distributions',
    # Utilities
    'generate_kmer_dictionary',
    'lempel_ziv_decomposition',
    # Exceptions
    'LZGraphError',
    'InputValidationError',
    'EmptyDataError',
    'MissingColumnError',
    'InvalidSequenceError',
    'InvalidProbabilityError',
    'GraphConstructionError',
    'EncodingError',
    'GeneDataError',
    'NoGeneDataError',
    'GeneAnnotationError',
    'WalkError',
    'NoValidPathError',
    'MissingNodeError',
    'MissingEdgeError',
    'SerializationError',
    'UnsupportedFormatError',
    'CorruptedFileError',
    'BOWError',
    'EncodingFunctionMismatchError',
    'UnfittedBOWError',
    'GraphOperationError',
    'IncompatibleGraphsError',
    'MetricsError',
    'InsufficientDataError',
]
