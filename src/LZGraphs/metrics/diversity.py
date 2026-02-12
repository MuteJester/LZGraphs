import warnings
from typing import Callable, List, Tuple, Union, Optional

import numpy as np
from scipy import stats

from .saturation import NodeEdgeSaturationProbe
from ..exceptions import EmptyDataError, InsufficientDataError


__all__ = [
    'lz_centrality',
    'k_diversity',
    'k100_diversity',
    'k500_diversity',
    'k1000_diversity',
    'k5000_diversity',
    'adaptive_k_diversity',
]


def lz_centrality(lzgraph, sequence: str) -> float:
    """
    Calculates the lz_centrality of a given CDR3 sequence in a repertoire represented by an LZGraph.

    Args:
        lzgraph (LZGraph): The LZGraph representing the repertoire.
        sequence (str): The CDR3 sequence for which lz_centrality needs to be calculated.

    Returns:
        float: The lz_centrality value for the given sequence.

    Calculates the out degree at each node of the given sequence using the `sequence_variation_curve` method
    of the lzgraph object. Missing nodes are penalized by assigning a value of -1. The average of the out degrees
    is then computed using `np.mean` and returned.

    Example:
       >>> graph = NDPLZGraph(Repertoire)
       >>> sequence = "ACCGACAGGATTTACGT"
       >>> lzcentrality = lz_centrality(graph, sequence)
       >>> print(lzcentrality)
    """
    # calculate out degree at each node of the sequence
    svc = lzgraph.sequence_variation_curve(sequence)[1]
    # penalize for missing nodes
    svc = [-1 if not isinstance(i, int) else i for i in svc]
    return np.mean(svc)


def k_diversity(
    list_of_sequences: List[str],
    lzgraph_encoding_function: Callable,
    sample_size: int = 1000,
    draws: int = 25,
    return_stats: bool = False,
    confidence_level: float = 0.95
) -> Union[float, Tuple[float, float, float, float]]:
    """
    Generalized K-Diversity index measuring repertoire diversity via graph node saturation.

    This metric quantifies how many unique LZ-subpatterns (nodes) emerge when sampling
    a fixed number of sequences from the repertoire. Higher values indicate greater diversity.

    Args:
        list_of_sequences (list): A list of CDR3 sequences.
        lzgraph_encoding_function (Callable): The LZGraph encoding function to be used.
            (e.g., AAPLZGraph.encode_sequence, NDPLZGraph.encode_sequence)
        sample_size (int): Number of sequences to sample per draw. Default is 1000.
            Will be automatically adjusted if the repertoire is smaller.
        draws (int): The number of resampling iterations. Defaults to 25.
            More draws provide more stable estimates.
        return_stats (bool): If True, return (mean, std, ci_lower, ci_upper).
            If False, return only the mean. Defaults to False.
        confidence_level (float): Confidence level for interval calculation.
            Defaults to 0.95 (95% confidence interval).

    Returns:
        float or tuple: Mean K-Diversity index, or (mean, std, ci_lower, ci_upper)
            if return_stats=True.

    Raises:
        ValueError: If list_of_sequences is empty or draws < 2.

    Example:
        >>> sequences = df['cdr3_amino_acid'].tolist()
        >>> encoding_function = AAPLZGraph.encode_sequence
        >>> # Simple usage
        >>> k_div = k_diversity(sequences, encoding_function, sample_size=1000)
        >>> # With statistics
        >>> mean, std, ci_low, ci_high = k_diversity(
        ...     sequences, encoding_function, sample_size=1000, return_stats=True
        ... )
    """
    if not list_of_sequences:
        raise EmptyDataError("list_of_sequences cannot be empty")
    if draws < 2:
        raise InsufficientDataError(
            required=2,
            available=draws,
            message="draws must be at least 2 for statistical calculations"
        )

    # Get unique sequences
    unique_sequences = list(set(list_of_sequences))
    n_unique = len(unique_sequences)

    # Auto-adjust sample size if needed
    effective_sample = min(sample_size, int(n_unique * 0.8))
    if effective_sample < sample_size:
        warnings.warn(
            f"Sample size reduced from {sample_size} to {effective_sample} "
            f"due to repertoire size ({n_unique} unique sequences). "
            f"Consider using a smaller sample_size or K{effective_sample}_Diversity.",
            UserWarning
        )

    if effective_sample < 10:
        raise InsufficientDataError(
            required=13,
            available=n_unique,
            message=f"Repertoire too small ({n_unique} unique sequences) for meaningful "
            f"K-Diversity calculation. Need at least 13 unique sequences."
        )

    # Run the saturation probe
    NESP = NodeEdgeSaturationProbe(node_function=lzgraph_encoding_function)
    result = NESP.resampling_test(
        unique_sequences,
        n_tests=draws,
        sample_size=effective_sample
    )

    # Extract node counts from each resample
    K_tests = []
    for res in result:
        # Get the final entry (last logged point)
        values = list(res.values())
        if values:
            K_tests.append(values[-1]['nodes'])

    K_tests = np.array(K_tests)

    # Calculate statistics
    mean = float(np.mean(K_tests))

    if not return_stats:
        return mean

    std = float(np.std(K_tests, ddof=1))
    se = std / np.sqrt(len(K_tests))

    # Calculate confidence interval using t-distribution
    t_crit = stats.t.ppf((1 + confidence_level) / 2, len(K_tests) - 1)
    ci_margin = t_crit * se
    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin

    return (mean, std, ci_lower, ci_upper)


def k1000_diversity(
    list_of_sequences: List[str],
    lzgraph_encoding_function: Callable,
    draws: int = 25,
    return_stats: bool = False,
    confidence_level: float = 0.95
) -> Union[float, Tuple[float, float, float, float]]:
    """
    Calculates the K1000 Diversity index (diversity at sample size 1000).

    This is the standard diversity metric for comparing repertoires of similar size.
    Uses 1000 randomly sampled sequences per draw to count unique LZ-subpatterns.

    Args:
        list_of_sequences (list): A list of CDR3 sequences.
        lzgraph_encoding_function (function): The LZGraph encoding function to be used.
            (e.g., AAPLZGraph.encode_sequence)
        draws (int, optional): The number of draws for the resampling test. Defaults to 25.
        return_stats (bool): If True, return (mean, std, ci_lower, ci_upper).
            If False, return only the mean. Defaults to False for backward compatibility.
        confidence_level (float): Confidence level for interval calculation.
            Defaults to 0.95 (95% confidence interval).

    Returns:
        float or tuple: Mean K1000 Diversity index, or (mean, std, ci_lower, ci_upper)
            if return_stats=True.

    Example:
        >>> sequences = ["CASSLGIRRTNTEAFF", "CASSLEGKYEQYF", ...]
        >>> encoding_function = AAPLZGraph.encode_sequence
        >>> diversity = k1000_diversity(sequences, encoding_function, draws=30)
        >>> print(diversity)

        >>> # With confidence intervals
        >>> mean, std, ci_low, ci_high = k1000_diversity(
        ...     sequences, encoding_function, return_stats=True
        ... )
        >>> print(f"K1000 = {mean:.1f} +/- {1.96*std:.1f}")
    """
    return k_diversity(
        list_of_sequences,
        lzgraph_encoding_function,
        sample_size=1000,
        draws=draws,
        return_stats=return_stats,
        confidence_level=confidence_level
    )


def k100_diversity(
    list_of_sequences: List[str],
    lzgraph_encoding_function: Callable,
    draws: int = 25,
    return_stats: bool = False,
    confidence_level: float = 0.95
) -> Union[float, Tuple[float, float, float, float]]:
    """
    Calculates the K100 Diversity index (diversity at sample size 100).

    Recommended for small repertoires (100-500 sequences) where K1000 would
    require too much downsampling.

    Args:
        list_of_sequences (list): A list of CDR3 sequences.
        lzgraph_encoding_function (function): The LZGraph encoding function to be used.
        draws (int, optional): The number of draws for the resampling test. Defaults to 25.
        return_stats (bool): If True, return (mean, std, ci_lower, ci_upper).
        confidence_level (float): Confidence level for interval calculation.

    Returns:
        float or tuple: Mean K100 Diversity index, or statistics tuple if return_stats=True.
    """
    return k_diversity(
        list_of_sequences,
        lzgraph_encoding_function,
        sample_size=100,
        draws=draws,
        return_stats=return_stats,
        confidence_level=confidence_level
    )


def k500_diversity(
    list_of_sequences: List[str],
    lzgraph_encoding_function: Callable,
    draws: int = 25,
    return_stats: bool = False,
    confidence_level: float = 0.95
) -> Union[float, Tuple[float, float, float, float]]:
    """
    Calculates the K500 Diversity index (diversity at sample size 500).

    Recommended for medium-sized repertoires (500-2000 sequences).

    Args:
        list_of_sequences (list): A list of CDR3 sequences.
        lzgraph_encoding_function (function): The LZGraph encoding function to be used.
        draws (int, optional): The number of draws for the resampling test. Defaults to 25.
        return_stats (bool): If True, return (mean, std, ci_lower, ci_upper).
        confidence_level (float): Confidence level for interval calculation.

    Returns:
        float or tuple: Mean K500 Diversity index, or statistics tuple if return_stats=True.
    """
    return k_diversity(
        list_of_sequences,
        lzgraph_encoding_function,
        sample_size=500,
        draws=draws,
        return_stats=return_stats,
        confidence_level=confidence_level
    )


def k5000_diversity(
    list_of_sequences: List[str],
    lzgraph_encoding_function: Callable,
    draws: int = 25,
    return_stats: bool = False,
    confidence_level: float = 0.95
) -> Union[float, Tuple[float, float, float, float]]:
    """
    Calculates the K5000 Diversity index (diversity at sample size 5000).

    Recommended for large repertoires (>10000 sequences) for higher resolution
    diversity measurement.

    Args:
        list_of_sequences (list): A list of CDR3 sequences.
        lzgraph_encoding_function (function): The LZGraph encoding function to be used.
        draws (int, optional): The number of draws for the resampling test. Defaults to 25.
        return_stats (bool): If True, return (mean, std, ci_lower, ci_upper).
        confidence_level (float): Confidence level for interval calculation.

    Returns:
        float or tuple: Mean K5000 Diversity index, or statistics tuple if return_stats=True.
    """
    return k_diversity(
        list_of_sequences,
        lzgraph_encoding_function,
        sample_size=5000,
        draws=draws,
        return_stats=return_stats,
        confidence_level=confidence_level
    )


def adaptive_k_diversity(
    list_of_sequences: List[str],
    lzgraph_encoding_function: Callable,
    draws: int = 25,
    return_stats: bool = False,
    confidence_level: float = 0.95
) -> Union[float, Tuple[float, float, float, float], Tuple[int, float], Tuple[int, float, float, float, float]]:
    """
    Automatically selects appropriate sample size based on repertoire size.

    This function chooses K100, K500, K1000, or K5000 based on the number of
    unique sequences in the repertoire, ensuring meaningful diversity estimates.

    Sample size selection:
        - n < 200: K100
        - 200 <= n < 1500: K500
        - 1500 <= n < 8000: K1000
        - n >= 8000: K5000

    Args:
        list_of_sequences (list): A list of CDR3 sequences.
        lzgraph_encoding_function (function): The LZGraph encoding function to be used.
        draws (int, optional): The number of draws for the resampling test. Defaults to 25.
        return_stats (bool): If True, return (sample_size, mean, std, ci_lower, ci_upper).
            If False, return (sample_size, mean).
        confidence_level (float): Confidence level for interval calculation.

    Returns:
        tuple: (sample_size, mean) or (sample_size, mean, std, ci_lower, ci_upper)
            The sample_size indicates which K-diversity was computed.

    Example:
        >>> sample_size, k_div = adaptive_k_diversity(sequences, encode_fn)
        >>> print(f"Used K{sample_size}, diversity = {k_div:.1f}")
    """
    n = len(set(list_of_sequences))

    if n < 200:
        sample_size = 100
    elif n < 1500:
        sample_size = 500
    elif n < 8000:
        sample_size = 1000
    else:
        sample_size = 5000

    result = k_diversity(
        list_of_sequences,
        lzgraph_encoding_function,
        sample_size=sample_size,
        draws=draws,
        return_stats=return_stats,
        confidence_level=confidence_level
    )

    if return_stats:
        mean, std, ci_lower, ci_upper = result
        return (sample_size, mean, std, ci_lower, ci_upper)
    else:
        return (sample_size, result)
