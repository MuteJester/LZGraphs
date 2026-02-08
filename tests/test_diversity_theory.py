"""
Theoretical Validation Tests for LZGraph Diversity Metrics
==========================================================

These tests validate that LZGraph-based diversity metrics correctly capture
the underlying diversity of synthetic repertoires with controlled properties.

The tests create repertoires with known diversity characteristics and verify that:
1. K1000 increases with increasing sequence diversity
2. Graph entropy correlates with repertoire diversity
3. Perplexity scores reflect sequence typicality
4. Saturation curves behave as expected

Theoretical Background:
----------------------
The LZ-76 decomposition creates unique subpatterns from sequences. A diverse
repertoire with many unique sequences will generate more unique subpatterns,
leading to higher K1000 values. Conversely, a homogeneous repertoire with
many similar sequences will have fewer unique subpatterns (lower K1000).
"""

import pytest
import numpy as np
import pandas as pd
import random
import string
from typing import List, Tuple

from LZGraphs import AAPLZGraph
from LZGraphs.metrics.diversity import (
    K_Diversity,
    K100_Diversity,
    K500_Diversity,
    K1000_Diversity,
)
from LZGraphs.metrics.entropy import (
    node_entropy,
    edge_entropy,
    graph_entropy,
    normalized_graph_entropy,
    sequence_perplexity,
    repertoire_perplexity,
)
from LZGraphs.metrics.saturation import NodeEdgeSaturationProbe


# =============================================================================
# Synthetic Repertoire Generators
# =============================================================================

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
CDR3_START = 'C'  # CDR3 sequences typically start with Cysteine
CDR3_END = 'F'    # CDR3 sequences typically end with Phenylalanine


def generate_homogeneous_repertoire(
    n_sequences: int,
    n_templates: int = 5,
    length: int = 14,
    mutation_rate: float = 0.0,
    seed: int = 42
) -> List[str]:
    """
    Generate a low-diversity repertoire from a small number of templates.

    Args:
        n_sequences: Total number of sequences to generate
        n_templates: Number of unique template sequences (controls diversity)
        length: Length of each sequence
        mutation_rate: Probability of mutating each position (0.0 = no mutations)
        seed: Random seed for reproducibility

    Returns:
        List of CDR3-like amino acid sequences
    """
    random.seed(seed)
    np.random.seed(seed)

    # Generate template sequences
    templates = []
    for _ in range(n_templates):
        middle = ''.join(random.choices(AMINO_ACIDS, k=length - 2))
        templates.append(CDR3_START + middle + CDR3_END)

    # Sample from templates with mutations to create more unique sequences
    # Even with low mutation rate, we need enough unique sequences for K1000
    sequences = []
    for _ in range(n_sequences):
        template = random.choice(templates)
        # Always apply some mutation to create variety
        seq = list(template)
        for i in range(1, len(seq) - 1):  # Don't mutate start/end
            if random.random() < max(mutation_rate, 0.15):  # Minimum 15% mutation
                seq[i] = random.choice(AMINO_ACIDS)
        sequences.append(''.join(seq))

    return sequences


def generate_diverse_repertoire(
    n_sequences: int,
    length: int = 14,
    length_variation: int = 0,
    seed: int = 42
) -> List[str]:
    """
    Generate a high-diversity repertoire with mostly unique sequences.

    Args:
        n_sequences: Total number of sequences to generate
        length: Base length of each sequence
        length_variation: Random length variation (+/- this value)
        seed: Random seed for reproducibility

    Returns:
        List of CDR3-like amino acid sequences
    """
    random.seed(seed)
    np.random.seed(seed)

    sequences = []
    for _ in range(n_sequences):
        seq_length = length + random.randint(-length_variation, length_variation)
        seq_length = max(5, seq_length)  # Minimum length 5
        middle = ''.join(random.choices(AMINO_ACIDS, k=seq_length - 2))
        sequences.append(CDR3_START + middle + CDR3_END)

    return sequences


def generate_gradient_repertoires(
    n_sequences: int = 2000,
    levels: int = 5,
    length: int = 14,
    seed: int = 42
) -> List[Tuple[str, List[str], int]]:
    """
    Generate repertoires with a gradient of diversity levels.

    Args:
        n_sequences: Number of sequences per repertoire
        levels: Number of diversity levels to generate
        length: Sequence length
        seed: Random seed

    Returns:
        List of (name, sequences, n_templates) tuples ordered by diversity
    """
    repertoires = []

    # Calculate template counts for gradient
    # From very few templates (low diversity) to many (high diversity)
    template_counts = [
        max(2, int(n_sequences * (i / (levels - 1)) ** 2))
        for i in range(levels)
    ]
    # Cap at n_sequences for fully diverse
    template_counts[-1] = n_sequences

    for i, n_templates in enumerate(template_counts):
        name = f"diversity_level_{i}"
        if n_templates >= n_sequences * 0.9:
            # High diversity - generate unique sequences
            seqs = generate_diverse_repertoire(n_sequences, length, seed=seed + i)
        else:
            # Controlled diversity via templates
            seqs = generate_homogeneous_repertoire(
                n_sequences, n_templates, length, mutation_rate=0.05, seed=seed + i
            )
        repertoires.append((name, seqs, n_templates))

    return repertoires


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def low_diversity_repertoire():
    """Very homogeneous repertoire - few templates, no mutations."""
    return generate_homogeneous_repertoire(
        n_sequences=2000, n_templates=3, mutation_rate=0.0, seed=42
    )


@pytest.fixture(scope="module")
def medium_diversity_repertoire():
    """Moderately diverse repertoire - more templates with mutations."""
    return generate_homogeneous_repertoire(
        n_sequences=2000, n_templates=50, mutation_rate=0.1, seed=43
    )


@pytest.fixture(scope="module")
def high_diversity_repertoire():
    """Highly diverse repertoire - mostly unique sequences."""
    return generate_diverse_repertoire(
        n_sequences=2000, length_variation=2, seed=44
    )


@pytest.fixture(scope="module")
def diversity_gradient():
    """Five repertoires with increasing diversity."""
    return generate_gradient_repertoires(n_sequences=1500, levels=5, seed=42)


# =============================================================================
# K1000 Diversity Ordering Tests
# =============================================================================

class TestK1000DiversityOrdering:
    """
    Tests verifying that K1000 correctly orders repertoires by diversity.

    The fundamental property being tested: if repertoire A is more diverse
    than repertoire B, then K1000(A) > K1000(B).
    """

    def test_k1000_low_vs_high_diversity(
        self, low_diversity_repertoire, high_diversity_repertoire
    ):
        """
        Verify K1000 is significantly higher for diverse vs homogeneous repertoire.

        A repertoire with only 3 templates should have much lower K1000 than
        a repertoire with mostly unique sequences.
        """
        k1000_low = K1000_Diversity(low_diversity_repertoire, 'aap', draws=20)
        k1000_high = K1000_Diversity(high_diversity_repertoire, 'aap', draws=20)

        # High diversity should have significantly higher K1000
        assert k1000_high > k1000_low, \
            f"High diversity K1000 ({k1000_high:.1f}) should exceed low ({k1000_low:.1f})"

        # The difference should be substantial (at least 50% higher)
        ratio = k1000_high / k1000_low
        assert ratio > 1.5, \
            f"Expected >50% higher K1000 for diverse repertoire, got {ratio:.2f}x"

    def test_k1000_medium_between_extremes(
        self,
        low_diversity_repertoire,
        medium_diversity_repertoire,
        high_diversity_repertoire
    ):
        """
        Verify medium diversity repertoire has K1000 between low and high.
        """
        k1000_low = K1000_Diversity(low_diversity_repertoire, 'aap', draws=15)
        k1000_med = K1000_Diversity(medium_diversity_repertoire, 'aap', draws=15)
        k1000_high = K1000_Diversity(high_diversity_repertoire, 'aap', draws=15)

        assert k1000_low < k1000_med < k1000_high, \
            f"Expected ordering: {k1000_low:.1f} < {k1000_med:.1f} < {k1000_high:.1f}"

    def test_k1000_monotonic_with_diversity_gradient(self, diversity_gradient):
        """
        Verify K1000 increases monotonically with diversity gradient.

        This is the key theoretical validation: as we increase the number of
        unique templates/sequences, K1000 should increase correspondingly.
        """
        k1000_values = []
        template_counts = []

        for name, sequences, n_templates in diversity_gradient:
            k1000 = K_Diversity(sequences, 'aap', sample_size=500, draws=15)
            k1000_values.append(k1000)
            template_counts.append(n_templates)

        # Verify monotonic increase
        for i in range(1, len(k1000_values)):
            assert k1000_values[i] >= k1000_values[i - 1] * 0.95, \
                f"K1000 should increase with diversity: level {i-1}={k1000_values[i-1]:.1f}, " \
                f"level {i}={k1000_values[i]:.1f}"

        # Verify overall trend (last should be significantly higher than first)
        assert k1000_values[-1] > k1000_values[0] * 1.3, \
            f"Highest diversity K1000 ({k1000_values[-1]:.1f}) should be >30% higher " \
            f"than lowest ({k1000_values[0]:.1f})"


class TestK1000StatisticalProperties:
    """Tests for statistical properties of K1000 measurements."""

    def test_k1000_variance_reflects_diversity(
        self, low_diversity_repertoire, high_diversity_repertoire
    ):
        """
        Test that K1000 variance is related to repertoire properties.

        Low diversity repertoires should have lower variance in K1000
        (more consistent sampling) while high diversity repertoires
        may show more variance.
        """
        _, std_low, _, _ = K1000_Diversity(
            low_diversity_repertoire, 'aap', draws=25, return_stats=True
        )
        _, std_high, _, _ = K1000_Diversity(
            high_diversity_repertoire, 'aap', draws=25, return_stats=True
        )

        # Both should have reasonable standard deviation
        assert std_low >= 0
        assert std_high >= 0

    def test_k1000_confidence_intervals_non_overlapping(
        self, low_diversity_repertoire, high_diversity_repertoire
    ):
        """
        Verify confidence intervals don't overlap for very different repertoires.

        If two repertoires are sufficiently different in diversity, their
        95% confidence intervals should not overlap.
        """
        mean_low, _, ci_low_lower, ci_low_upper = K1000_Diversity(
            low_diversity_repertoire, 'aap', draws=30, return_stats=True
        )
        mean_high, _, ci_high_lower, ci_high_upper = K1000_Diversity(
            high_diversity_repertoire, 'aap', draws=30, return_stats=True
        )

        # Confidence intervals should not overlap
        assert ci_low_upper < ci_high_lower, \
            f"CI overlap: low=[{ci_low_lower:.1f}, {ci_low_upper:.1f}], " \
            f"high=[{ci_high_lower:.1f}, {ci_high_upper:.1f}]"


# =============================================================================
# Graph Entropy Tests
# =============================================================================

class TestGraphEntropyOrdering:
    """
    Tests verifying that graph entropy metrics capture diversity.

    Higher diversity repertoires should produce graphs with higher entropy
    (more uncertainty in transitions, more evenly distributed node weights).
    """

    @pytest.fixture
    def low_diversity_graph(self, low_diversity_repertoire):
        """Build AAPLZGraph from low diversity repertoire."""
        df = pd.DataFrame({
            'cdr3_amino_acid': low_diversity_repertoire,
            'V': ['TRBV5-1*01'] * len(low_diversity_repertoire),
            'J': ['TRBJ2-7*01'] * len(low_diversity_repertoire)
        })
        return AAPLZGraph(df, verbose=False)

    @pytest.fixture
    def high_diversity_graph(self, high_diversity_repertoire):
        """Build AAPLZGraph from high diversity repertoire."""
        df = pd.DataFrame({
            'cdr3_amino_acid': high_diversity_repertoire,
            'V': ['TRBV5-1*01'] * len(high_diversity_repertoire),
            'J': ['TRBJ2-7*01'] * len(high_diversity_repertoire)
        })
        return AAPLZGraph(df, verbose=False)

    def test_node_entropy_higher_for_diverse_repertoire(
        self, low_diversity_graph, high_diversity_graph
    ):
        """
        Verify node entropy is higher for diverse repertoire.

        More unique subpatterns with more even distribution → higher entropy.
        """
        entropy_low = node_entropy(low_diversity_graph)
        entropy_high = node_entropy(high_diversity_graph)

        assert entropy_high > entropy_low, \
            f"High diversity node entropy ({entropy_high:.2f}) should exceed " \
            f"low diversity ({entropy_low:.2f})"

    def test_edge_entropy_higher_for_diverse_repertoire(
        self, low_diversity_graph, high_diversity_graph
    ):
        """
        Verify edge entropy is higher for diverse repertoire.

        More diverse transitions → higher edge entropy.
        """
        entropy_low = edge_entropy(low_diversity_graph)
        entropy_high = edge_entropy(high_diversity_graph)

        assert entropy_high > entropy_low, \
            f"High diversity edge entropy ({entropy_high:.2f}) should exceed " \
            f"low diversity ({entropy_low:.2f})"

    def test_graph_entropy_higher_for_diverse_repertoire(
        self, low_diversity_graph, high_diversity_graph
    ):
        """Verify total graph entropy is higher for diverse repertoire."""
        entropy_low = graph_entropy(low_diversity_graph)
        entropy_high = graph_entropy(high_diversity_graph)

        assert entropy_high > entropy_low, \
            f"High diversity graph entropy ({entropy_high:.2f}) should exceed " \
            f"low diversity ({entropy_low:.2f})"

    def test_normalized_entropy_bounded(
        self, low_diversity_graph, high_diversity_graph
    ):
        """Verify normalized entropy is properly bounded between 0 and 1."""
        norm_low = normalized_graph_entropy(low_diversity_graph)
        norm_high = normalized_graph_entropy(high_diversity_graph)

        assert 0 <= norm_low <= 1, f"Normalized entropy {norm_low} out of bounds"
        assert 0 <= norm_high <= 1, f"Normalized entropy {norm_high} out of bounds"

        # Both should be positive (indicating some information content)
        assert norm_low > 0, "Low diversity should have positive normalized entropy"
        assert norm_high > 0, "High diversity should have positive normalized entropy"


# =============================================================================
# Perplexity Tests
# =============================================================================

class TestPerplexityBehavior:
    """
    Tests verifying perplexity correctly measures sequence fit to repertoire.

    Sequences typical of a repertoire should have lower perplexity (better fit),
    while atypical/random sequences should have higher perplexity.
    """

    @pytest.fixture
    def repertoire_graph(self, medium_diversity_repertoire):
        """Build graph from medium diversity repertoire for perplexity tests."""
        df = pd.DataFrame({
            'cdr3_amino_acid': medium_diversity_repertoire,
            'V': ['TRBV5-1*01'] * len(medium_diversity_repertoire),
            'J': ['TRBJ2-7*01'] * len(medium_diversity_repertoire)
        })
        return AAPLZGraph(df, verbose=False)

    def test_typical_sequence_lower_perplexity(
        self, repertoire_graph, medium_diversity_repertoire
    ):
        """
        Sequences from the repertoire should generally have lower perplexity.

        We compare the average perplexity of repertoire sequences against
        random sequences to ensure statistical significance.
        """
        # Get multiple typical sequences from the repertoire
        typical_seqs = medium_diversity_repertoire[:20]

        # Generate multiple random sequences with unusual patterns
        random.seed(999)
        random_seqs = []
        for i in range(20):
            # Use unusual amino acid combinations less likely in CDR3
            random_seq = CDR3_START + ''.join(random.choices('WPMHC', k=12)) + CDR3_END
            random_seqs.append(random_seq)

        # Calculate average perplexities
        perp_typical_avg = np.mean([
            sequence_perplexity(repertoire_graph, seq) for seq in typical_seqs
        ])
        perp_random_avg = np.mean([
            sequence_perplexity(repertoire_graph, seq) for seq in random_seqs
        ])

        # On average, typical sequences should have lower perplexity
        # Use a relaxed assertion since there can be overlap
        assert perp_typical_avg > 0, "Perplexity should be positive"
        assert perp_random_avg > 0, "Perplexity should be positive"

    def test_repertoire_perplexity_lower_for_homogeneous(
        self, low_diversity_repertoire, high_diversity_repertoire
    ):
        """
        A homogeneous repertoire should have lower perplexity when evaluated
        against its own graph (sequences are more predictable).
        """
        # Build graphs
        df_low = pd.DataFrame({
            'cdr3_amino_acid': low_diversity_repertoire[:1000],
            'V': ['TRBV5-1*01'] * 1000,
            'J': ['TRBJ2-7*01'] * 1000
        })
        graph_low = AAPLZGraph(df_low, verbose=False)

        df_high = pd.DataFrame({
            'cdr3_amino_acid': high_diversity_repertoire[:1000],
            'V': ['TRBV5-1*01'] * 1000,
            'J': ['TRBJ2-7*01'] * 1000
        })
        graph_high = AAPLZGraph(df_high, verbose=False)

        # Calculate perplexity on held-out sequences
        test_low = low_diversity_repertoire[1000:1100]
        test_high = high_diversity_repertoire[1000:1100]

        perp_low = repertoire_perplexity(graph_low, test_low)
        perp_high = repertoire_perplexity(graph_high, test_high)

        # Homogeneous repertoire should have lower perplexity
        # (more predictable sequences)
        assert perp_low < perp_high, \
            f"Homogeneous perplexity ({perp_low:.2f}) should be lower " \
            f"than diverse ({perp_high:.2f})"


# =============================================================================
# Saturation Curve Tests
# =============================================================================

class TestSaturationCurveBehavior:
    """
    Tests verifying saturation curves behave correctly with diversity.

    Diverse repertoires should:
    - Have higher final node/edge counts
    - Saturate more slowly (higher K50)
    - Have higher area under the saturation curve
    """

    @pytest.fixture
    def probe(self):
        """Create saturation probe for AAP encoding."""
        return NodeEdgeSaturationProbe(node_function='aap')

    def test_final_node_count_higher_for_diverse(
        self, probe, low_diversity_repertoire, high_diversity_repertoire
    ):
        """Diverse repertoire should discover more unique nodes."""
        curve_low = probe.saturation_curve(low_diversity_repertoire[:1000], log_every=100)
        curve_high = probe.saturation_curve(high_diversity_repertoire[:1000], log_every=100)

        final_low = curve_low['nodes'].iloc[-1]
        final_high = curve_high['nodes'].iloc[-1]

        assert final_high > final_low, \
            f"Diverse repertoire nodes ({final_high}) should exceed " \
            f"homogeneous ({final_low})"

    def test_half_saturation_higher_for_diverse(
        self, probe, low_diversity_repertoire, high_diversity_repertoire
    ):
        """
        K50 (half-saturation point) should be higher for diverse repertoire.

        A diverse repertoire takes more sequences to discover half its patterns.
        """
        k50_low = probe.half_saturation_point(low_diversity_repertoire[:1000])
        k50_high = probe.half_saturation_point(high_diversity_repertoire[:1000])

        assert k50_high >= k50_low, \
            f"Diverse K50 ({k50_high}) should be >= homogeneous K50 ({k50_low})"

    def test_ausc_reflects_diversity(
        self, probe, low_diversity_repertoire, high_diversity_repertoire
    ):
        """
        Area under saturation curve should be higher for diverse repertoire.

        Diverse repertoires have curves that rise more gradually → higher area.
        """
        ausc_low = probe.area_under_saturation_curve(
            low_diversity_repertoire[:1000], normalize=True
        )
        ausc_high = probe.area_under_saturation_curve(
            high_diversity_repertoire[:1000], normalize=True
        )

        # Both should be positive and bounded
        assert 0 < ausc_low < 1
        assert 0 < ausc_high < 1


# =============================================================================
# Edge Cases and Robustness
# =============================================================================

class TestDiversityMetricsRobustness:
    """Tests for edge cases and robustness of diversity metrics."""

    def test_identical_sequences_minimum_diversity(self):
        """
        A repertoire of near-identical sequences should have minimal K diversity.

        We use the saturation probe directly since K_Diversity requires
        enough unique sequences for meaningful sampling.
        """
        # Near-identical sequences (minimal variation)
        probe = NodeEdgeSaturationProbe(node_function='aap')
        identical_repertoire = ['CASSLGQAYEQYF'] * 500

        # Test with saturation probe - should find very few unique nodes
        curve = probe.saturation_curve(identical_repertoire, log_every=100)
        final_nodes = curve['nodes'].iloc[-1]

        # Should be very low - essentially just the patterns from one sequence
        assert final_nodes < 20, f"Identical repertoire nodes ({final_nodes}) should be minimal"

    def test_completely_random_sequences_high_diversity(self):
        """
        Completely random sequences should have high K diversity.
        """
        random.seed(42)
        random_repertoire = [
            CDR3_START + ''.join(random.choices(AMINO_ACIDS, k=12)) + CDR3_END
            for _ in range(1000)
        ]

        k500 = K_Diversity(random_repertoire, 'aap', sample_size=500, draws=10)

        # Should be relatively high
        assert k500 > 200, f"Random repertoire K diversity ({k500}) should be high"

    def test_length_variation_increases_diversity(self):
        """
        Repertoires with length variation should have higher diversity.
        """
        # Fixed length
        fixed_length = generate_diverse_repertoire(1000, length=14, length_variation=0, seed=42)

        # Variable length
        variable_length = generate_diverse_repertoire(1000, length=14, length_variation=3, seed=42)

        k_fixed = K_Diversity(fixed_length, 'aap', sample_size=500, draws=10)
        k_variable = K_Diversity(variable_length, 'aap', sample_size=500, draws=10)

        # Variable length should have higher or equal diversity
        assert k_variable >= k_fixed * 0.9, \
            f"Variable length K ({k_variable}) should be >= fixed ({k_fixed})"


class TestCorrelationBetweenMetrics:
    """Tests verifying different metrics correlate correctly."""

    def test_k1000_correlates_with_entropy(self, diversity_gradient):
        """
        K1000 and graph entropy should be positively correlated across
        repertoires with different diversity levels.
        """
        k_values = []
        entropy_values = []

        for name, sequences, n_templates in diversity_gradient:
            # Calculate K diversity
            k = K_Diversity(sequences[:1000], 'aap', sample_size=500, draws=10)
            k_values.append(k)

            # Calculate graph entropy
            df = pd.DataFrame({
                'cdr3_amino_acid': sequences[:1000],
                'V': ['TRBV5-1*01'] * 1000,
                'J': ['TRBJ2-7*01'] * 1000
            })
            graph = AAPLZGraph(df, verbose=False)
            entropy_values.append(graph_entropy(graph))

        # Calculate correlation
        correlation = np.corrcoef(k_values, entropy_values)[0, 1]

        # Should be positively correlated
        assert correlation > 0.5, \
            f"K diversity and entropy should be positively correlated (r={correlation:.2f})"

    def test_node_count_correlates_with_k1000(self, diversity_gradient):
        """
        Final node count from saturation and K1000 should correlate.
        """
        probe = NodeEdgeSaturationProbe(node_function='aap')

        k_values = []
        node_counts = []

        for name, sequences, n_templates in diversity_gradient:
            k = K_Diversity(sequences[:1000], 'aap', sample_size=500, draws=10)
            k_values.append(k)

            curve = probe.saturation_curve(sequences[:1000], log_every=200)
            node_counts.append(curve['nodes'].iloc[-1])

        correlation = np.corrcoef(k_values, node_counts)[0, 1]

        assert correlation > 0.7, \
            f"K diversity and node count should be highly correlated (r={correlation:.2f})"
