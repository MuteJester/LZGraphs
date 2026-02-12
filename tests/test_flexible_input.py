"""
Tests for Flexible Constructor Input
=====================================

Verify that AAPLZGraph and NDPLZGraph accept lists, Series, and DataFrames
as input, with optional abundances/v_genes/j_genes keyword arguments.
"""

import pytest
import pandas as pd
import numpy as np

from LZGraphs import AAPLZGraph, NDPLZGraph


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aap_sequences():
    """A small list of amino acid CDR3 sequences."""
    return [
        "CASSLEPSGGTDTQYF",
        "CASSDTSGGTDTQYF",
        "CASSLEPQTFTDTFFF",
        "CASSQDSGANVLTF",
        "CASSLSRGQNTEAFF",
        "CASSDRGDSYEQYF",
        "CASSYRGDQETQYF",
        "CASSEALNEQFF",
        "CASSFRGADTQYF",
        "CASSQGRDTDTQYF",
    ]


@pytest.fixture(scope="module")
def ndp_sequences():
    """A small list of nucleotide CDR3 sequences."""
    return [
        "TGTGCCAGCAGTTTCAAGAT",
        "TGTGCCAGCAGCCAAAGCAG",
        "TGTGCCAGCAGTCTCAAAGA",
        "TGTGCCAGCAGTGATGTACA",
        "TGTGCCAGCAGTTACAAATC",
        "TGTGCCAGCAGCCTAGCAGT",
        "TGTGCCAGCAGCCAAGAGAC",
        "TGTGCCAGCAGATCGGGGAC",
        "TGTGCCAGCAGTATAGGCAA",
        "TGTGCCAGCAGCCAAATCCT",
    ]


# ---------------------------------------------------------------------------
# AAPLZGraph flexible input
# ---------------------------------------------------------------------------

class TestAAPFlexibleInput:
    """AAPLZGraph should accept list, Series, and DataFrame."""

    def test_list_input(self, aap_sequences):
        graph = AAPLZGraph(aap_sequences, verbose=False)
        assert graph.graph.number_of_nodes() > 0
        assert graph.has_gene_data is False

    def test_series_input(self, aap_sequences):
        series = pd.Series(aap_sequences)
        graph = AAPLZGraph(series, verbose=False)
        assert graph.graph.number_of_nodes() > 0
        assert graph.has_gene_data is False

    def test_dataframe_input(self, aap_sequences):
        df = pd.DataFrame({"cdr3_amino_acid": aap_sequences})
        graph = AAPLZGraph(df, verbose=False)
        assert graph.graph.number_of_nodes() > 0

    def test_list_with_abundances(self, aap_sequences):
        abundances = [10, 5, 8, 3, 7, 2, 4, 6, 1, 9]
        graph = AAPLZGraph(aap_sequences, abundances=abundances, verbose=False)
        assert graph.graph.number_of_nodes() > 0
        # Total sequences weighted by abundance
        total = sum(graph.lengths.values())
        assert total == sum(abundances)

    def test_list_with_genes(self, aap_sequences):
        v = ["TRBV16-1*01"] * len(aap_sequences)
        j = ["TRBJ1-2*01"] * len(aap_sequences)
        graph = AAPLZGraph(aap_sequences, v_genes=v, j_genes=j, verbose=False)
        assert graph.has_gene_data is True

    def test_list_with_genes_and_abundances(self, aap_sequences):
        v = ["TRBV16-1*01"] * len(aap_sequences)
        j = ["TRBJ1-2*01"] * len(aap_sequences)
        abundances = [2] * len(aap_sequences)
        graph = AAPLZGraph(
            aap_sequences,
            abundances=abundances, v_genes=v, j_genes=j,
            verbose=False,
        )
        assert graph.has_gene_data is True
        total = sum(graph.lengths.values())
        assert total == 2 * len(aap_sequences)

    def test_list_equivalent_to_dataframe(self, aap_sequences):
        """A list-built graph should be identical to a DataFrame-built one."""
        graph_list = AAPLZGraph(aap_sequences, verbose=False)
        df = pd.DataFrame({"cdr3_amino_acid": aap_sequences})
        graph_df = AAPLZGraph(df, verbose=False)

        assert graph_list.graph.number_of_nodes() == graph_df.graph.number_of_nodes()
        assert graph_list.graph.number_of_edges() == graph_df.graph.number_of_edges()
        assert graph_list.initial_state_counts == graph_df.initial_state_counts

    def test_walk_probability_after_list_input(self, aap_sequences):
        graph = AAPLZGraph(aap_sequences, verbose=False)
        pgen = graph.walk_probability(aap_sequences[0], use_log=True)
        assert isinstance(pgen, float)
        assert pgen < 0  # log probability is negative


# ---------------------------------------------------------------------------
# NDPLZGraph flexible input
# ---------------------------------------------------------------------------

class TestNDPFlexibleInput:
    """NDPLZGraph should accept list, Series, and DataFrame."""

    def test_list_input(self, ndp_sequences):
        graph = NDPLZGraph(ndp_sequences, verbose=False)
        assert graph.graph.number_of_nodes() > 0
        assert graph.has_gene_data is False

    def test_series_input(self, ndp_sequences):
        series = pd.Series(ndp_sequences)
        graph = NDPLZGraph(series, verbose=False)
        assert graph.graph.number_of_nodes() > 0

    def test_dataframe_input(self, ndp_sequences):
        df = pd.DataFrame({"cdr3_rearrangement": ndp_sequences})
        graph = NDPLZGraph(df, verbose=False)
        assert graph.graph.number_of_nodes() > 0

    def test_list_with_abundances(self, ndp_sequences):
        abundances = [10, 5, 8, 3, 7, 2, 4, 6, 1, 9]
        graph = NDPLZGraph(ndp_sequences, abundances=abundances, verbose=False)
        total = sum(graph.lengths.values())
        assert total == sum(abundances)

    def test_list_with_genes(self, ndp_sequences):
        v = ["TRBV16-1*01"] * len(ndp_sequences)
        j = ["TRBJ1-2*01"] * len(ndp_sequences)
        graph = NDPLZGraph(ndp_sequences, v_genes=v, j_genes=j, verbose=False)
        assert graph.has_gene_data is True

    def test_list_equivalent_to_dataframe(self, ndp_sequences):
        graph_list = NDPLZGraph(ndp_sequences, verbose=False)
        df = pd.DataFrame({"cdr3_rearrangement": ndp_sequences})
        graph_df = NDPLZGraph(df, verbose=False)

        assert graph_list.graph.number_of_nodes() == graph_df.graph.number_of_nodes()
        assert graph_list.graph.number_of_edges() == graph_df.graph.number_of_edges()

    def test_walk_probability_after_list_input(self, ndp_sequences):
        graph = NDPLZGraph(ndp_sequences, verbose=False)
        pgen = graph.walk_probability(ndp_sequences[0], use_log=True)
        assert isinstance(pgen, float)
        assert pgen < 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestFlexibleInputErrors:
    """Verify proper error messages for bad input."""

    def test_dataframe_with_abundances_kwarg_raises(self, aap_sequences):
        df = pd.DataFrame({"cdr3_amino_acid": aap_sequences})
        with pytest.raises(TypeError, match="Cannot pass abundances"):
            AAPLZGraph(df, abundances=[1] * len(aap_sequences), verbose=False)

    def test_dataframe_with_v_genes_kwarg_raises(self, aap_sequences):
        df = pd.DataFrame({"cdr3_amino_acid": aap_sequences})
        with pytest.raises(TypeError, match="Cannot pass abundances"):
            AAPLZGraph(df, v_genes=["X"] * len(aap_sequences),
                       j_genes=["Y"] * len(aap_sequences), verbose=False)

    def test_mismatched_abundance_length(self, aap_sequences):
        with pytest.raises(ValueError, match="abundances length"):
            AAPLZGraph(aap_sequences, abundances=[1, 2, 3], verbose=False)

    def test_v_genes_without_j_genes(self, aap_sequences):
        with pytest.raises(ValueError, match="both be provided"):
            AAPLZGraph(aap_sequences, v_genes=["X"] * len(aap_sequences), verbose=False)

    def test_j_genes_without_v_genes(self, aap_sequences):
        with pytest.raises(ValueError, match="both be provided"):
            AAPLZGraph(aap_sequences, j_genes=["X"] * len(aap_sequences), verbose=False)

    def test_mismatched_gene_length(self, aap_sequences):
        with pytest.raises(ValueError, match="must match"):
            AAPLZGraph(aap_sequences,
                       v_genes=["X"] * 3,
                       j_genes=["Y"] * 3,
                       verbose=False)

    def test_invalid_data_type(self):
        with pytest.raises(TypeError, match="must be a DataFrame"):
            AAPLZGraph(12345, verbose=False)

    def test_ndp_invalid_data_type(self):
        with pytest.raises(TypeError, match="must be a DataFrame"):
            NDPLZGraph({"not": "a list"}, verbose=False)
