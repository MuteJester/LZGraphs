import numpy as np
import pandas as pd
from ..utilities.misc import choice, _is_v_gene, _is_j_gene
from ..exceptions import NoGeneDataError, MetricsError

class GeneLogicMixin:
    """
    Mixin that handles all logic related to V and J gene loading,
    selection, and edge updates for gene-based edges.

    Requirements:
        - The parent class must define:
            self.graph (networkx.DiGraph)
            self.n_transitions (int)
            self.genetic (bool)
        - A function `choice(options, weights)` that picks one item from `options`
          with probability distribution `weights`.
    """

    def _raise_genetic_mode_error(self):
        """
        Raise an error if genetic mode is off but a genetic function is called.
        """
        if not self.genetic:
            raise NoGeneDataError(
                message="Genomic data function requires gene annotation data, "
                "but `self.genetic` is False."
            )

    def _load_gene_data(self, data: pd.DataFrame) -> None:
        """
        Load V and J gene data from the input DataFrame into
        marginal frequency distributions. Also track the combined frequency
        of V-J pairs (VJ).

        Args:
            data (pd.DataFrame): Must contain columns ['V', 'J'] at minimum,
                                 representing observed V/J genes for each sequence.
        """
        # Unique sets of V and J
        self.observed_vgenes = list(set(data['V']))
        self.observed_jgenes = list(set(data['J']))

        # Marginal distributions (normalized)
        self.marginal_vgenes = data['V'].value_counts(normalize=True)
        self.marginal_jgenes = data['J'].value_counts(normalize=True)

        # Combined VJ distribution
        self.vj_probabilities = (data['V'] + '_' + data['J']).value_counts(normalize=True)

    def _select_random_vj_genes(self, mode='marginal') -> tuple[str, str]:
        """
        Select random (V, J) genes based on:
            - 'marginal': pick V from marginal_vgenes, and J from marginal_jgenes, independently
            - 'combined': pick a single 'V_J' from vj_probabilities

        Args:
            mode (str): 'marginal' or 'combined'

        Returns:
            (V, J) (tuple[str, str]): The selected V and J genes.
        """
        self._raise_genetic_mode_error()
        if mode == 'marginal':
            V = choice(self.marginal_vgenes.index, self.marginal_vgenes.values)
            J = choice(self.marginal_jgenes.index, self.marginal_jgenes.values)
            return V, J
        elif mode == 'combined':
            VJ = choice(self.vj_probabilities.index, self.vj_probabilities.values)
            V, J = VJ.split('_')
            return V, J
        else:
            raise MetricsError(f"Unknown mode: {mode}. Use 'marginal' or 'combined'.")

    def _insert_edge_and_information(self, node_a: str, node_b: str, Vgene: str, Jgene: str) -> None:
        """
        Insert or update an edge (node_a -> node_b) with the relevant gene data.
        This increments the 'weight' by 1, as well as the counters for V and J usage.

        Args:
            node_a (str): The source node.
            node_b (str): The target node.
            Vgene (str): The V gene name.
            Jgene (str): The J gene name.
        """
        try:
            # If edge exists, update counts
            edge_data = self.graph[node_a][node_b]
            edge_data["weight"] += 1
            edge_data[Vgene] = edge_data.get(Vgene, 0) + 1
            edge_data[Jgene] = edge_data.get(Jgene, 0) + 1
            edge_data["Vsum"] += 1
            edge_data["Jsum"] += 1
        except KeyError:
            # Edge doesn't exist yet, create with initial counts
            self.graph.add_edge(
                node_a,
                node_b,
                weight=1,
                Vsum=1,
                Jsum=1,
                **{Vgene: 1, Jgene: 1}
            )

        # Track a global transition count (if needed by the parent class)
        self.n_transitions += 1

    def _normalize_gene_weights(self, edge_list) -> None:
        """
        For each edge in the given edge_list, normalize gene-based keys
        so that V genes sum to 1 and J genes sum to 1. This also assumes
        each edge has a 'Vsum' and 'Jsum' key that track the sum of all V and J counts.

        Args:
            edge_list (list[tuple[str, str]]): A list of edges (node_a, node_b).
        """
        for n_a, n_b in edge_list:
            e_data = self.graph.get_edge_data(n_a, n_b)
            vsum = e_data['Vsum']
            jsum = e_data['Jsum']

            # All keys besides 'weight', 'Vsum', and 'Jsum' are assumed to be gene counts
            genes = set(e_data) - {'Vsum', 'Jsum', 'weight'}

            for key in genes:
                if _is_v_gene(key):
                    e_data[key] /= vsum
                elif _is_j_gene(key):
                    e_data[key] /= jsum
