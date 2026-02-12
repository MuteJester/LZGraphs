import numpy as np
from ..utilities.misc import choice, _is_v_gene, _is_j_gene
from ..graphs.edge_data import EdgeData
from ..exceptions import NoGeneDataError, MetricsError

class GeneLogicMixin:
    """
    Mixin that handles all logic related to V and J gene loading,
    selection, and edge updates for gene-based edges.

    Requirements:
        - The parent class must define:
            self.graph (networkx.DiGraph)
            self.num_transitions (int)
            self.has_gene_data (bool)
        - A function `choice(options, weights)` that picks one item from `options`
          with probability distribution `weights`.
    """

    def _raise_genetic_mode_error(self):
        """
        Raise an error if genetic mode is off but a genetic function is called.
        """
        if not self.has_gene_data:
            raise NoGeneDataError(
                message="Genomic data function requires gene annotation data, "
                "but `self.has_gene_data` is False."
            )

    def _load_gene_data(self, data) -> None:
        """
        Load V and J gene data from the input DataFrame into
        marginal frequency distributions. Also track the combined frequency
        of V-J pairs (VJ).

        Args:
            data: Must contain columns ['V', 'J'] at minimum,
                  representing observed V/J genes for each sequence.
        """
        # Unique sets of V and J
        self.observed_v_genes = list(set(data['V']))
        self.observed_j_genes = list(set(data['J']))

        # Marginal distributions (normalized) — stored as plain dicts
        self.marginal_v_genes = dict(data['V'].value_counts(normalize=True))
        self.marginal_j_genes = dict(data['J'].value_counts(normalize=True))

        # Combined VJ distribution
        self.vj_probabilities = dict(
            (data['V'] + '_' + data['J']).value_counts(normalize=True)
        )

    def _select_random_vj_genes(self, mode='marginal') -> tuple[str, str]:
        """
        Select random (V, J) genes based on:
            - 'marginal': pick V from marginal_v_genes, and J from marginal_j_genes, independently
            - 'combined': pick a single 'V_J' from vj_probabilities

        Args:
            mode (str): 'marginal' or 'combined'

        Returns:
            (V, J) (tuple[str, str]): The selected V and J genes.
        """
        self._raise_genetic_mode_error()
        if mode == 'marginal':
            V = choice(list(self.marginal_v_genes.keys()), list(self.marginal_v_genes.values()))
            J = choice(list(self.marginal_j_genes.keys()), list(self.marginal_j_genes.values()))
            return V, J
        elif mode == 'combined':
            VJ = choice(list(self.vj_probabilities.keys()), list(self.vj_probabilities.values()))
            V, J = VJ.split('_')
            return V, J
        else:
            raise MetricsError(f"Unknown mode: {mode}. Use 'marginal' or 'combined'.")

    def _insert_edge_and_information(self, node_a: str, node_b: str, v_gene: str, j_gene: str, count: int = 1) -> None:
        """
        Insert or update an edge (node_a -> node_b) with the relevant gene data.

        Args:
            node_a (str): The source node.
            node_b (str): The target node.
            v_gene (str): The V gene name.
            j_gene (str): The J gene name.
            count (int): Number of traversals (abundance weight). Default 1.
        """
        if self.graph.has_edge(node_a, node_b):
            self.graph[node_a][node_b]['data'].record(v_gene=v_gene, j_gene=j_gene, count=count)
        else:
            ed = EdgeData()
            ed.record(v_gene=v_gene, j_gene=j_gene, count=count)
            self.graph.add_edge(node_a, node_b, data=ed)

        # Track a global transition count (if needed by the parent class)
        self.num_transitions += count
