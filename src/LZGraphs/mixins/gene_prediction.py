import heapq
from collections import Counter

from ..exceptions import MetricsError
from ..utilities.misc import _is_v_gene, _is_j_gene, choice

class GenePredictionMixin:
    """
    Mixin that provides different heuristics for predicting V/J genes
    from a given walk (list of nodes).

    This mixin assumes:
    - `self.graph` is a networkx.DiGraph.
    - Each edge has a 'data' attribute containing an EdgeData object.
    - A function `choice(options, weights)` is available for weighted selection.
    """

    def _max_sum_gene_prediction(self, walk, top_n=1):
        """
        Aggregate the sum of each V or J gene probability along the edges of the walk.
        Then pick either the single best gene (top_n=1) or top_n genes.

        Args:
            walk (list): The list of node names representing the walk.
            top_n (int): How many top genes to return.

        Returns:
            tuple:
              If top_n=1, returns (best_V_gene, best_J_gene) as strings.
              Otherwise, returns (set_of_top_V_genes, set_of_top_J_genes).
        """
        v_gene_agg = {}
        j_gene_agg = {}

        for i in range(len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i+1]):
                ed = self.graph[walk[i]][walk[i+1]]['data']
                for gene, count in ed.v_genes.items():
                    v_gene_agg[gene] = v_gene_agg.get(gene, 0) + ed.v_probability(gene)
                for gene, count in ed.j_genes.items():
                    j_gene_agg[gene] = j_gene_agg.get(gene, 0) + ed.j_probability(gene)

        if top_n == 1:
            best_v = max(v_gene_agg, key=v_gene_agg.get) if v_gene_agg else None
            best_j = max(j_gene_agg, key=j_gene_agg.get) if j_gene_agg else None
            return best_v, best_j
        else:
            top_vs = heapq.nlargest(top_n, v_gene_agg, key=v_gene_agg.get)
            top_js = heapq.nlargest(top_n, j_gene_agg, key=j_gene_agg.get)
            return set(top_vs), set(top_js)

    def _max_product_gene_prediction(self, walk, top_n=1):
        """
        Aggregate the product of each V or J gene probability along the edges.

        Args:
            walk (list): The list of node names representing the walk.
            top_n (int): How many top genes to return.

        Returns:
            tuple:
              If top_n=1, returns (best_V_gene, best_J_gene) as strings.
              Otherwise, returns (set_of_top_V_genes, set_of_top_J_genes).
        """
        v_gene_agg = {}
        j_gene_agg = {}

        for i in range(len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i+1]):
                ed = self.graph[walk[i]][walk[i+1]]['data']
                for gene in ed.v_genes:
                    prob = ed.v_probability(gene)
                    v_gene_agg[gene] = v_gene_agg.get(gene, 1.0) * prob
                for gene in ed.j_genes:
                    prob = ed.j_probability(gene)
                    j_gene_agg[gene] = j_gene_agg.get(gene, 1.0) * prob

        if top_n == 1:
            best_v = max(v_gene_agg, key=v_gene_agg.get) if v_gene_agg else None
            best_j = max(j_gene_agg, key=j_gene_agg.get) if j_gene_agg else None
            return best_v, best_j
        else:
            top_vs = heapq.nlargest(top_n, v_gene_agg, key=v_gene_agg.get)
            top_js = heapq.nlargest(top_n, j_gene_agg, key=j_gene_agg.get)
            return set(top_vs), set(top_js)

    def _sampling_gene_prediction(self, walk, top_n=1, n_samples=25):
        """
        For each edge in the walk, sample a V gene and a J gene multiple times
        according to the relative weights on that edge.

        Args:
            walk (list): The list of node names representing the walk.
            top_n (int): How many top genes to return.
            n_samples (int): Number of times to sample from each edge's distribution.

        Returns:
            tuple:
              If top_n=1, returns (best_V_gene, best_J_gene) as strings.
              Otherwise, returns (list_of_top_V_genes, list_of_top_J_genes).
        """
        V_samples = []
        J_samples = []

        for i in range(len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i+1]):
                ed = self.graph[walk[i]][walk[i+1]]['data']

                v_names = list(ed.v_genes.keys())
                v_vals = [ed.v_probability(g) for g in v_names]

                j_names = list(ed.j_genes.keys())
                j_vals = [ed.j_probability(g) for g in j_names]

                for _ in range(n_samples):
                    if v_names:
                        V_samples.append(self._choice_wrapper(v_names, v_vals))
                    if j_names:
                        J_samples.append(self._choice_wrapper(j_names, j_vals))

        v_counter = Counter(V_samples)
        j_counter = Counter(J_samples)

        if top_n == 1:
            best_v = v_counter.most_common(1)[0][0] if v_counter else None
            best_j = j_counter.most_common(1)[0][0] if j_counter else None
            return best_v, best_j
        else:
            top_vs = [item[0] for item in v_counter.most_common(top_n)]
            top_js = [item[0] for item in j_counter.most_common(top_n)]
            return top_vs, top_js

    def _full_appearance_gene_prediction(self, walk, alpha=0):
        """
        Returns all genes (V or J) that appear on *every* edge of the walk.

        Args:
            walk (list): The list of node names representing the walk.
            alpha (int): Optional offset (skip the last alpha edges).

        Returns:
            tuple: (list_of_consistent_V_genes, list_of_consistent_J_genes)
        """
        vgenes = []
        jgenes = []

        end_index = len(walk) - 1 - alpha
        for i in range(end_index):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                ed = self.graph[walk[i]][walk[i+1]]['data']
                vgenes.append(set(ed.v_genes.keys()))
                jgenes.append(set(ed.j_genes.keys()))

        if vgenes:
            common_vs = set.intersection(*vgenes)
        else:
            common_vs = set()

        if jgenes:
            common_js = set.intersection(*jgenes)
        else:
            common_js = set()

        return sorted(common_vs), sorted(common_js)

    def predict_vj_genes(self, walk, top_n=1, mode='max_sum', alpha=0):
        """
        Main entry point for gene prediction. Dispatches to an internal
        method depending on the mode.

        Args:
            walk (list): The list of node names representing the walk.
            top_n (int): How many top genes to return (applicable for sum, product, sampling).
            mode (str): Which prediction approach to use. One of:
                        ['max_sum', 'max_product', 'sampling', 'full'].
            alpha (int): Used only in 'full' mode, optionally ignoring last alpha edges.

        Returns:
            tuple: The result depends on the method chosen:
                   - 'max_sum' / 'max_product':
                      (best_V_gene, best_J_gene) if top_n=1, otherwise (set_of_V, set_of_J)
                   - 'sampling':
                      (best_V_gene, best_J_gene) if top_n=1, otherwise (list_of_V, list_of_J)
                   - 'full': (list_of_consistent_V_genes, list_of_consistent_J_genes)
        """
        if mode == 'max_sum':
            return self._max_sum_gene_prediction(walk, top_n)
        elif mode == 'max_product':
            return self._max_product_gene_prediction(walk, top_n)
        elif mode == 'sampling':
            return self._sampling_gene_prediction(walk, top_n)
        elif mode == 'full':
            return self._full_appearance_gene_prediction(walk, alpha)
        else:
            raise MetricsError(
                f"Unknown prediction mode: {mode}. "
                "Use 'max', 'sampling', or 'full'."
            )

    def _choice_wrapper(self, keys, values):
        """
        Utility to choose a gene from lists of keys and values.

        Args:
            keys (list): The list of gene names.
            values (list): The corresponding probabilities for each key.

        Returns:
            The selected key.
        """
        total = sum(values)
        if total <= 0:
            return None
        weights = [v / total for v in values]
        return choice(keys, weights)
