import heapq
from collections import Counter

class GenePredictionMixin:
    """
    Mixin that provides different heuristics for predicting V/J genes
    from a given walk (list of nodes).

    This mixin assumes:
    - `self.graph` is a networkx.DiGraph.
    - Each edge may have:
        - 'weight', 'Vsum', 'Jsum' as numeric keys.
        - Additional keys representing probabilities or counts for specific V/J genes.
    - A function `choice(options, weights)` is available for weighted selection.
    """

    def _max_sum_gene_prediction(self, walk, top_n=1):
        """
        Aggregate the sum of each V or J gene count along the edges of the walk.
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
                edge_data = self.graph[walk[i]][walk[i+1]]
                # Accumulate for each gene that starts with 'V' or 'J'
                for key, value in edge_data.items():
                    if key not in {'weight', 'Vsum', 'Jsum'}:
                        if key.startswith('V'):
                            v_gene_agg[key] = v_gene_agg.get(key, 0) + value
                        elif key.startswith('J'):
                            j_gene_agg[key] = j_gene_agg.get(key, 0) + value

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
        Aggregate the product of each V or J gene count along the edges.
        (Interpreted as probabilities, so we multiply them to get the overall
         probability if edges are viewed as independent factors.)

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
                edge_data = self.graph[walk[i]][walk[i+1]]
                for key, value in edge_data.items():
                    if key not in {'weight', 'Vsum', 'Jsum'}:
                        if key.startswith('V'):
                            # Use a product (start from 1)
                            v_gene_agg[key] = v_gene_agg.get(key, 1.0) * value
                        elif key.startswith('J'):
                            j_gene_agg[key] = j_gene_agg.get(key, 1.0) * value

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
        according to the relative weights on that edge. Aggregate these samples
        to find the most common V/J genes along the path.

        Args:
            walk (list): The list of node names representing the walk.
            top_n (int): How many top genes to return.
            n_samples (int): Number of times to sample from each edge's distribution
                             to build up frequencies.

        Returns:
            tuple:
              If top_n=1, returns (best_V_gene, best_J_gene) as strings.
              Otherwise, returns (list_of_top_V_genes, list_of_top_J_genes).
        """
        V_samples = []
        J_samples = []

        for i in range(len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i+1]):
                edge_data = self.graph[walk[i]][walk[i+1]]
                # Separate V and J genes, ignoring 'weight','Vsum','Jsum'
                v_keys = {}
                j_keys = {}
                for key, val in edge_data.items():
                    if key not in {'weight', 'Vsum', 'Jsum'}:
                        if key.startswith('V'):
                            v_keys[key] = val
                        elif key.startswith('J'):
                            j_keys[key] = val

                # Sampling from each edge distribution
                if v_keys:
                    v_names = list(v_keys.keys())
                    v_vals = list(v_keys.values())
                else:
                    v_names = []
                    v_vals = []

                if j_keys:
                    j_names = list(j_keys.keys())
                    j_vals = list(j_keys.values())
                else:
                    j_names = []
                    j_vals = []

                # Normalize if needed
                # For example:
                # sum_v = sum(v_vals) if v_vals else 1
                # v_probs = [x / sum_v for x in v_vals]
                # sum_j = sum(j_vals) if j_vals else 1
                # j_probs = [x / sum_j for x in j_vals]

                # Then sample n_samples times
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
        Returns all genes (V or J) that appear on *every* edge of the walk
        (minus an alpha offset if desired).

        Args:
            walk (list): The list of node names representing the walk.
            alpha (int): Optional offset (if you need to skip the last alpha edges, etc.).

        Returns:
            tuple: (list_of_consistent_V_genes, list_of_consistent_J_genes)
        """
        vgenes = []
        jgenes = []

        # If alpha is 0, we use all edges from 0 to len(walk)-2
        # If alpha is 1, we skip the last edge, etc.
        # (Adjust logic as needed, depending on your definition.)
        end_index = len(walk) - 1 - alpha
        for i in range(end_index):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                edge_data = self.graph[walk[i]][walk[i+1]]
                # Collect all V or J keys from this edge
                current_v = [k for k in edge_data if k.startswith('V')]
                current_j = [k for k in edge_data if k.startswith('J')]

                vgenes.append(set(current_v))
                jgenes.append(set(current_j))

        # Intersect across all edges to find those that appear in every edge
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
            # You can also pass n_samples if you wish, for example:
            # return self._sampling_gene_prediction(walk, top_n, n_samples=25)
            return self._sampling_gene_prediction(walk, top_n)
        elif mode == 'full':
            return self._full_appearance_gene_prediction(walk, alpha)
        else:
            raise ValueError(f"Unknown prediction mode: {mode}")

    def _choice_wrapper(self, keys, values):
        """
        Utility to choose a gene from lists of keys and values.
        You can adapt this if you have a custom choice function or you can do:
           return np.random.choice(keys, p=normalized_weights)

        Args:
            keys (list): The list of gene names (e.g. ['V1', 'V2', ...]).
            values (list): The corresponding values for each key (counts or probabilities).

        Returns:
            The selected key (e.g. 'V1' or 'J3').
        """
        # Here we assume 'choice' is a function that does weighted choice:
        total = sum(values)
        if total <= 0:
            # fallback: pick randomly or return None
            # raise an error, or just pick randomly among keys
            return None
        weights = [v / total for v in values]
        return choice(keys, weights)
