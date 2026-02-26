import networkx as nx


class GraphTopologyMixin:
    """Mixin providing graph topology inspection and mutation utilities.

    Requirements:
        - self.graph (networkx.DiGraph)
        - self.terminal_state_counts (dict)
        - self._topo_order (list or None)
        - self._walk_cache (object or None)
    """

    def _length_specific_terminal_state(self, length):
        """
        Return all terminal states whose suffix (split by '_') equals the given `length`.
        """
        return [
            state for state in self.terminal_state_counts
            if int(state.rsplit('_', 1)[-1]) == length
        ]

    @property
    def isolates(self):
        """
        Return a list of isolate nodes (nodes with zero edges).
        """
        return list(nx.isolates(self.graph))

    def drop_isolates(self):
        """
        Remove isolates (nodes with zero edges) from the graph.
        """
        self.graph.remove_nodes_from(self.isolates)
        self._walk_cache = None
        self._topo_order = None

    @property
    def is_dag(self):
        """
        Check whether the graph is a Directed Acyclic Graph (DAG).
        """
        return nx.is_directed_acyclic_graph(self.graph)

    def _get_topo_order(self):
        """Return cached topological order, rebuilding if needed.

        Raises:
            RuntimeError: If the graph contains cycles.
        """
        if self._topo_order is None:
            try:
                self._topo_order = list(nx.topological_sort(self.graph))
            except nx.NetworkXUnfeasible:
                raise RuntimeError(
                    "This operation requires a DAG-structured graph. "
                    "Use lzpgen_distribution() for graphs with cycles."
                )
        return self._topo_order
