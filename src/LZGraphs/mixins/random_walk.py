import numpy as np
from ..utilities.misc import choice

class RandomWalkMixin:
    """
    Mixin that handles random walk logic on the graph (both genomic and non-genomic),
    plus logic for blacklisted edges, etc.

    Requirements:
        - The parent class must define:
            self.graph (networkx.DiGraph)
            self.genetic_walks_black_list (dict)
            self.genetic (bool)
            self.is_stop_condition(state, selected_v=None, selected_j=None) -> bool
            self._select_random_vj_genes(vj_init) -> (str, str)
            self._raise_genetic_mode_error() -> None
            self._random_initial_state() -> str
            self._get_node_feature_info_df(node, feature, V=None, J=None, asdict=False) -> pd.DataFrame or dict
        - A function `choice(iterable, weights)` that returns one item from `iterable`
          with probability distribution `weights`.
    """

    def clear_blacklist(self):
        """
        Clear the genetic walks blacklist to free memory.
        Call this after completing a batch of random walks or when memory usage is a concern.
        """
        self.genetic_walks_black_list = {}

    def genomic_random_walk(self, initial_state=None, vj_init='marginal', clear_blacklist=False):
        """
        Perform a random walk over the graph in 'genetic' mode,
        ensuring selected V and J genes are used at each step.

        Args:
            initial_state (str, optional): The starting node. If None,
                selects randomly based on some logic in `_random_initial_state()`.
            vj_init (str): Strategy for selecting V/J genes
                (e.g. 'marginal' or 'combined'), as implemented by `_select_random_vj_genes`.
            clear_blacklist (bool): If True, clears the genetic walks blacklist before
                starting the walk. Useful to prevent memory accumulation across many calls.
                Default is False for backward compatibility.

        Returns:
            tuple: (walk, selected_v, selected_j)
                - walk (list[str]): The sequence of nodes visited.
                - selected_v (str): The chosen V gene for this walk.
                - selected_j (str): The chosen J gene for this walk.
        """
        self._raise_genetic_mode_error()  # Raises error if self.genetic == False

        if clear_blacklist:
            self.genetic_walks_black_list = {}

        selected_v, selected_j = self._select_random_vj_genes(vj_init)

        # Initialize the walk
        current_state = initial_state or self._random_initial_state()
        walk = [current_state]

        while not self.is_stop_condition(current_state, selected_v, selected_j):
            edge_info = self._get_node_feature_info_df(
                current_state,
                'weight',
                selected_v,
                selected_j,
                asdict=True  # Return as dict for easy popping
            )

            # Apply black list for (current_state, selected_v, selected_j)
            if (current_state, selected_v, selected_j) in self.genetic_walks_black_list:
                blacklist_edges = self.genetic_walks_black_list[(current_state, selected_v, selected_j)]
                for col in blacklist_edges:
                    edge_info.pop(col, None)

            # If no edges remain, we've hit a dead-end
            if len(edge_info) == 0:
                # Backtrack if possible
                if len(walk) > 2:
                    last_state = walk[-2]
                    # Mark the current edge in the blacklist from last_state -> current_state
                    self.genetic_walks_black_list[(last_state, selected_v, selected_j)] = \
                        self.genetic_walks_black_list.get((last_state, selected_v, selected_j), []) + [walk[-1]]
                    current_state = last_state
                    walk.pop()
                else:
                    # Reset if backtracking isn't possible
                    walk = walk[:1]
                    current_state = walk[0]
                    selected_v, selected_j = self._select_random_vj_genes(vj_init)
                continue

            # Otherwise, pick the next state based on weights
            weights = np.array([edge_info[i]['weight'] for i in edge_info])
            weights /= weights.sum()  # normalize
            current_state = choice(list(edge_info.keys()), weights)
            walk.append(current_state)

        return walk, selected_v, selected_j

    def random_walk(self, initial_state=None, clear_blacklist=False):
        """
        Perform a random walk over the graph without gene constraints.

        Args:
            initial_state (str, optional): The starting node.
                If None, selects randomly in `_random_initial_state()`.
            clear_blacklist (bool): If True, clears the walks blacklist before
                starting the walk. Useful to prevent memory accumulation across many calls.
                Default is False for backward compatibility.

        Returns:
            list[str]: The sequence of nodes visited.
        """
        if clear_blacklist:
            self.genetic_walks_black_list = {}

        current_state = initial_state or self._random_initial_state()
        walk = [current_state]

        while not self.is_stop_condition(current_state):
            edge_info = self._get_node_feature_info_df(
                current_state,
                'weight',
                asdict=True
            )

            # Apply blacklist if there's an entry for this state
            if current_state in self.genetic_walks_black_list:
                blacklist_edges = self.genetic_walks_black_list[current_state]
                for col in blacklist_edges:
                    edge_info.pop(col, None)

            if len(edge_info) == 0:
                # If no edges remain, attempt backtracking or reset
                if len(walk) > 2:
                    last_state = walk[-2]
                    self.genetic_walks_black_list[last_state] = \
                        self.genetic_walks_black_list.get(last_state, []) + [walk[-1]]
                    current_state = last_state
                    walk.pop()
                else:
                    # Reset if we can't backtrack
                    walk = walk[:1]
                    current_state = walk[0]
                continue

            # Pick next state based on edge weights
            weights = np.array([edge_info[i]['weight'] for i in edge_info])
            weights /= weights.sum()
            current_state = choice(list(edge_info.keys()), weights)
            walk.append(current_state)

        return walk
