import copy

from ..graphs.edge_data import EdgeData


class BayesianPosteriorMixin:
    """Mixin providing Bayesian posterior graph construction.

    Uses the current graph as a Dirichlet prior and observed sequences
    to produce a posterior graph via Dirichlet-Multinomial conjugacy.

    Requirements:
        - self.graph (networkx.DiGraph)
        - self.initial_state_counts (dict)
        - self.initial_state_probabilities (dict)
        - self.terminal_state_counts (dict)
        - self.node_outgoing_counts (dict)
        - self._stop_probability_cache (dict)
        - self.terminal_state_data (dict)
        - self.encode_sequence(seq) — abstract method from base
    """

    def get_posterior(self, sequences, abundances=None, kappa=1.0):
        """Create a posterior LZGraph by Bayesian updating from this graph.

        Uses this graph as a Dirichlet prior and the provided sequences as
        observed data to produce a posterior graph.  The posterior transition
        probabilities are:

            P_post(b|a) = (κ·P_prior(b|a) + c(a→b)) / (κ + n_a)

        where κ is the concentration parameter controlling prior strength,
        and c(a→b) are the observed edge counts from the individual's data.

        Args:
            sequences: Sequence data — list of strings, Series-like,
                or DataFrame-like with a sequence column.
            abundances: Optional per-sequence abundance counts.
                If None, each sequence counts as 1.
            kappa (float): Prior concentration parameter.  Lower values give
                more weight to the observed data.  Default 1.0.

        Returns:
            A new graph instance of the same type with posterior probabilities.
        """
        # 1. Parse sequences into edge/state counts
        seq_list, abd_list = self._parse_posterior_input(sequences, abundances)
        ind_counts = self._extract_counts_for_posterior(seq_list, abd_list)

        # 2. Create a lightweight copy (deep-copy only the nx.DiGraph)
        posterior = copy.copy(self)
        posterior.graph = copy.deepcopy(self.graph)

        # Deep copy mutable dicts that will be modified
        posterior.initial_state_counts = dict(self.initial_state_counts)
        posterior.terminal_state_counts = dict(self.terminal_state_counts)
        posterior.node_outgoing_counts = dict(self.node_outgoing_counts)
        posterior._walk_cache = None
        posterior._topo_order = None
        posterior._edges_cache = None

        # 3. Apply Bayesian update to edge weights
        self._apply_posterior_edges(posterior, ind_counts, kappa)

        # 4. Apply Bayesian update to initial state probabilities
        self._apply_posterior_initial_states(posterior, ind_counts, kappa)

        # 5. Apply Bayesian update to stop probabilities
        self._apply_posterior_stop_probs(posterior, ind_counts, kappa)

        return posterior

    def _parse_posterior_input(self, sequences, abundances):
        """Normalize posterior input to (list_of_strings, list_of_ints|None)."""
        if hasattr(sequences, 'columns'):
            # DataFrame-like: try known column names
            for col in ('cdr3_amino_acid', 'cdr3_rearrangement', 'amino_acid'):
                if col in sequences.columns:
                    seq_list = list(sequences[col])
                    break
            else:
                # Use first column
                cols = list(sequences.columns)
                seq_list = list(sequences[cols[0]])

            if abundances is None:
                for col in ('abundance', 'templates', 'count'):
                    if col in sequences.columns:
                        abd_list = list(sequences[col])
                        break
                else:
                    abd_list = None
            else:
                abd_list = list(abundances)
        elif hasattr(sequences, 'tolist') and not hasattr(sequences, 'columns'):
            # Series-like
            seq_list = sequences.tolist()
            abd_list = list(abundances) if abundances is not None else None
        else:
            seq_list = list(sequences)
            abd_list = list(abundances) if abundances is not None else None

        return seq_list, abd_list

    def _extract_counts_for_posterior(self, seq_list, abd_list):
        """Parse sequences through LZ encoding and count edges/states.

        Returns a dict with:
            edge_counts: {(a, b): count}
            initial_counts: {node: count}
            terminal_counts: {node: count}
            node_outgoing: {node: count}
        """
        edge_counts = {}
        initial_counts = {}
        terminal_counts = {}
        node_outgoing = {}

        if abd_list is not None:
            pairs = zip(seq_list, abd_list)
        else:
            pairs = ((seq, 1) for seq in seq_list)

        for seq, abd in pairs:
            count = int(abd)
            walk = self.encode_sequence(seq)
            if len(walk) == 0:
                continue

            first = walk[0]
            initial_counts[first] = initial_counts.get(first, 0) + count

            last = walk[-1]
            terminal_counts[last] = terminal_counts.get(last, 0) + count

            for i in range(len(walk) - 1):
                a, b = walk[i], walk[i + 1]
                edge_counts[(a, b)] = edge_counts.get((a, b), 0) + count
                node_outgoing[a] = node_outgoing.get(a, 0) + count

        return {
            'edge_counts': edge_counts,
            'initial_counts': initial_counts,
            'terminal_counts': terminal_counts,
            'node_outgoing': node_outgoing,
        }

    def _apply_posterior_edges(self, posterior, ind_counts, kappa):
        """Apply Dirichlet-Multinomial posterior update to edge weights."""
        edge_counts = ind_counts['edge_counts']
        ind_node_outgoing = ind_counts['node_outgoing']

        # Add novel edges from individual data to the posterior graph
        for (a, b), count in edge_counts.items():
            if not posterior.graph.has_node(a):
                posterior.graph.add_node(a)
            if not posterior.graph.has_node(b):
                posterior.graph.add_node(b)
            if not posterior.graph.has_edge(a, b):
                ed = EdgeData()
                ed.count = count
                ed._weight = 0.0  # will be set below
                posterior.graph.add_edge(a, b, data=ed)

        for node in list(posterior.graph.nodes()):
            succs = list(posterior.graph.successors(node))
            if not succs:
                continue

            # Only include κ in denominator if the prior had knowledge of
            # this node (i.e. it had outgoing edges). For entirely novel
            # nodes the prior contributes no mass, so κ should not inflate
            # the denominator — use pure MLE in that case.
            has_prior = self.graph.has_node(node) and self.graph.out_degree(node) > 0
            prior_total = kappa if has_prior else 0.0
            ind_total = ind_node_outgoing.get(node, 0)
            denom = prior_total + ind_total

            if denom <= 0:
                continue

            for succ in succs:
                ed = posterior.graph[node][succ]['data']
                prior_weight = prior_total * (ed.weight if ed.weight > 0 else 0.0)
                observed = edge_counts.get((node, succ), 0)
                ed._weight = (prior_weight + observed) / denom

            posterior.node_outgoing_counts[node] = (
                self.node_outgoing_counts.get(node, 0) + ind_total
            )

    def _apply_posterior_initial_states(self, posterior, ind_counts, kappa):
        """Apply Dirichlet posterior update to initial state distribution."""
        ind_init = ind_counts['initial_counts']
        all_states = set(self.initial_state_probabilities) | set(ind_init)

        posterior_counts = {}
        for state in all_states:
            # Ensure the state exists as a node in the posterior graph.
            # Prior initial states are valid starting points even if the
            # individual didn't happen to sample them — we keep them
            # weighted by their prior probability.
            if not posterior.graph.has_node(state):
                posterior.graph.add_node(state)
            prior_prob = self.initial_state_probabilities.get(state, 0.0)
            observed = ind_init.get(state, 0)
            posterior_counts[state] = kappa * prior_prob + observed

        total = sum(posterior_counts.values())
        if total > 0:
            posterior.initial_state_probabilities = {
                k: v / total for k, v in posterior_counts.items() if v > 0
            }
        for state, count in ind_init.items():
            posterior.initial_state_counts[state] = (
                posterior.initial_state_counts.get(state, 0) + count
            )

    def _apply_posterior_stop_probs(self, posterior, ind_counts, kappa):
        """Apply Dirichlet posterior update to stop probabilities."""
        ind_term = ind_counts['terminal_counts']
        ind_node_out = ind_counts['node_outgoing']

        all_terminal = set(self._stop_probability_cache) | set(ind_term)
        new_stop_cache = {}
        new_tsd = {}

        for state in all_terminal:
            f_term = self.terminal_state_counts.get(state, 0)
            f_out = self.node_outgoing_counts.get(state, 0)
            f_total = f_term + f_out

            if f_total > 0:
                prior_stop = kappa * (f_term / f_total)
                prior_cont = kappa * (f_out / f_total)
            else:
                f_stop = self._stop_probability_cache.get(state, 0.5)
                prior_stop = kappa * f_stop
                prior_cont = kappa * (1 - f_stop)

            obs_stop = ind_term.get(state, 0)
            obs_cont = ind_node_out.get(state, 0)

            post_stop = prior_stop + obs_stop
            post_cont = prior_cont + obs_cont
            denom = post_stop + post_cont

            if denom > 0:
                sp = post_stop / denom
            elif (prior_stop + prior_cont) > 0:
                sp = prior_stop / (prior_stop + prior_cont)
            else:
                sp = 0.5

            new_stop_cache[state] = sp
            new_tsd[state] = {
                'terminal_count': f_term + obs_stop,
                'outgoing_count': f_out + obs_cont,
                'stop_probability': sp,
            }

            posterior.terminal_state_counts[state] = (
                posterior.terminal_state_counts.get(state, 0) + obs_stop
            )

        posterior._stop_probability_cache = new_stop_cache
        posterior.terminal_state_data = new_tsd
