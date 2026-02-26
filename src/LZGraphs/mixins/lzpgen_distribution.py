import numpy as np

from ..constants import _EPS, _LOG_EPS


class LZPgenDistributionMixin:
    """Mixin providing LZPgen distribution computation (Monte Carlo and analytical).

    Requirements:
        - self._walk_cache (dict or None)
        - self._build_walk_cache(seed) — method (stays in base)
        - self._get_topo_order() — method (from GraphTopologyMixin)
        - self.initial_state_probabilities (dict)
        - self._stop_probability_cache (dict)
        - self.graph (networkx.DiGraph)
    """

    def lzpgen_distribution(self, n=100_000, seed=None):
        """Compute the empirical LZPgen distribution via Monte Carlo.

        Generates *n* random walks from the graph's generative model and
        computes the log-probability (LZPgen) for each walk simultaneously.
        This gives the probability-weighted distribution of generation
        probabilities *directly from the graph structure*, without needing
        the original input sequences.

        The log-probability for each walk is computed as::

            log P(init) + sum(log P(edge_i)) + log P(stop | last_node)

        consistent with :meth:`walk_log_probability`.

        Args:
            n (int): Number of sequences to sample.  Default 100,000.
            seed (int, optional): RNG seed for reproducibility.

        Returns:
            numpy.ndarray: Array of shape ``(n,)`` with log-probability values.

        Example::

            log_probs = graph.lzpgen_distribution(n=50_000, seed=42)
            plt.hist(log_probs, bins=100, density=True)
            plt.xlabel('log P(seq)')
            plt.ylabel('Density')
        """
        if n <= 0:
            return np.empty(0, dtype=np.float64)

        # Build / rebuild cache
        if self._walk_cache is None or seed is not None:
            self._build_walk_cache(seed)

        cache = self._walk_cache
        rng = cache['rng']
        initial_ids = cache['initial_ids']
        initial_probs = cache['initial_probs']
        stop_probs = cache['stop_probs']
        neighbor_ids = cache['neighbor_ids']
        neighbor_weights = cache['neighbor_weights']

        # Pre-compute log values for zero per-step overhead
        eps = _EPS
        initial_log_probs = np.log(np.maximum(initial_probs, eps))

        n_nodes = len(cache['id_to_node'])
        stop_log_probs = np.where(
            np.isnan(stop_probs), np.nan,
            np.log(np.maximum(stop_probs, eps))
        )

        neighbor_log_weights = [None] * n_nodes
        for i in range(n_nodes):
            if neighbor_weights[i] is not None:
                neighbor_log_weights[i] = np.log(
                    np.maximum(neighbor_weights[i], eps)
                )

        log_probs = np.empty(n, dtype=np.float64)
        n_initial = len(initial_ids)

        for seq_idx in range(n):
            # Pick initial state
            init_idx = rng.choice(n_initial, p=initial_probs)
            current = initial_ids[init_idx]
            log_p = initial_log_probs[init_idx]

            while True:
                # Check stop condition
                stop_p = stop_probs[current]
                if not np.isnan(stop_p):
                    if rng.random() < stop_p:
                        log_p += stop_log_probs[current]
                        break

                # Dead-end check
                nb_ids = neighbor_ids[current]
                if nb_ids is None:
                    log_p += np.log(eps)
                    break

                # Take a step
                n_nb = len(nb_ids)
                step_idx = rng.choice(n_nb, p=neighbor_weights[current])
                log_p += neighbor_log_weights[current][step_idx]
                current = nb_ids[step_idx]

            log_probs[seq_idx] = log_p

        return log_probs

    def lzpgen_moments(self):
        """Compute exact moments of the LZPgen distribution via forward propagation.

        For DAG-structured graphs (AAPLZGraph, NDPLZGraph), this computes the
        *exact* mean and variance of the log-probability distribution by
        propagating probability mass and moment accumulators through the graph
        in topological order.  This runs in O(|E|) time and gives deterministic
        results with no sampling required.

        The computed moments correspond to the distribution of
        :meth:`walk_log_probability` values over all possible walks, weighted
        by their generative probability.

        The forward propagation tracks five quantities per node:

        - **m0**: total probability mass arriving at the node
        - **m1**: first moment (mass-weighted sum of log-probabilities)
        - **m2**: second moment (mass-weighted sum of squared log-probabilities)
        - **m3**: third moment (for skewness)
        - **m4**: fourth moment (for kurtosis)

        At each terminal node, the stopping mass contributes to the overall
        distribution moments.  Continue-mass propagates forward via edge
        transitions.

        Returns:
            dict: Keys ``'mean'``, ``'variance'``, ``'std'``, ``'skewness'``,
                ``'kurtosis'``, ``'total_mass'``.
                ``total_mass`` should be close to 1.0 for a well-formed model.

        Raises:
            RuntimeError: If the graph contains cycles (use
                :meth:`lzpgen_distribution` for cyclic graphs instead).

        Example::

            moments = graph.lzpgen_moments()
            print(f"E[log P] = {moments['mean']:.4f}")
            print(f"Std[log P] = {moments['std']:.4f}")
        """
        topo_order = self._get_topo_order()
        eps = _EPS
        n_nodes = len(topo_order)

        # Moment accumulators per node (indexed by topological position)
        m0 = np.zeros(n_nodes)
        m1 = np.zeros(n_nodes)
        m2 = np.zeros(n_nodes)
        m3 = np.zeros(n_nodes)
        m4 = np.zeros(n_nodes)

        node_to_idx = {node: i for i, node in enumerate(topo_order)}

        # Seed initial states
        for state, p in self.initial_state_probabilities.items():
            if state in node_to_idx:
                idx = node_to_idx[state]
                lp = np.log(max(p, eps))
                lp2 = lp * lp
                m0[idx] += p
                m1[idx] += p * lp
                m2[idx] += p * lp2
                m3[idx] += p * lp2 * lp
                m4[idx] += p * lp2 * lp2

        # Terminal accumulators
        T0 = T1 = T2 = T3 = T4 = 0.0

        graph = self.graph

        for u in topo_order:
            ui = node_to_idx[u]
            if m0[ui] == 0:
                continue

            # Terminal contribution: mass that stops here
            stop_prob = self._stop_probability_cache.get(u)
            if stop_prob is not None and stop_prob > 0:
                ls = np.log(max(stop_prob, eps))
                ls2 = ls * ls
                ls3 = ls2 * ls
                ls4 = ls3 * ls

                T0 += stop_prob * m0[ui]
                T1 += stop_prob * (m1[ui] + ls * m0[ui])
                T2 += stop_prob * (m2[ui] + 2 * ls * m1[ui] + ls2 * m0[ui])
                T3 += stop_prob * (
                    m3[ui] + 3 * ls * m2[ui]
                    + 3 * ls2 * m1[ui] + ls3 * m0[ui]
                )
                T4 += stop_prob * (
                    m4[ui] + 4 * ls * m3[ui] + 6 * ls2 * m2[ui]
                    + 4 * ls3 * m1[ui] + ls4 * m0[ui]
                )
                continue_factor = 1.0 - stop_prob
            else:
                continue_factor = 1.0

            if continue_factor < eps:
                continue

            # Propagate to successors (binomial moment expansion)
            for v in graph.successors(u):
                vi = node_to_idx[v]
                w = graph[u][v]['data'].weight
                lw = np.log(max(w, eps))
                lw2 = lw * lw
                lw3 = lw2 * lw
                lw4 = lw3 * lw
                f = continue_factor * w

                m0[vi] += f * m0[ui]
                m1[vi] += f * (m1[ui] + lw * m0[ui])
                m2[vi] += f * (m2[ui] + 2 * lw * m1[ui] + lw2 * m0[ui])
                m3[vi] += f * (
                    m3[ui] + 3 * lw * m2[ui]
                    + 3 * lw2 * m1[ui] + lw3 * m0[ui]
                )
                m4[vi] += f * (
                    m4[ui] + 4 * lw * m3[ui] + 6 * lw2 * m2[ui]
                    + 4 * lw3 * m1[ui] + lw4 * m0[ui]
                )

        if T0 < eps:
            raise RuntimeError("No probability mass reached terminal states")

        # Raw moments about origin
        mu1 = T1 / T0
        mu2 = T2 / T0
        mu3 = T3 / T0
        mu4 = T4 / T0

        # Cumulants
        k1 = mu1
        k2 = mu2 - mu1 ** 2
        k3 = mu3 - 3 * mu2 * mu1 + 2 * mu1 ** 3
        k4 = (mu4 - 4 * mu3 * mu1 - 3 * mu2 ** 2
              + 12 * mu2 * mu1 ** 2 - 6 * mu1 ** 4)

        variance = max(k2, 0.0)
        std = np.sqrt(variance)

        return {
            'mean': float(k1),
            'variance': float(variance),
            'std': float(std),
            'skewness': float(k3 / k2 ** 1.5) if k2 > 0 else 0.0,
            'kurtosis': float(k4 / k2 ** 2) if k2 > 0 else 0.0,
            'total_mass': float(T0),
        }

    def lzpgen_analytical_distribution(self):
        """Derive a full analytical LZPgen distribution from graph structure.

        Combines two complementary techniques:

        1. **Length-conditional Gaussian mixture** — one Normal component per
           walk length, with exact per-length (weight, mean, variance) computed
           by length-stratified forward propagation.
        2. **Saddlepoint approximation** — exact kappa_1 … kappa_4 computed by
           extending the moment propagation to m3 and m4, enabling the
           Lugannani-Rice PDF/CDF.

        All parameters are derived deterministically from the graph's adjacency
        matrix, edge weights, and stop probabilities.  No Monte Carlo sampling.

        Time complexity: O(|E| * K_max) where K_max is the maximum walk length
        (typically 5–25).

        Returns:
            LZPgenDistribution: A scipy-like distribution object with
                ``.pdf()``, ``.cdf()``, ``.ppf()``, ``.rvs()``,
                ``.confidence_interval()``, ``.saddlepoint_pdf()``,
                ``.saddlepoint_cdf()``, ``.mean()``, ``.var()``,
                ``.skewness()``, ``.kurtosis()``, and ``.summary()``.

        Raises:
            RuntimeError: If the graph contains cycles.

        Example::

            dist = graph.lzpgen_analytical_distribution()
            print(dist.summary())

            x = np.linspace(-35, -10, 500)
            plt.plot(x, dist.pdf(x), label='Gaussian mixture')
            plt.plot(x, dist.saddlepoint_pdf(x), '--', label='Saddlepoint')
        """
        from ..metrics.pgen_distribution import LZPgenDistribution

        topo_order = self._get_topo_order()
        eps = _EPS
        n_nodes = len(topo_order)
        node_to_idx = {node: i for i, node in enumerate(topo_order)}

        # ── Accumulators: m0…m4 per node ──
        m0 = np.zeros(n_nodes)
        m1 = np.zeros(n_nodes)
        m2 = np.zeros(n_nodes)
        m3 = np.zeros(n_nodes)
        m4 = np.zeros(n_nodes)

        # ── Depth per node (= walk edge-count from any initial state) ──
        depth = np.full(n_nodes, -1, dtype=np.int32)

        # Seed initial states
        for state, p in self.initial_state_probabilities.items():
            if state in node_to_idx:
                idx = node_to_idx[state]
                lp = np.log(max(p, eps))
                lp2 = lp * lp
                lp3 = lp2 * lp
                lp4 = lp3 * lp
                m0[idx] += p
                m1[idx] += p * lp
                m2[idx] += p * lp2
                m3[idx] += p * lp3
                m4[idx] += p * lp4
                depth[idx] = 0

        # ── Per-length terminal accumulators ──
        # {walk_length: [tm0, tm1, tm2]}
        per_length = {}
        # Global terminal accumulators for cumulants
        T0 = T1 = T2 = T3 = T4 = 0.0

        graph = self.graph

        for u in topo_order:
            ui = node_to_idx[u]
            if m0[ui] == 0:
                continue

            # Terminal contribution
            stop_prob = self._stop_probability_cache.get(u)
            if stop_prob is not None and stop_prob > 0:
                ls = np.log(max(stop_prob, eps))
                ls2 = ls * ls
                ls3 = ls2 * ls
                ls4 = ls3 * ls

                sm0 = stop_prob * m0[ui]
                sm1 = stop_prob * (m1[ui] + ls * m0[ui])
                sm2 = stop_prob * (m2[ui] + 2 * ls * m1[ui] + ls2 * m0[ui])
                sm3 = stop_prob * (
                    m3[ui] + 3 * ls * m2[ui]
                    + 3 * ls2 * m1[ui] + ls3 * m0[ui]
                )
                sm4 = stop_prob * (
                    m4[ui] + 4 * ls * m3[ui] + 6 * ls2 * m2[ui]
                    + 4 * ls3 * m1[ui] + ls4 * m0[ui]
                )

                T0 += sm0; T1 += sm1; T2 += sm2; T3 += sm3; T4 += sm4

                # Per-length accumulation
                d = int(depth[ui])
                if d not in per_length:
                    per_length[d] = [0.0, 0.0, 0.0]
                per_length[d][0] += sm0
                per_length[d][1] += sm1
                per_length[d][2] += sm2

                continue_factor = 1.0 - stop_prob
            else:
                continue_factor = 1.0

            if continue_factor < eps:
                continue

            # Propagate to successors
            for v in graph.successors(u):
                vi = node_to_idx[v]
                w = graph[u][v]['data'].weight
                lw = np.log(max(w, eps))
                lw2 = lw * lw
                lw3 = lw2 * lw
                lw4 = lw3 * lw
                f = continue_factor * w

                m0[vi] += f * m0[ui]
                m1[vi] += f * (m1[ui] + lw * m0[ui])
                m2[vi] += f * (m2[ui] + 2 * lw * m1[ui] + lw2 * m0[ui])
                m3[vi] += f * (
                    m3[ui] + 3 * lw * m2[ui]
                    + 3 * lw2 * m1[ui] + lw3 * m0[ui]
                )
                m4[vi] += f * (
                    m4[ui] + 4 * lw * m3[ui] + 6 * lw2 * m2[ui]
                    + 4 * lw3 * m1[ui] + lw4 * m0[ui]
                )

                # Propagate depth
                if depth[vi] == -1:
                    depth[vi] = depth[ui] + 1

        if T0 < eps:
            raise RuntimeError("No probability mass reached terminal states")

        # ── Build cumulants from global moments ──
        mu1 = T1 / T0
        mu2 = T2 / T0
        mu3 = T3 / T0
        mu4 = T4 / T0

        k1 = mu1
        k2 = mu2 - mu1 ** 2
        k3 = mu3 - 3 * mu2 * mu1 + 2 * mu1 ** 3
        k4 = mu4 - 4 * mu3 * mu1 - 3 * mu2 ** 2 + 12 * mu2 * mu1 ** 2 - 6 * mu1 ** 4

        cumulants = {
            'kappa_1': float(k1),
            'kappa_2': float(max(k2, 0.0)),
            'kappa_3': float(k3),
            'kappa_4': float(k4),
            'skewness': float(k3 / k2 ** 1.5) if k2 > 0 else 0.0,
            'kurtosis': float(k4 / k2 ** 2) if k2 > 0 else 0.0,
            'total_mass': float(T0),
        }

        # ── Build Gaussian mixture from per-length terminal data ──
        sorted_lengths = sorted(per_length.keys())
        weights = []
        means = []
        stds = []
        walk_lengths = []

        for d in sorted_lengths:
            pm0, pm1, pm2 = per_length[d]
            if pm0 < 1e-12:
                continue
            wt = pm0 / T0
            mu_d = pm1 / pm0
            var_d = pm2 / pm0 - mu_d ** 2
            weights.append(wt)
            means.append(mu_d)
            stds.append(np.sqrt(max(var_d, 0.0)))
            walk_lengths.append(d)

        return LZPgenDistribution(
            weights=weights,
            means=means,
            stds=stds,
            walk_lengths=walk_lengths,
            cumulants=cumulants,
        )
