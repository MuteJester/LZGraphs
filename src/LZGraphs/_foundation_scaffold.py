"""Foundation-derived FlashBack scaffold alignment.

This module implements a pure-Python prototype aligner that sits on top of an
existing ``FlashBackGraph``. The alignment topology is a fixed scaffold of
bilateral structural slots ``(order, lane, offset)`` learned from the
foundation graph, while each aligned residue also carries a token-level
compatibility score against that slot.

The intent is to preserve the graph-native FlashBack structure without falling
back to classical pairwise-DP multiple alignment.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


_LANE_ORDER = {"front": 0, "center": 1, "back": 2}


def _coord_sort_key(coord):
    return (coord[0], _LANE_ORDER.get(coord[1], 99), coord[2])


def _parse_token(token_str):
    """Parse a FlashBack token string."""
    match = re.match(r"^(.+)_(\d+)\{(\d+)\}$", token_str)
    if not match:
        return None
    symbols = match.group(1)
    split_pos = int(match.group(2))
    order = int(match.group(3))
    is_terminal = split_pos == 0 and order > 0
    if is_terminal:
        front = symbols
        back = ""
    else:
        front = symbols[:split_pos]
        back = symbols[split_pos:]
    return {
        "symbols": symbols,
        "split_pos": split_pos,
        "order": order,
        "front": front,
        "back": back,
        "is_terminal": is_terminal,
    }


def _token_shape(token_str):
    parsed = _parse_token(token_str)
    if parsed is None:
        return None
    return (len(parsed["front"]), len(parsed["back"]), parsed["is_terminal"])


def _token_signature(token_str):
    parsed = _parse_token(token_str)
    if parsed is None:
        return None
    return (
        parsed["order"],
        len(parsed["front"]),
        len(parsed["back"]),
        parsed["is_terminal"],
    )


def _symbol_identity(token_a, token_b):
    pa = _parse_token(token_a)
    pb = _parse_token(token_b)
    if pa is None or pb is None:
        return 0.0
    sa = pa["symbols"]
    sb = pb["symbols"]
    if len(sa) != len(sb) or len(sa) == 0:
        return 1.0 if sa == sb else 0.0
    matches = sum(1 for a, b in zip(sa, sb) if a == b)
    return matches / len(sa)


def _fallback_token_meta(token_str):
    return {
        "token_idx": None,
        "shape": _token_shape(token_str),
        "pred_top": tuple(),
        "succ_top": tuple(),
        "in_degree": 0,
        "out_degree": 0,
    }


def _jaccard(a, b):
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _graph_similarity(meta_a, meta_b):
    """Similarity in [0, 1] based on foundation neighborhoods."""
    score = 0.0
    if meta_a["shape"] == meta_b["shape"]:
        score += 1.0
    score += 1.5 * _jaccard(meta_a["pred_top"], meta_b["pred_top"])
    score += 1.5 * _jaccard(meta_a["succ_top"], meta_b["succ_top"])
    if meta_a["token_idx"] is not None and meta_b["token_idx"] is not None:
        if abs(meta_a["in_degree"] - meta_b["in_degree"]) <= 2:
            score += 0.25
        if abs(meta_a["out_degree"] - meta_b["out_degree"]) <= 2:
            score += 0.25
    return score / 4.5


def _build_token_meta(foundation_graph, token_labels, top_k=5):
    cache = getattr(foundation_graph, "_foundation_scaffold_token_meta_cache", None)
    if cache is None:
        cache = {}

    missing = [token for token in token_labels if token not in cache]
    if missing:
        idx_map = foundation_graph._get_node_index()
        csr = foundation_graph._get_csr()
        rcsr = foundation_graph._get_reverse_csr()

        def top_neighbors(idx, csr_like):
            start = int(csr_like["row_offsets"][idx])
            end = int(csr_like["row_offsets"][idx + 1])
            entries = []
            for e in range(start, end):
                nbr = int(csr_like["col_indices"][e])
                count = int(csr_like["counts"][e])
                weight = float(csr_like["weights"][e])
                entries.append((count, weight, nbr))
            entries.sort(key=lambda x: (-x[0], -x[1], x[2]))
            top = entries[:top_k]
            return tuple(nbr for _, _, nbr in top), len(entries)

        for token in missing:
            idx = idx_map.get(token)
            if idx is None:
                cache[token] = _fallback_token_meta(token)
                continue
            succ_top, out_degree = top_neighbors(idx, csr)
            pred_top, in_degree = top_neighbors(idx, rcsr)
            cache[token] = {
                "token_idx": idx,
                "shape": _token_shape(token),
                "pred_top": pred_top,
                "succ_top": succ_top,
                "in_degree": in_degree,
                "out_degree": out_degree,
            }

        setattr(foundation_graph, "_foundation_scaffold_token_meta_cache", cache)

    return {token: cache[token] for token in token_labels}


def _foundation_node_masses(foundation_graph):
    cache = getattr(foundation_graph, "_foundation_scaffold_node_mass_cache", None)
    if cache is not None:
        return dict(cache)

    csr = foundation_graph._get_csr()
    rcsr = foundation_graph._get_reverse_csr()
    masses = {}
    for idx, token in enumerate(foundation_graph.all_nodes):
        if token.startswith("@") or "$" in token:
            continue
        in_total = int(rcsr["counts"][rcsr["row_offsets"][idx]:rcsr["row_offsets"][idx + 1]].sum())
        out_total = int(csr["counts"][csr["row_offsets"][idx]:csr["row_offsets"][idx + 1]].sum())
        masses[token] = max(in_total, out_total)

    setattr(foundation_graph, "_foundation_scaffold_node_mass_cache", dict(masses))
    return masses


def _foundation_slot_counts(foundation_graph):
    masses = _foundation_node_masses(foundation_graph)
    slot_counts = defaultdict(Counter)
    for token, mass in masses.items():
        if mass <= 0:
            continue
        parsed = _parse_token(token)
        if parsed is None or parsed["order"] == 0:
            continue
        order = parsed["order"]
        if parsed["is_terminal"]:
            for offset, _aa in enumerate(parsed["front"]):
                slot_counts[(order, "center", offset)][token] += mass
        else:
            for offset, _aa in enumerate(parsed["front"]):
                slot_counts[(order, "front", offset)][token] += mass
            for offset, _aa in enumerate(reversed(parsed["back"])):
                slot_counts[(order, "back", offset)][token] += mass
    return slot_counts


def _foundation_signature_counts(foundation_graph):
    masses = _foundation_node_masses(foundation_graph)
    sig_counts = defaultdict(Counter)
    for token, mass in masses.items():
        sig = _token_signature(token)
        if sig is None or sig[0] == 0 or mass <= 0:
            continue
        sig_counts[sig][token] += mass
    return sig_counts


def _extract_residue_events(sequence, decompose_fn):
    """Expand a sequence into residue-level structural events."""
    tokens_raw = decompose_fn(sequence)
    events = []
    left = 0
    right = len(sequence)

    def add_event(order, lane, offset, aa, seq_pos, token, token_index):
        if not (0 <= seq_pos < len(sequence)):
            raise ValueError(f"seq_pos out of bounds for {sequence}: {seq_pos}")
        if sequence[seq_pos] != aa:
            raise ValueError(
                f"AA mismatch for {sequence}: pos {seq_pos}, expected {sequence[seq_pos]}, got {aa}"
            )
        events.append(
            {
                "coord": (order, lane, offset),
                "order": order,
                "lane": lane,
                "offset": offset,
                "aa": aa,
                "seq_pos": seq_pos,
                "token": token,
                "token_index": token_index,
            }
        )

    for token_index, token in enumerate(tokens_raw):
        parsed = _parse_token(token)
        if parsed is None or parsed["order"] == 0:
            continue
        order = parsed["order"]
        if parsed["is_terminal"]:
            for offset, aa in enumerate(parsed["front"]):
                add_event(order, "center", offset, aa, left + offset, token, token_index)
            left += len(parsed["front"])
            right = left
        else:
            for offset, aa in enumerate(parsed["front"]):
                add_event(order, "front", offset, aa, left + offset, token, token_index)
            left += len(parsed["front"])
            for offset, aa in enumerate(reversed(parsed["back"])):
                add_event(order, "back", offset, aa, right - 1 - offset, token, token_index)
            right -= len(parsed["back"])

    events.sort(key=lambda ev: ev["seq_pos"])
    covered = {ev["seq_pos"] for ev in events}
    if covered != set(range(len(sequence))):
        raise ValueError(f"Residue coverage failed for {sequence}")
    if any(ev["token_index"] is None for ev in events):
        raise ValueError(f"Token index assignment failed for {sequence}")
    return events, tokens_raw


@dataclass(frozen=True, slots=True)
class ScaffoldPosteriorCandidate:
    """Candidate latent token state for a scaffold event."""

    token: str
    posterior: float
    prior: float
    similarity: float


@dataclass(frozen=True, slots=True)
class FoundationScaffoldEvent:
    """One aligned residue event on the scaffold."""

    coord: tuple[int, str, int]
    slot_index: int
    aa: str
    seq_pos: int
    token_index: int
    token: str
    latent_token: str
    emission_score: float
    emission_log_score: float
    posterior: tuple[ScaffoldPosteriorCandidate, ...]


@dataclass(frozen=True, slots=True)
class FoundationScaffoldAlignment:
    """Alignment result for a single sequence."""

    sequence: str
    tokens: tuple[str, ...]
    latent_tokens: tuple[str, ...]
    token_states: tuple[FoundationScaffoldTokenState, ...]
    row: str
    events: tuple[FoundationScaffoldEvent, ...]
    total_log_emission: float
    total_log_path: float


@dataclass(frozen=True, slots=True)
class FoundationScaffoldTokenState:
    """Latent foundation token assignment for one observed token."""

    token_index: int
    observed_token: str
    latent_token: str
    slot_signature: tuple[int, int, int, bool]
    local_log_emission: float
    incoming_log_transition: float
    total_log_score: float
    posterior: tuple[ScaffoldPosteriorCandidate, ...]


@dataclass(frozen=True, slots=True)
class FoundationScaffoldBatch:
    """Batch alignment result on a fixed scaffold."""

    alignments: tuple[FoundationScaffoldAlignment, ...]
    rows: tuple[str, ...]
    columns: tuple[tuple[int, str, int], ...]
    col_orders: np.ndarray
    col_sides: tuple[str, ...]
    occupancy: np.ndarray
    mean_log_emission: np.ndarray

    @property
    def n_sequences(self):
        return len(self.alignments)

    @property
    def n_columns(self):
        return len(self.columns)


class FoundationScaffoldAligner:
    """Fixed-slot aligner derived from a foundation ``FlashBackGraph``.

    The scaffold topology is the set of bilateral structural slots seen in the
    foundation graph. Aligning a sequence means decomposing it with FlashBack
    and mapping each residue event to its deterministic structural slot, then
    scoring the event against the slot's foundation token prior.
    """

    def __init__(
        self,
        foundation_graph,
        *,
        columns,
        slot_index,
        slot_token_counts,
        slot_totals,
        slot_top_tokens,
        signature_token_counts,
        signature_totals,
        signature_top_tokens,
        node_masses,
        token_meta,
        candidate_limit=32,
        similarity_floor=1e-6,
        emission_graph_weight=0.65,
        emission_symbol_weight=0.35,
        transition_weight=1.0,
        transition_smoothing=1e-9,
    ):
        self.foundation_graph = foundation_graph
        self.columns = tuple(columns)
        self.slot_index = dict(slot_index)
        self.slot_token_counts = dict(slot_token_counts)
        self.slot_totals = dict(slot_totals)
        self.slot_top_tokens = {coord: tuple(tokens) for coord, tokens in slot_top_tokens.items()}
        self.signature_token_counts = dict(signature_token_counts)
        self.signature_totals = dict(signature_totals)
        self.signature_top_tokens = {
            sig: tuple(tokens) for sig, tokens in signature_top_tokens.items()
        }
        self.node_masses = dict(node_masses)
        self.token_meta = dict(token_meta)
        self.candidate_limit = int(candidate_limit)
        self.similarity_floor = float(similarity_floor)
        self.emission_graph_weight = float(emission_graph_weight)
        self.emission_symbol_weight = float(emission_symbol_weight)
        self.transition_weight = float(transition_weight)
        self.transition_smoothing = float(transition_smoothing)
        self.col_orders = np.array([coord[0] for coord in self.columns], dtype=int)
        self.col_sides = tuple(coord[1] for coord in self.columns)
        self._node_index = foundation_graph._get_node_index()
        self._csr = foundation_graph._get_csr()
        self._edge_weight_cache = {}

    def __repr__(self):
        return (
            f"FoundationScaffoldAligner(columns={len(self.columns)}, "
            f"tokens={len(self.token_meta)}, candidate_limit={self.candidate_limit})"
        )

    @classmethod
    def from_graph(
        cls,
        foundation_graph,
        *,
        top_k_neighbors=5,
        top_tokens_per_slot=64,
        top_tokens_per_signature=64,
        candidate_limit=32,
        similarity_floor=1e-6,
        emission_graph_weight=0.65,
        emission_symbol_weight=0.35,
        transition_weight=1.0,
        transition_smoothing=1e-9,
    ):
        """Build a fixed scaffold and slot-token priors from a foundation graph."""
        slot_counts = _foundation_slot_counts(foundation_graph)
        signature_counts = _foundation_signature_counts(foundation_graph)
        node_masses = _foundation_node_masses(foundation_graph)
        columns = sorted(slot_counts.keys(), key=_coord_sort_key)
        slot_index = {coord: i for i, coord in enumerate(columns)}
        real_tokens = set(node_masses)
        token_meta = _build_token_meta(foundation_graph, real_tokens, top_k=top_k_neighbors)
        slot_totals = {coord: int(sum(counts.values())) for coord, counts in slot_counts.items()}
        slot_top_tokens = {
            coord: [token for token, _ in counts.most_common(top_tokens_per_slot)]
            for coord, counts in slot_counts.items()
        }
        signature_totals = {
            sig: int(sum(counts.values())) for sig, counts in signature_counts.items()
        }
        signature_top_tokens = {
            sig: [token for token, _ in counts.most_common(top_tokens_per_signature)]
            for sig, counts in signature_counts.items()
        }
        return cls(
            foundation_graph,
            columns=columns,
            slot_index=slot_index,
            slot_token_counts=slot_counts,
            slot_totals=slot_totals,
            slot_top_tokens=slot_top_tokens,
            signature_token_counts=signature_counts,
            signature_totals=signature_totals,
            signature_top_tokens=signature_top_tokens,
            node_masses=node_masses,
            token_meta=token_meta,
            candidate_limit=candidate_limit,
            similarity_floor=similarity_floor,
            emission_graph_weight=emission_graph_weight,
            emission_symbol_weight=emission_symbol_weight,
            transition_weight=transition_weight,
            transition_smoothing=transition_smoothing,
        )

    def _meta_for_token(self, token):
        meta = self.token_meta.get(token)
        if meta is None:
            meta = _fallback_token_meta(token)
        return meta

    def top_slot_tokens(self, coord, n=10):
        """Return the top foundation tokens for a scaffold slot."""
        counts = self.slot_token_counts.get(coord)
        if counts is None:
            raise KeyError(f"Unknown scaffold slot: {coord}")
        return counts.most_common(n)

    def top_signature_tokens(self, signature, n=10):
        """Return the top foundation tokens for a token signature."""
        counts = self.signature_token_counts.get(signature)
        if counts is None:
            raise KeyError(f"Unknown token signature: {signature}")
        return counts.most_common(n)

    def token_candidates(self, query_token, *, candidate_limit=None):
        """Return candidate latent foundation tokens for an observed token."""
        signature = _token_signature(query_token)
        if signature is None:
            raise KeyError(f"Unparseable FlashBack token: {query_token}")
        counts = self.signature_token_counts.get(signature)
        if counts is None:
            return [query_token]
        limit = self.candidate_limit if candidate_limit is None else int(candidate_limit)
        candidates = list(self.signature_top_tokens[signature][:limit])
        if query_token in counts and query_token not in candidates:
            candidates.append(query_token)
        return candidates

    def _slot_token_similarity(self, query_token, candidate_token, *, similarity_mode="graph"):
        query_meta = self._meta_for_token(query_token)
        cand_meta = self._meta_for_token(candidate_token)
        if similarity_mode == "graph":
            similarity = _graph_similarity(query_meta, cand_meta)
        elif similarity_mode == "symbol":
            similarity = _symbol_identity(query_token, candidate_token)
        elif similarity_mode == "hybrid":
            similarity = 0.5 * (
                _graph_similarity(query_meta, cand_meta)
                + _symbol_identity(query_token, candidate_token)
            )
        elif similarity_mode == "uniform":
            similarity = 1.0
        else:
            raise ValueError(f"Unknown similarity_mode: {similarity_mode}")
        return max(self.similarity_floor, similarity)

    def _transition_prob(self, src_token, dst_token):
        key = (src_token, dst_token)
        cached = self._edge_weight_cache.get(key)
        if cached is not None:
            return cached
        src = self._node_index.get(src_token)
        dst = self._node_index.get(dst_token)
        if src is None or dst is None:
            self._edge_weight_cache[key] = 0.0
            return 0.0
        start = int(self._csr["row_offsets"][src])
        end = int(self._csr["row_offsets"][src + 1])
        prob = 0.0
        for e in range(start, end):
            if int(self._csr["col_indices"][e]) == dst:
                prob = float(self._csr["weights"][e])
                break
        self._edge_weight_cache[key] = prob
        return prob

    def _token_emission_components(self, observed_token, latent_token):
        signature = _token_signature(observed_token)
        total = self.signature_totals.get(signature, 0)
        counts = self.signature_token_counts.get(signature)
        prior = 0.0
        if counts is not None and total:
            prior = counts[latent_token] / total
        elif latent_token in self.node_masses:
            prior = self.node_masses[latent_token] / max(sum(self.node_masses.values()), 1)

        graph_sim = _graph_similarity(
            self._meta_for_token(observed_token),
            self._meta_for_token(latent_token),
        )
        symbol_sim = _symbol_identity(observed_token, latent_token)
        similarity = max(
            self.similarity_floor,
            self.emission_graph_weight * graph_sim + self.emission_symbol_weight * symbol_sim,
        )
        emission_log = math.log(max(prior, self.similarity_floor)) + math.log(similarity)
        return prior, graph_sim, symbol_sim, similarity, emission_log

    def slot_token_posterior(
        self,
        coord,
        query_token,
        *,
        top_k=5,
        candidate_limit=None,
        similarity_mode="graph",
    ):
        """Posterior over latent foundation tokens for an observed token at a slot.

        The posterior is:

            p(z | slot, query) proportional to p(z | slot) * sim(query, z)

        where ``z`` is a latent foundation token and ``sim`` is a graph
        neighborhood similarity score.
        """
        if coord not in self.slot_token_counts:
            raise KeyError(f"Unknown scaffold slot: {coord}")

        limit = self.candidate_limit if candidate_limit is None else int(candidate_limit)
        counts = self.slot_token_counts[coord]
        total = self.slot_totals[coord]

        candidates = list(self.slot_top_tokens[coord][:limit])
        if query_token in counts and query_token not in candidates:
            candidates.append(query_token)

        weighted = []
        for token in candidates:
            prior = counts[token] / total if total else 0.0
            similarity = self._slot_token_similarity(
                query_token,
                token,
                similarity_mode=similarity_mode,
            )
            weighted.append((token, prior, similarity, prior * similarity))

        emission_score = sum(item[3] for item in weighted)
        if emission_score <= 0:
            posterior = tuple()
            log_score = float("-inf")
            return posterior, 0.0, log_score

        weighted.sort(key=lambda item: (-item[3], item[0]))
        posterior = tuple(
            ScaffoldPosteriorCandidate(
                token=item[0],
                posterior=item[3] / emission_score,
                prior=item[1],
                similarity=item[2],
            )
            for item in weighted[:top_k]
        )
        return posterior, emission_score, math.log(emission_score)

    @staticmethod
    def support_adaptive_blend(exact_value, smooth_value, support, kappa):
        """Blend exact and smoothed scalar values by support."""
        support = max(float(support), 0.0)
        kappa = max(float(kappa), 0.0)
        alpha = support / (support + kappa) if (support > 0 or kappa > 0) else 1.0
        return alpha * float(exact_value) + (1.0 - alpha) * float(smooth_value)

    @staticmethod
    def support_adaptive_distribution(exact_dist, smooth_dist, support, kappa):
        """Blend exact and smoothed categorical distributions by support."""
        support = max(float(support), 0.0)
        kappa = max(float(kappa), 0.0)
        alpha = support / (support + kappa) if (support > 0 or kappa > 0) else 1.0
        keys = set(exact_dist) | set(smooth_dist)
        out = {
            key: alpha * exact_dist.get(key, 0.0) + (1.0 - alpha) * smooth_dist.get(key, 0.0)
            for key in keys
        }
        total = sum(out.values())
        if total <= 0:
            return {}
        return {key: value / total for key, value in out.items()}

    def posterior_expectation(
        self,
        coord,
        query_token,
        value_by_token,
        *,
        default_value=0.0,
        top_k=32,
        candidate_limit=None,
        similarity_mode="graph",
    ):
        """Expectation of a token-associated scalar under the slot posterior."""
        posterior, _score, _log = self.slot_token_posterior(
            coord,
            query_token,
            top_k=top_k,
            candidate_limit=candidate_limit,
            similarity_mode=similarity_mode,
        )
        if not posterior:
            return float(default_value)

        total = 0.0
        mass = 0.0
        for cand in posterior:
            total += cand.posterior * float(value_by_token.get(cand.token, default_value))
            mass += cand.posterior
        if mass < 1.0:
            total += (1.0 - mass) * float(default_value)
        return float(total)

    def posterior_distribution(
        self,
        coord,
        query_token,
        distribution_by_token,
        *,
        default_distribution=None,
        top_k=32,
        candidate_limit=None,
        similarity_mode="graph",
    ):
        """Posterior-weighted categorical distribution over token-associated distributions."""
        if default_distribution is None:
            default_distribution = {}
        posterior, _score, _log = self.slot_token_posterior(
            coord,
            query_token,
            top_k=top_k,
            candidate_limit=candidate_limit,
            similarity_mode=similarity_mode,
        )
        if not posterior:
            return dict(default_distribution)

        out = defaultdict(float)
        mass = 0.0
        for cand in posterior:
            dist = distribution_by_token.get(cand.token, default_distribution)
            for key, value in dist.items():
                out[key] += cand.posterior * float(value)
            mass += cand.posterior
        if mass < 1.0:
            for key, value in default_distribution.items():
                out[key] += (1.0 - mass) * float(value)
        total = sum(out.values())
        if total <= 0:
            return dict(default_distribution)
        return {key: value / total for key, value in out.items()}

    def infer_token_path(self, sequence, *, posterior_top_k=5, candidate_limit=None):
        """Infer a latent foundation-token path for the sequence via Viterbi."""
        from . import flashback_decompose

        _events, tokens_raw = _extract_residue_events(sequence, flashback_decompose)
        observed_tokens = [tok for tok in tokens_raw if (_parse_token(tok) or {}).get("order", 0) > 0]
        if not observed_tokens:
            return tuple(), tuple()

        limit = self.candidate_limit if candidate_limit is None else int(candidate_limit)
        root = tokens_raw[0]
        token_states = []

        candidate_sets = []
        emission_tables = []
        local_posteriors = []
        for obs in observed_tokens:
            candidates = self.token_candidates(obs, candidate_limit=limit)
            candidate_sets.append(candidates)
            emission_map = {}
            weights = []
            for cand in candidates:
                prior, graph_sim, symbol_sim, similarity, emission_log = self._token_emission_components(obs, cand)
                emission_map[cand] = {
                    "prior": prior,
                    "graph_similarity": graph_sim,
                    "symbol_similarity": symbol_sim,
                    "similarity": similarity,
                    "emission_log": emission_log,
                }
                weights.append((cand, max(prior * similarity, self.similarity_floor)))
            total_weight = sum(weight for _, weight in weights)
            local_posteriors.append(
                tuple(
                    ScaffoldPosteriorCandidate(
                        token=token,
                        posterior=weight / total_weight if total_weight else 0.0,
                        prior=emission_map[token]["prior"],
                        similarity=emission_map[token]["similarity"],
                    )
                    for token, weight in sorted(weights, key=lambda item: (-item[1], item[0]))[:posterior_top_k]
                )
            )
            emission_tables.append(emission_map)

        backptr = []
        dp_prev = {}
        first_obs = observed_tokens[0]
        for cand in candidate_sets[0]:
            trans = self._transition_prob(root, cand)
            prior = emission_tables[0][cand]["prior"]
            trans_log = self.transition_weight * math.log(max(trans + self.transition_smoothing * prior, self.similarity_floor))
            dp_prev[cand] = emission_tables[0][cand]["emission_log"] + trans_log
        backptr.append({cand: None for cand in candidate_sets[0]})

        for pos in range(1, len(observed_tokens)):
            dp_cur = {}
            ptr_cur = {}
            for cand in candidate_sets[pos]:
                best_prev = None
                best_score = float("-inf")
                prior = emission_tables[pos][cand]["prior"]
                for prev in candidate_sets[pos - 1]:
                    trans = self._transition_prob(prev, cand)
                    trans_log = self.transition_weight * math.log(max(trans + self.transition_smoothing * prior, self.similarity_floor))
                    score = dp_prev[prev] + trans_log
                    if score > best_score:
                        best_score = score
                        best_prev = prev
                dp_cur[cand] = emission_tables[pos][cand]["emission_log"] + best_score
                ptr_cur[cand] = best_prev
            dp_prev = dp_cur
            backptr.append(ptr_cur)

        best_last = max(dp_prev.items(), key=lambda item: item[1])[0]
        latent_tokens = [best_last]
        for pos in range(len(observed_tokens) - 1, 0, -1):
            best_last = backptr[pos][best_last]
            latent_tokens.append(best_last)
        latent_tokens.reverse()

        prev = root
        for token_index, (obs, lat, posterior, emission_map) in enumerate(
            zip(observed_tokens, latent_tokens, local_posteriors, emission_tables)
        ):
            prior = emission_map[lat]["prior"]
            trans = self._transition_prob(prev, lat)
            trans_log = self.transition_weight * math.log(max(trans + self.transition_smoothing * prior, self.similarity_floor))
            token_states.append(
                FoundationScaffoldTokenState(
                    token_index=token_index + 1,
                    observed_token=obs,
                    latent_token=lat,
                    slot_signature=_token_signature(obs),
                    local_log_emission=emission_map[lat]["emission_log"],
                    incoming_log_transition=trans_log,
                    total_log_score=emission_map[lat]["emission_log"] + trans_log,
                    posterior=posterior,
                )
            )
            prev = lat

        return tuple(token_states), tuple(tokens_raw)

    def align(self, sequence, *, posterior_top_k=5):
        """Align one sequence onto the fixed scaffold."""
        return self.align_sequence(sequence, posterior_top_k=posterior_top_k)

    def align_sequence(self, sequence, *, posterior_top_k=5):
        """Align one sequence onto the fixed scaffold.

        This is the main single-sequence entrypoint.
        """
        from . import flashback_decompose

        events_raw, tokens_raw = _extract_residue_events(sequence, flashback_decompose)
        token_states, _ = self.infer_token_path(sequence, posterior_top_k=posterior_top_k)
        latent_by_index = {state.token_index: state.latent_token for state in token_states}
        row = ["-"] * len(self.columns)
        events = []
        total_log = 0.0
        for ev in events_raw:
            coord = ev["coord"]
            if coord not in self.slot_index:
                raise KeyError(
                    f"Sequence uses scaffold slot {coord} not present in the foundation-derived scaffold"
                )
            slot_index = self.slot_index[coord]
            posterior, emission_score, emission_log = self.slot_token_posterior(
                coord,
                ev["token"],
                top_k=posterior_top_k,
            )
            latent_token = latent_by_index.get(ev["token_index"], ev["token"])
            row[slot_index] = ev["aa"]
            events.append(
                FoundationScaffoldEvent(
                    coord=coord,
                    slot_index=slot_index,
                    aa=ev["aa"],
                    seq_pos=ev["seq_pos"],
                    token_index=ev["token_index"],
                    token=ev["token"],
                    latent_token=latent_token,
                    emission_score=emission_score,
                    emission_log_score=emission_log,
                    posterior=posterior,
                )
            )
            total_log += emission_log

        return FoundationScaffoldAlignment(
            sequence=sequence,
            tokens=tuple(tokens_raw),
            latent_tokens=tuple(latent_by_index.get(i, tok) for i, tok in enumerate(tokens_raw)),
            token_states=token_states,
            row="".join(row),
            events=tuple(events),
            total_log_emission=float(total_log),
            total_log_path=float(sum(state.total_log_score for state in token_states)),
        )

    def align_many(self, sequences, *, posterior_top_k=5):
        """Align a batch of sequences and return a matrix-style scaffold view."""
        alignments = tuple(self.align_sequence(seq, posterior_top_k=posterior_top_k) for seq in sequences)
        rows = tuple(aln.row for aln in alignments)
        n_cols = len(self.columns)
        n_rows = len(rows)
        occupancy = np.zeros(n_cols, dtype=np.float64)
        mean_log_emission = np.zeros(n_cols, dtype=np.float64)
        support = np.zeros(n_cols, dtype=np.int64)

        for aln in alignments:
            for event in aln.events:
                occupancy[event.slot_index] += 1.0
                mean_log_emission[event.slot_index] += event.emission_log_score
                support[event.slot_index] += 1

        if n_rows:
            occupancy /= n_rows
        nz = support > 0
        mean_log_emission[nz] /= support[nz]
        mean_log_emission[~nz] = np.nan

        return FoundationScaffoldBatch(
            alignments=alignments,
            rows=rows,
            columns=self.columns,
            col_orders=self.col_orders.copy(),
            col_sides=self.col_sides,
            occupancy=occupancy,
            mean_log_emission=mean_log_emission,
        )


__all__ = [
    "FoundationScaffoldAligner",
    "FoundationScaffoldAlignment",
    "FoundationScaffoldBatch",
    "FoundationScaffoldEvent",
    "FoundationScaffoldTokenState",
    "ScaffoldPosteriorCandidate",
]
