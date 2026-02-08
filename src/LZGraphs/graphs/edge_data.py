"""
EdgeData: Encapsulates all data for a single directed edge in an LZGraph.

Raw counts are the source of truth. Normalized probabilities are cached
after calling normalize() and are read-only.
"""

from ..utilities.misc import _is_v_gene, _is_j_gene

__all__ = ["EdgeData"]


class EdgeData:
    """Stores all data for a single directed edge in an LZGraph.

    Raw counts are the source of truth. Normalized probabilities
    are cached after calling normalize() and are read-only.

    Attributes:
        count (int): Raw transition count (source of truth).
        v_genes (dict): {gene_name: raw_count} for V genes.
        j_genes (dict): {gene_name: raw_count} for J genes.
    """
    __slots__ = ('count', '_weight', 'v_genes', 'j_genes')

    def __init__(self):
        self.count = 0
        self._weight = 0.0
        self.v_genes = {}
        self.j_genes = {}

    @property
    def weight(self):
        """Cached transition probability P(B|A), set by normalize()."""
        return self._weight

    @property
    def vsum(self):
        """Total count of V gene observations on this edge."""
        return sum(self.v_genes.values())

    @property
    def jsum(self):
        """Total count of J gene observations on this edge."""
        return sum(self.j_genes.values())

    @property
    def is_genetic(self):
        """Whether this edge has any gene data."""
        return bool(self.v_genes or self.j_genes)

    def record(self, v_gene=None, j_gene=None):
        """Record one traversal during graph construction.

        Args:
            v_gene (str, optional): V gene to record.
            j_gene (str, optional): J gene to record.
        """
        self.count += 1
        if v_gene is not None:
            self.v_genes[v_gene] = self.v_genes.get(v_gene, 0) + 1
        if j_gene is not None:
            self.j_genes[j_gene] = self.j_genes.get(j_gene, 0) + 1

    def unrecord(self, v_gene=None, j_gene=None):
        """Remove one traversal (for sequence removal).

        Args:
            v_gene (str, optional): V gene to decrement.
            j_gene (str, optional): J gene to decrement.
        """
        self.count = max(0, self.count - 1)
        if v_gene is not None and v_gene in self.v_genes:
            self.v_genes[v_gene] -= 1
            if self.v_genes[v_gene] <= 0:
                del self.v_genes[v_gene]
        if j_gene is not None and j_gene in self.j_genes:
            self.j_genes[j_gene] -= 1
            if self.j_genes[j_gene] <= 0:
                del self.j_genes[j_gene]

    def merge(self, other):
        """Merge another EdgeData into this one (for graph union).

        Args:
            other (EdgeData): The edge data to merge in.
        """
        self.count += other.count
        for g, c in other.v_genes.items():
            self.v_genes[g] = self.v_genes.get(g, 0) + c
        for g, c in other.j_genes.items():
            self.j_genes[g] = self.j_genes.get(g, 0) + c

    def normalize(self, node_frequency, alpha=0.0, n_successors=0):
        """Compute and cache transition probability from raw count.

        Args:
            node_frequency (int): Total outgoing count from source node.
            alpha (float): Laplace smoothing parameter.
            n_successors (int): Number of successors (for Laplace smoothing).
        """
        if alpha > 0:
            denom = node_frequency + alpha * n_successors
            self._weight = (self.count + alpha) / denom if denom > 0 else 0.0
        elif node_frequency > 0:
            self._weight = self.count / node_frequency
        else:
            self._weight = 0.0

    def v_probability(self, gene):
        """Return P(gene) among V genes on this edge."""
        vsum = self.vsum
        return self.v_genes.get(gene, 0) / vsum if vsum > 0 else 0.0

    def j_probability(self, gene):
        """Return P(gene) among J genes on this edge."""
        jsum = self.jsum
        return self.j_genes.get(gene, 0) / jsum if jsum > 0 else 0.0

    def has_gene(self, gene):
        """Check if a gene (V or J) is present on this edge."""
        return gene in self.v_genes or gene in self.j_genes

    def gene_dict(self):
        """Return {gene: probability} dict for all genes on this edge."""
        result = {}
        vsum, jsum = self.vsum, self.jsum
        for g, c in self.v_genes.items():
            result[g] = c / vsum if vsum > 0 else 0.0
        for g, c in self.j_genes.items():
            result[g] = c / jsum if jsum > 0 else 0.0
        return result

    def to_legacy_dict(self):
        """Convert to flat dict matching old edge attribute format.

        Returns:
            dict: {weight, count, Vsum, Jsum, gene_name: probability, ...}
        """
        d = {'weight': self._weight, 'count': self.count}
        if self.v_genes:
            d['Vsum'] = self.vsum
            for g in self.v_genes:
                d[g] = self.v_probability(g)
        if self.j_genes:
            d['Jsum'] = self.jsum
            for g in self.j_genes:
                d[g] = self.j_probability(g)
        return d

    @classmethod
    def from_legacy_dict(cls, d, node_frequency=0):
        """Reconstruct EdgeData from an old-format flat dict.

        Used for loading old saves where edge data was stored as
        {weight, Vsum, Jsum, gene_name: probability, ...}.

        Args:
            d (dict): Old-format edge attribute dictionary.
            node_frequency (int): Per-node observed frequency for count recovery.

        Returns:
            EdgeData: Reconstructed edge data.
        """
        edge = cls()
        edge._weight = d.get('weight', 0.0)
        edge.count = d.get('count', 0)
        if edge.count == 0 and node_frequency > 0:
            edge.count = int(round(edge._weight * node_frequency))

        vsum = d.get('Vsum', 0)
        jsum = d.get('Jsum', 0)
        for key, val in d.items():
            if key in ('weight', 'count', 'Vsum', 'Jsum'):
                continue
            if _is_v_gene(key) and vsum > 0:
                edge.v_genes[key] = int(round(val * vsum))
            elif _is_j_gene(key) and jsum > 0:
                edge.j_genes[key] = int(round(val * jsum))
        return edge

    def __getstate__(self):
        return (self.count, self._weight, self.v_genes, self.j_genes)

    def __setstate__(self, state):
        self.count, self._weight, self.v_genes, self.j_genes = state

    def __eq__(self, other):
        if not isinstance(other, EdgeData):
            return NotImplemented
        return (self.count == other.count
                and self.v_genes == other.v_genes
                and self.j_genes == other.j_genes)

    def __repr__(self):
        return (f"EdgeData(count={self.count}, weight={self._weight:.4f}, "
                f"v={len(self.v_genes)}, j={len(self.j_genes)})")
