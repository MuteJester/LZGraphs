from ..utilities.decomposition import lempel_ziv_decomposition
from ..exceptions import NoGeneDataError, GeneAnnotationError


class WalkAnalysisMixin:
    """Mixin providing per-walk gene usage analysis and variation curves.

    Requirements:
        - self.graph (networkx.DiGraph)
        - self.has_gene_data (bool)
        - self.marginal_v_genes (dict)
        - self.marginal_j_genes (dict)
        - self.encode_sequence(seq) — abstract method from base
    """

    def sequence_variation_curve(self, cdr3_sample):
        """
        Given a sequence, return the encoded subpatterns and the out-degree
        (number of possible transitions) at each position.

        Args:
            cdr3_sample (str): A sequence to analyze.

        Returns:
            tuple: (encoded_subpatterns, out_degrees) where both are lists.
        """
        encoded = self.encode_sequence(cdr3_sample)
        curve = [self.graph.out_degree(node) for node in encoded]
        return encoded, curve

    def walk_genes(self, walk, dropna=True, raise_error=True):
        """
        Given a walk (list of nodes), return gene usage at each edge.

        Args:
            walk (list): The node path.
            dropna (bool): If True, drop genes that are absent from all edges.
            raise_error (bool): If True and result is empty, raise an error.

        Returns:
            dict: Outer keys are gene names, inner keys are edge labels
                (``"nodeA->nodeB"``).  Each gene entry also has ``'type'``
                (``"V"``/``"J"``/``"Unknown"``) and ``'sum'`` keys.
        """
        # Collect {edge_label: {gene: prob}} for each edge in the walk
        edge_genes = {}
        for i in range(len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                edge_attrs = self.graph[walk[i]][walk[i + 1]]['data'].gene_dict()
                edge_genes[f"{walk[i]}->{walk[i + 1]}"] = edge_attrs

        if not edge_genes:
            if raise_error:
                raise GeneAnnotationError("No gene data found in the edges for the given walk.")
            return {}

        # Pivot: {gene_name: {edge_label: prob, ...}}
        all_genes = set()
        for attrs in edge_genes.values():
            all_genes.update(attrs.keys())

        result = {}
        for gene in sorted(all_genes):
            row = {}
            all_none = True
            for edge_label, attrs in edge_genes.items():
                val = attrs.get(gene)
                row[edge_label] = val
                if val is not None:
                    all_none = False
            if dropna and all_none:
                continue
            gene_type = "V" if "v" in gene.lower() else ("J" if "j" in gene.lower() else "Unknown")
            row['type'] = gene_type
            row['sum'] = sum(v for v in row.values() if isinstance(v, (int, float)) and v is not None)
            result[gene] = row

        if not result and raise_error:
            raise GeneAnnotationError("No gene data found in the edges for the given walk.")

        return result

    def path_gene_table(self, cdr3_sample, threshold=None):
        """
        Return two tables (V genes, J genes) representing which genes could
        generate the given sequence. Genes missing from more than *threshold*
        edges are dropped.

        Args:
            cdr3_sample (str): The sequence to examine.
            threshold (float, optional): NaN threshold. Defaults to length/4
                for V genes and length/2 for J genes.

        Returns:
            tuple: ``(vgene_dict, jgene_dict)`` — each is a dict of dicts
                keyed by gene name, sorted by ascending NA count.
        """
        encoded = self.encode_sequence(cdr3_sample)
        length = len(encoded)

        if threshold is None:
            threshold_v = length * 0.25
            threshold_j = length * 0.5
        else:
            threshold_v = threshold
            threshold_j = threshold

        gene_table = self.walk_genes(encoded, dropna=False, raise_error=False)

        # Count NAs per gene (edge values that are None)
        edge_keys = [k for k in next(iter(gene_table.values()), {}).keys()
                     if k not in ('type', 'sum')] if gene_table else []

        def na_count(row):
            return sum(1 for k in edge_keys if row.get(k) is None)

        vgene_table = {}
        jgene_table = {}
        for gene, row in gene_table.items():
            nc = na_count(row)
            gene_lower = gene.lower()
            if 'v' in gene_lower and nc < threshold_v:
                vgene_table[gene] = row
            elif 'j' in gene_lower and nc < threshold_j:
                jgene_table[gene] = row

        # Sort by ascending NA count
        vgene_table = dict(sorted(vgene_table.items(), key=lambda kv: na_count(kv[1])))
        jgene_table = dict(sorted(jgene_table.items(), key=lambda kv: na_count(kv[1])))

        return vgene_table, jgene_table

    def gene_variation(self, cdr3):
        """
        Return a list showing how many V and J genes are possible at
        each subpattern position in the given sequence.

        Args:
            cdr3 (str): The sequence to analyze.

        Returns:
            list[dict]: Each dict has keys 'genes', 'type', and 'sp'.

        Raises:
            NoGeneDataError: If the graph has no gene data.
        """
        if not self.has_gene_data:
            raise NoGeneDataError(
                operation="gene_variation",
                message="Cannot compute gene variation: this LZGraph has no gene data (genetic=False)."
            )

        encoded = self.encode_sequence(cdr3)

        n_v = [len(self.marginal_v_genes)]
        n_j = [len(self.marginal_j_genes)]

        for node in encoded[1:]:
            in_edges = self.graph.in_edges(node)
            v_candidates = set()
            j_candidates = set()
            for ea, eb in in_edges:
                ed = self.graph[ea][eb]['data']
                v_candidates |= set(ed.v_genes.keys())
                j_candidates |= set(ed.j_genes.keys())

            n_v.append(len(v_candidates))
            n_j.append(len(j_candidates))

        lz_subpatterns = lempel_ziv_decomposition(cdr3)
        genes = n_v + n_j
        types = (["V"] * len(n_v)) + (["J"] * len(n_j))
        sps = lz_subpatterns + lz_subpatterns
        return [
            {"genes": g, "type": t, "sp": s}
            for g, t, s in zip(genes, types, sps)
        ]
