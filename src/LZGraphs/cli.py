"""LZGraphs command-line interface.

Usage: lzg <command> [options] [files...]
"""

import argparse
import json
import os
import sys
import time

from . import __version__


def _stderr(msg):
    print(msg, file=sys.stderr)


def _tagged(prefix, key, value):
    """Print a prefix-tagged line (samtools-style)."""
    print(f"{prefix}\t{key}\t{value}")


# ── Input helpers ───────────────────────────────────────────


def _add_seq_column_arg(p):
    p.add_argument('-s', '--seq-column', default=None,
                   help='Sequence column name [default: auto-detect]')


def _add_variant_arg(p):
    p.add_argument('-V', '--variant', default='aap', choices=['aap', 'ndp', 'naive'],
                   help='Graph variant [default: aap]')


def _add_output_arg(p, required=False):
    p.add_argument('-o', '--output', default=None, required=required,
                   help='Output file [default: stdout]')


def _add_json_arg(p):
    p.add_argument('--json', action='store_true', help='JSON output')


def _get_graph(path):
    """Load a graph, with timing."""
    from . import LZGraph
    t0 = time.time()
    g = LZGraph.load(path)
    _stderr(f"[loaded] {os.path.basename(path)}: {g.n_nodes} nodes, "
            f"{g.n_edges} edges ({time.time()-t0:.2f}s)")
    return g


def _out(args):
    """Get output file handle."""
    if args.output:
        return open(args.output, 'w')
    return sys.stdout


# ── Commands ────────────────────────────────────────────────


def cmd_build(args):
    from . import LZGraph, set_log_level
    from ._io import read_sequences

    if not args.quiet:
        set_log_level('info')

    t0 = time.time()
    data = read_sequences(
        args.input, seq_column=args.seq_column,
        v_column=args.v_column, j_column=args.j_column,
        abundance_column=args.abundance_column,
        variant=args.variant, no_genes=args.no_genes,
    )
    n = len(data['sequences'])
    has_genes = data['v_genes'] is not None and data['j_genes'] is not None
    if not args.quiet:
        gene_info = ""
        if has_genes:
            nv = len(set(data['v_genes']))
            nj = len(set(data['j_genes']))
            gene_info = f" ({nv} V genes, {nj} J genes)"
        _stderr(f"[build] {n} sequences read{gene_info} ({time.time()-t0:.2f}s)")

    t1 = time.time()
    g = LZGraph(
        data['sequences'],
        variant=args.variant,
        abundances=data['abundances'],
        v_genes=data['v_genes'],
        j_genes=data['j_genes'],
        smoothing=args.smoothing,
    )
    if not args.quiet:
        _stderr(f"[build] {g.n_nodes} nodes, {g.n_edges} edges ({time.time()-t1:.2f}s)")

    g.save(args.output)
    sz = os.path.getsize(args.output)
    if not args.quiet:
        _stderr(f"[build] saved {args.output} ({sz/1024:.1f} KB)")

    set_log_level('none')


def cmd_info(args):
    from . import LZGraph
    g = _get_graph(args.graph) if not args.quiet else LZGraph.load(args.graph)

    if hasattr(args, 'json') and args.json:
        d = g.summary()
        d['variant'] = g.variant
        d['has_gene_data'] = g.has_gene_data
        d['path_count'] = g.path_count
        d.update(g.diversity_profile())
        d.update(g.pgen_moments())
        print(json.dumps(d, indent=2))
        return

    print(f"# lzg info v{__version__} — {os.path.basename(args.graph)}")
    s = g.summary()
    _tagged('GR', 'variant', g.variant)
    _tagged('GR', 'nodes', s['n_nodes'])
    _tagged('GR', 'edges', s['n_edges'])
    _tagged('GR', 'initial_states', s['n_initial'])
    _tagged('GR', 'terminal_states', s['n_terminal'])
    _tagged('GR', 'is_dag', 'yes' if s['is_dag'] else 'no')
    _tagged('GR', 'has_gene_data', 'yes' if g.has_gene_data else 'no')
    _tagged('GR', 'path_count', f"{g.path_count:.6g}")

    dp = g.diversity_profile()
    _tagged('DV', 'effective_diversity', f"{dp['effective_diversity']:.4f}")
    _tagged('DV', 'entropy_nats', f"{dp['entropy_nats']:.4f}")
    _tagged('DV', 'entropy_bits', f"{dp['entropy_bits']:.4f}")
    _tagged('DV', 'uniformity', f"{dp['uniformity']:.4f}")

    m = g.pgen_moments()
    _tagged('PR', 'pgen_mean', f"{m['mean']:.4f}")
    _tagged('PR', 'pgen_std', f"{m['std']:.4f}")
    _tagged('PR', 'dynamic_range_decades', f"{g.pgen_dynamic_range():.4f}")

    diag = g.pgen_diagnostics()
    _tagged('PR', 'is_proper', 'yes' if diag['is_proper'] else 'no')

    if args.genes or args.all:
        if g.has_gene_data:
            for name, prob in sorted(g.v_marginals.items(), key=lambda x: -x[1]):
                _tagged('VG', name, f"{prob:.6f}")
            for name, prob in sorted(g.j_marginals.items(), key=lambda x: -x[1]):
                _tagged('JG', name, f"{prob:.6f}")
        else:
            _stderr("[info] no gene data in this graph")

    if args.all:
        hc = g.hill_curve()
        for o, h in zip(hc['orders'], hc['values']):
            _tagged('HL', f"{o:.2f}", f"{h:.4f}")


def cmd_score(args):
    from . import LZGraph
    from ._io import read_sequences_simple
    import numpy as np

    g = _get_graph(args.graph) if not args.quiet else LZGraph.load(args.graph)
    seqs = read_sequences_simple(args.input, seq_column=args.seq_column,
                                  variant=g.variant)
    if not seqs:
        _stderr("[score] no sequences to score")
        return

    lps = g.lzpgen(seqs)
    if not args.prob:
        values = lps
    else:
        values = np.exp(lps)

    out = _out(args)
    col_name = 'lzpgen' if not args.prob else 'pgen'

    if hasattr(args, 'json') and args.json:
        result = [{'sequence': s, col_name: float(v)}
                  for s, v in zip(seqs, values)]
        out.write(json.dumps(result, indent=2) + '\n')
    else:
        out.write(f"sequence\t{col_name}\n")
        for s, v in zip(seqs, values):
            out.write(f"{s}\t{v:.6f}\n")

    if out is not sys.stdout:
        out.close()
    if not args.quiet:
        _stderr(f"[score] scored {len(seqs)} sequences")


def cmd_simulate(args):
    from . import LZGraph

    g = _get_graph(args.graph) if not args.quiet else LZGraph.load(args.graph)

    kwargs = dict(n=args.count, seed=args.seed)
    if args.v_gene:
        kwargs['v_gene'] = args.v_gene
    if args.j_gene:
        kwargs['j_gene'] = args.j_gene
    if args.sample_genes:
        kwargs['sample_genes'] = True

    result = g.simulate(**kwargs)
    out = _out(args)

    has_genes = result.v_genes is not None
    detailed = args.with_details or has_genes

    if hasattr(args, 'json') and args.json:
        items = []
        for i in range(len(result)):
            d = {'sequence': result.sequences[i]}
            if detailed:
                d['lzpgen'] = float(result.log_probs[i])
                d['n_tokens'] = int(result.n_tokens[i])
            if has_genes:
                d['v_gene'] = result.v_genes[i]
                d['j_gene'] = result.j_genes[i]
            items.append(d)
        out.write(json.dumps(items, indent=2) + '\n')
    elif detailed:
        cols = ['sequence', 'lzpgen', 'n_tokens']
        if has_genes:
            cols += ['v_gene', 'j_gene']
        out.write('\t'.join(cols) + '\n')
        for i in range(len(result)):
            row = [result.sequences[i], f"{result.log_probs[i]:.6f}",
                   str(int(result.n_tokens[i]))]
            if has_genes:
                row += [result.v_genes[i], result.j_genes[i]]
            out.write('\t'.join(row) + '\n')
    else:
        for s in result:
            out.write(s + '\n')

    if out is not sys.stdout:
        out.close()
    if not args.quiet:
        _stderr(f"[simulate] generated {len(result)} sequences")


def cmd_diversity(args):
    from . import LZGraph

    g = _get_graph(args.graph) if not args.quiet else LZGraph.load(args.graph)

    # Parse hill orders
    orders = []
    for s in args.hill.split(','):
        s = s.strip()
        if s == 'inf':
            orders.append(float('inf'))
        else:
            orders.append(float(s))

    # Compute Hill numbers (skip inf, handle separately)
    finite_orders = [o for o in orders if o != float('inf')]
    hills = {}
    if finite_orders:
        import numpy as np
        vals = g.hill_numbers(finite_orders)
        for o, v in zip(finite_orders, vals):
            hills[o] = v

    # D(inf) = 1/max(pi) — use dynamic range
    if float('inf') in orders:
        dr = g.pgen_dynamic_range_detail()
        import math
        hills[float('inf')] = math.exp(-dr['max_log_prob'])

    dp = g.diversity_profile()
    m = g.pgen_moments()

    if hasattr(args, 'json') and args.json:
        d = {'hill_numbers': {str(k): v for k, v in hills.items()}}
        d.update(dp)
        d['pgen_mean'] = m['mean']
        d['pgen_std'] = m['std']
        d['dynamic_range_decades'] = g.pgen_dynamic_range()
        print(json.dumps(d, indent=2))
        return

    print(f"# lzg diversity v{__version__}")
    for o in orders:
        label = 'inf' if o == float('inf') else f"{o:g}"
        _tagged('HL', label, f"{hills[o]:.4f}")
    _tagged('DV', 'effective_diversity', f"{dp['effective_diversity']:.4f}")
    _tagged('DV', 'entropy_nats', f"{dp['entropy_nats']:.4f}")
    _tagged('DV', 'entropy_bits', f"{dp['entropy_bits']:.4f}")
    _tagged('DV', 'uniformity', f"{dp['uniformity']:.4f}")
    _tagged('DR', 'dynamic_range_decades', f"{g.pgen_dynamic_range():.4f}")
    _tagged('DR', 'pgen_mean', f"{m['mean']:.4f}")
    _tagged('DR', 'pgen_std', f"{m['std']:.4f}")


def cmd_compare(args):
    from . import LZGraph, jensen_shannon_divergence

    a = _get_graph(args.graph_a) if not args.quiet else LZGraph.load(args.graph_a)
    b = _get_graph(args.graph_b) if not args.quiet else LZGraph.load(args.graph_b)

    jsd = jensen_shannon_divergence(a, b)
    sa = a.summary()
    sb = b.summary()

    # Structural overlap via intersection
    inter = a & b
    si = inter.summary()

    if hasattr(args, 'json') and args.json:
        print(json.dumps({
            'jsd': jsd, 'nodes_a': sa['n_nodes'], 'nodes_b': sb['n_nodes'],
            'nodes_shared': si['n_nodes'], 'edges_a': sa['n_edges'],
            'edges_b': sb['n_edges'], 'edges_shared': si['n_edges'],
        }, indent=2))
        return

    na, nb = os.path.basename(args.graph_a), os.path.basename(args.graph_b)
    print(f"# lzg compare v{__version__} — {na} vs {nb}")
    _tagged('CP', 'jsd', f"{jsd:.6f}")
    _tagged('CP', 'nodes_a', sa['n_nodes'])
    _tagged('CP', 'nodes_b', sb['n_nodes'])
    _tagged('CP', 'nodes_shared', si['n_nodes'])
    _tagged('CP', 'edges_a', sa['n_edges'])
    _tagged('CP', 'edges_b', sb['n_edges'])
    _tagged('CP', 'edges_shared', si['n_edges'])

    total_n = sa['n_nodes'] + sb['n_nodes'] - si['n_nodes']
    total_e = sa['n_edges'] + sb['n_edges'] - si['n_edges']
    _tagged('CP', 'jaccard_nodes', f"{si['n_nodes']/max(total_n,1):.4f}")
    _tagged('CP', 'jaccard_edges', f"{si['n_edges']/max(total_e,1):.4f}")


def cmd_decompose(args):
    from . import lz76_decompose
    from ._io import read_sequences_simple

    seqs = read_sequences_simple(args.input or '-', seq_column=args.seq_column)
    out = _out(args)

    if hasattr(args, 'json') and args.json:
        result = []
        for s in seqs:
            tokens = lz76_decompose(s)
            result.append({'sequence': s, 'tokens': tokens, 'n_tokens': len(tokens)})
        out.write(json.dumps(result, indent=2) + '\n')
    else:
        d = args.delimiter
        out.write("sequence\ttokens\tn_tokens\n")
        for s in seqs:
            tokens = lz76_decompose(s)
            out.write(f"{s}\t{d.join(tokens)}\t{len(tokens)}\n")

    if out is not sys.stdout:
        out.close()


def cmd_saturation(args):
    from . import saturation_curve

    from ._io import read_sequences_simple
    seqs = read_sequences_simple(args.input, seq_column=args.seq_column,
                                  variant=args.variant)
    if not args.quiet:
        _stderr(f"[saturation] {len(seqs)} sequences, variant={args.variant}")

    points = saturation_curve(seqs, variant=args.variant, log_every=args.log_every)
    out = _out(args)

    if hasattr(args, 'json') and args.json:
        out.write(json.dumps(points, indent=2) + '\n')
    else:
        out.write("n_sequences\tn_nodes\tn_edges\n")
        for p in points:
            out.write(f"{p['n_sequences']}\t{p['n_nodes']}\t{p['n_edges']}\n")

    if out is not sys.stdout:
        out.close()


def cmd_predict(args):
    from . import LZGraph

    g = _get_graph(args.graph) if not args.quiet else LZGraph.load(args.graph)
    out = _out(args)

    if args.predict_cmd == 'richness':
        depths = _parse_depths(args.depths)
        import numpy as np
        values = g.richness_curve(depths)

        if hasattr(args, 'json') and args.json:
            out.write(json.dumps([{'depth': d, 'predicted_richness': float(v)}
                                   for d, v in zip(depths, values)], indent=2) + '\n')
        else:
            out.write("depth\tpredicted_richness\n")
            for d, v in zip(depths, values):
                out.write(f"{d:g}\t{v:.4f}\n")

    elif args.predict_cmd == 'overlap':
        v = g.predicted_overlap(args.di, args.dj)
        if hasattr(args, 'json') and args.json:
            out.write(json.dumps({'d_i': args.di, 'd_j': args.dj,
                                   'predicted_overlap': v}, indent=2) + '\n')
        else:
            _tagged('PO', 'd_i', f"{args.di:g}")
            _tagged('PO', 'd_j', f"{args.dj:g}")
            _tagged('PO', 'predicted_overlap', f"{v:.4f}")

    elif args.predict_cmd == 'sharing':
        draws = [float(x) for x in args.draws.split(',')]
        max_k = args.max_k or len(draws)
        result = g.predict_sharing(draws, max_k=max_k)

        if hasattr(args, 'json') and args.json:
            out.write(json.dumps(result, indent=2, default=_json_default) + '\n')
        else:
            import numpy as np
            out.write("k\texpected_count\n")
            for k, v in enumerate(result['spectrum']):
                if k == 0:
                    continue
                out.write(f"{k}\t{v:.6f}\n")

    if out is not sys.stdout:
        out.close()


def cmd_posterior(args):
    from . import LZGraph, set_log_level
    from ._io import read_sequences

    if not args.quiet:
        set_log_level('info')

    g = LZGraph.load(args.prior)
    data = read_sequences(args.new_data, seq_column=args.seq_column,
                          abundance_column=args.abundance_column,
                          variant=g.variant)

    if not args.quiet:
        _stderr(f"[posterior] {len(data['sequences'])} new sequences, kappa={args.kappa}")

    post = g.posterior(data['sequences'], abundances=data['abundances'],
                       kappa=args.kappa)
    post.save(args.output)

    if not args.quiet:
        _stderr(f"[posterior] saved {args.output}")
    set_log_level('none')


def _parse_depths(s):
    """Parse depths: comma-separated or START:END:N (log-spaced)."""
    import numpy as np
    if ':' in s:
        parts = s.split(':')
        start, end, n = float(parts[0]), float(parts[1]), int(parts[2])
        return np.logspace(np.log10(start), np.log10(end), n).tolist()
    return [float(x) for x in s.split(',')]


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


# ── Argument parser ─────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(
        prog='lzg',
        description='LZGraphs — LZ76 compression graphs for immune repertoire analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--version', action='version', version=f'lzg {__version__}')
    p.add_argument('-q', '--quiet', action='store_true', help='suppress progress')

    sub = p.add_subparsers(dest='command', title='commands')

    # ── build ──
    b = sub.add_parser('build', help='Build a graph from sequences')
    b.add_argument('input', help='Input file (.txt, .tsv, .csv, .gz, or - for stdin)')
    b.add_argument('-o', '--output', required=True, help='Output .lzg file')
    _add_variant_arg(b)
    _add_seq_column_arg(b)
    b.add_argument('--v-column', default='v_call', help='V gene column [default: v_call]')
    b.add_argument('--j-column', default='j_call', help='J gene column [default: j_call]')
    b.add_argument('-a', '--abundance-column', default=None, help='Abundance column')
    b.add_argument('--no-genes', action='store_true', help='Ignore gene columns')
    b.add_argument('--smoothing', type=float, default=0.0, help='Laplace smoothing [default: 0.0]')
    # min-initial-count removed (sentinel model)

    # ── info ──
    i = sub.add_parser('info', help='Inspect a saved graph')
    i.add_argument('graph', help='.lzg file')
    i.add_argument('--genes', action='store_true', help='Show V/J gene marginals')
    i.add_argument('--all', action='store_true', help='Show everything')
    _add_json_arg(i)

    # ── score ──
    sc = sub.add_parser('score', help='Compute LZPGEN for sequences')
    sc.add_argument('graph', help='.lzg graph file')
    sc.add_argument('input', nargs='?', default='-', help='Sequence file [default: stdin]')
    _add_seq_column_arg(sc)
    _add_output_arg(sc)
    sc.add_argument('--prob', action='store_true', help='Output probability (not log)')
    sc.add_argument('--append', action='store_true', help='Pass through input columns')
    _add_json_arg(sc)

    # ── simulate ──
    sm = sub.add_parser('simulate', help='Generate sequences from a graph')
    sm.add_argument('graph', help='.lzg graph file')
    sm.add_argument('-n', '--count', type=int, required=True, help='Number of sequences')
    _add_output_arg(sm)
    sm.add_argument('--seed', type=int, default=None, help='RNG seed')
    sm.add_argument('--v-gene', default=None, help='Constrain V gene')
    sm.add_argument('--j-gene', default=None, help='Constrain J gene')
    sm.add_argument('--sample-genes', action='store_true', help='Sample VJ from joint dist')
    sm.add_argument('--with-details', action='store_true', help='Include lzpgen, n_tokens')
    _add_json_arg(sm)

    # ── diversity ──
    dv = sub.add_parser('diversity', help='Diversity metrics')
    dv.add_argument('graph', help='.lzg graph file')
    dv.add_argument('--hill', default='0,1,2,5,inf', help='Hill orders [default: 0,1,2,5,inf]')
    _add_json_arg(dv)

    # ── compare ──
    cp = sub.add_parser('compare', help='Compare two repertoires')
    cp.add_argument('graph_a', help='First .lzg file')
    cp.add_argument('graph_b', help='Second .lzg file')
    _add_json_arg(cp)

    # ── decompose ──
    dc = sub.add_parser('decompose', help='LZ76-decompose sequences')
    dc.add_argument('input', nargs='?', default='-', help='Sequence file [default: stdin]')
    _add_seq_column_arg(dc)
    _add_output_arg(dc)
    dc.add_argument('-d', '--delimiter', default='|', help='Token delimiter [default: |]')
    _add_json_arg(dc)

    # ── saturation ──
    st = sub.add_parser('saturation', help='Node/edge saturation curve')
    st.add_argument('input', help='Sequence file')
    _add_variant_arg(st)
    _add_seq_column_arg(st)
    _add_output_arg(st)
    st.add_argument('--log-every', type=int, default=100, help='Record every N seqs [default: 100]')
    _add_json_arg(st)

    # ── predict ──
    pr = sub.add_parser('predict', help='Occupancy predictions')
    pr_sub = pr.add_subparsers(dest='predict_cmd', title='prediction type')

    pr_r = pr_sub.add_parser('richness', help='Predicted richness at given depths')
    pr_r.add_argument('graph', help='.lzg graph file')
    pr_r.add_argument('--depths', required=True,
                      help='Depths: comma-separated or START:END:N (log-spaced)')
    _add_output_arg(pr_r)
    _add_json_arg(pr_r)

    pr_o = pr_sub.add_parser('overlap', help='Predicted overlap between two samples')
    pr_o.add_argument('graph', help='.lzg graph file')
    pr_o.add_argument('--di', type=float, required=True, help='Depth of sample i')
    pr_o.add_argument('--dj', type=float, required=True, help='Depth of sample j')
    _add_output_arg(pr_o)
    _add_json_arg(pr_o)

    pr_s = pr_sub.add_parser('sharing', help='Predicted sharing spectrum')
    pr_s.add_argument('graph', help='.lzg graph file')
    pr_s.add_argument('--draws', required=True, help='Draws per donor: D1,D2,...')
    pr_s.add_argument('--max-k', type=int, default=None, help='Max sharing degree')
    _add_output_arg(pr_s)
    _add_json_arg(pr_s)

    # ── posterior ──
    po = sub.add_parser('posterior', help='Bayesian posterior update')
    po.add_argument('prior', help='Prior .lzg graph file')
    po.add_argument('new_data', help='New observation file')
    po.add_argument('-o', '--output', required=True, help='Output .lzg file')
    _add_seq_column_arg(po)
    po.add_argument('-a', '--abundance-column', default=None, help='Abundance column')
    po.add_argument('--kappa', type=float, default=1.0, help='Prior strength [default: 1.0]')

    return p


# ── Main ────────────────────────────────────────────────────

_DISPATCH = {
    'build': cmd_build,
    'info': cmd_info,
    'score': cmd_score,
    'simulate': cmd_simulate,
    'diversity': cmd_diversity,
    'compare': cmd_compare,
    'decompose': cmd_decompose,
    'saturation': cmd_saturation,
    'predict': cmd_predict,
    'posterior': cmd_posterior,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    fn = _DISPATCH.get(args.command)
    if fn:
        try:
            fn(args)
        except KeyboardInterrupt:
            _stderr("\ninterrupted")
            sys.exit(130)
        except Exception as e:
            _stderr(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
