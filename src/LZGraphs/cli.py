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


_LOG_PRIORITIES = {
    'none': 0,
    'error': 1,
    'warn': 2,
    'info': 3,
    'debug': 4,
    'trace': 5,
}


def _effective_log_level(args, default='info'):
    level = getattr(args, 'log_level', None)
    if level:
        return level
    if getattr(args, 'quiet', False):
        return 'none'
    return default


def _cli_log_enabled(args, level='info'):
    current = _effective_log_level(args)
    return _LOG_PRIORITIES.get(current, 0) >= _LOG_PRIORITIES.get(level, 0)


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


def _add_input_contract_args(p):
    from ._io import VALID_FORMATS

    p.add_argument('--strict-input', action='store_true',
                   help='fail on mixed or malformed input records')
    p.add_argument('--expect-format', choices=sorted(VALID_FORMATS),
                   default=None,
                   help='require a specific input format')


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


def _emit_validation_report(args, report):
    if hasattr(args, 'json') and args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"# lzg validate-input v{__version__} — {os.path.basename(report['path'])}")
    _tagged('VL', 'ok', 'yes' if report['ok'] else 'no')
    _tagged('VL', 'detected_kind', report['detected_kind'])
    _tagged('VL', 'mode', report['mode'] or 'none')
    _tagged('VL', 'strict_input', 'yes' if report['strict_input'] else 'no')
    _tagged('VL', 'total_lines', report['total_lines'])
    _tagged('VL', 'records', report['records'])
    _tagged('VL', 'blank_lines', report['blank_lines'])
    _tagged('VL', 'plain_records', report['plain_records'])
    _tagged('VL', 'seqcount_records', report['seqcount_records'])
    _tagged('VL', 'tabular_rows', report['tabular_rows'])
    _tagged('VL', 'errors', report['error_count'])
    _tagged('VL', 'warnings', report['warning_count'])
    if report['expect_format']:
        _tagged('VL', 'expect_format', report['expect_format'])
    if report['seq_column']:
        _tagged('VL', 'seq_column', report['seq_column'])
    if report['abundance_column']:
        _tagged('VL', 'abundance_column', report['abundance_column'])
    if report['v_column']:
        _tagged('VL', 'v_column', report['v_column'])
    if report['j_column']:
        _tagged('VL', 'j_column', report['j_column'])
    for issue in report['issues']:
        line = issue.get('line')
        label = issue['level'].upper()
        msg = f"line={line} {issue['message']}" if line is not None else issue['message']
        _tagged(label, 'issue', msg)


def cmd_validate_input(args):
    from ._io import validate_input

    report = validate_input(
        args.input,
        seq_column=args.seq_column,
        v_column=args.v_column,
        j_column=args.j_column,
        abundance_column=args.abundance_column,
        variant=args.variant,
        no_genes=args.no_genes,
        strict_input=args.strict_input,
        expect_format=args.expect_format,
    )
    _emit_validation_report(args, report)
    if not report['ok']:
        sys.exit(2)


# ── Commands ────────────────────────────────────────────────


def cmd_build(args):
    from . import LZGraph, set_log_level
    from ._io import (
        FormatError,
        detect_format,
        raw_prefix_is_streamable,
        read_sequences,
        validate_input,
    )

    active_log_level = _effective_log_level(args, default='info')
    set_log_level(active_log_level)

    # The raw-streaming fast path (LZGraph.from_file) reads bytes straight
    # off disk with no decompression, so it is only safe when the file is
    # actually uncompressed plain/plain_seqcount content. Deciding that from
    # the filename (e.g. a ".gz" suffix) is not enough: a bzip2, xz, or
    # misnamed-gzip file has none of those suffixes yet is still compressed,
    # and streaming its raw bytes into the C builder silently produces a
    # garbage graph instead of failing loudly. detect_format inspects the
    # actual magic bytes, so it is the only reliable signal here. --expect-format
    # is passed through as override= here on purpose (this is the one place in
    # this function that still coerces rather than asserts): a user-declared
    # format decides whether the fast path is even attempted, so genuinely
    # ambiguous-but-declared content (see raw_prefix_is_streamable's docstring)
    # still gets a chance at it instead of being second-guessed by content
    # sniffing. This is safe because raw_prefix_is_streamable independently
    # verifies the coerced interpretation actually produces well-formed lines,
    # and because the validate_input call below re-checks expect_format as a
    # true assertion (comparing the real auto-detected format, uncoerced)
    # before anything is actually built, so a genuine mismatch is still caught
    # loudly rather than silently streamed. read_sequences, in the safe branch
    # further below, treats expect_format as an assertion too, not a coercion:
    # see _detect_format_for_read. A detection failure here (empty/binary
    # input) must not crash this fast-path decision; it just means "don't
    # stream", leaving the real error to surface from the safe branch below.
    #
    # detect_format's format/compression verdict is not the whole story: the
    # C reader also has none of read_sequences' per-line normalisation (no
    # utf-8-sig BOM stripping, no universal-newline translation, no
    # _is_wellformed filtering), so a file that is genuinely uncompressed
    # plain/plain_seqcount content can still make the fast path disagree with
    # the rest of the CLI (a BOM glued onto the first sequence, CR-only line
    # endings collapsing the whole file into one sequence, a header line or
    # other malformed record ingested as data). raw_prefix_is_streamable
    # inspects the file's real bytes for exactly those cases and declines the
    # fast path when any of them would matter, falling through to the safe
    # (and here, correct) non-streaming branch below instead.
    #
    # raw_prefix_is_streamable only checks the same bounded sniff prefix
    # detect_format already read (32 lines), not the whole file: a malformed
    # record beyond that prefix (say, line 10000 of a foundation-scale file)
    # still takes the fast path and is ingested rather than dropped. This is
    # a deliberate trade, not an oversight: catching it would mean reading
    # the entire file up front, which defeats the constant-memory streaming
    # this path exists to provide for exactly those large files. A user who
    # needs full per-record validation already has two ways to get it:
    # compressed input never reaches this gate at all (falls to
    # read_sequences below), and --strict-input runs validate_input over the
    # whole file before cmd_build streams anything.
    can_stream_plain = False
    if args.input != '-':
        try:
            spec = detect_format(
                args.input, variant=args.variant, override=args.expect_format
            )
        except FormatError:
            spec = None
        if spec is not None:
            can_stream_plain = (
                spec.compression == 'none'
                and spec.format in ('plain', 'plain_seqcount')
                and raw_prefix_is_streamable(args.input, spec.format)
            )

    if can_stream_plain and (args.strict_input or args.expect_format is not None):
        if _cli_log_enabled(args, 'info'):
            _stderr(f"[build] phase=validate-input status=start input={args.input}")
        t_validate = time.time()
        report = validate_input(
            args.input,
            variant=args.variant,
            strict_input=args.strict_input,
            expect_format=args.expect_format,
        )
        if _cli_log_enabled(args, 'info'):
            _stderr(
                f"[build] phase=validate-input status={'ok' if report['ok'] else 'error'} "
                f"kind={report['detected_kind']} mode={report['mode']} "
                f"records={report['records']} errors={report['error_count']} "
                f"warnings={report['warning_count']} elapsed={time.time()-t_validate:.2f}s"
            )
        if not report['ok']:
            raise ValueError(report['summary'])

    if can_stream_plain:
        t1 = time.time()
        g = LZGraph.from_file(
            args.input,
            variant=args.variant,
            smoothing=args.smoothing,
        )
        if _cli_log_enabled(args, 'info'):
            _stderr(f"[build] phase=construct mode=stream variant={args.variant} "
                    f"nodes={g.n_nodes} edges={g.n_edges} elapsed={time.time()-t1:.2f}s")
        if _cli_log_enabled(args, 'info'):
            _stderr(f"[build] phase=save status=start output={args.output}")
        t2 = time.time()
        g.save(args.output)
        sz = os.path.getsize(args.output)
        if _cli_log_enabled(args, 'info'):
            _stderr(f"[build] phase=save status=done output={args.output} "
                    f"size_kb={sz/1024:.1f} elapsed={time.time()-t2:.2f}s")
        set_log_level('none')
        return

    t0 = time.time()
    data = read_sequences(
        args.input, seq_column=args.seq_column,
        v_column=args.v_column, j_column=args.j_column,
        abundance_column=args.abundance_column,
        variant=args.variant, no_genes=args.no_genes,
        strict_input=args.strict_input,
        expect_format=args.expect_format,
    )
    n = len(data['sequences'])
    stats = data['stats']
    has_genes = data['v_genes'] is not None and data['j_genes'] is not None
    if _cli_log_enabled(args, 'info'):
        gene_info = ""
        if has_genes:
            nv = len(set(data['v_genes']))
            nj = len(set(data['j_genes']))
            gene_info = f" ({nv} V genes, {nj} J genes)"
        _stderr(f"[build] phase=read sequences={n}{gene_info} total={stats.total} "
                f"malformed={stats.malformed} nonproductive={stats.nonproductive} "
                f"elapsed={time.time()-t0:.2f}s")
    dropped = stats.malformed + stats.nonproductive
    if dropped and _cli_log_enabled(args, 'warn'):
        _stderr(f"[build] phase=read status=warn dropped={dropped} "
                f"malformed={stats.malformed} nonproductive={stats.nonproductive}")
    if n == 0:
        raise ValueError(
            f"{args.input}: read {stats.total} record(s), 0 kept "
            f"(malformed={stats.malformed}, nonproductive={stats.nonproductive}); "
            "nothing left to build a graph from"
        )

    t1 = time.time()
    g = LZGraph(
        data['sequences'],
        variant=args.variant,
        abundances=data['abundances'],
        v_genes=data['v_genes'],
        j_genes=data['j_genes'],
        smoothing=args.smoothing,
    )
    if _cli_log_enabled(args, 'info'):
        _stderr(f"[build] phase=construct mode=in_memory variant={args.variant} "
                f"nodes={g.n_nodes} edges={g.n_edges} elapsed={time.time()-t1:.2f}s")

    if _cli_log_enabled(args, 'info'):
        _stderr(f"[build] phase=save status=start output={args.output}")
    t2 = time.time()
    g.save(args.output)
    sz = os.path.getsize(args.output)
    if _cli_log_enabled(args, 'info'):
        _stderr(f"[build] phase=save status=done output={args.output} "
                f"size_kb={sz/1024:.1f} elapsed={time.time()-t2:.2f}s")

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

    lps = g.pgen(seqs)
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

    active_log_level = _effective_log_level(args, default='info')
    set_log_level(active_log_level)

    g = LZGraph.load(args.prior)
    data = read_sequences(args.new_data, seq_column=args.seq_column,
                          abundance_column=args.abundance_column,
                          variant=g.variant)

    if _cli_log_enabled(args, 'info'):
        _stderr(f"[posterior] phase=update sequences={len(data['sequences'])} kappa={args.kappa}")

    post = g.posterior(data['sequences'], abundances=data['abundances'],
                       kappa=args.kappa)
    if _cli_log_enabled(args, 'info'):
        _stderr(f"[posterior] phase=save status=start output={args.output}")
    post.save(args.output)

    if _cli_log_enabled(args, 'info'):
        _stderr(f"[posterior] phase=save status=done output={args.output}")
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


# ── FlashBack command group ─────────────────────────────────
#
# FlashBackGraph is a separate class from LZGraph and a saved .lzg file does
# not record which family produced it, so these commands always load via
# FlashBackGraph and live under their own `lzg flashback ...` namespace rather
# than auto-dispatching from the generic loaders.


def _get_fb_graph(path, quiet):
    """Load a FlashBackGraph, with timing unless quiet."""
    from . import FlashBackGraph
    if quiet:
        return FlashBackGraph.load(path)
    t0 = time.time()
    g = FlashBackGraph.load(path)
    _stderr(f"[loaded] {os.path.basename(path)}: {g.n_nodes} nodes, "
            f"{g.n_edges} edges ({time.time()-t0:.2f}s)")
    return g


def _fb_build(args):
    from . import FlashBackGraph, set_log_level

    set_log_level(_effective_log_level(args, default='info'))
    if args.input == '-':
        raw = sys.stdin.read().strip().split('\n')
        seqs, abundances = [], []
        for line in raw:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                s, c = line.split('\t', 1)
                seqs.append(s)
                abundances.append(int(c))
            else:
                seqs.append(line)
                abundances.append(1)
        g = FlashBackGraph(seqs, abundances=abundances, smoothing=args.smoothing)
    else:
        # from_file streams plain "seq" or "seq<TAB>abundance" with constant memory.
        g = FlashBackGraph.from_file(args.input, smoothing=args.smoothing)
    if _cli_log_enabled(args, 'info'):
        _stderr(f"[flashback build] phase=construct nodes={g.n_nodes} "
                f"edges={g.n_edges}")
    g.save(args.output)
    if _cli_log_enabled(args, 'info'):
        sz = os.path.getsize(args.output)
        _stderr(f"[flashback build] phase=save status=done output={args.output} "
                f"size_kb={sz/1024:.1f}")
    set_log_level('none')


def _fb_info(args):
    g = _get_fb_graph(args.graph, args.quiet)
    if getattr(args, 'json', False):
        d = dict(g.summary())
        d['variant'] = g.variant
        d['n_nodes'] = g.n_nodes
        d['n_edges'] = g.n_edges
        d['path_count'] = g.path_count
        d.update(g.diversity_profile())
        print(json.dumps(d, indent=2, default=_json_default))
        return
    print(f"# lzg flashback info v{__version__} — {os.path.basename(args.graph)}")
    _tagged('GR', 'variant', g.variant)
    _tagged('GR', 'nodes', g.n_nodes)
    _tagged('GR', 'edges', g.n_edges)
    _tagged('GR', 'is_dag', 'yes' if g.is_dag else 'no')
    _tagged('GR', 'path_count', f"{g.path_count:.6g}")
    dp = g.diversity_profile()
    _tagged('DV', 'effective_diversity', f"{dp['effective_diversity']:.4f}")


def _fb_score(args):
    from ._io import read_sequences_simple
    import numpy as np

    g = _get_fb_graph(args.graph, args.quiet)
    seqs = read_sequences_simple(args.input, seq_column=args.seq_column)
    if not seqs:
        _stderr("[flashback score] no sequences to score")
        return
    values = g.pgen(seqs)
    if args.prob:
        values = np.exp(values)
    col = 'prob' if args.prob else 'pgen'
    out = _out(args)
    values = np.atleast_1d(values)
    if getattr(args, 'json', False):
        result = [{'sequence': s, col: float(v)} for s, v in zip(seqs, values)]
        out.write(json.dumps(result, indent=2) + '\n')
    else:
        out.write(f"sequence\t{col}\n")
        for s, v in zip(seqs, values):
            out.write(f"{s}\t{float(v):.6f}\n")
    if out is not sys.stdout:
        out.close()
    if not args.quiet:
        _stderr(f"[flashback score] scored {len(seqs)} sequences")


def _fb_simulate(args):
    g = _get_fb_graph(args.graph, args.quiet)
    result = g.simulate(args.count, seed=args.seed)
    out = _out(args)
    if getattr(args, 'json', False):
        items = [{'sequence': result.sequences[i],
                  'log_prob': float(result.log_probs[i])}
                 for i in range(len(result))]
        out.write(json.dumps(items, indent=2) + '\n')
    elif args.with_details:
        out.write("sequence\tlog_prob\n")
        for i in range(len(result)):
            out.write(f"{result.sequences[i]}\t{result.log_probs[i]:.6f}\n")
    else:
        for s in result:
            out.write(s + '\n')
    if out is not sys.stdout:
        out.close()
    if not args.quiet:
        _stderr(f"[flashback simulate] generated {len(result)} sequences")


def _fb_diversity(args):
    import numpy as np

    g = _get_fb_graph(args.graph, args.quiet)
    orders = [float(s) for s in args.hill.split(',') if s.strip()]
    hills = {}
    if orders:
        vals = np.atleast_1d(g.hill_numbers(orders))
        hills = {o: float(v) for o, v in zip(orders, vals)}
    dp = g.diversity_profile()

    if getattr(args, 'json', False):
        d = {'hill_numbers': {str(k): v for k, v in hills.items()}}
        d.update(dp)
        print(json.dumps(d, indent=2, default=_json_default))
        return

    print(f"# lzg flashback diversity v{__version__}")
    for o in orders:
        _tagged('HL', f"{o:g}", f"{hills[o]:.4f}")
    _tagged('DV', 'effective_diversity', f"{dp['effective_diversity']:.4f}")
    if 'entropy_nats' in dp:
        _tagged('DV', 'entropy_nats', f"{dp['entropy_nats']:.4f}")


def _fb_scale(args):
    from . import ScaleCalibration
    from ._io import read_sequences_simple
    import numpy as np

    g = _get_fb_graph(args.graph, args.quiet)
    if args.calibration:
        cal = ScaleCalibration.load(args.calibration)
    else:
        if not args.quiet:
            _stderr(f"[flashback scale] calibrating "
                    f"(n_sim={args.n_sim}, seed={args.seed})...")
        cal = g.calibrate_scale(n_sim=args.n_sim, seed=args.seed)
        if args.save_calibration:
            cal.save(args.save_calibration)
            if not args.quiet:
                _stderr(f"[flashback scale] saved calibration to "
                        f"{args.save_calibration}")
    seqs = read_sequences_simple(args.input, seq_column=args.seq_column)
    if not seqs:
        _stderr("[flashback scale] no sequences to score")
        return
    scores = np.atleast_1d(g.scale_score(seqs, cal))
    out = _out(args)
    if getattr(args, 'json', False):
        items = [{'sequence': s, 'scale': float(v)}
                 for s, v in zip(seqs, scores)]
        out.write(json.dumps(items, indent=2) + '\n')
    else:
        out.write("sequence\tscale\n")
        for s, v in zip(seqs, scores):
            out.write(f"{s}\t{float(v):.6f}\n")
    if out is not sys.stdout:
        out.close()
    if not args.quiet:
        _stderr(f"[flashback scale] scored {len(seqs)} sequences")


def cmd_flashback(args):
    handler = {
        'build': _fb_build,
        'info': _fb_info,
        'score': _fb_score,
        'simulate': _fb_simulate,
        'diversity': _fb_diversity,
        'scale': _fb_scale,
    }.get(getattr(args, 'flashback_cmd', None))
    if handler is None:
        raise ValueError(
            "a flashback subcommand is required "
            "(build, info, score, simulate, diversity, scale)"
        )
    handler(args)


# ── Argument parser ─────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(
        prog='lzg',
        description='LZGraphs — LZ76 compression graphs for immune repertoire analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--version', action='version', version=f'lzg {__version__}')
    p.add_argument('-q', '--quiet', action='store_true', help='suppress progress')
    p.add_argument('--log-level', choices=['none', 'error', 'warn', 'info', 'debug', 'trace'],
                   default=None,
                   help="logging verbosity [default: info unless --quiet]")

    sub = p.add_subparsers(dest='command', title='commands')

    # ── build ──
    b = sub.add_parser('build', help='Build a graph from sequences')
    b.add_argument('input', help='Input file (.txt, .tsv, .csv, .gz, or - for stdin)')
    b.add_argument('-o', '--output', required=True, help='Output .lzg file')
    _add_variant_arg(b)
    _add_seq_column_arg(b)
    _add_input_contract_args(b)
    b.add_argument('--v-column', default='v_call', help='V gene column [default: v_call]')
    b.add_argument('--j-column', default='j_call', help='J gene column [default: j_call]')
    b.add_argument('-a', '--abundance-column', default=None, help='Abundance column')
    b.add_argument('--no-genes', action='store_true', help='Ignore gene columns')
    b.add_argument('--smoothing', type=float, default=0.0, help='Laplace smoothing [default: 0.0]')
    # min-initial-count removed (sentinel model)

    # ── validate-input ──
    vi = sub.add_parser('validate-input', help='Validate an input file without building')
    vi.add_argument('input', help='Input file (.txt, .tsv, .csv, .gz, or - for stdin)')
    _add_variant_arg(vi)
    _add_seq_column_arg(vi)
    _add_input_contract_args(vi)
    vi.add_argument('--v-column', default='v_call', help='V gene column [default: v_call]')
    vi.add_argument('--j-column', default='j_call', help='J gene column [default: j_call]')
    vi.add_argument('-a', '--abundance-column', default=None, help='Abundance column')
    vi.add_argument('--no-genes', action='store_true', help='Ignore gene columns')
    _add_json_arg(vi)

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

    # ── flashback ──
    fb = sub.add_parser('flashback',
                        help='FlashBackGraph commands (Markovian DAG, exact analytics)')
    fb_sub = fb.add_subparsers(dest='flashback_cmd', title='flashback commands')

    fb_b = fb_sub.add_parser('build', help='Build a FlashBackGraph from sequences')
    fb_b.add_argument('input', help='Input file (.txt, .tsv, .gz, or - for stdin)')
    fb_b.add_argument('-o', '--output', required=True, help='Output .lzg file')
    _add_seq_column_arg(fb_b)
    fb_b.add_argument('--smoothing', type=float, default=0.0,
                      help='Laplace smoothing [default: 0.0]')

    fb_i = fb_sub.add_parser('info', help='Inspect a saved FlashBackGraph')
    fb_i.add_argument('graph', help='.lzg file')
    _add_json_arg(fb_i)

    fb_sc = fb_sub.add_parser('score', help='Exact generation probability for sequences')
    fb_sc.add_argument('graph', help='.lzg file')
    fb_sc.add_argument('input', nargs='?', default='-', help='Sequence file [default: stdin]')
    _add_seq_column_arg(fb_sc)
    _add_output_arg(fb_sc)
    fb_sc.add_argument('--prob', action='store_true', help='Output probability (not log)')
    _add_json_arg(fb_sc)

    fb_sm = fb_sub.add_parser('simulate', help='Generate sequences from a FlashBackGraph')
    fb_sm.add_argument('graph', help='.lzg file')
    fb_sm.add_argument('-n', '--count', type=int, required=True, help='Number of sequences')
    _add_output_arg(fb_sm)
    fb_sm.add_argument('--seed', type=int, default=None, help='RNG seed')
    fb_sm.add_argument('--with-details', action='store_true', help='Include log_prob')
    _add_json_arg(fb_sm)

    fb_dv = fb_sub.add_parser('diversity', help='Exact diversity metrics')
    fb_dv.add_argument('graph', help='.lzg file')
    fb_dv.add_argument('--hill', default='0,1,2', help='Hill orders [default: 0,1,2]')
    _add_json_arg(fb_dv)

    fb_scale = fb_sub.add_parser(
        'scale', help='SCALE anomaly score (self-calibrated -log Pgen)')
    fb_scale.add_argument('graph', help='.lzg file')
    fb_scale.add_argument('input', nargs='?', default='-',
                          help='Sequence file [default: stdin]')
    _add_seq_column_arg(fb_scale)
    _add_output_arg(fb_scale)
    fb_scale.add_argument('--calibration', default=None,
                          help='Load a saved calibration JSON instead of calibrating')
    fb_scale.add_argument('--save-calibration', default=None,
                          help='Save the freshly built calibration to this JSON path')
    fb_scale.add_argument('--n-sim', type=int, default=200_000,
                          help='Sequences to simulate for calibration [default: 200000]')
    fb_scale.add_argument('--seed', type=int, default=None,
                          help='Calibration RNG seed')
    _add_json_arg(fb_scale)

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
    'validate-input': cmd_validate_input,
    'info': cmd_info,
    'score': cmd_score,
    'simulate': cmd_simulate,
    'diversity': cmd_diversity,
    'compare': cmd_compare,
    'decompose': cmd_decompose,
    'saturation': cmd_saturation,
    'predict': cmd_predict,
    'posterior': cmd_posterior,
    'flashback': cmd_flashback,
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
