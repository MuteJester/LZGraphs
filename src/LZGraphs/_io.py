"""Input file reading utilities for CLI and programmatic use."""

import csv
import gzip
import sys


# Variant-aware sequence column auto-detection (case-insensitive, first match)
_SEQ_COLUMNS = {
    'aap':   ['junction_aa', 'cdr3_amino_acid', 'cdr3_aa', 'aminoacid'],
    'ndp':   ['junction', 'cdr3_rearrangement', 'cdr3_nt', 'nucleotide'],
    'naive': ['junction_aa', 'cdr3_amino_acid', 'junction', 'cdr3_rearrangement'],
}
_SEQ_FALLBACK = ['sequence', 'cdr3', 'seq']


def _detect_seq_column(headers, variant='aap'):
    """Find the sequence column from headers, variant-aware."""
    lower = {h.lower(): h for h in headers}
    candidates = _SEQ_COLUMNS.get(variant, []) + _SEQ_FALLBACK
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _open_file(path):
    """Open a file, handling gzip and stdin."""
    if path == '-':
        return sys.stdin
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def read_sequences(path, seq_column=None, v_column=None, j_column=None,
                   abundance_column=None, variant='aap', no_genes=False):
    """Read sequences from a file, auto-detecting format.

    Returns dict with keys:
        'sequences': list[str]
        'abundances': list[int] | None
        'v_genes': list[str] | None
        'j_genes': list[str] | None
    """
    fh = _open_file(path)
    try:
        first_line = fh.readline()
        if not first_line.strip():
            raise ValueError(f"empty input: {path}")

        # Detect format: does first line look like a header?
        has_tab = '\t' in first_line
        has_comma = ',' in first_line and not has_tab

        if has_tab or has_comma:
            return _read_tabular(fh, first_line, has_tab,
                                 seq_column, v_column, j_column,
                                 abundance_column, variant, no_genes)
        else:
            return _read_plain(fh, first_line)
    finally:
        if fh is not sys.stdin:
            fh.close()


def _read_plain(fh, first_line):
    """Read plain text: one sequence per line, or sequence\\tabundance."""
    sequences = []
    abundances = None

    # Check if first line is seq\tabundance
    parts = first_line.strip().split('\t')
    if len(parts) == 2 and parts[1].isdigit():
        # seq\tabundance format
        abundances = []
        sequences.append(parts[0])
        abundances.append(int(parts[1]))
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split('\t')
            sequences.append(p[0])
            abundances.append(int(p[1]) if len(p) > 1 else 1)
    else:
        # Pure one-per-line
        seq = first_line.strip()
        if seq:
            sequences.append(seq)
        for line in fh:
            line = line.strip()
            if line:
                sequences.append(line)

    return {
        'sequences': sequences,
        'abundances': abundances,
        'v_genes': None,
        'j_genes': None,
    }


def _read_tabular(fh, header_line, is_tsv, seq_column, v_column, j_column,
                   abundance_column, variant, no_genes):
    """Read TSV/CSV with header."""
    delimiter = '\t' if is_tsv else ','
    headers = [h.strip() for h in header_line.strip().split(delimiter)]

    # Resolve sequence column
    if seq_column:
        if seq_column not in headers:
            raise ValueError(
                f"column '{seq_column}' not found in headers: {headers}")
        scol = seq_column
    else:
        scol = _detect_seq_column(headers, variant)
        if scol is None:
            raise ValueError(
                f"could not auto-detect sequence column for variant '{variant}' "
                f"in headers: {headers}. Use --seq-column to specify.")

    # Resolve optional columns
    vcol = None if no_genes else (v_column if v_column in headers else
                                   ('v_call' if 'v_call' in headers else None))
    jcol = None if no_genes else (j_column if j_column in headers else
                                   ('j_call' if 'j_call' in headers else None))
    acol = abundance_column if abundance_column and abundance_column in headers else (
        'duplicate_count' if 'duplicate_count' in headers else None)

    si = headers.index(scol)
    vi = headers.index(vcol) if vcol and vcol in headers else None
    ji = headers.index(jcol) if jcol and jcol in headers else None
    ai = headers.index(acol) if acol and acol in headers else None

    sequences = []
    v_genes = [] if vi is not None else None
    j_genes = [] if ji is not None else None
    abundances = [] if ai is not None else None

    reader = csv.reader(fh, delimiter=delimiter)
    for row in reader:
        if not row or not row[si].strip():
            continue
        sequences.append(row[si].strip())
        if v_genes is not None:
            v_genes.append(row[vi].strip())
        if j_genes is not None:
            j_genes.append(row[ji].strip())
        if abundances is not None:
            try:
                abundances.append(int(row[ai]))
            except (ValueError, IndexError):
                abundances.append(1)

    return {
        'sequences': sequences,
        'abundances': abundances,
        'v_genes': v_genes,
        'j_genes': j_genes,
    }


def read_sequences_simple(path, seq_column=None, variant='aap'):
    """Read just sequences (no genes/abundances). For score, decompose, etc."""
    if path == '-':
        lines = sys.stdin.read().strip().split('\n')
    else:
        fh = _open_file(path)
        try:
            lines = fh.read().strip().split('\n')
        finally:
            if fh is not sys.stdin:
                fh.close()

    if not lines or not lines[0].strip():
        return []

    # Check for header
    first = lines[0]
    if '\t' in first or ',' in first:
        delim = '\t' if '\t' in first else ','
        headers = [h.strip() for h in first.split(delim)]
        col = seq_column or _detect_seq_column(headers, variant)
        if col and col in headers:
            idx = headers.index(col)
            return [row.split(delim)[idx].strip()
                    for row in lines[1:] if row.strip()]

    # Plain text
    return [l.strip() for l in lines if l.strip()]
