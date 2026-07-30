"""Input file reading and validation utilities for CLI and programmatic use."""

import csv
import gzip
import sys


_MAX_U64 = 2**64 - 1

# Variant-aware sequence column auto-detection (case-insensitive, first match)
_SEQ_COLUMNS = {
    'aap': ['junction_aa', 'cdr3_amino_acid', 'cdr3_aa', 'aminoacid'],
    'ndp': ['junction', 'cdr3_rearrangement', 'cdr3_nt', 'nucleotide'],
    'naive': ['junction_aa', 'cdr3_amino_acid', 'junction', 'cdr3_rearrangement'],
}
_SEQ_FALLBACK = ['sequence', 'cdr3', 'seq']
_VALID_EXPECT_FORMATS = {'plain', 'plain_seqcount', 'tabular'}


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


def _chain_first_line(first_line, fh):
    yield first_line
    yield from fh


def _looks_like_plain_seqcount(first_line):
    parts = first_line.strip().split('\t')
    if len(parts) != 2 or not parts[0].strip():
        return False
    try:
        value = int(parts[1])
    except ValueError:
        return False
    return 0 <= value <= _MAX_U64


def _parse_u64(text):
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"invalid abundance '{text}'") from exc
    if value < 0:
        raise ValueError(f"invalid abundance '{text}': must be non-negative")
    if value > _MAX_U64:
        raise ValueError(f"abundance '{text}' exceeds uint64 limit")
    return value


def _expected_format_for_plain_record(line):
    parts = line.strip().split('\t')
    if len(parts) == 2 and parts[0].strip():
        try:
            _parse_u64(parts[1].strip() or '0')
            return 'plain_seqcount'
        except ValueError:
            pass
    return 'plain'


def detect_input_kind(path, variant='aap'):
    """Classify a sequence input file from its first line."""
    fh = _open_file(path)
    try:
        first_line = fh.readline()
        if not first_line.strip():
            return 'empty'
        if _looks_like_plain_seqcount(first_line):
            return 'plain_seqcount'
        has_tab = '\t' in first_line
        has_comma = ',' in first_line and not has_tab
        if has_tab or has_comma:
            return 'tabular'
        return 'plain'
    finally:
        if fh is not sys.stdin:
            fh.close()


def _new_validation_report(path, detected_kind, *, strict_input=False,
                           expect_format=None, variant='aap'):
    return {
        'path': path,
        'variant': variant,
        'detected_kind': detected_kind,
        'expect_format': expect_format,
        'strict_input': bool(strict_input),
        'ok': True,
        'mode': detected_kind if detected_kind != 'empty' else None,
        'total_lines': 0,
        'records': 0,
        'blank_lines': 0,
        'plain_records': 0,
        'seqcount_records': 0,
        'tabular_rows': 0,
        'warning_count': 0,
        'error_count': 0,
        'issues': [],
        'seq_column': None,
        'v_column': None,
        'j_column': None,
        'abundance_column': None,
        'has_header': False,
    }


def _add_issue(report, level, message, *, line=None):
    item = {'level': level, 'message': message}
    if line is not None:
        item['line'] = int(line)
    report['issues'].append(item)
    if level == 'error':
        report['error_count'] += 1
        report['ok'] = False
    else:
        report['warning_count'] += 1


def _summarize_report(report):
    status = 'ok' if report['ok'] else 'error'
    parts = [
        f"status={status}",
        f"kind={report['detected_kind']}",
        f"mode={report['mode']}",
        f"lines={report['total_lines']}",
        f"records={report['records']}",
        f"errors={report['error_count']}",
        f"warnings={report['warning_count']}",
    ]
    if report['expect_format']:
        parts.append(f"expect={report['expect_format']}")
    return ' '.join(parts)


def _parse_plain_record(line, *, strict_input=False, preferred_mode=None,
                        expect_format=None):
    raw = line.strip()
    if not raw:
        return None

    parts = raw.split('\t')
    if len(parts) == 1:
        seq = parts[0].strip()
        if not seq:
            raise ValueError("empty sequence")
        kind = 'plain'
        count = 1
    elif len(parts) == 2:
        seq = parts[0].strip()
        if not seq:
            raise ValueError("empty sequence")
        count_text = parts[1].strip()
        kind = 'plain_seqcount'
        if count_text == '':
            if strict_input:
                raise ValueError("missing abundance value")
            count = 1
        else:
            count = _parse_u64(count_text)
    else:
        raise ValueError("expected 1 or 2 tab-separated fields")

    if preferred_mode and kind != preferred_mode and strict_input:
        raise ValueError(f"mixed '{preferred_mode}' and '{kind}' records")
    if expect_format and kind != expect_format:
        raise ValueError(f"expected format '{expect_format}', saw '{kind}'")
    return kind, seq, count


def _resolve_tabular_columns(headers, seq_column, v_column, j_column,
                             abundance_column, variant, no_genes):
    if seq_column:
        if seq_column not in headers:
            raise ValueError(f"column '{seq_column}' not found in headers: {headers}")
        scol = seq_column
    else:
        scol = _detect_seq_column(headers, variant)
        if scol is None:
            raise ValueError(
                f"could not auto-detect sequence column for variant '{variant}' "
                f"in headers: {headers}. Use --seq-column to specify."
            )

    vcol = None if no_genes else (
        v_column if v_column in headers else ('v_call' if 'v_call' in headers else None)
    )
    jcol = None if no_genes else (
        j_column if j_column in headers else ('j_call' if 'j_call' in headers else None)
    )
    acol = abundance_column if abundance_column and abundance_column in headers else (
        'duplicate_count' if 'duplicate_count' in headers else None
    )
    return scol, vcol, jcol, acol


def validate_input(path, *, seq_column=None, v_column=None, j_column=None,
                   abundance_column=None, variant='aap', no_genes=False,
                   strict_input=False, expect_format=None):
    """Validate an input file without building a graph."""
    if expect_format is not None and expect_format not in _VALID_EXPECT_FORMATS:
        raise ValueError(
            f"expect_format must be one of {sorted(_VALID_EXPECT_FORMATS)}, got '{expect_format}'"
        )

    fh = _open_file(path)
    try:
        first_line = fh.readline()
        detected_kind = 'empty'
        if first_line.strip():
            if _looks_like_plain_seqcount(first_line):
                detected_kind = 'plain_seqcount'
            else:
                has_tab = '\t' in first_line
                has_comma = ',' in first_line and not has_tab
                detected_kind = 'tabular' if (has_tab or has_comma) else 'plain'
        report = _new_validation_report(
            path, detected_kind, strict_input=strict_input,
            expect_format=expect_format, variant=variant,
        )
        if detected_kind == 'empty':
            _add_issue(report, 'error', 'empty input')
            report['summary'] = _summarize_report(report)
            return report

        if detected_kind in ('plain', 'plain_seqcount'):
            _validate_plain_stream(
                fh, first_line, report, strict_input=strict_input,
                expect_format=expect_format,
            )
        else:
            _validate_tabular_stream(
                fh, first_line, report,
                seq_column=seq_column, v_column=v_column, j_column=j_column,
                abundance_column=abundance_column, variant=variant,
                no_genes=no_genes, strict_input=strict_input,
                expect_format=expect_format,
            )
        report['summary'] = _summarize_report(report)
        return report
    finally:
        if fh is not sys.stdin:
            fh.close()


def _validate_plain_stream(fh, first_line, report, *, strict_input=False,
                           expect_format=None):
    first_mode = report['detected_kind']
    report['mode'] = first_mode

    for idx, raw_line in enumerate(_chain_first_line(first_line, fh), start=1):
        report['total_lines'] += 1
        try:
            shape = _parse_plain_record(
                raw_line,
                strict_input=False,
                preferred_mode=None,
                expect_format=None,
            )
        except ValueError as exc:
            _add_issue(report, 'error', str(exc), line=idx)
            continue

        if shape is None:
            report['blank_lines'] += 1
            continue

        kind, _, _ = shape
        if kind != first_mode:
            report['mode'] = 'mixed'
            if strict_input:
                _add_issue(
                    report,
                    'error',
                    f"mixed '{first_mode}' and '{kind}' records",
                    line=idx,
                )
                continue
            if report['warning_count'] == 0:
                _add_issue(
                    report,
                    'warning',
                    f"mixed '{first_mode}' and '{kind}' records",
                    line=idx,
                )

        if expect_format and kind != expect_format:
            _add_issue(
                report,
                'error',
                f"expected format '{expect_format}', saw '{kind}'",
                line=idx,
            )
            continue

        if kind == 'plain':
            report['plain_records'] += 1
        else:
            report['seqcount_records'] += 1
        report['records'] += 1

    if expect_format and report['records'] > 0:
        if report['mode'] == 'mixed':
            _add_issue(
                report,
                'error',
                f"expected format '{expect_format}', saw mixed plain/plain_seqcount records",
            )
        elif report['mode'] != expect_format:
            _add_issue(
                report,
                'error',
                f"expected format '{expect_format}', saw '{report['mode']}'",
            )

    if report['records'] == 0 and report['error_count'] == 0:
        _add_issue(report, 'error', 'input contains no sequence records')


def _validate_tabular_stream(fh, header_line, report, *, seq_column=None,
                             v_column=None, j_column=None,
                             abundance_column=None, variant='aap',
                             no_genes=False, strict_input=False,
                             expect_format=None):
    if expect_format and expect_format != 'tabular':
        _add_issue(
            report,
            'error',
            f"expected format '{expect_format}', saw 'tabular'",
        )
        return

    delimiter = '\t' if '\t' in header_line else ','
    headers = [h.strip() for h in header_line.strip().split(delimiter)]
    report['has_header'] = True
    report['total_lines'] = 1
    report['mode'] = 'tabular'

    try:
        scol, vcol, jcol, acol = _resolve_tabular_columns(
            headers, seq_column, v_column, j_column,
            abundance_column, variant, no_genes,
        )
    except ValueError as exc:
        _add_issue(report, 'error', str(exc), line=1)
        return

    report['seq_column'] = scol
    report['v_column'] = vcol
    report['j_column'] = jcol
    report['abundance_column'] = acol

    si = headers.index(scol)
    vi = headers.index(vcol) if vcol else None
    ji = headers.index(jcol) if jcol else None
    ai = headers.index(acol) if acol else None
    reader = csv.reader(fh, delimiter=delimiter)

    for row_idx, row in enumerate(reader, start=2):
        report['total_lines'] += 1
        if not row or not any(col.strip() for col in row):
            report['blank_lines'] += 1
            continue
        if si >= len(row):
            _add_issue(report, 'error', f"missing sequence column '{scol}'", line=row_idx)
            continue
        seq = row[si].strip()
        if not seq:
            if strict_input:
                _add_issue(report, 'error', f"empty sequence value in column '{scol}'", line=row_idx)
            else:
                report['blank_lines'] += 1
            continue
        if ai is not None:
            if ai >= len(row):
                _add_issue(report, 'error', f"missing abundance column '{acol}'", line=row_idx)
                continue
            aval = row[ai].strip()
            if aval == '':
                if strict_input:
                    _add_issue(report, 'error', f"empty abundance value in column '{acol}'", line=row_idx)
                    continue
            else:
                try:
                    _parse_u64(aval)
                except ValueError as exc:
                    if strict_input:
                        _add_issue(report, 'error', str(exc), line=row_idx)
                        continue
                    _add_issue(report, 'warning', str(exc), line=row_idx)

        if vi is not None and strict_input and vi >= len(row):
            _add_issue(report, 'error', f"missing V gene column '{vcol}'", line=row_idx)
            continue
        if ji is not None and strict_input and ji >= len(row):
            _add_issue(report, 'error', f"missing J gene column '{jcol}'", line=row_idx)
            continue

        report['tabular_rows'] += 1
        report['records'] += 1

    if report['records'] == 0 and report['error_count'] == 0:
        _add_issue(report, 'error', 'tabular input contains no sequence records')


def read_sequences(path, seq_column=None, v_column=None, j_column=None,
                   abundance_column=None, variant='aap', no_genes=False,
                   strict_input=False, expect_format=None):
    """Read sequences from a file, auto-detecting format."""
    fh = _open_file(path)
    try:
        first_line = fh.readline()
        if not first_line.strip():
            raise ValueError(f"empty input: {path}")

        if _looks_like_plain_seqcount(first_line):
            return _read_plain(
                fh,
                first_line,
                strict_input=strict_input,
                expect_format=expect_format,
            )

        has_tab = '\t' in first_line
        has_comma = ',' in first_line and not has_tab
        if has_tab or has_comma:
            return _read_tabular(
                fh, first_line, has_tab,
                seq_column, v_column, j_column,
                abundance_column, variant, no_genes,
                strict_input=strict_input,
                expect_format=expect_format,
            )
        return _read_plain(
            fh,
            first_line,
            strict_input=strict_input,
            expect_format=expect_format,
        )
    finally:
        if fh is not sys.stdin:
            fh.close()


def _read_plain(fh, first_line, *, strict_input=False, expect_format=None):
    """Read plain text: one sequence per line, or sequence\\tabundance."""
    first_kind = 'plain_seqcount' if _looks_like_plain_seqcount(first_line) else 'plain'
    if expect_format and expect_format not in ('plain', 'plain_seqcount'):
        raise ValueError(f"expected format '{expect_format}', saw '{first_kind}'")

    sequences = []
    counts = []
    saw_seqcount = False

    for raw_line in _chain_first_line(first_line, fh):
        parsed = _parse_plain_record(
            raw_line,
            strict_input=strict_input,
            preferred_mode=first_kind,
            expect_format=expect_format,
        )
        if parsed is None:
            continue
        kind, seq, count = parsed
        sequences.append(seq)
        counts.append(count)
        if kind == 'plain_seqcount':
            saw_seqcount = True

    return {
        'sequences': sequences,
        'abundances': counts if saw_seqcount else None,
        'v_genes': None,
        'j_genes': None,
    }


def _read_tabular(fh, header_line, is_tsv, seq_column, v_column, j_column,
                  abundance_column, variant, no_genes,
                  *, strict_input=False, expect_format=None):
    """Read TSV/CSV with header."""
    if expect_format and expect_format != 'tabular':
        raise ValueError(f"expected format '{expect_format}', saw 'tabular'")

    delimiter = '\t' if is_tsv else ','
    headers = [h.strip() for h in header_line.strip().split(delimiter)]
    scol, vcol, jcol, acol = _resolve_tabular_columns(
        headers, seq_column, v_column, j_column,
        abundance_column, variant, no_genes,
    )

    si = headers.index(scol)
    vi = headers.index(vcol) if vcol is not None else None
    ji = headers.index(jcol) if jcol is not None else None
    ai = headers.index(acol) if acol is not None else None

    sequences = []
    v_genes = [] if vi is not None else None
    j_genes = [] if ji is not None else None
    abundances = [] if ai is not None else None

    reader = csv.reader(fh, delimiter=delimiter)
    for row_idx, row in enumerate(reader, start=2):
        if not row or not any(col.strip() for col in row):
            continue
        if si >= len(row):
            if strict_input:
                raise ValueError(f"line {row_idx}: missing sequence column '{scol}'")
            continue
        seq = row[si].strip()
        if not seq:
            if strict_input:
                raise ValueError(f"line {row_idx}: empty sequence value in column '{scol}'")
            continue

        sequences.append(seq)
        if v_genes is not None:
            if vi >= len(row):
                if strict_input:
                    raise ValueError(f"line {row_idx}: missing V gene column '{vcol}'")
                v_genes.append('')
            else:
                v_genes.append(row[vi].strip())
        if j_genes is not None:
            if ji >= len(row):
                if strict_input:
                    raise ValueError(f"line {row_idx}: missing J gene column '{jcol}'")
                j_genes.append('')
            else:
                j_genes.append(row[ji].strip())
        if abundances is not None:
            if ai >= len(row):
                if strict_input:
                    raise ValueError(f"line {row_idx}: missing abundance column '{acol}'")
                abundances.append(1)
            else:
                aval = row[ai].strip()
                if aval == '':
                    if strict_input:
                        raise ValueError(f"line {row_idx}: empty abundance value in column '{acol}'")
                    abundances.append(1)
                else:
                    try:
                        abundances.append(_parse_u64(aval))
                    except ValueError:
                        if strict_input:
                            raise ValueError(f"line {row_idx}: invalid abundance '{aval}'")
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

    first = lines[0]
    if '\t' in first or ',' in first:
        delim = '\t' if '\t' in first else ','
        headers = [h.strip() for h in first.split(delim)]
        col = seq_column or _detect_seq_column(headers, variant)
        if col and col in headers:
            idx = headers.index(col)
            return [row.split(delim)[idx].strip() for row in lines[1:] if row.strip()]

    return [l.strip() for l in lines if l.strip()]
