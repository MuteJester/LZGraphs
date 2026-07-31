"""Header parity between ``detect_format``'s resolved columns and what
``iter_tabular_rows`` actually reads.

Measured regression: ``detect_format`` built ``InputSpec.header`` and resolved
``seq_column``/``abundance_column``/``v_column``/``j_column`` by naive
``line.split(delimiter)`` followed by ``.strip()``, while ``iter_tabular_rows``
looks those resolved names up in ``csv.DictReader`` keys, which are neither
stripped nor quote-normalised the same way ``split`` is. When the two parses
disagree on a real header, the lookup silently returns nothing and the field
is treated as absent: exit 0, wrong data, no error. Three such corruptions
were measured:

- a padded abundance column ('junction_aa, duplicate_count') resolves the
  name correctly but reads abundance as 1 for every row instead of the real
  counts,
- a padded 'productive' column keeps a non-productive row that should be
  dropped,
- a duplicated column name reachable only through quoting
  ('junction_aa,"junction_aa"') is missed by the duplicate-column guard
  (which also used the naive split) and silently reads the wrong column's
  data under the right column's name.

This module pins the fix: one parse (via the ``csv`` module) and one
normalisation (stripping), applied identically on both sides, so a header
that ``detect_format`` resolves is always the same header ``iter_tabular_rows``
reads from.
"""
from __future__ import annotations

import csv
import io

import pytest

from LZGraphs._io._public import read_sequences
from LZGraphs._io._sniff import detect_format
from LZGraphs._io._spec import FormatError
from LZGraphs._io._validate import validate_input


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_padded_abundance_column_reads_the_real_counts(tmp_path):
    """header 'junction_aa, duplicate_count' must resolve AND read
    duplicate_count, not silently default abundance to 1 for every row.
    """
    text = (
        "junction_aa, duplicate_count\n"
        "CASSLGQAYEQYF,30\n"
        "CASSPGTGVYGYTF,20\n"
    )
    path = _write(tmp_path, "padded_abundance.csv", text)

    spec = detect_format(path)
    assert spec.abundance_column == "duplicate_count"

    result = read_sequences(path)
    assert result["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert result["abundances"] == [30, 20]


def test_padded_productive_column_drops_the_nonproductive_row(tmp_path):
    """header 'junction_aa\\t productive' must drop the F row, not keep it."""
    text = (
        "junction_aa\t productive\n"
        "CASSLGQAYEQYF\tT\n"
        "CASSPGTGVYGYTF\tF\n"
    )
    path = _write(tmp_path, "padded_productive.tsv", text)

    result = read_sequences(path)
    assert result["sequences"] == ["CASSLGQAYEQYF"]
    assert result["stats"].nonproductive == 1

    report = validate_input(path)
    assert report["records"] == 1


def test_quoted_duplicate_column_raises_instead_of_reading_wrong_value(tmp_path):
    """header 'junction_aa,"junction_aa"' is a duplicate reachable only
    through quoting: naive split sees the second field as the literal text
    '"junction_aa"' (quotes intact), a different string from the first
    field, and misses the collision. A real csv parse sees two columns
    named identically. This must raise, not silently read WRONGVALUE
    (csv.DictReader keeps only the last occurrence of a repeated key).
    """
    text = (
        'junction_aa,"junction_aa"\n'
        'CASSLGQAYEQYF,WRONGVALUE\n'
    )
    path = _write(tmp_path, "quoted_dup.csv", text)

    with pytest.raises(FormatError) as excinfo:
        detect_format(path)
    assert "junction_aa" in str(excinfo.value)

    with pytest.raises(FormatError):
        read_sequences(path)


def test_fully_quoted_header_resolves_columns(tmp_path):
    """Hazard 3: a fully quoted header ('"junction_aa","duplicate_count"')
    must resolve both columns, not fail sequence-column detection.
    """
    text = (
        '"junction_aa","duplicate_count"\n'
        'CASSLGQAYEQYF,7\n'
        'CASSPGTGVYGYTF,2\n'
    )
    path = _write(tmp_path, "quoted_header.csv", text)

    spec = detect_format(path)
    assert spec.seq_column == "junction_aa"
    assert spec.abundance_column == "duplicate_count"

    result = read_sequences(path)
    assert result["sequences"] == ["CASSLGQAYEQYF", "CASSPGTGVYGYTF"]
    assert result["abundances"] == [7, 2]


# General structural guard.
#
# The four cases above each pin one concrete symptom. This guard instead
# checks the *mechanism*: for a battery of awkward-but-legitimate headers,
# spec.header (however detect_format derives it) must be exactly what
# csv.DictReader will produce as fieldnames for the same line, once both are
# normalised the same way. If a future change reintroduces a second,
# differently-normalised parse of the header line, this fails structurally
# instead of waiting for a specific corruption to be noticed.

_AWKWARD_HEADERS = [
    "junction_aa, duplicate_count, v_call , j_call ",
    '"junction_aa","duplicate_count","v_call","j_call"',
    "junction_aa\t productive \tduplicate_count",
    '"junction_aa"\t"v_call"\t"j_call"',
    " junction_aa , v_call , j_call ",
]


@pytest.mark.parametrize("header_line", _AWKWARD_HEADERS)
def test_spec_header_agrees_with_dictreader_fieldnames(tmp_path, header_line):
    text = header_line + "\n"
    path = _write(tmp_path, "awkward.txt", text)

    spec = detect_format(path)
    delimiter = spec.delimiter or "\t"

    expected = next(csv.reader(io.StringIO(header_line), delimiter=delimiter))
    expected = [name.strip() for name in expected]

    assert list(spec.header) == expected

    # And the reader must agree independently: DictReader's own fieldnames,
    # once stripped the same way, must match too. This is the actual
    # mechanism iter_tabular_rows looks names up against.
    dict_fieldnames = csv.DictReader(io.StringIO(text), delimiter=delimiter).fieldnames
    assert [name.strip() for name in dict_fieldnames] == expected
