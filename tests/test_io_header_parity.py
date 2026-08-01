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

import pytest

from LZGraphs._io._compress import open_text
from LZGraphs._io._public import read_sequences
from LZGraphs._io._readers import iter_tabular_rows
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
# checks the *mechanism*, and does so by actually exercising it: for a
# battery of awkward-but-legitimate headers (padded, quoted, reordered,
# comma- and tab-delimited), every column detect_format resolves
# (seq_column, and abundance_column/v_column/j_column/a productive column
# where the case has one) must be genuinely honoured when
# iter_tabular_rows reads a row, not merely equal, in isolation, to a
# second independently-recomputed csv.DictReader.fieldnames list.
#
# That distinction matters: an earlier version of this guard compared
# spec.header against a fieldnames list recomputed locally in the test,
# and never called iter_tabular_rows at all. A reviewer mutated
# _readers.py to delete the fieldnames-normalisation line (reintroducing
# the reader-side half of the original bug) and all five parametrized
# cases of that guard kept passing, because the guard's two sides were
# both re-derivations of the header line, not the real reader. This
# version instead builds one data row per case with a distinctive value in
# every resolved column, runs the real iter_tabular_rows against the real
# resolved InputSpec, and asserts the yielded tuple carries those
# distinctive values. A column silently falling back to None, or an
# abundance silently defaulting to 1, fails the assertion directly.
#
# Each case is (delimiter, [(role, header_field_text), ...]): the header
# field text carries whatever padding/quoting is under test, and "role"
# says which resolved spec attribute (seq/abundance/v/j) or which ad hoc
# scan (productive) that field is supposed to satisfy.
_AWKWARD_CASES = [
    (",", [
        ("seq", "junction_aa"),
        ("abundance", " duplicate_count"),
        ("v", " v_call "),
        ("j", " j_call "),
        ("productive", " productive"),
    ]),
    (",", [
        # Quoting alone (no internal whitespace) is handled entirely by the
        # csv module and never touches normalize_header_field, so this also
        # pads *inside* the quotes: that whitespace is preserved literally
        # by csv (quoted content is taken verbatim) and can only be removed
        # by the same stripping iter_tabular_rows applies to fieldnames.
        ("seq", '"junction_aa"'),
        ("abundance", '" duplicate_count "'),
        ("v", '" v_call "'),
        ("j", '" j_call "'),
        ("productive", '"productive"'),
    ]),
    ("\t", [
        # Padding here sits on abundance/v/j, not just productive: the
        # productive-key scan strips its own key independently
        # (k.strip().lower()), so a header where only productive is padded
        # cannot, by itself, catch a missing fieldnames normalisation.
        ("seq", "junction_aa"),
        ("productive", " productive "),
        ("abundance", " duplicate_count"),
        ("v", " v_call "),
        ("j", " j_call "),
    ]),
    ("\t", [
        ("seq", '"junction_aa"'),
        ("v", '" v_call "'),
        ("j", '" j_call "'),
        ("abundance", '" duplicate_count "'),
        ("productive", '"productive"'),
    ]),
    (",", [
        ("seq", " junction_aa "),
        ("v", " v_call "),
        ("j", " j_call "),
        ("abundance", " duplicate_count "),
        ("productive", " productive "),
    ]),
]

_ROLE_TO_SPEC_ATTR = {
    "abundance": "abundance_column",
    "v": "v_column",
    "j": "j_column",
}


@pytest.mark.parametrize("case_index", range(len(_AWKWARD_CASES)))
def test_iter_tabular_rows_honours_every_column_detect_format_resolved(
    tmp_path, case_index,
):
    delimiter, fields = _AWKWARD_CASES[case_index]
    distinctive = {
        "seq": f"CASSDISTINCT{case_index}",
        "abundance": str(300 + case_index),
        "v": f"IGHV{case_index}-1*01",
        "j": f"IGHJ{case_index}*01",
        "productive": f"PRODMARK{case_index}",
    }

    header_line = delimiter.join(field_text for _role, field_text in fields)
    data_line = delimiter.join(distinctive[role] for role, _field_text in fields)
    text = header_line + "\n" + data_line + "\n"
    path = _write(tmp_path, f"awkward_case_{case_index}.txt", text)

    spec = detect_format(path)
    assert spec.seq_column is not None, "the seq column itself must resolve"

    stream, _codec = open_text(path)
    try:
        rows = list(iter_tabular_rows(stream, spec))
    finally:
        stream.close()
    assert len(rows) == 1
    sequence, abundance, v_call, j_call, productive = rows[0]

    assert sequence == distinctive["seq"]
    for role, spec_attr in _ROLE_TO_SPEC_ATTR.items():
        if getattr(spec, spec_attr) is None:
            continue
        actual = {"abundance": abundance, "v": v_call, "j": j_call}[role]
        if role == "abundance":
            assert actual == int(distinctive["abundance"])
        else:
            assert actual == distinctive[role]
    if any(role == "productive" for role, _ in fields):
        assert productive == distinctive["productive"]
