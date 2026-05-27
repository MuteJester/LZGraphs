"""Tier 1 regression: lz76 and flashback tokenizers.

Snapshots the decomposition output for a fixed list of CDR3 sequences,
plus a sanity-check on the round-trip property of flashback.
"""
from __future__ import annotations

from LZGraphs import flashback_decompose, flashback_reverse, lz76_decompose

from .snapshot_utils import assert_snapshot_match


# A diverse set covering: short, long, repeated chars, all-same, mixed.
TOKENIZER_PROBES = [
    "A",
    "AA",
    "AAA",
    "AB",
    "ABBA",
    "ABCDEF",
    "FFFFFFFFF",
    "AABCA",
    "AABBCCDD",
    "AAABBBCCCDDDEEE",
    "CASSLGIRRT",
    "CASSAYFF",
    "CASSLEPSGGTDTQYF",
    "CASSFSTCSANYGYTF",
    "CARSLVGGLAGGPQNTEAFF",
]


def test_lz76_decompose_snapshot():
    out = {
        seq: list(lz76_decompose(seq))
        for seq in TOKENIZER_PROBES
    }
    assert_snapshot_match("tokenizers_lz76", out)


def test_flashback_decompose_snapshot():
    out = {
        seq: list(flashback_decompose(seq))
        for seq in TOKENIZER_PROBES
    }
    assert_snapshot_match("tokenizers_flashback", out)


def test_flashback_reverse_round_trip():
    """flashback_reverse(flashback_decompose(s)) == s for every probe."""
    for seq in TOKENIZER_PROBES:
        tokens = flashback_decompose(seq)
        recovered = flashback_reverse(tokens)
        assert recovered == seq, (
            f"round-trip failed for {seq!r}: "
            f"decompose={tokens!r} reverse={recovered!r}"
        )
