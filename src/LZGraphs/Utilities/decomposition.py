from typing import List


def lempel_ziv_decomposition(sequence: str) -> List[str]:
    """
    An optimized implementation of the LZ76 compression algorithm.

    Given a string, the function returns all unique sub-patterns derived
    from the input string using the Lempel-Ziv 1976 parsing algorithm.

    The algorithm parses a sequence left-to-right, at each position extending
    the current pattern until it is not found in the set of previously seen
    patterns. This new pattern is then added to the vocabulary.

    Complexity:
        - Time: O(n * avg_pattern_length) where n is the sequence length
        - Space: O(n) for the set and list of subpatterns

    This implementation uses a set for O(1) membership testing instead of
    the original O(n) list-based approach, providing significant speedup
    for long sequences.

    Args:
        sequence (str): A string from which to derive sub-patterns.
            Typically an amino acid or nucleotide sequence.

    Returns:
        list: A list of unique sub-patterns in order of discovery.

    Example:
        >>> lempel_ziv_decomposition("ABCABCDEF")
        ['A', 'B', 'C', 'AB', 'CD', 'E', 'F']
        >>> lempel_ziv_decomposition("CASSLGIRRTNTEAFF")
        ['C', 'A', 'SS', 'L', 'G', 'I', 'RR', 'T', 'N', 'TE', 'AF', 'F']
    """
    if not sequence:
        return []

    sub_strings = []
    seen = set()  # O(1) membership testing
    n = len(sequence)
    ind = 0

    while ind < n:
        inc = 1
        # Extend pattern until it's new or we hit the end
        while ind + inc <= n:
            sub_str = sequence[ind:ind + inc]

            if sub_str not in seen:
                # Found a new pattern
                sub_strings.append(sub_str)
                seen.add(sub_str)
                ind += inc
                break

            inc += 1

            # Handle end-of-sequence edge case
            if ind + inc > n:
                # We've reached the end but current pattern is already seen
                # Append it anyway (LZ76 behavior for sequence termination)
                sub_str = sequence[ind:]
                if sub_str:
                    sub_strings.append(sub_str)
                ind = n  # Exit the outer loop
                break

    return sub_strings
