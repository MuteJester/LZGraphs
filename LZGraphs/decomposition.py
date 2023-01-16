from collections import OrderedDict
from typing import List
from numba import jit

#@jit(nopython=True)
def lempel_ziv_decomposition(sequence:str) -> List[str]:
    """
          an implementation of the LZ76 compression algorithm,
          Given a string the function will return all unique sub-patterns derived from the input string

                  Args:
                          sequence (str): a string from which to derive sub-patterns

                  Returns:
                          list : a list of unique sub-patterns
   """
    sub_strings = list()
    n = len(sequence)
    ind = 0
    inc = 1
    while True:
        if ind + inc > n:
            break
        if ind + inc == n and sequence[ind:ind + inc] in sub_strings:
            sub_str = sequence[ind: ind + inc]  # +sequence[ind : ind + inc]
            sub_strings.append(sub_str)
            break
        else:
            sub_str = sequence[ind: ind + inc]

        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.append(sub_str)
            ind += inc
            inc = 1
    return sub_strings


