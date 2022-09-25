from collections import OrderedDict


def lempel_ziv_decomposition(sequence):
    sub_strings = list()
    n = len(sequence)

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break

        if ind + inc == len(sequence) and sequence[ind:ind + inc] in sub_strings:
            sub_str = sequence[ind: ind + inc]  # +sequence[ind : ind + inc]
            sub_strings.append(sub_str)
            break
        else:
            sub_str = sequence[ind: ind + inc]

        # print(sub_str, ind, inc)
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.append(sub_str)
            ind += inc
            inc = 1
    return list(sub_strings)


