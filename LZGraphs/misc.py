import networkx as nx
from itertools import tee


def has_path(G, source, target):
    try:
        sp = nx.dijkstra_path(G, source, target, weight=lambda x, y, z: 1 - z['weight'])
    except nx.NetworkXNoPath:
        return False
    return True


def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


def chunkify(L, n):
    """ Yield successive n-sized chunks from L.
    """
    for i in range(0, len(L), n):
        yield L[i:i+n]
