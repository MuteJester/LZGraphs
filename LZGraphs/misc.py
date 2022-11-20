import networkx as nx
from itertools import tee
import numpy as np

def has_path(G, source, target):
    try:
        sp = nx.dijkstra_path(G, source, target, weight=lambda x, y, z: 1 - z['weight'])
    except nx.NetworkXNoPath:
        return False
    return True

def choice(options,probs):
    x = np.random.rand()
    cum = 0
    i = None
    for i,p in enumerate(probs):
        cum += p
        if x < cum:
            break
    return options[i]
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
