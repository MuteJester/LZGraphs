import random

import numpy as np

from LZGraphs import derive_lz_reading_frame_position, derive_lz_and_position, lempel_ziv_decomposition, window
from tqdm.auto import tqdm
import itertools


def naive_node_extractor(seq):
    return lempel_ziv_decomposition(seq)


def ndp_node_extractor(seq):
    LZ, POS, locations = derive_lz_reading_frame_position(seq)
    nodes_local = list(map(lambda x, y, z: x + str(y) + '_' + str(z), LZ, POS, locations))
    return nodes_local


def aap_node_extractor(seq):
    LZ, locations = derive_lz_and_position(seq)
    nodes_local = list(map(lambda x, y: x + '_' + str(y), LZ, locations))
    return nodes_local


class NodeEdgeSaturationProbe:
    def __init__(self, node_function='naive',log_level=1):
        self.nodes = set()
        self.edges = set()
        self.log_memory = dict()
        self.log_level = log_level
        self.node_function = None
        if node_function == 'naive':
            self.node_function = naive_node_extractor
        elif node_function == 'ndp':
            self.node_function = ndp_node_extractor
        elif node_function == 'aap':
            self.node_function = aap_node_extractor


    def log(self,args):
        if self.log_level == 1:
            self.log_memory[args] = {'nodes': len(self.nodes), 'edges': len(self.edges)}


    def test_sequences(self, sequence_list,log_every = 1000,iteration_number=None):
        for ax,seq in tqdm(enumerate(sequence_list,start=1), leave=False, position=0,total=len(sequence_list)):
            nodes_local = self.node_function(seq)
            self.nodes.update(nodes_local)
            self.edges.update((window(nodes_local, 2)))

            if ax % log_every == 0:
                self.log(ax)

    def _reset(self):
        self.nodes = set()
        self.edges = set()
        self.log_memory = dict()
    def resampling_test(self,sequence_list,n_tests,log_every=1000,sample_size=0):
        result = []
        if sample_size == 0:
            for n in range(n_tests):
                random.shuffle(sequence_list)
                self.test_sequences(sequence_list,log_every,n)
                # save logs
                # reset aux
                result.append(self.log_memory.copy())
                self._reset()
        else:
            for n in range(n_tests):
                random.shuffle(sequence_list)
                self.test_sequences(random.sample(sequence_list,sample_size), log_every,n)
                # save logs
                # reset aux
                result.append(self.log_memory.copy())
                self._reset()
        return result
