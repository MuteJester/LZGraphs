


class LZGraphBase:
    def __int__(self):
        self.genetic = False
        # a list of invalid genetic walks
        self.genetic_walks_black_list = None
        # total number of sub-patterns
        self.n_subpatterns = 0
        # init and term states sets
        self.initial_states, self.terminal_states = [], []
        self.lengths = []
        self.cac_graphs = dict()
        self.n_neighbours = dict()
        dictionary_flag = False
        dictionary = set()


    def __load_gene_data(self, data):
        self.observed_vgenes = list(set(data['V']))
        self.observed_jgenes = list(set(data['J']))

        self.marginal_vgenes = data['V'].value_counts()
        self.marginal_jgenes = data['J'].value_counts()
        self.marginal_vgenes /= self.marginal_vgenes.sum()
        self.marginal_jgenes /= self.marginal_jgenes.sum()

        self.vj_probabilities = (data['V'] + '_' + data['J']).value_counts()
        self.vj_probabilities /= self.vj_probabilities.sum()
