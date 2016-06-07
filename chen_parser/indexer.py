class Indexer(object):
    def __init__(self):
        self.k_to_ind = {}
        self.ind_to_k = {}

    def add(self, k):
        if k in self.k_to_ind:
            return self.k_to_ind[k]
        else:
            ind = len(self.k_to_ind)
            self.k_to_ind[k] = ind
            self.ind_to_k[ind] = k
            return ind

    def ind(self, k):
        return self.k_to_ind[k]

    def k(self, ind):
        return self.ind_to_k[ind]

    def size(self): return len(self.k_to_ind)
