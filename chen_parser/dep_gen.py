from collections import defaultdict
import math
import deptree

class DepGen(object):
    def __init__(self, dep_filename):
        self.prior = defaultdict(float) #prior[rel]
        self.posterior = defaultdict(lambda: defaultdict(float)) #posterior[parent-rel][rel]
        trees = deptree.DeptreeExample.load_examples_from_file(dep_filename)
        
        for t in trees:
            for parent, rel in t.parents.itervalues():
                self.prior[rel]+=1
                if parent == deptree.ROOT: continue
                parent_rel = t.parents[parent][1]
                self.posterior[parent_rel][rel]+=1

        def to_log_probs(d):
            logZ = math.log(sum(d.itervalues()))
            for k,v in d.iteritems():
                d[k] = math.log(v) - logZ

        to_log_probs(self.prior)
        for k in self.posterior:
            to_log_probs(self.posterior[k])

    def tree_logprob(self, parents):# TODO laplace smoothing
        logprob = 0.0
        for child, rel in parents.itervalues():
            if child not in parents: continue
            parent_rel = parents[child][1]
            logprob += self.posterior[parent_rel][rel]
        return logprob

    def tree_prob(self, parents):
        return math.exp(self.tree_logprob(parents))

if __name__ == '__main__':
    depgen = DepGen('ds/deptrees/train')
    print depgen.prior
    print depgen.posterior