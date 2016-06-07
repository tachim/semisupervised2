import numpy as np
import sys
import argparse
import io
import re
import os
from os.path import join, isfile
from deptree import DeptreeExample
from collections import defaultdict

class Reader:

    def __init__(self, ptbfile, unvfile, brownfile, window_size, verbose=False):
        '''
        Class for sampling instance of three word sequences and pos tags for training
        INPUTS : ptbfile - path to pentreeback file
                 unvfile - ptb to universal pos tag mapping
                 brownfile - output brown clustering file for universe of words
                 verbose - print debugging statements
        '''
        self.verbose = verbose
        self.examples = DeptreeExample.load_examples_from_file(ptbfile)
        self.examples = remove_short(self.examples, window_size)
        self.pos_to_unv, self.unv_to_ind = parse_unv(unvfile)
        print 'Number of Examples : %d' % len(self.examples)

        if isfile(brownfile):
            self.word_to_cluster = self.get_mappings(brownfile)
        else:
            self.word_to_cluster = {}

    def sample(self, batch_size):
        '''
        Returns batch of word features, brown clusters, and pos tag indices
        '''
        self.log('Sampling batch...')
        seq_word = []
        pos = []
        for i in range(batch_size):
            # Address edge case where sentence contains less than SEQ_SIZE elements
            ex_ind = np.random.randint(len(self.examples))
            seq_toks = self.examples[ex_ind].gt_tokens
            seq_wd = [st[0] for st in seq_toks]
            seq_tag = [self.pos_to_ind(st[1]) for st in seq_toks]
            seq_word.append(seq_wd)
            pos.append(seq_tag)

        clusters = [[self.word_to_cluster.get(wd, wd) for wd in seq] for seq in seq_word]
        self.log('Number of unique brown clusters : %d' % len(set(self.word_to_cluster.values())))
        return seq_word, clusters, pos

    def get_mappings(self, brown_filename):
        '''
        Generates mappings between words x_i -> features sub_j(x_i) and cluster(x_i)
        INPUT : brown_filepath - location of path file output from brownclustering script
        OUTPUT : indcluster_mapping - dict mapping x_i -> cluster(x_i)
        '''
        brown_file = io.open(brown_filename, encoding='utf8', mode='r')
        cluster_mapping = {}

        for line in brown_file:
          splits = line.split('\t')
          if len(splits) != 3:
            print 'len(splits) = ', len(splits), ' at line ', counter
            print splits
            assert False
          cluster, word, frequency = splits
          cluster_mapping[word] = cluster
        brown_file.close()

        # Convert cluster from binary representation to indices
        cluster_str = list(set(cluster_mapping.values()))
        cluster_to_ind = {cluster_str[ind]:ind for ind in range(len(cluster_str))}
        indcluster_mapping = {word:cluster_to_ind[cluster_mapping[word]] for word in cluster_mapping}

        return indcluster_mapping

    # Returns index given ptb POS
    def pos_to_ind(self, pos):
        return self.unv_to_ind[self.pos_to_unv[pos]]

    def log(self, msg):
        if self.verbose:
            print msg

# Returns list of examples with sequences less then min_len removed
def remove_short(examples, min_len):
    result = []
    for ex in examples:
        if len(ex.gt_tokens) >= min_len:
            result.append(ex)
    return result

def parse_unv(filepath):
    pos_to_unv = {}
    with open(filepath, 'r') as f:
        for line in f:
            ptb, unv = line.strip().split('\t')
            pos_to_unv[ptb] = unv
    unv_tags = list(set(pos_to_unv.values()))
    print 'Number of univeral pos tags : %d' % len(unv_tags)
    unv_to_ind = {unv:ind for ind,unv in enumerate(unv_tags)}
    return pos_to_unv, unv_to_ind

# Returns list where each element is a full sentence string from the text corpus
def flatten(examples):
    result = []
    for ex in examples:
        buff = []
        for tok in ex.gt_tokens:
            word = tok[0]
            buff.append(word)
        result.append(' '.join(buff))
    return result

class FeatureGenerator:
    def __init__(self, brown_filename):
        self.digit_regex = re.compile('\d')
        self.hyphen_regex = re.compile('-')

        self.have_brown_clusters = isfile(brown_filename)
        if not self.have_brown_clusters:
            return
        brown_file = io.open(brown_filename, encoding='utf8', mode='r')

        self.suffix_counts = defaultdict(int)
        self.prefix_counts = defaultdict(int)
        self.frequency = defaultdict(int)
        counter = 0
        for line in brown_file:
          counter += 1
          splits = line.split('\t')
          if len(splits) != 3:
            print 'len(splits) = ', len(splits), ' at line ', counter
            print splits
            assert False
          cluster, word, frequency = splits
          self.frequency[word] = frequency
          
          for suffix_length in range(1,4):
            if len(word) > suffix_length:
              self.suffix_counts[word[-suffix_length:]] += 1
              self.prefix_counts[word[0:suffix_length]] += 1

        ## This is the value "f" in Table 2 of Ammar et al. 2014
        self.min_affix_count = counter / 500.0

    def hypthen(self, word):
        if self.hyphen_regex.search(word):
            return u'contain_hyphen'
        return None

    def digit(self, word):
        if self.digit_regex.search(word):
            return u'contain_digit'
        return None

    def prefix2(self, word):
        if not self.have_brown_clusters: return 0

        affix_length = 2
        if len(word) > affix_length and self.prefix_counts[word[0:affix_length]] > self.min_affix_count:
            return u'{}-pref-{}'.format(affix_length, word[0:affix_length])
        return None

    def prefix3(self, word):
        if not self.have_brown_clusters: return 0

        affix_length = 3
        if len(word) > affix_length and self.prefix_counts[word[0:affix_length]] > self.min_affix_count:
            return u'{}-pref-{}'.format(affix_length, word[0:affix_length])
        return 0

    def suffix2(self, word):
        if not self.have_brown_clusters: return 0

        affix_length = 2
        if len(word) > affix_length and self.suffix_counts[word[-affix_length:]] > self.min_affix_count:
            return u'{}-suff-{}'.format(affix_length, word[-affix_length:])
        return None

    def suffix3(self, word):
        if not self.have_brown_clusters: return 0

        affix_length = 3
        if len(word) > affix_length and self.suffix_counts[word[-affix_length:]] > self.min_affix_count:
            return u'{}-suff-{}'.format(affix_length, word[-affix_length:])
        return None

    def freq(self, word):
        if not self.have_brown_clusters: return 0

        # only fire an emission for frequent words
        min_word_frequency=100
        if self.frequency >= min_word_frequency:
            return word.lower().replace(u'=', u'eq')
        return None

    def shape(self, word):
        if not self.have_brown_clusters: return 0

        # word shape
        shape=[u'^']
        for c in word:
            if c.isdigit():
                if shape[-1] != u'0': shape.append(u'0')
            elif c.isalpha() and c.isupper():
                if shape[-1] != u'A': shape.append(u'A')
            elif c.isalpha():
                if shape[-1] != u'a': shape.append(u'a')
            else: 
                if shape[-1] != u'#': shape.append(u'#')
        return u''.join(shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate files necessary for feature generation')
    parser.add_argument('--outdir', help='Output directory for txt file')
    parser.add_argument('-gen_bfile', action='store_true', help='Generate flat txt file for brown clustering')
    args = parser.parse_args()

    if args.outdir is None:
        print 'Using output/ as default outdir path...'
        args.outdir = 'output/'
    
    from main import GlobalOpts
    opts = GlobalOpts()
    
    if args.gen_bfile and args.outdir is not None:
        print 'Generating txt file for brown clustering...'
        reader = Reader(opts, 'ds/deptrees/dev', load_brownfile=False)
        lst = reader.flatten()
        with open(join(args.outdir, 'plain.txt'), 'w') as o:
            for sentence in lst:
                o.write(sentence + '\n')

    else:
        print 'Tests for reader class...'
        reader = Reader(opts, 'ds/deptrees/dev')

        word_mapping, cluster_mapping = get_mappings(join(args.outdir, 'paths'))
        dense = dense_rep(word_mapping)
        print 'Vector length : %d' % len(dense.values()[0])

        features, clusters, postags = reader.sample()
        print features.shape
        print clusters.shape
        print postags.shape

        print reader.cocurrence().shape

