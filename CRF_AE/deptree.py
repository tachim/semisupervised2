#!/usr/bin/env python
'''
Parse Dependency tree files
'''


import collections

class DeptreeExample(object):
    def __init__(self, parents, children, tokens):
        self.gt_parents = parents
        self.gt_children = children
        self.gt_tokens = tokens

    def __repr__(self):
        return str(('parents', self.gt_parents, 
            'children', self.gt_children, 
            'tokens', self.gt_tokens))

    @classmethod
    def read_example(cls, f):
        '''
        Parse a single line of treebank data file
        INPUT : f - file stream from data file
        OUTPUT : DeptreeExample data structure
                 .gt_parents : child_ind -> (parent_ind, label)
                 .gt_children : parent_ind -> [child_ind*]
                 .gt_tokens : [(word, pos)*]
        '''
        
        header = f.readline()
        if not header or not header.startswith('example'):
            return None
        nodes = []
        while True:
            line = f.readline().strip()
            if not line: break
            elts = line.split()
            word, pos, parent, label = elts[2], elts[5], elts[7], elts[8]
            parent = int(parent) - 1
            nodes.append((word, pos, parent, label))
        parents = dict((i, (parent, label)) for i, (_, _, parent, label) in enumerate(nodes))

        children = collections.defaultdict(set)
        for child, (parent, _) in parents.iteritems():
            children[parent].add(child)
        tokens = [(word, pos) for (word, pos, _, _) in nodes]
        return DeptreeExample(parents, children, tokens)

    @classmethod
    def load_examples_from_file(cls, fname):
        ret = []
        with open(fname, 'r') as f:
            while True:
                ex = cls.read_example(f)
                if ex is None: break
                ret.append(ex)
        return ret

PUNCTUATION = set(["``", "''", ".", ",", ":"])

class DeptreeSearchState(object):
    KEYS = dict(
            s1w=0, s1pos=1,
            s2w=2, s2pos=3,
            s3w=4, s3pos=5,
            b1w=6, b1pos=7,
            b2w=8, b2pos=9,
            b3w=10, b3pos=11,
            s1lcw=12, s1lcpos=13,
            s1rcw=14, s1rcpos=15)

    def __init__(self, example):
        self.example = example
        self.buf = collections.deque(range(len(self.example.gt_tokens)))
        self.stack = collections.deque([-1])
        self.children = collections.defaultdict(set)
        self.parents = {}

    def _add_parent_child(self, parent, child, label):
        self.parents[child] = (parent, label)
        self.children[parent].add(child)

    def compute_metric(self):
        denom = 0
        num = 0
        for child, (parent, label) in self.parents.iteritems():
            if child == -1: continue
            tok = self.example.gt_tokens[child][0]
            if tok in PUNCTUATION: continue
            denom += 1
            num += int(parent == self.example.gt_parents[child][0])
        if denom == 0: 
            return None
        else:
            return float(num) / denom

    def is_done(self):
        return len(self.buf) == 0 and len(self.stack) == 1

    def can_apply_action(self, action):
        if action == 'SHIFT':
            return len(self.buf) >= 1
        else:
            return len(self.stack) >= 2

    def apply_action(self, action):
        if action == 'SHIFT':
            assert len(self.buf) >= 1
            self.stack.appendleft(self.buf.popleft())
        else:
            assert len(self.stack) >= 2
            s1 = self.stack.popleft()
            s2 = self.stack.popleft()
            if action[0] == 'L':
                parent, child = s1, s2
            else:
                parent, child = s2, s1
            self.stack.appendleft(parent)
            self._add_parent_child(parent, child, action[2:])

    def iter_oracle_states(self):
        while len(self.buf) > 0 or len(self.stack) > 1:
            did_action = False
            if len(self.stack) >= 2:
                s1, s2 = self.stack[0], self.stack[1]
                if self.example.gt_parents[s1][0] == s2:
                    if len(self.children[s1]) == len(self.example.gt_children[s1]):
                        action = 'R-' + self.example.gt_parents[s1][1]
                        yield (self, action)
                        self.apply_action(action)
                        did_action = True

                elif not did_action and s2 != -1 and self.example.gt_parents[s2][0] == s1:
                    if len(self.children[s2]) == len(self.example.gt_children[s2]):
                        action = 'L-' + self.example.gt_parents[s2][1]
                        yield (self, action)
                        self.apply_action(action)
                        did_action = True

            if not did_action and len(self.buf) > 0:
                yield (self, 'SHIFT')
                self.stack.appendleft(self.buf.popleft())
                did_action = True

            assert did_action

    def extract_tokens(self):
        """
        s1w=0, s1pos=1,
        s2w=2, s2pos=3,
        s3w=4, s3pos=5,
        b1w=6, b1pos=7,
        b2w=8, b2pos=9,
        b3w=10, b3pos=11,
        """
        tokens = ["<NULL>"] * len(self.KEYS)
        s1 = self.stack[0] if len(self.stack) >= 1 else None
        s2 = self.stack[1] if len(self.stack) >= 2 else None
        s3 = self.stack[2] if len(self.stack) >= 3 else None
        b1 = self.buf[0] if len(self.buf) >= 1 else None
        b2 = self.buf[1] if len(self.buf) >= 2 else None
        b3 = self.buf[2] if len(self.buf) >= 3 else None
        K = self.KEYS
        gt_tok = self.example.gt_tokens
        if s1: tokens[K['s1w']] = gt_tok[s1][0]; tokens[K['s1pos']] = gt_tok[s1][1]; 
        if s2: tokens[K['s2w']] = gt_tok[s2][0]; tokens[K['s2pos']] = gt_tok[s2][1]; 
        if s3: tokens[K['s3w']] = gt_tok[s3][0]; tokens[K['s3pos']] = gt_tok[s3][1]; 
        if b1: tokens[K['b1w']] = gt_tok[b1][0]; tokens[K['b1pos']] = gt_tok[b1][1]; 
        if b2: tokens[K['b2w']] = gt_tok[b2][0]; tokens[K['b2pos']] = gt_tok[b2][1]; 
        if b3: tokens[K['b3w']] = gt_tok[b3][0]; tokens[K['b3pos']] = gt_tok[b3][1]; 
        return tokens

if __name__ == '__main__':
    examples = DeptreeExample.load_examples_from_file('ds/deptrees/dev')
    n_valid = 0
    for ex in examples:
        search = DeptreeSearchState(ex)
        try:
            for state, decision in search.iter_oracle_states():
                state.extract_tokens()
        except:
            pass
