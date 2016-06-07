#!/usr/bin/env python

import collections
from copy import deepcopy

class DeptreeExample(object):
    def __init__(self, tokens, parents=None, children=None):
        self.tokens = tokens
        self.parents = parents
        self.children = children

    def __repr__(self):
        return str(('parents', self.parents, 
            'children', self.children, 
            'tokens', self.tokens))

    def iter_oracle_states(self):
        state = DeptreeSearchState.from_tokens(self.tokens)
        while not state.is_done():
            action = None
            if len(state.stack) >= 2:
                s1, s2 = state.stack[0], state.stack[1]
                if self.parents[s1][0] == s2:
                    if len(state.children[s1]) == len(self.children[s1]):
                        action = 'R-' + self.parents[s1][1]
                elif action is None and s2 != -1 and self.parents[s2][0] == s1:
                    if len(state.children[s2]) == len(self.children[s2]):
                        action = 'L-' + self.parents[s2][1]

            if action is None and len(state.buf) > 0:
                action = 'SHIFT'
            assert action is not None
            yield (state, action)
            state.apply_action(action)

    #TODO include root node as a token so features are extracted properly
    #currently it is stored as -1 so we extract full stop
    @classmethod
    def read_example(cls, f):
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
        return DeptreeExample(tokens, parents, children)

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
ROOT = -1

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

    def __init__(self, tokens, buf, stack, children, parents):
        self.tokens = tokens
        self.buf = buf
        self.stack = stack
        self.children = children
        self.parents = parents

    @classmethod
    def from_tokens(cls, tokens):
        buf = collections.deque(range(len(tokens)))
        stack = collections.deque([-1])
        children = collections.defaultdict(set)
        parents = {}
        return cls(tokens, buf, stack, children, parents)

    @classmethod
    def from_state(cls, state):
        buf = deepcopy(state.buf)
        stack = deepcopy(state.stack)
        children = deepcopy(state.children)
        parents = deepcopy(state.parents)
        return cls(state.tokens, buf, stack, children, parents)
        

    def _add_parent_child(self, parent, child, label):
        self.parents[child] = (parent, label)
        self.children[parent].add(child)

    def compute_metric(self, gt):
        denom = 0
        num = 0
        for child, (parent, label) in self.parents.iteritems():
            tok = gt.tokens[child][0]
            if tok in PUNCTUATION: continue
            denom += 1
            if child != -1 and int(parent == gt.parents[child][0]):
                num += 1
        if denom == 0: 
            return None
        else:
            return num, denom

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


    def extract_features(self):
        """
        s1w=0, s1pos=1,
        s2w=2, s2pos=3,
        s3w=4, s3pos=5,
        b1w=6, b1pos=7,
        b2w=8, b2pos=9,
        b3w=10, b3pos=11,
        """

        # TODO
        # The choice of Sw, St, Sl:
        # Following (Zhang and Nivre, 2011), we pick a
        # rich set of elements for our final parser. In detail,
        # Sw contains nw = 18 elements: (1) The top 3
        # words on the stack and buffer: s1, s2, s3, b1, b2, b3;
        # (2) The first and second leftmost / rightmost
        # children of the top two words on the stack:
        # lc1(si), rc1(si), lc2(si), rc2(si), i = 1, 2. (3)
        # The leftmost of leftmost / rightmost of rightmost
        # children of the top two words on the stack:
        # lc1(lc1(si)), rc1(rc1(si)), i = 1, 2.
        # We use the corresponding POS tags for St
        # (nt = 18), and the corresponding arc labels of
        # words excluding those 6 words on the stack/buffer
        # for Sl (nl = 12). A good advantage of our parser
        # is that we can add a rich set of elements cheaply,
        # instead of hand-crafting many more indicator features.

        feats = ["<NULL>"] * len(self.KEYS)
        s1 = self.stack[0] if len(self.stack) >= 1 else None
        s2 = self.stack[1] if len(self.stack) >= 2 else None
        s3 = self.stack[2] if len(self.stack) >= 3 else None
        b1 = self.buf[0] if len(self.buf) >= 1 else None
        b2 = self.buf[1] if len(self.buf) >= 2 else None
        b3 = self.buf[2] if len(self.buf) >= 3 else None
        K = self.KEYS
        if s1: feats[K['s1w']] = self.tokens[s1][0]; feats[K['s1pos']] = self.tokens[s1][1]; 
        if s2: feats[K['s2w']] = self.tokens[s2][0]; feats[K['s2pos']] = self.tokens[s2][1]; 
        if s3: feats[K['s3w']] = self.tokens[s3][0]; feats[K['s3pos']] = self.tokens[s3][1]; 
        if b1: feats[K['b1w']] = self.tokens[b1][0]; feats[K['b1pos']] = self.tokens[b1][1]; 
        if b2: feats[K['b2w']] = self.tokens[b2][0]; feats[K['b2pos']] = self.tokens[b2][1]; 
        if b3: feats[K['b3w']] = self.tokens[b3][0]; feats[K['b3pos']] = self.tokens[b3][1]; 
        return feats

if __name__ == '__main__':
    with open('ds/deptrees/dev.1','r') as f:
        tree = DeptreeExample.read_example(f)
    
    for state, decision in tree.iter_oracle_states():
        print state.extract_features(), decision
