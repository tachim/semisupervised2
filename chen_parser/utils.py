import numpy as np
from collections import defaultdict

class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, count=0)

    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def add_words(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)

def data_iterator(orig_X, orig_y=None, batch_size=32, shuffle=False):
    # Optionally shuffle the data before training
    if shuffle:
        indices = np.random.permutation(len(orig_X))
        data_X = orig_X[indices]
        data_y = orig_y[indices] if orig_y else None
    else:
        data_X = orig_X
        data_y = orig_y
    ###
    total_processed_examples = 0
    total_steps = int(np.ceil(len(data_X) / float(batch_size)))
    for step in xrange(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        x = data_X[batch_start:batch_start + batch_size]
        y = data_y[batch_start:batch_start + batch_size] if data_y else None
        # Convert our target from the class index to a one hot vector
        # y = None
        # if np.any(data_y):
        #     y_indices = data_y[batch_start:batch_start + batch_size]
        #     y = np.zeros((len(x), label_size), dtype=np.int32)
        #     y[np.arange(len(y_indices)), y_indices] = 1
        ###
        yield x, y
        total_processed_examples += len(x)