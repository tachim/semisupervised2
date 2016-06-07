#!/usr/bin/env python

import sys
import os
import time
import pickle
import math
from copy import deepcopy
import numpy as np
from scipy.misc import logsumexp
import heapq
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from utils import data_iterator, Vocab
import deptree
from dep_gen import DepGen


def unlabaled(parents):
    res = dict()
    for child, (parent, label) in parents.iteritems():
        res[child] = parent
    return res

class DepNet(object):
    def __init__(self, n_inputs, X_vocab, y_vocab, hidden_dim=200, embed_dim=50, l2=1e-8, lr=1e-2, batch_size=512):
        self.n_inputs = n_inputs
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.X_vocab = X_vocab
        self.y_vocab = y_vocab
        self.n_outputs = len(y_vocab)
        self.l2 = l2
        self.lr = lr
        self.batch_size=batch_size
        self.depgen = DepGen('ds/deptrees/train')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.input_placeholder = tf.placeholder(tf.int32, (None, self.n_inputs), 'inputs')
        self.labels_placeholder = tf.placeholder(tf.int64, (None, ), 'labels')

        # self.beamfeats_placeholder = tf.placeholder(tf.int32, (None, None, self.n_inputs))#beam_size * sent_len * n_inputs
        # self.beamactions_placeholder = tf.placeholder(tf.int32, (None, None))

        self.build_model()
        self.predictions()
        self.setup_loss()
        self.training()

        self.saver = tf.train.Saver(tf.all_variables())

    def build_model(self):
        embedding = tf.get_variable('embedding', [len(self.X_vocab), self.embed_dim])
        x = tf.nn.embedding_lookup(embedding, self.input_placeholder)
        x = tf.reshape(x, [-1, self.n_inputs * self.embed_dim])

        with tf.variable_scope('Cubic'):
            W = tf.get_variable('W', [self.embed_dim * self.n_inputs, self.hidden_dim])
            b = tf.get_variable('b', [1, self.hidden_dim])
            z = tf.matmul(x, W) + b
            a = tf.tanh(z)
            tf.add_to_collection('l2_loss', tf.nn.l2_loss(W))
        with tf.variable_scope('Softmax'):
            W = tf.get_variable('W', [self.hidden_dim, self.n_outputs])
            b = tf.get_variable('b', [1, self.n_outputs])
            self.logits = tf.matmul(a, W) + b
            self.probs = tf.nn.softmax(self.logits)
            self.logprobs = tf.log(self.probs)
            tf.add_to_collection('l2_loss', tf.nn.l2_loss(W))

    def predictions(self):
        self.predictions = tf.argmax(self.logits, 1)
        correct_prediction = tf.equal(self.predictions, self.labels_placeholder)
        self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))

    def setup_loss(self):
        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.labels_placeholder))
        l2_loss = tf.add_n(tf.get_collection('l2_loss'))
        self.loss = ce_loss + self.l2 * l2_loss
        
    def training(self):
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        #self.grads = self.opt.compute_gradients(self.loss)
        self.grads = [(tf.convert_to_tensor(g[0]), g[1]) for g in self.opt.compute_gradients(self.loss)]

        self.grad_placeholder = [(tf.placeholder("float", g[1].get_shape(), 'grads'), g[1]) for g in self.grads]
        self.apply_grad_op = self.opt.apply_gradients(self.grad_placeholder, global_step=self.global_step)

    # def manual_gradients(self):
    #     self.grads = self.opt.compute_gradients(self.loss)

    def create_feed_dict(self, input_batch, label_batch=None):
        feed_dict = {
            self.input_placeholder: input_batch,
        }
        if label_batch is not None:
            feed_dict[self.labels_placeholder] = label_batch
        return feed_dict

    def predict(self, session, X, y=None):
        losses = []
        results = []
        for step, (x, y) in enumerate(data_iterator(X, y, batch_size=self.batch_size, shuffle=False)):
            feed = self.create_feed_dict(input_batch=x, label_batch=y)
            if np.any(y):
                loss, preds = session.run(
                        [self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)
            else:
                preds = session.run(self.predictions, feed_dict=feed)
            results.extend(preds)
        return results, np.mean(losses)

    @profile
    def beam_search(self, session, tokens, beam_size=1, backward=False):
        beam = [([], deptree.DeptreeSearchState.from_tokens(tokens), [])]#prob, state, grads_list
        for step in xrange(2 * len(tokens)):
            X = []
            for _, state, _ in beam:
                X.append([self.X_vocab.encode(f) for f in state.extract_features()])
            feed = self.create_feed_dict(input_batch=X)
            logprobs = session.run(self.logprobs, feed_dict=feed)
            newbeam = []
            for b, ray in enumerate(beam):
                probs, state, grads = ray
                for a in xrange(len(logprobs[0])):
                    action = self.y_vocab.decode(a)
                    if not state.can_apply_action(action): continue
                    newprob = sum(probs, logprobs[b,a])
                    if len(newbeam)>0 and sum(newbeam[0][0])<newprob:
                        heapq.heappop(newbeam)
                    if len(newbeam) < beam_size:
                        newray = (probs+[logprobs[b,a]], state, grads, action)
                        heapq.heappush(newbeam, newray)

            beam = []
            for probs, state, grads, action in newbeam:
                x = [self.X_vocab.encode(f) for f in state.extract_features()]
                feed = self.create_feed_dict(input_batch=[x], label_batch=[self.y_vocab.encode(action)])
                grad = session.run([g[0] for g in self.grads], feed_dict=feed) if backward else None
                grads.append(grad)
                state = deptree.DeptreeSearchState.from_state(state)
                state.apply_action(action)
                beam.append((probs, state, grads))

        beam.sort(reverse=True)
        # print unlabaled(tree.parents)
        # for prob, state in beam:
        #     assert state.is_done()
        #     print state.compute_metric(tree), prob
        #     print unlabaled(state.parents)
        return beam
        # Z = tf.reduce_sum([prob for prob, _ in beam])
        # self.varobj = tf.reduce_sum([prob*(self.depgen.tree_prob(state.children) - tf.log(prob)) for prob, state, in beam])/Z


    def unsup_update(self, session, tree):
        start = time.time()
        beam = self.beam_search(session, tree.tokens, beam_size=10, backward=True)
        @profile
        def actual_update():
            logQ = [sum(logprobs) for logprobs, _, _ in beam]
            logZ = logsumexp(logQ)
            logP = [self.depgen.tree_logprob(state.parents) for _, state, _ in beam]
            grads = [np.zeros_like(g, dtype=np.float) for g in beam[0][2][0]]
            var_objective = 0.0
            for b, ray in enumerate(beam):
                var_objective += math.exp(logQ[b]-logZ) * (logP[b] - logQ[b])
                logprobs, _, ray_grads = ray
                for i in xrange(len(ray_grads)):
                    mult = math.exp(sum(logprobs[:i]) + sum(logprobs[i+1:]) - logZ) * (logP[b] - logQ[b] + 1)
                    for g in xrange(1,len(ray_grads[0])):
                        grads[g] += mult*ray_grads[i][g]
            feed = {}
            for i in xrange(len(self.grad_placeholder)):
                feed[self.grad_placeholder[i][0]] = grads[i]
            session.run(self.apply_grad_op, feed_dict=feed)
            best_state = beam[0][1]
            return var_objective, best_state.compute_metric(tree)
        print 'MY TIME = %f'%(time.time()-start)
        return actual_update()


    # def beam_search(self, session, tree, beam_size=1):
    #     beam = [(0.0, deptree.DeptreeSearchState.from_tokens(tree.tokens))]
    #     for _ in xrange(2 * len(tree.tokens)):
    #         X = []
    #         for _, state in beam:
    #             X.append([self.X_vocab.encode(f) for f in state.extract_features()])
    #         feed = self.create_feed_dict(input_batch=X)
    #         logprobs = session.run(self.logprobs, feed_dict=feed)
    #         newbeam = []
    #         for b, ray in enumerate(beam):
    #             prob, state = ray
    #             for a in xrange(len(logprobs[0])):
    #                 action = self.y_vocab.decode(a)
    #                 if not state.can_apply_action(action): continue
    #                 newprob = prob + logprobs[b,a]
    #                 newray = (newprob, state, action)
    #                 if len(newbeam) < beam_size:
    #                     heapq.heappush(newbeam, newray)
    #                 else:
    #                     heapq.heappushpop(newbeam, newray)
    #         beam = []
    #         for prob, state, action in newbeam:
    #             state = deepcopy(state)
    #             state.apply_action(action)
    #             beam.append((prob, state))
    #     beam.sort(reverse=True)
    #     # print unlabaled(tree.parents)
    #     # for prob, state in beam:
    #     #     assert state.is_done()
    #     #     print state.compute_metric(tree), prob
    #     #     print unlabaled(state.parents)
    #     return beam

    def run_unsupervised_epoch(self, session, trees, verbose=True):
        var_objective_history = []
        total_correct_examples, total_processed_examples = 0, 0
        for step in xrange(len(trees)):
            #profile
            import profile, pstats
            profiler = profile.Profile()
            result = profiler.runcall(self.unsup_update, session, trees[step])
            stats = pstats.Stats(profiler).strip_dirs()
            stats.sort_stats('cumtime').print_stats(20)

            var_objective, (num, denom) = self.unsup_update(session, trees[step])
            var_objective_history.append(var_objective)
            total_processed_examples += denom
            total_correct_examples += num
            sys.exit(0)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : var_objective = {}'.format(
                        step, len(trees), np.mean(var_objective)))
                sys.stdout.flush()
        if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()
        return var_objective_history, total_correct_examples / float(total_processed_examples)


    def run_epoch(self, session, X, y, trees, shuffle=False, verbose=True):
        loss_history = []
        var_objective_history = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(X) / self.batch_size
        for step, (X_batch, y_batch) in enumerate(data_iterator(X, y, batch_size=self.batch_size, shuffle=shuffle)):
            #supervised
            feed = self.create_feed_dict(input_batch=X_batch, label_batch=y_batch)
            loss, total_correct, _ = session.run(
                    [self.loss, self.correct_predictions, self.train_op],
                    feed_dict=feed)
            total_processed_examples += len(X_batch)
            total_correct_examples += total_correct
            loss_history.append(loss)

            #unsupervised
            var_objective, accuracy = self.unsup_update(trees[step])
            var_objective_history.append(var_objective)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}, var_objective = {}'.format(
                        step, total_steps, np.mean(loss_history), np.mean(var_objective)))
                sys.stdout.flush()
        if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()
        return loss_history, total_correct_examples / float(total_processed_examples)



def prep_data(trees, X_vocab=None, y_vocab=None):
    update_vocab = False
    if X_vocab is None:
        X_vocab, y_vocab = Vocab(), Vocab()
        update_vocab = True
    X, y = [], []
    for tree in tqdm(trees):
        if len(tree.tokens) < 2: continue
        #TODO accumulate features without iterating over all states
        try:
            for state, decision in tree.iter_oracle_states():
                feats = state.extract_features()
                if update_vocab:
                    X_vocab.add_words(feats)
                    y_vocab.add_word(decision)
                X.append([X_vocab.encode(f) for f in feats])
                y.append(y_vocab.encode(decision))
        except:
            pass
    return X, y, X_vocab, y_vocab

def train_DepNet():
    train_trees = deptree.DeptreeExample.load_examples_from_file('ds/deptrees/train')
    train_trees.sort(key= lambda t:len(t.tokens))
    train_trees = train_trees[1000:1100]
    dev_trees = deptree.DeptreeExample.load_examples_from_file('ds/deptrees/dev')[:1]
    X_train, y_train, X_vocab, y_vocab = prep_data(train_trees)
    X_dev, y_dev ,_ ,_ = prep_data(dev_trees, X_vocab, y_vocab)
    pickle.dump((X_vocab, y_vocab), open( "vocabs.p", "wb" ))
    print 'train=%d dev=%d'%(len(train_trees), len(dev_trees))
    print 'xvocab=%d yvocab=%d'%(len(X_vocab), len(y_vocab))

    n_inputs = len(deptree.DeptreeSearchState.KEYS)
    print '#labels=%d'%len(y_vocab)

    with tf.Graph().as_default():
        model = DepNet(n_inputs, X_vocab, y_vocab)#, l2=1e-2, lr=1e-3)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session() as session:
            best_val_loss = float('inf')
            best_val_epoch = 0

            session.run(init)
            train_loss_history = []
            for epoch in xrange(20):
                tic = time.time()
                #train_losses, train_acc = model.run_epoch(session, X_train, y_train)
                train_losses, train_acc = model.run_unsupervised_epoch(session, train_trees)
                val_predictions, val_loss = model.predict(session, X_dev, y_dev)
                toc = time.time()
                epoch_time = toc - tic
                train_loss_history.extend(train_losses)

                print('epoch %d training loss %f training acc %f validation loss %f validation acc %f time %f'%
                    (epoch, np.mean(train_losses), train_acc, val_loss, np.equal(y_dev, val_predictions).mean(), epoch_time))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if not os.path.exists("./weights"):
                        os.makedirs("./weights")
                    saver.save(session, './weights/depnet.ckpt', global_step=model.global_step)

                # if epoch - best_val_epoch > 2:
                #     print 'early stopping'
                #     break
    print len(X_vocab)
    # pickle.dump(train_loss_history, open( "train_loss.p", "wb" ))
    # plt.plot(train_loss_history)
    # plt.title('Training Loss history')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.savefig("loss_history.png")
    # plt.show()

def test_Depnet():
    trees = deptree.DeptreeExample.load_examples_from_file('ds/deptrees/train')
    trees.sort(key= lambda t:len(t.tokens))
    trees = trees[1000:1003]
    X_vocab, y_vocab = pickle.load(open( "vocabs.p", "rb" ))
    n_inputs = len(deptree.DeptreeSearchState.KEYS)
    with tf.Graph().as_default():
        with tf.Session() as session:
            model = DepNet(n_inputs, X_vocab, y_vocab)
            session.run(tf.initialize_all_variables())
            for tree in trees:
                model.beam_search(session, tree, beam_size=4)
                print
            ckpt = tf.train.get_checkpoint_state("./weights")
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                model.saver.restore(session, ckpt.model_checkpoint_path)
            for tree in trees:
                model.beam_search(session, tree, beam_size=4)
                print


if __name__ == "__main__":
    train_DepNet()
    test_Depnet()
