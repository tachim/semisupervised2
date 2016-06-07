import numpy as np
import itertools
from sklearn.feature_extraction import DictVectorizer

from collections import defaultdict as dd
import rtk
import bw

class Generative(object):
    def __init__(self, latent_states, clusters, word_to_cluster):
        self.word_to_cluster = word_to_cluster
        self.conditional = dict(
            (ls, dict((k, np.random.rand()) for k in clusters))
            for ls in latent_states
            )
        for z in self.conditional:
            norm = np.sum(self.conditional[z].values())
            for cluster in self.conditional[z]:
                self.conditional[z][cluster] /= norm

    def px_given_z(self, x, z):
        cluster = self.word_to_cluster.get(x, x)
        return self.conditional[z][cluster]

    def coordinate_ascent(self, crf):
        """ Perform EM on the CRF / generative model pair. """
        # z -> x -> [ log q(zi | xi) ]
        weights = dd(lambda: dd(list))
        for (i, latent_state), logq in crf.iter_latent():
            weights[latent_state[-1]][crf.sentence[i+len(latent_state)-1]].append(logq)
        for z, d in weights.iteritems():
            xs = sorted(d.keys())
            vals = [rtk.m.logsumexp(weights[z][x]) for x in xs]
            prbs = rtk.m.normalize_logprobs(vals)
            for x, prb in zip(xs, prbs):
                self.conditional[z][x] = prb

class HMMGenerative(object):
    def __init__(self, latent_states, clusters, word_to_cluster):
        self.cluster2ind = dict((c, i) for i, c in enumerate(clusters))
        self.latent2ind = dict((ls, i) for i, ls in enumerate(latent_states))

        self.word_to_cluster = word_to_cluster
        self.nhidden = len(self.latent2ind)

        self.A0 = np.random.random(self.nhidden)
        self.A = np.random.random((self.nhidden, self.nhidden))
        self.B = np.random.random((self.nhidden, len(self.cluster2ind)))
        self.A0 /= self.A0.sum()
        self.A /= self.A.sum(axis=1)[:, None]
        self.B /= self.B.sum(axis=1)[:, None]

    def cluster_ind(self, word):
        return self.cluster2ind[self.word_to_cluster.get(word, word)]

    def update_dp_tables(self, sequence):
        self.sequence = [self.cluster2ind[self.word_to_cluster.get(w, w)] for w in sequence]
        self.log_px, self.log_alphas, self.log_betas = \
                bw.forward_backward(self.A0, self.A, self.B, self.sequence)

    def logpzx(self, i, latent_states):
        latent_states = [self.latent2ind[ls] for ls in latent_states]
        logprb = self.log_alphas[latent_states[0], i]
        for j in xrange(1, len(latent_states)):
            logprb += np.log(self.A[latent_states[j-1], latent_states[j]])
            logprb += np.log(self.B[latent_states[j], self.sequence[j]])
        return logprb

    def compute_gradient(self, crf):
        self.update_dp_tables(crf.sentence)

        g_A0 = np.zeros(self.nhidden)
        g_A = np.zeros((self.nhidden, self.nhidden))
        g_B = np.zeros((self.nhidden, len(self.cluster2ind)))

        def accumulate_logprbs(latent_states, logprbs, fcn):
            ls2logprbs = dd(list)
            for ls, logprb in zip(latent_states, logprbs):
                ls2logprbs[fcn(ls)].append(logprb)
            ls_logprbs = [(ls, rtk.m.logsumexp(logprbs)) for ls, logprbs in ls2logprbs.iteritems()]
            return ls_logprbs

        accumulate_single = lambda ls, lp: accumulate_logprbs(ls, lp, lambda l: l[0])
        accumulate_two_first = lambda ls, lp: accumulate_logprbs(ls, lp, lambda l: (l[0], l[1]))
        accumulate_two_last = lambda ls, lp: accumulate_logprbs(ls, lp, lambda l: (l[1], l[2]))

        ret = 0
        for i, latent_states in crf._i_chunked_iter_latent():
            latent_states, logprbs = zip(*latent_states)
            if i == 0:
                for ls, logprb in accumulate_single(latent_states, logprbs):
                    g_A0[self.latent2ind[ls]] += np.exp(logprb) / self.A0[self.latent2ind[ls]]
                    g_B[self.latent2ind[ls], self.cluster_ind(crf.sentence[0])] += \
                            np.exp(logprb) / self.B[self.latent2ind[ls], self.cluster_ind(crf.sentence[0])]
                for (z1, z2), logprb in accumulate_two_first(latent_states, logprbs):
                    z1, z2 = map(self.latent2ind.get, [z1, z2])
                    g_A += np.exp(logprb) / self.A[z1, z2]
                    g_B += np.exp(logprb) / self.B[z2, self.cluster_ind(crf.sentence[1])]

            for (z1, z2), logprb in accumulate_two_last(latent_states, logprbs):
                z1, z2 = map(self.latent2ind.get, [z1, z2])
                g_A += np.exp(logprb) / self.A[z1, z2]
                g_B += np.exp(logprb) / self.B[z2, self.cluster_ind(crf.sentence[i+1])]

        return g_A0, g_A, g_B

    def batch_update(self, crf, sentences):
        N = len(sentences)
        g_A0 = np.zeros(self.A0.shape)
        g_A = np.zeros(self.A.shape)
        g_B = np.zeros(self.B.shape)

        for i, sentence in enumerate(sentences):
            crf.update_sentence(sentence)
            g_A0_tmp, g_A_tmp, g_B_tmp = self.compute_gradient(crf)
            g_A0 += g_A0_tmp
            g_A += g_A_tmp
            g_B += g_B_tmp

        THRESHOLD = 0.1

        g_A0 = np.minimum(THRESHOLD, np.maximum(-THRESHOLD, g_A0 / N))
        g_A = np.minimum(THRESHOLD, np.maximum(-THRESHOLD, g_A / N))
        g_B = np.minimum(THRESHOLD, np.maximum(-THRESHOLD, g_B / N))

        self.A0 = np.maximum(1e-5, self.A0 + g_A0)
        self.A = np.maximum(1e-5, self.A + g_A)
        self.B = np.maximum(1e-5, self.B + g_B)
        self.A0 /= self.A0.sum()
        self.A /= self.A.sum(axis=1)[:, None]
        self.B /= self.B.sum(axis=1)[:, None]

class CRFAE(object):
    #@classmethod
    def feature_functions(self):
        return {
                1: [
                    lambda i, x, z1: ('single', x[i], z1),
                    #lambda i, x, z1: ('single', self.fgen.hypthen(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.digit(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.prefix2(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.prefix3(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.suffix2(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.suffix3(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.freq(x[i]), z1),
                    #lambda i, x, z1: ('single', self.fgen.shape(x[i]), z1),
                    ],
                3: [
                    lambda i, x, z1, z2, z3: ('triple', z1, z2, z3),
                    ],
                }
    
    def __init__(self, feature_generator, latent_states=range(12), window_size=3):
        self.weights = dd(float)
        self.fgen = feature_generator
        self.latent_states = latent_states
        self.window_size = window_size

    def update_sentence(self, sentence):
        assert isinstance(sentence, list)
        self.sentence = sentence
        self.forward_partition = self._compute_forward_partition()
        self.backward_partition = self._compute_backward_partition()
        self.feature_cache = {}

    def _forward_latent_iter(self, window_size):
        assert window_size <= self.window_size
        for i in xrange(len(self.sentence)-self.window_size+1):
            for latent in itertools.product(self.latent_states, repeat=self.window_size):
                yield i, latent

    def _backward_latent_iter(self, window_size):
        assert window_size <= self.window_size
        for i in xrange(len(self.sentence)-self.window_size, -1, -1):
            for latent in itertools.product(self.latent_states, repeat=self.window_size):
                yield i, latent

    def _compute_backward_partition(self, unary_weight_function=None, maxprod=False):
        '''
        Computes DP table backward_partition[ind][seq] where index is the index of the last
        word in the three part sequecnce and seq is a pos tuple assignment
        OUTPUT : when unary=None : [ind][seq] = log sum_{1:ind} \lambda * features
                 when unary!=None : [ind][seq] = log sum_{1:ind} \lambda * features + \theta_x|z
        '''

        backward_partition = dd(lambda: dd(float))

        def _unary(ind, sentence, z):
            return unary_weight_function(ind, sentence, z) \
                    if unary_weight_function is not None \
                    else 0

        for i, latent_states in self._forward_latent_iter(self.window_size):
            logprevsums = []
            for prev_latent in self.latent_states:
                logprevsum = backward_partition.get(i-1, {})\
                        .get((prev_latent,) + latent_states[:-1])
                if logprevsum is not None:
                    logprevsums.append(logprevsum)
            if maxprod:
                logprevsum = np.max(logprevsums) if logprevsums else 0
            else:
                logprevsum = rtk.m.logsumexp(logprevsums) if logprevsums else 0

            for fcn in self.feature_functions()[1]:
                if i == 0:
                    logprevsum += sum(
                            self.weights[fcn(i + j, self.sentence, z)]
                            for j, z in enumerate(latent_states))
                else:
                    logprevsum += self.weights[fcn(i+self.window_size-1, self.sentence, latent_states[-1])]

            # Generative weights
            if i == 0:
                logprevsum += sum((_unary(i+j, self.sentence, z) 
                                   for j, z in enumerate(latent_states)))
            else:
                logprevsum += _unary(i+self.window_size-1, self.sentence, latent_states[-1])

            for fcn in self.feature_functions()[self.window_size]:
                logprevsum += self.weights[fcn(i, self.sentence, *latent_states)]
            backward_partition[i][latent_states] = logprevsum

        return backward_partition

    def _compute_forward_partition(self, unary_weight_function=None):
        '''
        Same as _compute_backward_partition except in other direction and without unary calc
        OUTPUT : [ind][seq] = log sum_{ind:end} \lambda * features
        '''

        forward_partition = dd(lambda: dd(float))

        def _unary(ind, sentence, z):
            return unary_weight_function(ind, sentence, z) \
                    if unary_weight_function is not None \
                    else 0

        for i, latent_states in self._backward_latent_iter(self.window_size):
            lognextsums = []
            for next_latent in self.latent_states:
                lognextsum = forward_partition.get(i+1, {}) \
                        .get(latent_states[1:] + (next_latent,))
                if lognextsum is not None:
                    lognextsums.append(lognextsum)
            lognextsum = rtk.m.logsumexp(lognextsums) if lognextsums else 0

            for fcn in self.feature_functions()[1]:
                if i == len(self.sentence)-self.window_size:
                    lognextsum += sum(
                            self.weights[fcn(i + j, self.sentence, z)]
                            for j, z in enumerate(latent_states)
                            )
                else:
                    lognextsum += self.weights[fcn(i, self.sentence, latent_states[0])]

            # Generative weights
            if i == len(self.sentence)-self.window_size:
                lognextsum += sum((_unary(i+j, self.sentence, z) 
                                   for j, z in enumerate(latent_states)))
            else:
                lognextsum += _unary(i, self.sentence, latent_states[0])

            for fcn in self.feature_functions()[self.window_size]:
                lognextsum += self.weights[fcn(i, self.sentence, *latent_states)]
            forward_partition[i][latent_states] = lognextsum
        return forward_partition

    def logZ(self):
        logZs = []
        for latent_states in itertools.product(self.latent_states, repeat=self.window_size):
            logZ_config, _ = self.logZ_config((0, latent_states))
            logZs.append(logZ_config)
        return rtk.m.logsumexp(logZs)

    def logZ_config(self, (i, latent_states)):
        # Log partition function for all assignments to latent variables
        # that contain latent_states starting at position i
        assert len(latent_states) == self.window_size
        assert 0 <= i <= len(self.sentence) - self.window_size
        
        logZ = self.forward_partition[i][latent_states] \
                + self.backward_partition[i][latent_states]

        # When multiplying the two together we double multiplied the features
        # for the current window. So subtract the log weights since we're operating
        # in log space.
        window_logweight = sum(
                [self.weights[k] * v 
                 for k, v in self.extract_window_features((i, latent_states)).iteritems()
                 ])

        return logZ - window_logweight, window_logweight

    def extract_window_features(self, (i, latent_states)):
        if (i, latent_states) not in self.feature_cache:
            ret = dd(float)
            for fcn in self.feature_functions()[1]:
                for j, z in enumerate(latent_states):
                    ret[fcn(i + j, self.sentence, z)] += 1
            for fcn in self.feature_functions()[self.window_size]:
                ret[fcn(i, self.sentence, *latent_states)] += 1
            self.feature_cache[(i, latent_states)] = ret
        return self.feature_cache[(i, latent_states)]

    def iter_latent(self):
        logZ = self.logZ()
        for i in xrange(len(self.sentence)-self.window_size+1):
            for latent_state in itertools.product(self.latent_states, repeat=self.window_size):
                logZ_config, window_logweight = self.logZ_config((i, latent_state))
                logq = logZ_config - logZ
                assert logq <= 1e-8, logq
                yield (i, latent_state), logq

    
    def iter_latent_generative(self, generative_model):
        def logZ_config((i, latent_states)):
            window_logweight = sum([self.weights[k] * v 
                 for k, v in self.extract_window_features((i, latent_states)).iteritems()])
            window_logweight += sum((unary(i+j, self.sentence, z) 
                                     for j, z in enumerate(latent_states)))
            logqp = forward_partition[i][latent_states] \
                    + backward_partition[i][latent_states] - window_logweight
            return logqp, window_logweight

        unary = lambda ind, sentence, z: np.log(generative_model.px_given_z( self.sentence[ind], z))
        forward_partition = self._compute_forward_partition(unary)
        backward_partition = self._compute_backward_partition(unary)

        for i in xrange(len(self.sentence) - self.window_size+1):
            logZs = []
            for latent_states in itertools.product(self.latent_states, repeat=self.window_size):
                lprob, _ = logZ_config((i, latent_states))
                logZs.append(lprob)
            logZ = rtk.m.logsumexp(logZs)

            for latent_states in itertools.product(self.latent_states, repeat=self.window_size):
                logqp, _ = logZ_config((i, latent_states))
                yield (i, latent_states), logqp - logZ
    

    # Computes p(\hat{x} | x)
    def compute_objective(self, generative_model):
        dp_table = self._compute_backward_partition(
                lambda ind, sentence, z: np.log(generative_model.px_given_z(
                    self.sentence[ind], z
                    )))
        last_window_ind = sorted(dp_table.keys())[-1]
        unnormalized_obj = rtk.m.logsumexp([v for _, v in dp_table[last_window_ind].iteritems()])
        return unnormalized_obj - self.logZ()

    def extract_features_for_expectation(self, i, latent_state):
        if i == 0:
            features = self.extract_window_features((i, latent_state))
        else:
            features = dd(float)
            for fcn in self.feature_functions()[1]:
                features[fcn(i + 2, self.sentence, latent_state[-1])] += 1
            for fcn in self.feature_functions()[self.window_size]:
                features[fcn(i, self.sentence, *latent_state)] += 1
        return features

    def compute_crfae_gradient(self, generative_model):
        gradient = dd(float)
        local_expectation = dd(float)

        for (i, latent_state), logqp, in self.iter_latent_generative(generative_model):
            for feature, val in self.extract_features_for_expectation(i, latent_state).iteritems():
                gradient[feature] += val * np.exp(logqp)

        for (i, latent_state), logq in self.iter_latent():
            for feature, val in self.extract_features_for_expectation(i, latent_state).iteritems():
                local_expectation[feature] += np.exp(logq) * val

        for feature in local_expectation:
            gradient[feature] -= local_expectation[feature]
        return gradient

    def compute_supervised_gradient(self, pos):
        pos_signal = dd(float)
        neg_signal = dd(float)

        for i in xrange(len(pos)-self.window_size+1):
            latent_state = tuple(pos[i:i+self.window_size])
            for feature, val in self.extract_features_for_expectation(i, latent_state).iteritems():
                pos_signal[feature] += val

        for (i, latent_state), logq in self.iter_latent():
            for feature, val in self.extract_features_for_expectation(i, latent_state).iteritems():
                neg_signal[feature] += val * np.exp(logq)

        gradient = dd(float)
        for k, v in pos_signal.iteritems(): gradient[k] += v
        for k, v in neg_signal.iteritems(): gradient[k] -= v
        return gradient

    def _i_chunked_iter_latent(self):
        prev_i = 0
        chunk = []
        for (i, latent_state), logq in self.iter_latent():
            if i != prev_i:
                yield prev_i, chunk
                prev_i = i
                chunk = [(latent_state, logq)]
            else:
                chunk.append((latent_state, logq))
        yield prev_i, chunk

    def compute_vae_objective(self, hmm):
        hmm.update_dp_tables(self.sentence)

        def accumulate_logprbs(latent_states, logprbs, fcn):
            ls2logprbs = dd(list)
            for ls, logprb in zip(latent_states, logprbs):
                ls2logprbs[fcn(ls)].append(logprb)
            ls_logprbs = [(ls, rtk.m.logsumexp(logprbs)) for ls, logprbs in ls2logprbs.iteritems()]
            return ls_logprbs

        accumulate_single = lambda ls, lp: accumulate_logprbs(ls, lp, lambda l: l[0])
        accumulate_two_first = lambda ls, lp: accumulate_logprbs(ls, lp, lambda l: (l[0], l[1]))
        accumulate_two_last = lambda ls, lp: accumulate_logprbs(ls, lp, lambda l: (l[1], l[2]))

        ret = 0
        for i, latent_states in self._i_chunked_iter_latent():
            latent_states, logprbs = zip(*latent_states)
            if i == 0:
                for ls, logprb in accumulate_single(latent_states, logprbs):
                    ret += np.exp(logprb) * (
                        np.log(hmm.A0[hmm.latent2ind[ls]])
                        + np.log(hmm.B[hmm.latent2ind[ls], hmm.cluster_ind(self.sentence[0])])
                        )
                for (z1, z2), logprb in accumulate_two_first(latent_states, logprbs):
                    z1, z2 = map(hmm.latent2ind.get, [z1, z2])
                    ret += np.exp(logprb) * (
                        np.log(hmm.A[z1, z2]) 
                        + np.log(hmm.B[z2, hmm.cluster_ind(self.sentence[1])])
                        )
            for (z1, z2), logprb in accumulate_two_last(latent_states, logprbs):
                z1, z2 = map(hmm.latent2ind.get, [z1, z2])
                ret += np.exp(logprb) * (
                    np.log(hmm.A[z1, z2]) 
                    + np.log(hmm.B[z2, hmm.cluster_ind(self.sentence[i+2])])
                    )
        return ret

    def compute_vae_gradient(self, hmm_gen_model):
        gradient = dd(float)
        hmm_gen_model.update_dp_tables(self.sentence)

        gradient = dd(float)

        """ First do windowsize=1 """
        def single_features():
            def accumulate_single_logprbs(latent_states, logprbs, ind):
                ls2logprbs = dd(list)
                for ls, logprb in zip(latent_states, logprbs):
                    ls2logprbs[ls[ind]].append(logprb)
                ls_logprbs = [(ls, rtk.m.logsumexp(logprbs)) for ls, logprbs in ls2logprbs.iteritems()]
                return ls_logprbs

            for i, latent_states in self._i_chunked_iter_latent():
                latent_states, logprbs = zip(*latent_states)
                assert abs(rtk.m.logsumexp(logprbs)) < 1e-6

                inds = range(self.window_size) if i == 0 else [-1]
                for ind in inds:
                    ls_logprbs = accumulate_single_logprbs(latent_states, logprbs, ind)
                    weights = rtk.m.normalize_logprobs([logq for _, logq in ls_logprbs])
                    for fcn in self.feature_functions()[1]:
                        feature_vals = [fcn(i+ind, self.sentence, ls) for ls, _ in ls_logprbs]
                        expected_fvals = dd(float)
                        for feature, weight in zip(feature_vals, weights):
                            expected_fvals[feature] += weight

                        for (ls, logq), feature in zip(ls_logprbs, feature_vals):
                            gradient[feature] += np.exp(logq) * \
                                    (1 - expected_fvals[feature]) * \
                                    ( hmm_gen_model.logpzx(i+ind, [ls]) - 1 - \
                                    logq \
                                    )

        """ Now do windowsize=self.windowsize """
        def window_features():
            for i, latent_states in self._i_chunked_iter_latent():
                latent_states, logprbs = zip(*latent_states)
                assert abs(rtk.m.logsumexp(logprbs)) < 1e-6
                weights = rtk.m.normalize_logprobs(logprbs)

                for fcn in self.feature_functions()[self.window_size]:
                    feature_vals = [fcn(i, self.sentence, *ls) for ls in latent_states]

                    expected_feature_val = dd(float)
                    for fval, w in zip(feature_vals, weights): expected_feature_val[fval] += w

                    for (ls, logq), feature in zip(zip(latent_states, logprbs), feature_vals):
                        gradient[feature] += np.exp(logq) * \
                                (1 - expected_feature_val[feature]) * \
                                ( hmm_gen_model.logpzx(i, ls) - 1 - \
                                logq \
                                )
        single_features()
        window_features()
        return gradient

    def batch_vae_update(self, sentences, hmm, stepsize=0.001):
        N = len(sentences)
        gradient = dd(float)
        for i, sentence in enumerate(sentences):
            self.update_sentence(sentence)
            exgrad = self.compute_vae_gradient(hmm)
            for g in exgrad:
                gradient[g] += exgrad[g] / float(N)

        gvs = np.array([v for k, v in gradient.iteritems()])
        #print 'GRADIENT STATS:', gvs.min(), gvs.mean(), gvs.max()

        self.gradient_step(stepsize, gradient)

    # Need to implement max product algorithm to get MAP assignment
    def predict(self, generative_model):
        if generative_model is not None:
            dp_table = self._compute_backward_partition(
                    lambda ind, sentence, z: np.log(generative_model.px_given_z(
                        self.sentence[ind], z
                        )), maxprod=True)
        else:
            dp_table = self._compute_backward_partition(maxprod=True)

        for i in xrange(len(self.sentence)-self.window_size, -1, -1):
            if i == (len(self.sentence) - self.window_size):
                last_table = dp_table[i]
                keys = list(last_table.keys())
                values = list(last_table.values())
                assignment = keys[values.index(np.max(values))]
                maxval = np.max(values)
            else:
                curr_table = dp_table[i]
                keys = [k for k in curr_table if k[-2:] == assignment[:2]]
                values = [curr_table[k] for k in keys]
                prev_latent = keys[values.index(np.max(values))][0]
                assignment = (prev_latent,) + assignment

        return assignment, maxval - self.logZ()


    def batch_crfae_update(self, sentences, generative_model, stepsize=0.001):
        N = len(sentences)
        gradient = dd(float)
        for i, sentence in enumerate(sentences):
            self.update_sentence(sentence)
            exgrad = self.compute_crfae_gradient(generative_model)
            for g in exgrad:
                gradient[g] += exgrad[g] / float(N)

        gvs = np.array([v for k, v in gradient.iteritems()])
        #print 'GRADIENT STATS:', gvs.min(), gvs.mean(), gvs.max()

        self.gradient_step(stepsize, gradient)

    def batch_supervised_update(self, sentences, pos, stepsize=0.001):
        N = len(sentences)
        gradient = dd(float)
        for i, sentence in enumerate(sentences):
            self.update_sentence(sentence)
            exgrad = self.compute_supervised_gradient(pos[i])
            for g in exgrad:
                gradient[g] += exgrad[g] / float(N)

        gvs = np.array([v for k, v in gradient.iteritems()])
        #print 'GRADIENT STATS:', gvs.min(), gvs.mean(), gvs.max()

        self.gradient_step(stepsize, gradient)

    def gradient_step(self, stepsize, g):
        for k, v in g.iteritems():
            self.weights[k] += stepsize * v
        self.forward_partition = self._compute_forward_partition()
        self.backward_partition = self._compute_backward_partition()

    def numeric_gradient(self, generative_model):
        gradient = {}
        h = 1e-8
        for feature in self.weights:
            self.weights[feature] -= h
            self.forward_partition = self._compute_forward_partition()
            self.backward_partition = self._compute_backward_partition()
            low = self.compute_objective(generative_model)

            self.weights[feature] += 2* h
            self.forward_partition = self._compute_forward_partition()
            self.backward_partition = self._compute_backward_partition()
            high = self.compute_objective(generative_model)

            self.weights[feature] -= h
            self.forward_partition = self._compute_forward_partition()
            self.backward_partition = self._compute_backward_partition()
            gradient[feature] = (high - low) / (2 * h)
        return gradient
