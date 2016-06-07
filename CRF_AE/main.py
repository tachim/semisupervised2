from CRFAE_var import CRFAE, Generative, HMMGenerative
from rtk.tests import *
from os.path import join

from collections import defaultdict as dd
from Reader import Reader, FeatureGenerator
import numpy as np
import pickle

import rtk
import os

import sklearn.metrics

def gen_pos_remap(reader, K):
    # Keep the top K, remap all the rest to K+1
    import collections
    all_tags = []
    for ex in reader.examples:
        seq_toks = ex.gt_tokens
        seq_wd = [st[0] for st in seq_toks]
        seq_tag = [reader.pos_to_ind(st[1]) for st in seq_toks]
        all_tags.extend(seq_tag)
    tag_counts = sorted(collections.Counter(all_tags).items(), key=lambda (k, v): v, reverse=True)
    top_tags = set([k for k, _ in tag_counts[:K]])
    new_tag = 500

    def pos_remap(pos):
        return [k if k in top_tags else new_tag for k in pos]

    return list(top_tags) + [new_tag], pos_remap

def v_measure(predicted_clusters, gt_labels):
    # predicted_clusters is a list of indices into clusters
    # gt_labels is a list of indices into classes
    # This function returns the V-measure of the clustering w.r.t. those classes.

    all_clusters = set([c for l in predicted_clusters for c in l])
    all_labels = set([label for lis in gt_labels for label in lis ])
    assert len(predicted_clusters) == len(gt_labels)
    assert all(len(l1) == len(l2) for l1, l2 in zip(predicted_clusters, gt_labels))

    predicted_clusters = sum(map(list, predicted_clusters), [])
    gt_labels = sum(gt_labels, [])

    return sklearn.metrics.v_measure_score(gt_labels, predicted_clusters)

def evaluate_v(crf, generative_model, sentences, pos):
    '''
    Evaluation with greedy assignment of latent variables to pos tags
    - Returns a pos tag accuracy across elements - not mean across examples
    '''
    # determine the most frequent
    def opt_mapping(lst, candidates):
        assert len(lst) > 0 and len(candidates) > 0
        max_freq = 0
        max_candidate = candidates[0]
        for c in candidates:
            freq = np.sum([elem for elem in lst if elem == c])
            if freq > max_freq:
                max_freq = freq
                max_candidate = c
        return max_candidate

    N = len(sentences)
    predictions = []
    for i in range(N):
        crf.update_sentence(sentences[i])
        predictions.append(crf.predict(generative_model)[0])

    return v_measure(predictions, pos)

def evaluate_accuracy(crf, generative_model, sentences, pos):
    n_correct = 0
    for i, sentence in enumerate(sentences):
        crf.update_sentence(sentence)
        predictions = crf.predict(generative_model)[0]
        n_correct += sum([1 for pred, true in zip(predictions, pos[i]) if pred == true])
    return float(n_correct) / sum(map(len, sentences))

def evaluate_vae(crf, generative_model, sentences):
    ret = []
    for i, sentence in enumerate(sentences):
        crf.update_sentence(sentence)
        ret.append(crf.compute_vae_objective(generative_model))
    return np.array(ret).mean()

def compute_log_likelihood(crf, generative, sentences):
    ret = 0
    for sentence in sentences:
        crf.update_sentence(sentence)
        ret += crf.compute_objective(generative)
    return ret / len(sentences)

def unsupervised_learning_step(crf, generative, sentences, step_size):
    crf.batch_crfae_update(sentences, generative_model, stepsize=step_size)
    generative_model.coordinate_ascent(crf)

def supervised_learning_step(crf, sentences, pos, step_size):
    crf.batch_supervised_update(sentences, pos, stepsize=step_size)

def main_crfae(MAX_ITERS):
    N_POS = 3
    BATCH_SIZE = 5
    WINDOW_SIZE = 3 # for removal of short sequences
    
    EVAL_CHK = 3
    
    # File paths
    ptbfile = join('ds','deptrees','dev')
    unvfile = join('universal-pos-tags', 'en-ptb.map')
    brownfile = join('output','paths')

    reader = Reader(ptbfile, unvfile, brownfile, WINDOW_SIZE)
    top_tags, pos_remap = gen_pos_remap(reader, N_POS)
    LATENT_STATES = top_tags

    fgen = FeatureGenerator(brownfile)
    all_tokens = set()
    for ex in reader.examples:
        all_tokens |= set([t[0] for t in ex.gt_tokens])

    generative_model = Generative(latent_states=LATENT_STATES, 
                                  clusters=all_tokens,
                                  word_to_cluster=reader.word_to_cluster)
    crf = CRFAE(fgen, latent_states=LATENT_STATES)

    eval_sentences, _, eval_pos = reader.sample(30)
    eval_pos = map(pos_remap, eval_pos)

    learning_rate = 0.01
    train_hist = []
    eval_hist = []
    for it in xrange(MAX_ITERS):
        with rtk.timing.Logger('batch', print_every=20):
            step_size = 100 * 1. / (it + 1)

            sentences, clusters, pos = reader.sample(BATCH_SIZE)
            pos = map(pos_remap, pos)

            if True:
                supervised_learning_step(crf, sentences, pos, step_size)
                print 'Accuracy:', evaluate_accuracy(crf, None, eval_sentences, eval_pos)
            else:
                unsupervised_learning_step(crf, generative_model, sentences, step_size=step_size)
                obj = compute_log_likelihood(crf, generative_model, eval_sentences)
                train_hist.append((it, obj))
                print 'Iter : %d Objective : %f' % (it, obj)

                if it % EVAL_CHK == 0:
                    acc = evaluate_v(crf, generative_model, eval_sentences, eval_pos)
                    print 'Evaluation : %f' % acc
                    eval_hist.append((it, acc))

    # Pickle training history
    with open(join('results', 'unsupervised_data2.pkl'), 'wb') as fid:
        pickle.dump({'train_hist':train_hist, 'eval_hist':eval_hist, 
                     'crf_weights':crf.weights, 'gen_weights':generative_model.conditional}, fid)

@rtk.dist.mgr.distributed
def main_vae():
    rtk.rand.seed(rtk.dist.mgr.p('rand_seed'))

    N_POS = 3
    BATCH_SIZE = 5
    WINDOW_SIZE = 3 # for removal of short sequences
    
    EVAL_CHK = 3
    
    # File paths
    prefix = '/atlas/u/tachim/w/semisupervised/CRF_AE'
    ptbfile = join(prefix, 'ds','deptrees','dev')
    unvfile = join(prefix, 'universal-pos-tags', 'en-ptb.map')
    brownfile = join(prefix, 'output','paths')

    reader = Reader(ptbfile, unvfile, brownfile, WINDOW_SIZE)
    top_tags, pos_remap = gen_pos_remap(reader, N_POS)
    LATENT_STATES = top_tags

    fgen = FeatureGenerator(brownfile)
    all_tokens = set()
    for ex in reader.examples:
        all_tokens |= set([t[0] for t in ex.gt_tokens])

    generative_model = HMMGenerative(latent_states=LATENT_STATES, 
            clusters=all_tokens,
            word_to_cluster=reader.word_to_cluster)
    crf = CRFAE(fgen, latent_states=LATENT_STATES)

    train_sentences, _, train_pos = reader.sample(20)
    train_pos = map(pos_remap, train_pos)
    eval_sentences, _, eval_pos = reader.sample(20)
    eval_pos = map(pos_remap, eval_pos)

    train_hist = []
    eval_hist = []

    stats = []
    for it in xrange(rtk.dist.mgr.p('n_iters')):
        with rtk.timing.Logger('batch', print_every=1):
            step_size = 1. / (it + 1)

            if it < 5:
                supervised_learning_step(crf, train_sentences, train_pos, step_size)
            crf.batch_vae_update(eval_sentences, generative_model, stepsize=step_size)
            generative_model.batch_update(crf, eval_sentences)

            vae_test = evaluate_vae(crf, generative_model, eval_sentences)
            crf_test_accuracy = evaluate_accuracy(crf, None, eval_sentences, eval_pos)
            crf_train_accuracy = evaluate_accuracy(crf, None, train_sentences, train_pos)
            v_measure = evaluate_v(crf, None, eval_sentences, eval_pos)

            print 'VAE on test:', vae_test
            print 'Accuracy on test:', crf_test_accuracy
            print 'Accuracy on train:', crf_train_accuracy
            print 'V measure:', v_measure

            stats.append(dict(
                vae_test=vae_test,
                crf_test_accuracy=crf_test_accuracy,
                crf_train_accuracy=crf_train_accuracy,
                v_measure=v_measure,
                ))
    return stats

if __name__ == '__main__':
    """ Charts:
    1. Accuracy on labeled test set over time
    2. VAE lower bound on test set
    3. VAE lower bound on training set
    """

    if False:
        import profile, pstats, IPython
        profiler = profile.Profile()
        result = profiler.runcall(main_vae, MAX_ITERS=1)
        stats = pstats.Stats(profiler).strip_dirs()
        stats.sort_stats('tot').print_stats(20)
        IPython.embed()
    else:
        with rtk.dist.mgr.Dist(os.path.realpath(__file__)):
            for rand_seed in xrange(40, 80):
                with rtk.dist.mgr.Params(rand_seed=rand_seed, n_iters=12):
                    main_vae()
