from CRFAE_var import CRFAE, Generative
from CRFAE_orig_test import CRFAE as CRFAE_test
from rtk.tests import *

from Reader import Reader, FeatureGenerator
from os.path import join

from collections import defaultdict as dd
import numpy as np

# File paths
ptbfile = join('ds','deptrees','dev')
unvfile = join('universal-pos-tags', 'en-ptb.map')
brownfile = join('output','paths')

reader = Reader(ptbfile, unvfile, brownfile, 3)
fgen = FeatureGenerator(brownfile)

def test_logZ():
    crf = CRFAE(fgen, latent_states=range(2), window_size=3)
    test = CRFAE_test(fgen, latent_states=range(2), window_size=3)

    sentence = 'this is a test'.split()
    crf.update_sentence(sentence)
    test.update_sentence(sentence)
    assert_eq(test.logZ(), crf.logZ(), eps=1e-6)

    crf.weights = dd(lambda: 1)
    crf.update_sentence(sentence)
    test.weights = dd(lambda: 1)
    test.update_sentence(sentence)
    assert_eq(test.logZ(), crf.logZ(), eps=1e-6)
    
    logZ_config, _ = crf.logZ_config((1, (0,0,0)))
    test_config = test.logZ_config((1, (0,0,0)))
    print 'CRF : %f Test : %f' % (logZ_config, test_config)
    assert_eq(test_config, logZ_config, eps=1e-6)

def test_objective():
    crf = CRFAE(fgen, latent_states=range(2), window_size=3)
    test = CRFAE_test(fgen, latent_states=range(2), window_size=3)

    sentence = 'this is a test'.split()
    generative = Generative(latent_states=range(2), 
                            clusters=sentence,
                            word_to_cluster=reader.word_to_cluster)

    crf.update_sentence(sentence)
    test.update_sentence(sentence)
    assert_eq(test.compute_objective(generative), crf.compute_objective(generative), eps=1e-6)


def test_predict():
    crf = CRFAE(fgen, latent_states=range(2), window_size=3)
    test = CRFAE_test(fgen, latent_states=range(2), window_size=3)

    sentence = 'this is a test'.split()
    generative = Generative(latent_states=range(2), 
                            clusters=sentence,
                            word_to_cluster=reader.word_to_cluster)

    crf.update_sentence(sentence)
    test.update_sentence(sentence)
    pred_crf = crf.predict(generative)
    pred_test = test.predict(generative)

    assert(pred_test[0] == pred_crf[0])
    assert_eq(pred_test[1], pred_crf[1], eps=1e-6)


def test_gradient():
    crf = CRFAE(fgen, latent_states=range(2), window_size=3)
    test = CRFAE_test(fgen, latent_states=range(2), window_size=3)

    sentence = 'this is a test'.split()
    generative = Generative(latent_states=range(2), 
                            clusters=sentence,
                            word_to_cluster=reader.word_to_cluster)

    crf.update_sentence(sentence)
    test.update_sentence(sentence)
    
    crf_grad =  crf.compute_crfae_gradient(generative)
    test_grad = test.compute_gradient(generative)
    num_grad = test.numeric_gradient(generative)

    print 'MISSING KEYS:', set(test_grad.keys()) - set(crf_grad.keys())

    '''
    print 'Numerical Gradient differences...'
    for key in num_grad:
        diff = np.abs(num_grad[key] - test_grad[key])
        print '%s : \t\t %f' % (key, diff / np.abs(test_grad[key]))
    '''

    assert_eq(len(crf_grad.keys()), len(test_grad.keys()), eps=1e-8)
    assert sorted(crf_grad.keys()) == sorted(test_grad.keys())
    for key in sorted(crf_grad):
        #diff = np.abs()
        #print '%s : \t\t %f' % (key, diff / np.abs(test_grad[key]))
        assert_eq(crf_grad[key], test_grad[key], eps=1e-6)
        assert_eq(num_grad[key], test_grad[key], eps=1e-6)
    


'''
def test_state():
    crf = CRFAE(fgen, latent_states=range(2), window_size=3)

    crf.weights = dd(lambda: 1)
    crf.update_sentence('this is a test sentence'.split())
    assert_eq(np.log(32 * np.exp(8)), crf.logZ(), eps=1e-6)

    for (i, latent_state), logq in crf.iter_latent():
        assert_eq(8., 1/np.exp(logq), eps=1e-6)

def test_gradient():
    sentence = 'this is a test sentence'
    crf = CRFAE(fgen, latent_states=range(2), window_size=3)

    crf.weights = dd(lambda: 0)
    crf.update_sentence(sentence.split())
    assert_eq(np.log(32), crf.logZ(), eps=1e-6)

    generative_model = Generative(crf.latent_states, sentence.split())

    print crf.compute_objective(generative_model)

    learning_rate = 0.001
    for i in xrange(10):
        g = crf.compute_gradient(generative_model)
        crf.gradient_step(learning_rate, g)
        generative_model.coordinate_ascent(crf)
        print crf.compute_objective(generative_model)
        print 'log partition:', crf.logZ()


    print generative_model.conditional

    assert crf.compute_objective(generative_model) <= 0
    assert False # for debugging
'''
