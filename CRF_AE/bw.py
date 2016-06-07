import numpy as np
import random

import rtk
from rtk.tests import *

def logsumexp(A):
    logc = max(A)
    return logc + np.log(sum(np.exp(a - logc) for a in A))

def normalize_logprobs(log_probs):
    # Given n unnormalized log weights, return the normalized 
    # version of exp(log_probs). This involves a neat log-space
    # trick to prevent underflow.
    log_probs -= np.min(log_probs)
    np.exp(log_probs, out=log_probs)
    rowsum = log_probs.sum()
    log_probs /= rowsum
    assert_eq(log_probs.sum(), 1, eps=1e-6)
    assert not np.any(np.isnan(log_probs))
    return log_probs

def forward_backward(A0, A, B, sequence):
    # Compute the forward and backward calculations in log-space. Pretty straightforward
    # calculation, but use logsumexp to prevent underflow.
    nhidden = A.shape[0]
    T = len(sequence)

    log_alphas = np.zeros((nhidden, T))
    log_betas = np.zeros((nhidden, T))

    for t in xrange(T):
        for h in xrange(nhidden):
            if t == 0:
                log_alphas[h, t] = np.log(A0[h]) + np.log(B[h, sequence[t]])
            else:
                log_alphas[h, t] = logsumexp([
                    log_alphas[hprime, t-1] + np.log(A[hprime, h]) + np.log(B[h, sequence[t]])
                    for hprime in xrange(nhidden)
                    ])
    for t in xrange(T-2, -1, -1):
        next_betas = log_betas[:, t+1].squeeze()
        for h in xrange(nhidden):
            log_betas[h, t] = logsumexp([
                np.log(A[h, hprime]) + next_betas[hprime] + np.log(B[hprime, sequence[t+1]])
                for hprime in xrange(nhidden)
                ])

    log_beta0s = np.zeros(nhidden)
    for h in xrange(nhidden):
        log_beta0s[h] = log_betas[h, 0] + np.log(B[h, sequence[0]]) + np.log(A0[h])
    assert_eq(logsumexp(log_alphas[:, -1]), logsumexp(log_beta0s), eps=1e-5)
    log_px = logsumexp(log_beta0s)

    return log_px, log_alphas, log_betas

def baumwelch(seq, nhidden, n_iter=10, A=None, B=None, A0=None):
    # Run Baum-Welch on a given sequence, assuming nhidden hidden states
    # and optional initial guesses for A, B, and A0.

    T = len(seq)
    nobs = max(seq) + 1
    A = A if A is not None else np.random.random((nhidden, nhidden))
    B = B if B is not None else np.random.random((nhidden, nobs))
    A0 = A0 if A0 is not None else np.ones(nhidden) / nhidden
    for i in xrange(nhidden):
        A[i] /= A[i].sum()
        B[i] /= B[i].sum()

    # Expected count at timestep t.
    def log_gamma(log_alphas, log_betas, t, i, j):
        log_b = log_betas[j, t+1]
        log_a = log_alphas[i, t]
        log_emission = np.log(B[j, seq[t+1]])
        return log_a + np.log(A[i, j]) + log_emission + log_b
 
    prev_log_px = None

    for iteration in xrange(n_iter):
        _, log_alphas, log_betas = forward_backward(A0, A, B, seq)
        log_g = lambda t, i, j: log_gamma(log_alphas, log_betas, t, i, j)

        new_A = np.zeros(A.shape)
        new_B = np.zeros(B.shape)

        for i in xrange(nhidden):
            for j in xrange(nhidden):
                new_A[i, j] = logsumexp([log_g(t, i, j) for t in xrange(T-1)])
                assert np.isfinite(new_A[i, j]) and not np.isnan(np.exp(new_A[i, j]))
            normalize_logprobs(new_A[i])

        for j in xrange(nhidden):
            for k in xrange(nobs):
                new_B[j, k] = logsumexp([log_g(t, i, j) 
                        for t in xrange(T-1) 
                        for i in xrange(nhidden)
                        if seq[t+1] == k])
                assert np.isfinite(new_B[j, k]) and not np.isnan(np.exp(new_B[j, k]))
            normalize_logprobs(new_B[j])

        A = new_A
        B = new_B
        log_px, _, __ = forward_backward(A0, A, B, seq)

        if prev_log_px is not None and abs(prev_log_px - log_px) < 1e-2:
            prev_log_px = log_px
            break
        else:
            prev_log_px = log_px
        print 'log likelihood of data AFTER UPDATE:', log_px

    return prev_log_px, A0, A, B

def generate_biased_coinflips(probs, coin_change_prob, length):
    ret = []
    coin_ind = random.choice(range(len(probs)))
    for i in xrange(length):
        ret.append(int(np.random.random() < probs[coin_ind]))
        if np.random.random() < coin_change_prob:
            coin_ind = random.choice(range(len(probs)))
    return ret

def main():
    rtk.rand.seed(5)
    GT_A0 = np.array([0.5, 0.5])
    GT_A = np.array([[0.6, 0.4], [0.4, 0.6]])
    GT_B = np.array([[0.55, 0.45], [0.45, 0.55]])

    # 4.c.ii GENERATE SAMPLES
    sequence = generate_biased_coinflips([0.45, 0.55], 0.8, 10000)

    log_px, _, _ = forward_backward(GT_A0, GT_A, GT_B, sequence)
    print 'LOG LIKELIHOOD OF GROUND TRUTH MODEL:', log_px
    bad_init_loglik, _, bad_A, bad_B = baumwelch(sequence, 2, n_iter=100)

    perturb = np.zeros((2,2))
    perturb[0, 0] = 1e-3
    good_init_loglik, _, good_A, good_B = baumwelch(sequence, 2, n_iter=100, 
            A=GT_A + perturb,
            B=GT_B + perturb)
    print 'Bad model:', bad_init_loglik
    print bad_A
    print bad_B

    print 'Good model:', good_init_loglik
    print good_A
    print good_B

if __name__ == '__main__':
    rtk.debug.wrap(main)()
