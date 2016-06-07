import argparse as ap
from collections import defaultdict as dd
import os
if True:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

import rtk

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('eids')
    args = parser.parse_args()

    improvements = dd(lambda: dd(list))

    for fn, args, duration, result in rtk.dist.db.iter_results(args.eids):
        for r in result:
            for k, v in r.items():
                improvements[k][args['rand_seed']].append(v)

    for k in improvements.keys():
        all_results = np.vstack(improvements[k].values())
        assert all_results.shape == (len(improvements[k]), len(improvements[k].values()[0]))
        improvements[k] = (all_results.mean(axis=0), all_results.std(axis=0))

    x = range(1, improvements.iteritems().next()[1][0].shape[0]+1)
    for k, (mean, err) in improvements.iteritems():
        if 'vae' in k: continue
        plt.errorbar(x, mean, yerr=err, label=k)
    plt.axvline(x=5, color='purple')
    plt.axhline(y=0.25, color='green')
    plt.legend(loc='center right')
    plt.title('Performance Metrics for CRF Semi-supervised')
    plt.xlabel('Iteration #')
    plt.savefig('crf.pdf')

if __name__ == '__main__':
    rtk.debug.wrap(main)()
