"""Aggregate statistics for a handful of runs, and the risk of choosing a checkpoint.

Agarwal et al. (NeurIPS 2021) show that point estimates from few runs mislead
and recommend the interquartile mean with stratified bootstrap confidence
intervals. Bailey, Borwein, Lopez de Prado & Zhu (J. Computational Finance 2017)
show that picking the best of several backtested configurations overfits, and
estimate the probability that it did by combinatorially symmetric
cross-validation (CSCV). Both are implemented here exactly as published.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np


def interquartile_mean(values):
    """Mean of the middle half; rliable's aggregate_iqm, scipy's trim_mean(x, 0.25)."""
    flat = np.sort(np.asarray(values, dtype=np.float64).ravel())
    cut = int(0.25 * len(flat))
    return float(flat[cut:len(flat) - cut].mean())


def stratified_bootstrap_interval(scores, statistic, replications, confidence, seed=0):
    """Percentile interval of ``statistic`` over runs resampled within each task.

    ``scores`` has shape (runs, tasks). rliable resamples the runs of every task
    independently, which is what "stratified" means in Agarwal et al.
    """
    scores = np.asarray(scores, dtype=np.float64)
    runs, tasks = scores.shape
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, runs, size=(replications, runs, tasks))
    resampled = np.take_along_axis(np.broadcast_to(scores, (replications, runs, tasks)), picks, axis=1)
    values = np.array([statistic(sample) for sample in resampled]) if statistic is not interquartile_mean \
        else _vectorised_iqm(resampled.reshape(replications, -1))
    tail = 50 * (1 - confidence)
    lower, upper = np.percentile(values, [tail, 100 - tail])
    return {'point': float(statistic(scores)), 'lower': float(lower), 'upper': float(upper),
            'confidence': confidence, 'replications': replications, 'runs': runs, 'tasks': tasks}


def _vectorised_iqm(rows):
    ordered = np.sort(rows, axis=1)
    cut = int(0.25 * ordered.shape[1])
    return ordered[:, cut:ordered.shape[1] - cut].mean(1)


def probability_of_backtest_overfitting(performance):
    """CSCV estimate of the probability that the in-sample best is below median out of sample.

    ``performance`` has shape (slices, runs, trials): a contiguous time slice, an
    independent training seed, a candidate checkpoint. Each slice is one CSCV
    partition, so S is the number of slices and must be even. For every way of
    choosing half the slices as in-sample, the trial with the best in-sample
    interquartile mean is ranked out of sample; the logit of its relative rank is
    at or below zero when it falls in the bottom half.
    """
    performance = np.asarray(performance, dtype=np.float64)
    slices, _, trials = performance.shape
    if slices < 2 or slices % 2 or trials < 2:
        return None
    logits = []
    for chosen in combinations(range(slices), slices // 2):
        inside = np.zeros(slices, bool)
        inside[list(chosen)] = True
        in_sample = np.array([interquartile_mean(performance[inside, :, t]) for t in range(trials)])
        out_sample = np.array([interquartile_mean(performance[~inside, :, t]) for t in range(trials)])
        best = int(np.argmax(in_sample))
        below = np.sum(out_sample < out_sample[best])
        ties = np.sum(out_sample == out_sample[best]) - 1
        omega = (1 + below + 0.5 * ties) / (trials + 1)
        logits.append(math.log(omega / (1 - omega)))
    logits = np.asarray(logits)
    return {'probability': float(np.mean(logits <= 0)), 'combinations': int(len(logits)),
            'median_logit': float(np.median(logits)), 'slices': slices, 'trials': trials}
