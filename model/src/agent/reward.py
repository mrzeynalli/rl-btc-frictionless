"""Reward: the exact per-bar increment of an exponentially weighted performance ratio.

Moody & Saffell (IEEE TNN 2001) track exponential estimates A of the mean return and
B of a risk second moment, and obtain their differential ratios by "expanding to
first order in the adaptation rate" eta. Two ratios are available:

- 'sharpe': risk = variance, B - A^2, with B the second moment of returns;
- 'sortino': risk = downside deviation, B the second moment of min(R, 0) - their
  downside deviation ratio, which has "no penalty for large positive returns".

The reward is the exact change of the ratio per bar divided by eta, not the
first-order term. For ordinary returns the two agree to O(eta); for extreme losses
the exact change stays bounded where the first-order term explodes, so no clip is
needed. The Sharpe ratio is invariant to the level of wealth, so termination at
liquidation - which removes every future reward - carries the cost of ruin.

A ratio pays for skill rather than exposure: across 78 predecessor runs trained on
plain log growth, net directional exposure alone explained R-squared of 0.96, 0.82
and 0.63 of validation return.
"""
from __future__ import annotations

import numpy as np


def risk_target(returns, kind):
    """The per-bar quantity whose exponential average is the risk second moment."""
    return returns * returns if kind == 'sharpe' else np.minimum(returns, 0.0) ** 2


def performance_ratio(mean, second_moment, kind):
    """Mean over risk. The guard is numerical only: it binds at float64 round-off."""
    floor = np.finfo(np.float64).eps * second_moment
    risk = second_moment - mean * mean if kind == 'sharpe' else second_moment
    return mean / np.sqrt(np.maximum(risk, floor))


def differential_ratio(returns, mean, second_moment, config, active=None):
    """One bar of the reward and the updated statistics.

    Terminated accounts contribute nothing and keep their statistics frozen.
    """
    kind = config.reward
    returns = np.asarray(returns, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    second_moment = np.asarray(second_moment, dtype=np.float64)
    active = np.ones(returns.shape, bool) if active is None else np.asarray(active, bool)
    eta = config.dsr_eta
    updated_mean = mean + eta * (returns - mean)
    updated_second = second_moment + eta * (risk_target(returns, kind) - second_moment)
    value = (performance_ratio(updated_mean, updated_second, kind)
             - performance_ratio(mean, second_moment, kind)) / eta
    value = np.where(active, value, 0.0)
    return value, np.where(active, updated_mean, mean), np.where(active, updated_second, second_moment)


def first_order_differential_sharpe(returns, mean, second_moment):
    """Moody & Saffell's D_t (eq. 14), kept to verify the exact increment reduces to it."""
    delta_mean = returns - mean
    delta_second = returns * returns - second_moment
    return (second_moment * delta_mean - 0.5 * mean * delta_second) / (second_moment - mean * mean) ** 1.5


def first_order_downside_ratio(returns, mean, second_moment):
    """Moody & Saffell's differential downside deviation ratio, kept for the same check.

    (R - A/2) / DD for a gain; (DD^2 (R - A/2) - A R^2 / 2) / DD^3 for a loss.
    """
    deviation = np.sqrt(second_moment)
    gain = (returns - 0.5 * mean) / deviation
    loss = (second_moment * (returns - 0.5 * mean) - 0.5 * mean * returns ** 2) / deviation ** 3
    return np.where(returns > 0, gain, loss)


def observation_features(mean, second_moment, config):
    """The reward state the policy must see for the reward to be Markov.

    Without these the same action in the same market state would pay differently
    according to unobserved recent performance, which the critic cannot fit.
    """
    return (performance_ratio(mean, second_moment, config.reward),
            0.5 * np.log(second_moment / config.initial_reward_second_moment))
