"""Reward: the exact per-step change in an exponentially weighted Sharpe ratio.

Moody & Saffell (IEEE TNN 2001, eqs. 13-15) obtain the differential Sharpe ratio
by "expanding to first order in the adaptation rate" eta the change in a Sharpe
ratio built from exponential moving estimates of the first two moments of return.
The previous agent used that first-order term and clipped it at +-10, and the
clip bound on exactly the events that matter most: a 46% slide and a 100% wipeout
paid the same penalty. The exact change needs no clip. For ordinary returns it
equals the first-order term to O(eta); for extreme ones it stays bounded, because
one huge return inflates the ratio's denominator as fast as its numerator.

That same saturation is the Sharpe ratio's blind spot for ruin - it is invariant
to the level of wealth. Termination at liquidation, which removes every future
reward, carries that loss; the reward does not pretend to.

A Sharpe-type reward pays for skill rather than exposure: across 78 predecessor
runs trained on plain log growth, net directional exposure alone explained
R-squared of 0.96, 0.82 and 0.63 of validation return. Raising exposure scales
return and volatility together, so the ratio does not reward leverage resets,
while their fees are pure loss.
"""
from __future__ import annotations

import numpy as np


def sharpe_ratio(mean, second_moment):
    """Exponentially weighted Sharpe ratio from its first two moments.

    The variance guard is numerical, not a modelling choice: a proper prior keeps
    second_moment - mean**2 positive, and the guard can only bind at float64
    round-off, where it keeps the ratio finite.
    """
    variance = np.maximum(second_moment - mean * mean, np.finfo(np.float64).eps * second_moment)
    return mean / np.sqrt(variance)


def differential_sharpe(returns, mean, second_moment, config, active=None):
    """One step of the reward and the updated statistics.

    ``returns`` are per-step net log equity changes; ``mean`` and ``second_moment``
    are the exponential estimates carried from the previous step. Terminated
    accounts contribute nothing and keep their statistics frozen.
    """
    returns = np.asarray(returns, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    second_moment = np.asarray(second_moment, dtype=np.float64)
    active = np.ones(returns.shape, bool) if active is None else np.asarray(active, bool)
    eta = config.dsr_eta
    updated_mean = mean + eta * (returns - mean)
    updated_second = second_moment + eta * (returns * returns - second_moment)
    value = (sharpe_ratio(updated_mean, updated_second) - sharpe_ratio(mean, second_moment)) / eta
    value = np.where(active, value, 0.0)
    return value, np.where(active, updated_mean, mean), np.where(active, updated_second, second_moment)


def first_order_differential_sharpe(returns, mean, second_moment):
    """Moody & Saffell's D_t (eq. 14), kept to verify the exact increment reduces to it."""
    delta_mean = returns - mean
    delta_second = returns * returns - second_moment
    return (second_moment * delta_mean - 0.5 * mean * delta_second) / (second_moment - mean * mean) ** 1.5


def observation_features(mean, second_moment, config):
    """The reward state the policy must see for the reward to be Markov.

    Without these the same action in the same market state would pay differently
    according to unobserved recent performance, which the critic cannot fit.
    """
    return (sharpe_ratio(mean, second_moment),
            0.5 * np.log(second_moment / config.dsr_initial_second_moment))
