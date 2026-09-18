"""Differential Sharpe ratio (Moody & Saffell 2001) as a per-step reward.

The plain objective, net log BTC equity growth, pays for exposure rather than
for skill: in a year BTC rose, the argmax of that objective is to go long in
January and never think again. Measured across 78 predecessor runs, net
directional exposure alone explained R-squared of 0.96, 0.82 and 0.63 of the
validation return, so seeds that guessed the year's direction were being scored
as if they had learned something.

The differential Sharpe ratio is the marginal contribution of the current bar to
an exponentially weighted Sharpe ratio. Holding one position for a year earns a
large return while eating the full variance of the asset, which is a mediocre
Sharpe; capturing moves and standing aside through chop earns a smaller gross
return with far smaller swings, which is a better one. It therefore prefers
activity exactly when activity adds skill, and prefers holding when it does not,
without naming a direction or referencing any benchmark policy.

It also removes the incentive behind the leverage-reset flicker: raising
exposure scales return and volatility together, leaving the ratio unchanged,
while the fees paid to do it are pure loss.
"""
from __future__ import annotations

import numpy as np


def variance_floor(config):
    """Smallest second-central-moment allowed under the ratio's denominator."""
    return config.dsr_variance_floor_fraction * config.dsr_initial_second_moment


def differential_sharpe(returns, mean, second_moment, config, active=None):
    """One step of the ratio and its updated statistics.

    ``returns`` are per-step net log equity changes. ``mean`` and
    ``second_moment`` are the exponentially weighted estimates A and B carried
    from the previous step; the value returned uses the PREVIOUS estimates, as
    the derivation requires, and the updated ones are returned alongside it.

    Terminated accounts contribute nothing and keep their statistics frozen,
    so a liquidated account cannot drag the baseline of a fresh one.
    """
    returns = np.asarray(returns, dtype=np.float64)
    active = np.ones(returns.shape, bool) if active is None else np.asarray(active, bool)
    delta_mean = returns - mean
    delta_second = returns * returns - second_moment
    variance = np.maximum(second_moment - mean * mean, variance_floor(config))
    value = (second_moment * delta_mean - 0.5 * mean * delta_second) / variance ** 1.5
    value = np.clip(value, -config.dsr_clip, config.dsr_clip)
    value = np.where(active, value, 0.0)
    updated_mean = np.where(active, mean + config.dsr_eta * delta_mean, mean)
    updated_second = np.where(active, second_moment + config.dsr_eta * delta_second, second_moment)
    return value, updated_mean, updated_second


def observation_features(mean, second_moment, config):
    """The two dimensionless views of the reward state the policy can see.

    Without these the reward depends on history the agent cannot observe: the
    same action in the same market state would pay differently according to
    recent performance, which is not a Markov decision process and which the
    critic has no way to fit.
    """
    deviation = np.sqrt(np.maximum(second_moment, variance_floor(config)))
    reference = np.sqrt(config.dsr_initial_second_moment)
    running_sharpe = np.clip(mean / np.maximum(deviation, 1e-12), -5, 5)
    relative_volatility = np.clip(np.log(deviation / reference), -5, 5)
    return running_sharpe, relative_volatility
