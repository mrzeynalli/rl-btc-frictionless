"""A market with a known, exploitable pattern, for verifying the learner works.

The central question before spending days of compute is not "is the agent
profitable" but "can this pipeline learn anything at all". A synthetic market
whose edge we planted ourselves answers it: if the agent cannot find a signal we
know is there and know pays several times its own trading costs, then no amount
of real data will help and the fault is in the machinery.

This is standard practice for debugging reinforcement learning - solve a problem
you know is solvable before trusting a result on one you do not.

The construction is deliberately simple. A hidden regime flips sign with a mean
dwell time of about two days and adds a constant drift to the price while it
holds. The regime as of the PREVIOUS minute is written into one observation
column, so it is causally available and predicts the drift ahead. The optimal
policy is therefore: hold long while the column is positive, short while it is
negative, and change side only when it flips. That is exactly the behaviour the
real design targets - a directional decision held for tens of hours - so a
failure here is a failure of the learner, not a mismatch of task shape.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import FIVE_MINUTE_RETURN_SD, Config
from .data import FAST_NAMES, SLOW_NAMES, Market, Scaler

REGIME_COLUMN = 0
MINUTES_PER_YEAR = 365 * 24 * 60


def planted_market(minutes=600_000, seed=0, oracle_sharpe=3.0, regime_hours=48.0,
                   start='2020-01-01', noise=0.05):
    """Build a Market whose drift is predictable from one observation column.

    ``oracle_sharpe`` is the annualised Sharpe ratio a perfectly informed,
    cost-free trader would achieve by following the regime.
    """
    generator = np.random.default_rng(seed)
    minute_sd = FIVE_MINUTE_RETURN_SD / math.sqrt(5)
    drift = oracle_sharpe * minute_sd / math.sqrt(MINUTES_PER_YEAR)
    flip = generator.random(minutes) < 1.0 / (regime_hours * 60)
    regime = np.cumprod(np.where(flip, -1.0, 1.0))
    returns = generator.normal(drift * regime, minute_sd)
    price = 100.0 * np.exp(np.cumsum(returns))

    prices = np.repeat(price[:, None], 4, axis=1)
    fast = np.zeros((minutes, len(FAST_NAMES)), np.float32)
    fast[1:, 0] = np.diff(np.log(price))
    fast[:, 4] = 1.0
    slow = np.zeros((minutes, len(SLOW_NAMES)), np.float32)
    # Strictly causal: the value at minute t is the regime as of t-1, matching
    # how the real feature builder shifts its slow block by one minute.
    slow[1:, REGIME_COLUMN] = regime[:-1]
    slow[:, REGIME_COLUMN] += generator.normal(0, noise, minutes)
    # A few uninformative columns, so the signal has to be found rather than
    # being the only thing varying.
    for column in range(1, 4):
        slow[:, column] = generator.normal(0, 1, minutes)
    epoch = int(pd.Timestamp(start, tz='UTC').value // 60_000_000_000)
    audit = {'synthetic': True, 'oracle_sharpe': oracle_sharpe, 'regime_hours': regime_hours,
             'drift_per_minute': drift, 'minute_sd': minute_sd, 'seed': seed,
             'regime_column': REGIME_COLUMN, 'noise': noise}
    market = Market(prices, np.ones(minutes, bool), np.zeros(minutes), fast, slow, epoch, audit)
    return market, regime


def scaler_for(market, config=Config()):
    """Standardise against the whole synthetic series; no leakage concern here."""
    fast = market.fast
    slow = market.slow[::5]
    return Scaler(fast.mean(0).astype(np.float32),
                  np.maximum(fast.std(0), 1e-6).astype(np.float32),
                  slow.mean(0).astype(np.float32),
                  np.maximum(slow.std(0), 1e-6).astype(np.float32),
                  str(market.timestamp(len(market) - 1)))


def oracle_actions(market, regime, bounds):
    """The best available policy: follow the regime, change side only on a flip."""
    from .ledger import LONG, SHORT
    first, last = bounds
    steps = (last - first) // 5
    at = first + np.arange(steps) * 5
    return np.where(regime[at - 1] > 0, LONG, SHORT)


def edge_economics(market, config=Config(), hold_hours=None):
    """How much the planted edge pays per position, against its own cost."""
    audit = market.audit
    hold_hours = audit['regime_hours'] if hold_hours is None else hold_hours
    minutes = hold_hours * 60
    gain = audit['drift_per_minute'] * minutes
    return {'expected_move_per_position_bp': gain * 1e4,
            'round_trip_cost_bp': config.round_trip_cost * 1e4,
            'edge_to_cost_ratio': gain / config.round_trip_cost,
            'oracle_sharpe': audit['oracle_sharpe'],
            'mean_regime_hours': audit['regime_hours']}
