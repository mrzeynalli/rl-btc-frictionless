"""Markets with a known, exploitable pattern, for verifying the learner works.

Before spending days of compute the question is not "is the agent profitable"
but "can this pipeline learn anything, and does it learn the right thing". A
hidden regime flips sign with a mean dwell of about two days and adds drift
while it holds; the regime as of the PREVIOUS minute is written into one feature
column, so it is causally available and predicts the drift ahead.

Two variants test the two failures of the previous agent:

- symmetric: no unconditional drift. The optimal policy follows the regime long
  and short. Failure here is a failure of the machinery.
- drift: an unconditional drift equal to BTC's own measured in-sample Sharpe
  ratio (0.63, five-minute returns 2017-2023) is added. In expectation most
  training episodes then rise - the attractor that locked the previous agent
  onto one side - while the regime is still worth following. Whether a given
  realisation actually carries that pull depends on the span's length and must
  be measured (scripts/planted_edge_test.py records it).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import FIVE_MINUTE_RETURN_SD, Config
from .data import BAR_MINUTES, DAY_MINUTES, FEATURE_NAMES, Market, Scaler

REGIME_FEATURE = 'momentum_1w'
DISTRACTOR_FEATURES = ('volatility_regime', 'volume_regime', 'funding_rate')
MINUTES_PER_YEAR = 365 * 24 * 60
# Measured on BitMEX XBTUSD five-minute log returns, 2017-01-01 to 2024-01-01.
BTC_IN_SAMPLE_SHARPE = 0.627


def planted_market(minutes=600_000, seed=0, oracle_sharpe=3.0, regime_hours=48.0,
                   start='2020-01-01', noise=0.05, unconditional_sharpe=0.0):
    """``oracle_sharpe`` is the cost-free Sharpe ratio of following the regime alone."""
    generator = np.random.default_rng(seed)
    minute_sd = FIVE_MINUTE_RETURN_SD / math.sqrt(BAR_MINUTES)
    drift = oracle_sharpe * minute_sd / math.sqrt(MINUTES_PER_YEAR)
    bias = unconditional_sharpe * minute_sd / math.sqrt(MINUTES_PER_YEAR)
    flip = generator.random(minutes) < 1.0 / (regime_hours * 60)
    regime = np.cumprod(np.where(flip, -1.0, 1.0))
    returns = generator.normal(bias + drift * regime, minute_sd)
    price = 100.0 * np.exp(np.cumsum(returns))
    prices = np.repeat(price[:, None], 4, axis=1)
    features = np.zeros((minutes, len(FEATURE_NAMES)), np.float32)
    column = FEATURE_NAMES.index(REGIME_FEATURE)
    # Strictly causal: the value at boundary t is the regime as of minute t-1.
    features[1:, column] = regime[:-1]
    features[:, column] += generator.normal(0, noise, minutes)
    # Uninformative columns, so the signal has to be found rather than being the
    # only thing that varies.
    for name in DISTRACTOR_FEATURES:
        features[:, FEATURE_NAMES.index(name)] = generator.normal(0, 1, minutes)
    features[:, FEATURE_NAMES.index('bar_coverage')] = 1.0
    daily_volatility = np.full(minutes, minute_sd * math.sqrt(DAY_MINUTES), np.float32)
    epoch = int(pd.Timestamp(start, tz='UTC').value // 60_000_000_000)
    audit = {'synthetic': True, 'oracle_sharpe': oracle_sharpe, 'regime_hours': regime_hours,
             'drift_per_minute': drift, 'bias_per_minute': bias, 'unconditional_sharpe': unconditional_sharpe,
             'minute_sd': minute_sd, 'seed': seed, 'regime_feature': REGIME_FEATURE, 'noise': noise}
    market = Market(prices, np.ones(minutes, bool), np.zeros(minutes), features, daily_volatility,
                    epoch, audit, history_minutes=1)
    return market, regime


def scaler_for(market):
    """Standardise against the whole synthetic series; there is nothing to leak."""
    rows = np.asarray(market.features[::BAR_MINUTES], dtype=np.float64)
    return Scaler(rows.mean(0).astype(np.float32), np.maximum(rows.std(0), 1e-6).astype(np.float32),
                  str(market.timestamp(len(market) - 1)))


def oracle_actions(market, regime, bounds):
    """Hold the side of the total drift in force, changing only when the regime flips."""
    from .ledger import LONG, SHORT
    first, last = bounds
    at = first + np.arange((last - first) // 5) * 5
    audit = market.audit
    total = audit.get('bias_per_minute', 0.0) + audit['drift_per_minute'] * regime[at - 1]
    return np.where(total > 0, LONG, SHORT)


def edge_economics(market, config=Config(), hold_hours=None):
    """How much the planted edge pays per position, against its own cost."""
    audit = market.audit
    hold_hours = audit['regime_hours'] if hold_hours is None else hold_hours
    gain = audit['drift_per_minute'] * hold_hours * 60
    return {'expected_move_per_position_bp': gain * 1e4,
            'round_trip_cost_bp': config.round_trip_cost * 1e4,
            'edge_to_cost_ratio': gain / config.round_trip_cost,
            'oracle_sharpe': audit['oracle_sharpe'],
            'unconditional_sharpe': audit.get('unconditional_sharpe', 0.0),
            'mean_regime_hours': audit['regime_hours']}
