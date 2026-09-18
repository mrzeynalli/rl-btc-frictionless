"""Markets with a known, exploitable pattern, for verifying the learner works.

Before spending days of compute the question is not "is the agent profitable"
but "can this pipeline learn anything, and does it learn the right thing".

- observable: a hidden regime flips sign with a mean dwell of two days and adds
  drift while it holds; the regime as of the PREVIOUS minute is written into the
  trend feature, so it is causally available. The signal is handed over, and only
  PPO, the ledger and the network are under test.
- latent: a Markov-switching market (Hamilton 1989) of bear, range and bull
  regimes that differ in drift, volatility and volume. Minute bars are generated
  and passed through the same causal feature pipeline as BTC, and the regime is
  never shown. The only way to trade it is to recognise it from the features -
  the skill the real agent needs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Config
from .data import (BAR_MINUTES, CANDIDATE_FEATURES, DAY_MINUTES, FEATURE_NAMES, Market, Scaler, bar_return_sd,
                   causal_arrays, feature_columns)

# v2: the planted regime is written into the one-week return; the distractors are other columns of the state.
REGIME_FEATURE = 'return_1w'
DISTRACTOR_FEATURES = ('volatility_regime', 'return_1h', 'return_26w')
MINUTES_PER_YEAR = 365 * 24 * 60
# Measured on BitMEX XBTUSD five-minute log returns, 2017-01-01 to 2024-01-01: the Sharpe
# ratio of holding long, and the standard deviation Scaler.fit measures on that span (26.55 bp),
# which scales the observable market.
BTC_IN_SAMPLE_SHARPE = 0.627
BTC_TRAINING_FIVE_MINUTE_SD = 0.002655
BEAR, RANGE, BULL = 0, 1, 2
REGIME_NAMES = ('bear', 'range', 'bull')


def planted_market(minutes=600_000, seed=0, oracle_sharpe=3.0, regime_hours=48.0,
                   start='2020-01-01', noise=0.05, unconditional_sharpe=0.0):
    """``oracle_sharpe`` is the cost-free Sharpe ratio of following the regime alone."""
    generator = np.random.default_rng(seed)
    minute_sd = BTC_TRAINING_FIVE_MINUTE_SD / math.sqrt(BAR_MINUTES)
    drift = oracle_sharpe * minute_sd / math.sqrt(MINUTES_PER_YEAR)
    bias = unconditional_sharpe * minute_sd / math.sqrt(MINUTES_PER_YEAR)
    flip = generator.random(minutes) < 1.0 / (regime_hours * 60)
    regime = np.cumprod(np.where(flip, -1.0, 1.0))
    returns = generator.normal(bias + drift * regime, minute_sd)
    price = 100.0 * np.exp(np.cumsum(returns))
    prices = np.repeat(price[:, None], 4, axis=1)
    features = np.zeros((minutes, len(CANDIDATE_FEATURES)), np.float32)
    column = CANDIDATE_FEATURES.index(REGIME_FEATURE)
    # Strictly causal: the value at boundary t is the regime as of minute t-1.
    features[1:, column] = regime[:-1]
    features[:, column] += generator.normal(0, noise, minutes)
    # Uninformative columns, so the signal has to be found rather than being the
    # only thing that varies.
    for name in DISTRACTOR_FEATURES:
        features[:, CANDIDATE_FEATURES.index(name)] = generator.normal(0, 1, minutes)
    daily_volatility = np.full(minutes, minute_sd * math.sqrt(DAY_MINUTES), np.float32)
    epoch = int(pd.Timestamp(start, tz='UTC').value // 60_000_000_000)
    audit = {'synthetic': 'observable', 'oracle_sharpe': oracle_sharpe, 'regime_hours': regime_hours,
             'drift_per_minute': drift, 'bias_per_minute': bias, 'unconditional_sharpe': unconditional_sharpe,
             'minute_sd': minute_sd, 'seed': seed, 'regime_feature': REGIME_FEATURE, 'noise': noise}
    market = Market(prices, np.ones(minutes, bool), np.zeros(minutes), features, daily_volatility,
                    epoch, audit, history_minutes=1)
    return market, regime


def latent_regime_market(years=4.0, seed=7, start='2012-01-01', regime_days=10.0, trend_sharpe=0.35,
                         daily_volatility=(0.045, 0.025, 0.030), volume_multiple=(1.3, 0.8, 1.0)):
    """Bear, range and bull regimes, observable only through the bars they produce.

    ``trend_sharpe`` is a trending regime's drift per day in units of that regime's
    daily volatility. Dwell times are geometric with mean ``regime_days``; a switch
    moves to either other regime with equal probability, so bull and bear are
    equally common and there is no unconditional drift. Volatilities bracket BTC's
    2017-2023 median daily 3.7% and are highest in the bear regime (the leverage
    effect); funding is zero-mean noise, so it carries no regime information.
    """
    generator = np.random.default_rng(seed)
    minutes = int(years * MINUTES_PER_YEAR)
    switch = generator.random(minutes) < 1.0 / (regime_days * DAY_MINUTES)
    step = np.where(switch, generator.integers(1, 3, minutes), 0)
    regime = (int(generator.integers(0, 3)) + np.cumsum(step)) % 3
    volatility = np.asarray(daily_volatility, dtype=np.float64)[regime]
    minute_sd = volatility / math.sqrt(DAY_MINUTES)
    minute_drift = (regime - RANGE) * trend_sharpe * volatility / DAY_MINUTES
    log_close = math.log(10_000.0) + np.cumsum(generator.normal(minute_drift, minute_sd))
    log_open = np.r_[math.log(10_000.0), log_close[:-1]]
    wick = np.abs(generator.normal(0.0, 0.5, (2, minutes))) * minute_sd
    stamps = pd.date_range(pd.Timestamp(start, tz='UTC'), periods=minutes, freq='min')
    bars = pd.DataFrame({'open_time': stamps, 'open': np.exp(log_open),
                         'high': np.exp(np.maximum(log_open, log_close) + wick[0]),
                         'low': np.exp(np.minimum(log_open, log_close) - wick[1]),
                         'close': np.exp(log_close),
                         'volume': 1e6 * np.asarray(volume_multiple)[regime] * generator.lognormal(0.0, 0.5, minutes)})
    end = stamps[-1] + pd.Timedelta(minutes=1)
    settlements = pd.date_range(stamps[0].floor('D') + pd.Timedelta(hours=4), end, freq='8h', inclusive='left')
    funding = pd.DataFrame({'timestamp': settlements, 'fundingRate': generator.normal(0.0, 5e-5, len(settlements)),
                            'funding_interval_hours': 8.0})
    market = causal_arrays(bars, funding, stamps[0], end)
    market.audit.update({'synthetic': 'latent_regime', 'seed': seed, 'years': years, 'regime_days': regime_days,
                         'trend_sharpe': trend_sharpe, 'daily_volatility': list(daily_volatility),
                         'volume_multiple': list(volume_multiple),
                         'regime_share': (np.bincount(regime, minlength=3) / minutes).tolist()})
    return market, regime


def scaler_for(market, end=None):
    """Measure on the synthetic training span, or the whole series when no end is given."""
    first = market.history_minutes + (-(market.epoch_start + market.history_minutes)) % BAR_MINUTES
    last = len(market) if end is None else end
    rows = np.asarray(market.features[first:last:BAR_MINUTES][:, feature_columns()], dtype=np.float64)
    return Scaler(rows.mean(0).astype(np.float32), np.maximum(rows.std(0), 1e-6).astype(np.float32),
                  str(market.timestamp(last - 1)), bar_return_sd(market, first, last), tuple(FEATURE_NAMES))


def span_decisions(bounds, config):
    """How many decisions one pass over a span holds."""
    return (bounds[1] - bounds[0]) // (config.decision_bars * BAR_MINUTES)


def decision_minutes(bounds, config):
    span = config.decision_bars * BAR_MINUTES
    first, last = bounds
    return first + np.arange((last - first) // span) * span


def oracle_actions(market, regime, bounds, config=Config()):
    """Hold the side of the drift in force as of the previous minute."""
    from .ledger import FLAT, LONG, SHORT
    at = decision_minutes(bounds, config)
    audit = market.audit
    if audit.get('synthetic') == 'latent_regime':
        state = regime[at - 1]
        return np.select([state == BULL, state == BEAR], [LONG, SHORT], FLAT)
    total = audit.get('bias_per_minute', 0.0) + audit['drift_per_minute'] * regime[at - 1]
    return np.where(total > 0, LONG, SHORT)


def trend_rule_actions(market, bounds, config=Config()):
    """The simplest causal use of the state: hold the sign of the weekly return."""
    from .ledger import LONG, SHORT
    at = decision_minutes(bounds, config)
    trend = np.asarray(market.features[at, CANDIDATE_FEATURES.index('return_1w')])
    return np.where(trend > 0, LONG, SHORT)


def edge_economics(market, config=Config()):
    """How much the planted edge pays per position, against its own cost."""
    audit = market.audit
    cost = config.round_trip_cost
    if audit.get('synthetic') == 'latent_regime':
        volatility = np.asarray(audit['daily_volatility'])
        move = audit['trend_sharpe'] * volatility[[BEAR, BULL]].mean() * audit['regime_days']
        return {'expected_move_per_trend_regime_bp': move * 1e4, 'round_trip_cost_bp': cost * 1e4,
                'edge_to_cost_ratio': move / cost,
                'trend_regime_annualised_sharpe': audit['trend_sharpe'] * math.sqrt(365),
                'mean_regime_days': audit['regime_days'], 'regime_share': audit['regime_share']}
    gain = audit['drift_per_minute'] * audit['regime_hours'] * 60
    return {'expected_move_per_position_bp': gain * 1e4, 'round_trip_cost_bp': cost * 1e4,
            'edge_to_cost_ratio': gain / cost, 'oracle_sharpe': audit['oracle_sharpe'],
            'unconditional_sharpe': audit.get('unconditional_sharpe', 0.0),
            'mean_regime_hours': audit['regime_hours']}
