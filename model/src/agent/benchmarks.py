"""Benchmarks for any agent experiment, traded through the agent's own ledger.

Every benchmark trades the same USDT linear account as the agent: hourly decisions, orders filled at
the next traded open with the real 5 bp fee and 1 bp slippage per leg, funding, minute margin checks,
and each validation quarter started from a flat account. Kept as a permanent reference set
(ARCHITECTURE.md §9):

| name | rule | source |
|---|---|---|
| `buy_and_hold`, `always_short`, `cash` | a constant 1x long, 1x short, or flat | passive references |
| `momentum_1w`, `momentum_26w` | hold the sign of the volatility-normalised return over the look-back, revised daily at 00:00 UTC | Liu & Tsyvinski (RFS 2021) weekly; Moskowitz, Ooi & Pedersen (JFE 2012) |
| `ma_50`, `ma_200` | long when the daily close is above its 50- or 200-day simple moving average, short below, revised daily | Brock, Lakonishok & LeBaron (JF 1992), VMA(1, 50) and VMA(1, 200); Corbet, Eraslan, Lucey & Sensoy (FRL 2019) find the variable-length moving average best on Bitcoin |
| `vol_scaled_hold` | long with weight c / previous four-week realised daily variance, rebalanced at each month start, capped at 2x; c makes the strategy's variance on the training span equal buy-and-hold's | Moreira & Muir (JF 2017) |
| `random_uniform` | each hour, short, flat or long with equal probability | Biondo, Pluchino, Rapisarda & Helbing (PLoS ONE 2013), random trading |
| `random_matched` | a Markov chain over short/flat/long with the agent's own hour-to-hour transition probabilities: the same holding periods, exposure and turnover, and no information | a turnover-matched null (DESIGN CHOICE; this project's walk-forward study used a matched-turnover null) |

The daily revision of the rules and the 2x cap are DESIGN CHOICES.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import BAR_MINUTES
from .data import CANDIDATE_FEATURES, DAY_MINUTES
from .ledger import FLAT, LONG, SHORT, Ledger, advance

DETERMINISTIC = ('buy_and_hold', 'always_short', 'cash', 'momentum_1w', 'momentum_26w', 'ma_50', 'ma_200',
                 'vol_scaled_hold')
RANDOM = ('random_uniform', 'random_matched')
BENCHMARKS = DETERMINISTIC + RANDOM
MOMENTUM = {'momentum_1w': 'return_1w', 'momentum_26w': 'return_26w'}
MOVING_AVERAGE_DAYS = {'ma_50': 50, 'ma_200': 200}
VOLATILITY_CAP = 2.0
SIDE_INDEX = {-1: SHORT, 0: FLAT, 1: LONG}


def is_midnight(market, minute):
    return (market.epoch_start + int(minute)) % DAY_MINUTES == 0


def month_start(market, minute):
    stamp = market.timestamp(minute)
    return stamp.day == 1 and stamp.hour == 0 and stamp.minute == 0


def moving_average_signal(market, minute, days):
    """+1 when the last daily close is above the mean of the last ``days`` daily closes, else -1."""
    closes = np.asarray(market.prices[minute - 1 - np.arange(days) * DAY_MINUTES, 3], dtype=np.float64)
    return 1 if closes[0] > closes.mean() else -1


def volatility_scale(market, first, last):
    """Moreira & Muir's c on a training span [first, last): managed variance equals buy-and-hold variance.

    Daily returns from midnight closes; the weight 1 / sigma^2 is fixed at each month start from the
    four-week realised daily volatility known then, and held for the month.
    """
    first += (-(market.epoch_start + first)) % DAY_MINUTES
    midnights = np.arange(first, last, DAY_MINUTES)
    closes = np.asarray(market.prices[midnights - 1, 3], dtype=np.float64)
    returns = np.diff(closes) / closes[:-1]
    stamps = pd.DatetimeIndex([market.timestamp(m) for m in midnights[:-1]])
    months = stamps.to_period('M')
    starts = pd.Series(midnights[:-1]).groupby(np.asarray(months)).transform('first').to_numpy()
    variance = np.asarray(market.daily_volatility[starts], dtype=np.float64) ** 2
    usable = variance > 0
    weights = np.where(usable, 1.0 / np.where(usable, variance, 1.0), 0.0)
    return float(math.sqrt(returns[usable].var() / (weights[usable] * returns[usable]).var()))


def fit_transitions(sides_by_path):
    """Hour-to-hour transition probabilities over short/flat/long; every path starts flat."""
    counts = np.zeros((3, 3))
    for sides in sides_by_path:
        states = np.r_[FLAT, [SIDE_INDEX[int(s)] for s in np.sign(sides)]]
        np.add.at(counts, (states[:-1], states[1:]), 1)
    rows = counts.sum(1, keepdims=True)
    return np.where(rows > 0, counts / np.maximum(rows, 1), np.eye(3))


def simulate(name, market, config, begin, end, paths=1, seed=0, transitions=None, scale=None):
    """One benchmark over [begin, end) from flat accounts; returns per-decision arrays of every path."""
    first, last = market.bounds(begin, end)
    span = config.decision_bars * BAR_MINUTES
    steps = (last - first) // span
    paths = paths if name in RANDOM else 1
    rng = np.random.default_rng(seed)
    ledger = Ledger(paths, config)
    cursor = np.full(paths, first, dtype=np.int64)
    log_return = np.zeros((paths, steps))
    equity = np.zeros((paths, steps))
    side = np.zeros((paths, steps), dtype=np.int8)
    weight = np.zeros(steps)
    target = np.full(paths, FLAT)
    column = CANDIDATE_FEATURES.index(MOMENTUM[name]) if name in MOMENTUM else None
    for step in range(steps):
        minute = int(cursor[0])
        sizes = None
        if name == 'buy_and_hold':
            target[:] = LONG
        elif name == 'always_short':
            target[:] = SHORT
        elif name == 'cash':
            target[:] = FLAT
        elif column is not None:
            if step == 0 or is_midnight(market, minute):
                signal = float(market.features[minute, column])
                target[:] = LONG if signal > 0 else SHORT if signal < 0 else FLAT
        elif name in MOVING_AVERAGE_DAYS:
            if step == 0 or is_midnight(market, minute):
                target[:] = LONG if moving_average_signal(market, minute, MOVING_AVERAGE_DAYS[name]) > 0 else SHORT
        elif name == 'vol_scaled_hold':
            target[:] = LONG
            if step == 0 or month_start(market, minute):
                variance = float(market.daily_volatility[minute]) ** 2
                size = min(VOLATILITY_CAP, scale / variance) if variance > 0 else 0.0
                sizes = np.full(paths, size)
                weight[step] = size
            else:
                weight[step] = weight[step - 1]
        elif name == 'random_uniform':
            target = rng.integers(0, 3, paths)
        elif name == 'random_matched':
            current = np.sign(ledger.contracts).astype(np.int64) + 1
            uniform = rng.random(paths)[:, None]
            target = (uniform > np.cumsum(transitions[current], axis=1)).sum(1).clip(0, 2)
        else:
            raise ValueError(f'unknown benchmark {name}')
        _, _, info = advance(ledger, market, cursor, target.copy(), config, last, sizes)
        cursor += span
        log_return[:, step] = info['log_return']
        equity[:, step] = info['equity']
        side[:, step] = np.sign(info['contracts'])
    times = pd.date_range(market.timestamp(first), periods=steps, freq=f'{span}min')
    return {'log_return': log_return, 'equity': equity, 'side': side, 'times': times, 'weight': weight,
            'orders': ledger.orders.copy(), 'fees': ledger.fees_total.copy(), 'funding': ledger.funding_total.copy(),
            'turnover': ledger.turnover_total.copy(), 'liquidations': ledger.liquidations.copy()}


def path_metrics(log_return, equity, side, times):
    """Per path: total log return, daily returns, max drawdown, exposure - evaluate.metrics' definitions."""
    days = pd.DatetimeIndex(times).floor('D')
    codes, _ = pd.factorize(days)
    daily_log = np.zeros((log_return.shape[0], codes.max() + 1))
    np.add.at(daily_log.T, codes, log_return.T)
    curve = np.concatenate([np.ones((equity.shape[0], 1)), equity], axis=1)
    drawdown = (curve / np.maximum.accumulate(curve, axis=1) - 1).min(1)
    return {'total_log_return': log_return.sum(1), 'daily_returns': np.expm1(daily_log), 'max_drawdown': drawdown,
            'long_share': (side > 0).mean(1), 'short_share': (side < 0).mean(1)}


def annualised_sharpe(daily_returns):
    spread = daily_returns.std(axis=-1, ddof=1)
    return np.where(spread > 0, daily_returns.mean(-1) / np.where(spread > 0, spread, 1) * math.sqrt(365), 0.0)
