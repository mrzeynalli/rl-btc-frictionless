"""Causal development evaluation with account-independent market caching.

Actions are SAMPLED from the policy, never taken by argmax, and every evaluation
runs several seeds and reports the distribution. Validation years are scored one
calendar quarter at a time, each from a flat account: the same length as a
training episode, and the window FinRL's ensemble strategy validates and trades
on (Yang et al. 2020).
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import torch

from .config import development_range
from .ledger import Environment, FLAT, LONG, SHORT
from .network import market_tensor

COLUMNS = (('action', np.int64), ('executed_action', np.int64), ('contracts', float),
           ('log_return', float), ('differential_sharpe', float), ('equity_btc', float),
           ('equity_usd', float), ('mark', float), ('funding_btc', float), ('fees_btc', float),
           ('filled', bool), ('liquidation', bool),
           ('p_short', float), ('p_flat', float), ('p_long', float))
QUARTER_MONTHS = (1, 4, 7, 10)


def metrics(frame, ledger, initial_price):
    returns = frame.log_return.to_numpy()
    daily = np.expm1(frame.groupby(frame.time.dt.floor('D')).log_return.sum().to_numpy())
    equity = np.r_[1.0, frame.equity_btc.to_numpy()]
    drawdown = equity / np.maximum.accumulate(equity) - 1
    daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    positions = np.sign(frame.contracts.to_numpy())
    previous = np.r_[0, positions[:-1]]
    closes = int(((previous != 0) & (positions != previous)).sum())
    changes = np.flatnonzero(positions != previous)
    spell_starts = np.unique(np.r_[0, changes])
    spells = np.diff(np.r_[spell_starts, len(frame)])
    held_spells = spells[positions[spell_starts] != 0]
    # A single-bar position is a leverage reset, not a trade: counted apart.
    flickers = int((held_spells <= 1).sum()) if len(held_spells) else 0
    quantiles = (np.percentile(held_spells, [10, 50, 90]) / 12).tolist() if len(held_spells) else [0., 0., 0.]
    entries = (previous == 0) & (positions != 0)
    entry_count = int(entries.sum())
    return {'total_return_btc': float(equity[-1] - 1), 'total_log_return_btc': float(returns.sum()),
            'total_return_usd': float(frame.equity_usd.iloc[-1] / initial_price - 1),
            'collateral_btc_hold_return_usd': float(frame.mark.iloc[-1] / initial_price - 1),
            'daily_sharpe_btc': float(daily.mean() / daily_std * math.sqrt(365)) if daily_std else 0.0,
            'mean_differential_sharpe': float(frame.differential_sharpe.mean()),
            'max_drawdown_btc': float(drawdown.min()), 'decisions': len(frame), 'days': len(daily),
            'orders': int(ledger.orders[0]), 'completed_round_trips': closes,
            'single_bar_round_trips': flickers,
            'flicker_share': float(flickers / closes) if closes else 0.0,
            'position_changes': int(len(changes)),
            'entries': entry_count,
            'long_entry_share': float((entries & (positions > 0)).sum() / entry_count) if entry_count else None,
            'reversals': int(((previous != 0) & (positions == -previous)).sum()),
            'median_position_spell_hours': float(np.median(spells) / 12),
            'median_holding_spell_hours': float(np.median(held_spells) / 12) if len(held_spells) else 0.0,
            'holding_spell_hours_p10': quantiles[0], 'holding_spell_hours_p50': quantiles[1],
            'holding_spell_hours_p90': quantiles[2],
            'side_share': {name: float(np.mean(positions == value))
                           for name, value in (('short', -1), ('flat', 0), ('long', 1))},
            'fees_btc': float(ledger.fees_total[0]), 'funding_btc': float(ledger.funding_total[0]),
            'turnover': float(ledger.turnover_total[0]), 'liquidations': int(ledger.liquidations[0]),
            'test_locked': True, 'test_performance_accessed': False}


@torch.no_grad()
def market_cache(branch, market, scaler, minutes, config):
    """Recurrent market memory for every step; it depends on the market alone.

    Matches training exactly: the state is rebuilt from a burn-in prefix at each
    rollout boundary.
    """
    features = market_tensor(scaler, market, minutes, config)
    memory = torch.zeros(len(minutes), branch.hidden)
    for begin in range(config.burn_in, len(minutes), config.rollout):
        end = min(len(minutes), begin + config.rollout)
        encoded, _ = branch.encode(features[begin - config.burn_in:end][None])
        memory[begin:end] = encoded[0, config.burn_in:]
    return memory


@torch.no_grad()
def rollout(models, market, scaler, config, bounds, memories, seed, baseline=None):
    """One sampled evaluation path. Always starts flat."""
    env = Environment(market, bounds, config, np.random.default_rng(0), training=False)
    first, last = bounds
    steps = (last - first) // 5
    initial_price = market.prices[first - 1, 3]
    generator = torch.Generator().manual_seed(seed)
    columns = {k: np.zeros(steps, dtype=d) for k, d in COLUMNS}
    for step in range(steps):
        _, account = env.observation()
        if baseline is None:
            offset = config.burn_in + step
            account_tensor = torch.from_numpy(account[:, None])
            probability = torch.stack([model.actor.read(memory[offset][None, None], account_tensor)[0, 0].exp()
                                       for model, memory in zip(models, memories)]).mean(0)
            action = int(torch.multinomial(probability, 1, generator=generator))
            columns['p_short'][step], columns['p_flat'][step], columns['p_long'][step] = probability.numpy()
        else:
            action = {'flat': FLAT, 'long': LONG, 'short': SHORT}[baseline]
        _, terminal, info = env.step(np.array([action]))
        columns['action'][step] = action
        for name in ('executed_action', 'contracts', 'log_return', 'differential_sharpe',
                     'equity_btc', 'equity_usd', 'funding_btc', 'fees_btc', 'filled'):
            columns[name][step] = info[name][0]
        columns['mark'][step] = market.prices[env.cursor[0] - 1, 3]
        columns['liquidation'][step] = terminal[0]
    frame = pd.DataFrame({'time': pd.date_range(market.timestamp(first), periods=steps, freq='5min'), **columns})
    result = metrics(frame, env.ledger, initial_price)
    ledger = env.ledger.state_dict()
    env.ledger._close(env.ledger.contracts != 0, market.prices[env.cursor - 1, 3])
    result.update({'return_btc_after_final_close': float(env.ledger.wallet[0] - 1),
                   'final_close_is_proxy': True, 'seed': seed,
                   'mean_probabilities': [float(frame[k].mean()) for k in ('p_short', 'p_flat', 'p_long')]})
    env.ledger.load_state_dict(ledger)
    return result, frame


SUMMARY_KEYS = ('total_log_return_btc', 'daily_sharpe_btc', 'orders', 'median_holding_spell_hours',
                'max_drawdown_btc', 'single_bar_round_trips', 'mean_differential_sharpe', 'entries')


def summarise(results, keys=SUMMARY_KEYS):
    """Distribution across sampling seeds, not a single number."""
    summary = {}
    for key in keys:
        values = [float(r[key]) for r in results]
        summary[key] = float(np.median(values))
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_min'] = float(np.min(values))
        summary[f'{key}_max'] = float(np.max(values))
    shares = [r['long_entry_share'] for r in results if r['long_entry_share'] is not None]
    summary['long_entry_share'] = float(np.median(shares)) if shares else None
    for side in ('short', 'flat', 'long'):
        summary[f'{side}_share'] = float(np.mean([r['side_share'][side] for r in results]))
    summary['positive_seed_share'] = float(np.mean([r['total_log_return_btc'] > 0 for r in results]))
    summary['liquidations'] = int(sum(r['liquidations'] for r in results))
    return summary


@torch.no_grad()
def evaluate(models, market, scaler, config, start, end, seeds=None, baseline=None, slippage=None):
    start, end = development_range(start, end)
    c = replace(config, slippage=config.slippage if slippage is None else slippage)
    bounds = market.bounds(start, end)
    steps = (bounds[1] - bounds[0]) // 5
    if steps < 1:
        raise ValueError('empty development interval')
    if baseline is None and not models:
        raise ValueError('policy required')
    memories = []
    if baseline is None:
        minutes = np.r_[bounds[0] - np.arange(c.burn_in, 0, -1) * 5, bounds[0] + np.arange(steps) * 5]
        for model in models:
            model.eval()
            memories.append(market_cache(model.actor, market, scaler, minutes, c))
    if baseline is not None:
        chosen = (0,)
    elif seeds is None:
        chosen = tuple(range(c.evaluation_seeds))
    elif isinstance(seeds, int):
        chosen = tuple(range(seeds))
    else:
        chosen = tuple(seeds)
    results, frames = [], []
    for seed in chosen:
        result, frame = rollout(models, market, scaler, c, bounds, memories, seed, baseline)
        results.append(result)
        frames.append(frame.assign(seed=seed))
    aggregate = {**summarise(results), 'per_seed': results, 'seeds': list(chosen),
                 'period': [str(start), str(end)], 'slippage': c.slippage,
                 'test_locked': True, 'test_performance_accessed': False}
    return aggregate, pd.concat(frames, ignore_index=True)


def quarter_windows(year):
    edges = [pd.Timestamp(f'{year}-{month:02d}-01', tz='UTC') for month in QUARTER_MONTHS]
    edges.append(pd.Timestamp(f'{year + 1}-01-01', tz='UTC'))
    return list(zip(edges[:-1], edges[1:]))


@torch.no_grad()
def evaluate_quarters(models, market, scaler, config, year, seeds=None, baseline=None, slippage=None):
    """Each calendar quarter of ``year`` from a flat account."""
    quarters, frames = [], []
    for number, (begin, finish) in enumerate(quarter_windows(year), 1):
        result, frame = evaluate(models, market, scaler, config, begin, finish, seeds=seeds,
                                 baseline=baseline, slippage=slippage)
        quarters.append({'quarter': number, **result})
        frames.append(frame.assign(quarter=number))
    return quarters, pd.concat(frames, ignore_index=True)
