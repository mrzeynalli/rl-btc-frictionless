"""Causal development evaluation with account-independent market caching.

Actions are the policy's most probable action ('argmax', USER 2026-09-15), so an
evaluation is one deterministic path; with evaluation_policy 'sampled' they are drawn
from the policy over several seeds and the distribution is reported, the protocol of
every result up to 2026-09-13. Training always samples. Validation folds are scored one
calendar quarter at a time, each from a flat account: the same length as a
training episode, and the window FinRL's ensemble strategy validates and trades
on (Yang et al. 2020).

Equity is USDT (v2). Passive references - constant long, short, flat - and two
time-series momentum rules are evaluated through the same ledger; they are reported and
never used for selection.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import torch

from .config import DEVELOPMENT_FOLDS, development_range
from .data import CANDIDATE_FEATURES, FEATURE_NAMES
from .ledger import Environment, FLAT, LONG, PORTFOLIO_NAMES, PORTFOLIO_WIDTH, SHORT
from .network import market_tensor

COLUMNS = (('action', np.int64), ('executed_action', np.int64), ('contracts', float),
           ('log_return', float), ('differential_ratio', float), ('equity', float),
           ('mark', float), ('funding', float), ('fees', float),
           ('filled', bool), ('liquidation', bool),
           ('p_short', float), ('p_flat', float), ('p_long', float), ('value', float))
QUARTER_MONTHS = (1, 4, 7, 10)
BASELINES = ('long', 'short', 'flat', 'momentum_1w', 'momentum_26w')
# Time-series momentum references: hold the sign of the volatility-normalised return over
# the look-back, revised once a day at 00:00 UTC. One week: Liu & Tsyvinski (2021);
# 26 weeks: Moskowitz, Ooi & Pedersen (2012). The daily revision is a DESIGN CHOICE that
# keeps the rules' turnover far below the agent's.
MOMENTUM_FEATURES = {'momentum_1w': 'return_1w', 'momentum_26w': 'return_26w'}


def metrics(frame, ledger, initial_price, hours_per_row):
    returns = frame.log_return.to_numpy()
    daily = np.expm1(frame.groupby(frame.time.dt.floor('D')).log_return.sum().to_numpy())
    equity = np.r_[1.0, frame.equity.to_numpy()]
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
    quantiles = (np.percentile(held_spells, [10, 50, 90]) * hours_per_row).tolist() if len(held_spells) else [0., 0., 0.]
    entries = (previous == 0) & (positions != 0)
    entry_count = int(entries.sum())
    return {'total_return': float(equity[-1] - 1), 'total_log_return': float(returns.sum()),
            'btc_price_return': float(frame.mark.iloc[-1] / initial_price - 1),
            'daily_sharpe': float(daily.mean() / daily_std * math.sqrt(365)) if daily_std else 0.0,
            'mean_differential_ratio': float(frame.differential_ratio.mean()),
            'max_drawdown': float(drawdown.min()), 'decisions': len(frame), 'days': len(daily),
            'orders': int(ledger.orders[0]), 'completed_round_trips': closes,
            'single_bar_round_trips': flickers,
            'flicker_share': float(flickers / closes) if closes else 0.0,
            'position_changes': int(len(changes)),
            'entries': entry_count,
            'long_entry_share': float((entries & (positions > 0)).sum() / entry_count) if entry_count else None,
            'reversals': int(((previous != 0) & (positions == -previous)).sum()),
            'median_position_spell_hours': float(np.median(spells) * hours_per_row),
            'median_holding_spell_hours': float(np.median(held_spells) * hours_per_row) if len(held_spells) else 0.0,
            'holding_spell_hours_p10': quantiles[0], 'holding_spell_hours_p50': quantiles[1],
            'holding_spell_hours_p90': quantiles[2],
            'side_share': {name: float(np.mean(positions == value))
                           for name, value in (('short', -1), ('flat', 0), ('long', 1))},
            'fees': float(ledger.fees_total[0]), 'funding': float(ledger.funding_total[0]),
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
def rollout(models, market, scaler, config, bounds, memories, seed, baseline=None, critic_memories=None):
    """One evaluation path, argmax or sampled as the configuration says. Always starts flat."""
    env = Environment(market, bounds, config, np.random.default_rng(0), training=False)
    first, last = bounds
    span = config.decision_bars * 5
    steps = (last - first) // span
    initial_price = market.prices[first - 1, 3]
    generator = torch.Generator().manual_seed(seed)
    columns = {k: np.zeros(steps, dtype=d) for k, d in COLUMNS}
    # Every input of every decision, so any decision can be reconstructed later.
    accounts = np.zeros((steps, PORTFOLIO_WIDTH), dtype=np.float32)
    momentum_column = CANDIDATE_FEATURES.index(MOMENTUM_FEATURES[baseline]) if baseline in MOMENTUM_FEATURES else None
    target = FLAT
    for step in range(steps):
        _, account = env.observation()
        accounts[step] = account[0]
        if baseline is None:
            offset = config.burn_in + step
            account_tensor = torch.from_numpy(account[:, None])
            probability = torch.stack([model.actor.read(memory[offset][None, None], account_tensor)[0, 0].exp()
                                       for model, memory in zip(models, memories)]).mean(0)
            if config.evaluation_policy == 'argmax':
                # torch.argmax returns the first maximum on an exact tie: SHORT, then FLAT, then LONG.
                action = int(probability.argmax())
            else:
                action = int(torch.multinomial(probability, 1, generator=generator))
            columns['p_short'][step], columns['p_flat'][step], columns['p_long'][step] = probability.numpy()
            if critic_memories:
                columns['value'][step] = float(torch.stack([
                    model.critic.read(memory[offset][None, None], account_tensor)[0, 0, 0]
                    for model, memory in zip(models, critic_memories)]).mean())
        elif momentum_column is not None:
            minute = int(env.cursor[0])
            if step == 0 or (market.epoch_start + minute) % 1440 == 0:
                signal = float(market.features[minute, momentum_column])
                target = LONG if signal > 0 else SHORT if signal < 0 else FLAT
            action = target
        else:
            action = {'flat': FLAT, 'long': LONG, 'short': SHORT}[baseline]
        _, terminal, info = env.step(np.array([action]))
        columns['action'][step] = action
        for name in ('executed_action', 'contracts', 'log_return', 'differential_ratio',
                     'equity', 'funding', 'fees', 'filled'):
            columns[name][step] = info[name][0]
        columns['mark'][step] = market.prices[env.cursor[0] - 1, 3]
        columns['liquidation'][step] = terminal[0]
    frame = pd.DataFrame({'time': pd.date_range(market.timestamp(first), periods=steps, freq=f'{span}min'), **columns})
    if baseline is None:
        minutes = first + np.arange(steps) * span
        observed = scaler.observations(market, minutes, config.observation_clip)
        frame = pd.concat([frame, pd.DataFrame(observed, columns=[f'z_{name}' for name in FEATURE_NAMES]),
                           pd.DataFrame(accounts, columns=[f'account_{name}' for name in PORTFOLIO_NAMES])], axis=1)
    result = metrics(frame, env.ledger, initial_price, span / 60)
    ledger = env.ledger.state_dict()
    env.ledger._close(env.ledger.contracts != 0, market.prices[env.cursor - 1, 3])
    result.update({'return_after_final_close': float(env.ledger.wallet[0] - 1),
                   'final_close_is_proxy': True, 'seed': seed,
                   'mean_probabilities': [float(frame[k].mean()) for k in ('p_short', 'p_flat', 'p_long')]})
    env.ledger.load_state_dict(ledger)
    return result, frame


SUMMARY_KEYS = ('total_log_return', 'daily_sharpe', 'orders', 'median_holding_spell_hours',
                'max_drawdown', 'single_bar_round_trips', 'mean_differential_ratio', 'entries')


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
    summary['positive_seed_share'] = float(np.mean([r['total_log_return'] > 0 for r in results]))
    summary['liquidations'] = int(sum(r['liquidations'] for r in results))
    return summary


@torch.no_grad()
def evaluate(models, market, scaler, config, start, end, seeds=None, baseline=None, slippage=None):
    start, end = development_range(start, end)
    c = replace(config, slippage=config.slippage if slippage is None else slippage)
    bounds = market.bounds(start, end)
    span = c.decision_bars * 5
    steps = (bounds[1] - bounds[0]) // span
    if steps < 1:
        raise ValueError('empty development interval')
    if baseline is not None and baseline not in BASELINES:
        raise ValueError(f'unknown baseline {baseline}')
    if baseline is None and not models:
        raise ValueError('policy required')
    memories, critic_memories = [], []
    if baseline is None:
        minutes = np.r_[bounds[0] - np.arange(c.burn_in, 0, -1) * span, bounds[0] + np.arange(steps) * span]
        for model in models:
            model.eval()
            memories.append(market_cache(model.actor, market, scaler, minutes, c))
            critic_memories.append(market_cache(model.critic, market, scaler, minutes, c))
    if baseline is not None or c.evaluation_policy == 'argmax':
        # A constant or a deterministic policy has one path; more seeds would repeat it.
        chosen = (0,)
    elif seeds is None:
        chosen = tuple(range(c.evaluation_seeds))
    elif isinstance(seeds, int):
        chosen = tuple(range(seeds))
    else:
        chosen = tuple(seeds)
    results, frames = [], []
    for seed in chosen:
        result, frame = rollout(models, market, scaler, c, bounds, memories, seed, baseline, critic_memories)
        results.append(result)
        frames.append(frame.assign(seed=seed))
    aggregate = {**summarise(results), 'per_seed': results, 'seeds': list(chosen),
                 'period': [str(start), str(end)], 'slippage': c.slippage,
                 'test_locked': True, 'test_performance_accessed': False}
    return aggregate, pd.concat(frames, ignore_index=True)


def quarter_windows(year, count=4):
    """The first ``count`` calendar quarters of ``year``."""
    edges = [pd.Timestamp(f'{year}-{month:02d}-01', tz='UTC') for month in QUARTER_MONTHS]
    edges.append(pd.Timestamp(f'{year + 1}-01-01', tz='UTC'))
    return list(zip(edges[:-1], edges[1:]))[:count]


@torch.no_grad()
def evaluate_quarters(models, market, scaler, config, year, seeds=None, baseline=None, slippage=None, count=None):
    """Each validation quarter of the fold ``year`` from a flat account."""
    count = config.fold_quarters.get(year, DEVELOPMENT_FOLDS.get(year)) if count is None else count
    quarters, frames = [], []
    for number, (begin, finish) in enumerate(quarter_windows(year, count), 1):
        result, frame = evaluate(models, market, scaler, config, begin, finish, seeds=seeds,
                                 baseline=baseline, slippage=slippage)
        quarters.append({'quarter': number, **result})
        frames.append(frame.assign(quarter=number))
    return quarters, pd.concat(frames, ignore_index=True)
