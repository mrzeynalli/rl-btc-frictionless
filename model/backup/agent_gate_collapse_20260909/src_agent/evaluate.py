"""Causal development evaluation with account-independent market caching.

Actions are SAMPLED from the policy, never taken by argmax.

That is not a stylistic preference. The policy factorises into a gate,
P(keep the current side), and a direction, P(side | switch). Leaving flat under
argmax would require P(switch) > 0.5, that is a mean flat spell under ten
minutes, while the whole design targets holds of roughly a day. Argmax and a
correctly calibrated holding prior are mutually exclusive: the predecessor
evaluated by argmax and placed exactly zero orders in 420,000 decisions across
every completed run, while the same checkpoints sampled placed 101 to 1272.
Argmax does not summarise a persistence-gated policy, it replaces it with a
different and degenerate one.

Because a single sampled path is noisy, every evaluation runs several seeds and
reports the distribution rather than one number.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import torch

from .config import development_range
from .ledger import Environment, FLAT, LONG, SHORT

COLUMNS = (('action', np.int64), ('executed_action', np.int64), ('contracts', float),
           ('log_return', float), ('differential_sharpe', float), ('equity_btc', float),
           ('equity_usd', float), ('mark', float), ('funding_btc', float), ('fees_btc', float),
           ('filled', bool), ('liquidation', bool),
           ('p_short', float), ('p_flat', float), ('p_long', float))


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
    # A single-bar position is a leverage reset, not a trade: closing and
    # reopening resizes the notional to 1x of the now larger equity. Counting
    # those as round trips is what made the predecessor's best run look like it
    # had taken sixteen trades when it had taken none.
    flickers = int((held_spells <= 1).sum()) if len(held_spells) else 0
    quantiles = (np.percentile(held_spells, [10, 50, 90]) / 12).tolist() if len(held_spells) else [0., 0., 0.]
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
    """Depends only on the market, so it is shared across sampling seeds."""
    size = len(minutes)
    local = torch.empty(size, branch.local_width)
    slow = torch.empty(size, market.slow.shape[-1])
    for begin in range(0, size, 512):
        end = min(size, begin + 512)
        fast, context = scaler.observations(market, minutes[begin:end], config.fast_minutes)
        local[begin:end] = branch.local(torch.from_numpy(fast))
        slow[begin:end] = torch.from_numpy(context)
    encoded = branch.market_input(torch.cat([local, slow], -1))
    memory = torch.empty(size, branch.hidden)
    memory[:config.burn_in] = 0
    for begin in range(config.burn_in, size, config.rollout):
        end = min(size, begin + config.rollout)
        _, hidden = branch.market(encoded[begin - config.burn_in:begin][None])
        memory[begin:end] = branch.market(encoded[begin:end][None], hidden)[0][0]
    return local, slow, memory


@torch.no_grad()
def rollout(models, market, scaler, config, bounds, caches, seed, baseline=None):
    """One sampled evaluation path. Always starts flat."""
    env = Environment(market, bounds, config, np.random.default_rng(0), training=False)
    first, last = bounds
    steps = (last - first) // 5
    initial_price = market.prices[first - 1, 3]
    prefix_minute, prefix_account = env.initial_prefix()
    account_history = np.zeros((steps + config.burn_in, prefix_account.shape[-1]), np.float32)
    account_history[:config.burn_in] = prefix_account[0]
    generator = torch.Generator().manual_seed(seed)
    hidden = [None] * len(models)
    columns = {k: np.zeros(steps, dtype=d) for k, d in COLUMNS}
    for step in range(steps):
        _, account = env.observation()
        offset = config.burn_in + step
        account_history[offset] = account[0]
        if baseline is None:
            opinions = []
            for i, model in enumerate(models):
                if step % config.rollout == 0:
                    _, hidden[i] = model.actor.position(
                        torch.from_numpy(account_history[offset - config.burn_in:offset])[None])
                local, slow, memory = caches[i]
                logits, hidden[i] = model.actor.decision(
                    local[offset][None, None], slow[offset][None, None],
                    torch.from_numpy(account[:, None]), memory[offset][None, None], hidden[i])
                opinions.append(logits[0, 0].exp())
            probability = torch.stack(opinions).mean(0)
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


def summarise(results, keys=('total_log_return_btc', 'daily_sharpe_btc', 'orders',
                             'median_holding_spell_hours', 'max_drawdown_btc',
                             'single_bar_round_trips', 'mean_differential_sharpe')):
    """Distribution across sampling seeds, not a single number."""
    summary = {}
    for key in keys:
        values = [float(r[key]) for r in results]
        summary[key] = float(np.median(values))
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_min'] = float(np.min(values))
        summary[f'{key}_max'] = float(np.max(values))
    summary['positive_seed_share'] = float(np.mean([r['total_log_return_btc'] > 0 for r in results]))
    summary['liquidations'] = int(sum(r['liquidations'] for r in results))
    return summary


@torch.no_grad()
def evaluate(models, market, scaler, config, start, end, seeds=None, baseline=None, slippage=None):
    start, end = development_range(start, end)
    c = replace(config, slippage=config.slippage if slippage is None else slippage)
    bounds = market.bounds(start, end)
    if (bounds[1] - bounds[0]) // 5 < 1:
        raise ValueError('empty development interval')
    if baseline is None and not models:
        raise ValueError('policy required')
    caches = []
    if baseline is None:
        steps = (bounds[1] - bounds[0]) // 5
        prefix = Environment(market, bounds, c, np.random.default_rng(0), training=False).initial_prefix()[0][0]
        minutes = np.r_[prefix, bounds[0] + np.arange(steps) * 5]
        for model in models:
            model.eval()
            caches.append(market_cache(model.actor, market, scaler, minutes, c))
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
        result, frame = rollout(models, market, scaler, c, bounds, caches, seed, baseline)
        results.append(result)
        frames.append(frame.assign(seed=seed))
    aggregate = {**summarise(results), 'per_seed': results, 'seeds': list(chosen),
                 'period': [str(start), str(end)], 'slippage': c.slippage,
                 'test_locked': True, 'test_performance_accessed': False}
    return aggregate, pd.concat(frames, ignore_index=True)


def block_windows(year, config):
    """Overlapping fixed-length windows inside one validation year.

    Three annual folds give three samples, which is why the predecessor's
    architecture comparison could not resolve anything: the outcome of a year is
    dominated by that year's direction. Overlapping blocks spanning trend and
    chop turn the same data into a distribution.
    """
    start = pd.Timestamp(f'{year}-01-01', tz='UTC')
    stop = pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
    windows, cursor = [], start
    while cursor + pd.Timedelta(days=config.block_days) <= stop:
        windows.append((cursor, cursor + pd.Timedelta(days=config.block_days)))
        cursor += pd.Timedelta(days=config.block_step_days)
    return windows


@torch.no_grad()
def evaluate_blocks(models, market, scaler, config, year, seeds=None, slippage=None):
    """Distribution over overlapping blocks; each block starts flat."""
    rows = []
    for begin, finish in block_windows(year, config):
        result, _ = evaluate(models, market, scaler, config, begin, finish, seeds=seeds, slippage=slippage)
        rows.append({'start': str(begin), 'end': str(finish),
                     **{k: result[k] for k in ('total_log_return_btc', 'daily_sharpe_btc', 'orders',
                                               'median_holding_spell_hours', 'max_drawdown_btc',
                                               'positive_seed_share')}})
    values = np.array([r['total_log_return_btc'] for r in rows]) if rows else np.zeros(1)
    return {'year': year, 'blocks': rows, 'block_count': len(rows),
            'median_block_log_return': float(np.median(values)),
            'mean_block_log_return': float(np.mean(values)),
            'positive_block_share': float(np.mean(values > 0)),
            'worst_decile_log_return': float(np.percentile(values, 10)),
            'block_days': config.block_days, 'block_step_days': config.block_step_days,
            'slippage': config.slippage if slippage is None else slippage,
            'test_locked': True, 'test_performance_accessed': False}
