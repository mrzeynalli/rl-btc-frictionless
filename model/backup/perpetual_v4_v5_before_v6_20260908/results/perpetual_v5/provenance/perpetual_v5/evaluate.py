"""Causal development evaluation with account-independent market caching."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
import torch

from perpetual.config import development_range
from perpetual.env import Environment, FLAT, LONG, SHORT
from perpetual.evaluate import metrics


@torch.no_grad()
def market_cache(branch, market, scaler, minutes, config):
    size = len(minutes)
    local = torch.empty(size, config.local_width)
    slow = torch.empty(size, market.slow.shape[-1])
    for begin in range(0, size, 512):
        end = min(size, begin+512)
        fast, context = scaler.observations(market, minutes[begin:end], config.fast_minutes)
        local[begin:end] = branch.local(torch.from_numpy(fast))
        slow[begin:end] = torch.from_numpy(context)
    encoded = branch.market_input(torch.cat([local, slow], -1))
    memory = torch.empty(size, config.hidden)
    memory[:config.burn_in] = 0
    for begin in range(config.burn_in, size, config.rollout):
        end = min(size, begin+config.rollout)
        _, hidden = branch.market(encoded[begin-config.burn_in:begin][None])
        memory[begin:end] = branch.market(encoded[begin:end][None], hidden)[0][0]
    return local, slow, memory


@torch.no_grad()
def evaluate(models, market, scaler, config, start, end, baseline=None, slippage=None):
    start, end = development_range(start, end)
    c = replace(config, slippage=config.slippage if slippage is None else slippage)
    bounds = market.bounds(start, end)
    env = Environment(market, bounds, c, np.random.default_rng(0), training=False)
    first, last = bounds
    steps = (last-first)//5
    if not steps:
        raise ValueError('empty development interval')
    initial_price = market.prices[first-1, 3]
    prefix_minute, prefix_account = env.initial_prefix()
    minutes = np.r_[prefix_minute[0], first+np.arange(steps)*5]
    account_history = np.zeros((len(minutes), 16), np.float32)
    account_history[:c.burn_in] = prefix_account[0]
    cache = []
    if baseline is None:
        if not models:
            raise ValueError('policy required')
        for model in models:
            model.eval()
            cache.append(market_cache(model.actor, market, scaler, minutes, c))
    hidden = [None]*len(models)
    columns = {k: np.zeros(steps, dtype=d) for k, d in (
        ('action', np.int64), ('executed_action', np.int64), ('contracts', float),
        ('log_return', float), ('equity_btc', float), ('equity_usd', float),
        ('mark', float), ('funding_btc', float), ('fees_btc', float),
        ('filled', bool), ('liquidation', bool), ('p_short', float), ('p_flat', float), ('p_long', float))}
    for step in range(steps):
        _, account = env.observation()
        offset = c.burn_in+step
        account_history[offset] = account[0]
        if baseline is None:
            opinions = []
            for i, model in enumerate(models):
                if step % c.rollout == 0:
                    _, hidden[i] = model.actor.position(torch.from_numpy(account_history[offset-c.burn_in:offset])[None])
                local, slow, memory = cache[i]
                logits, hidden[i] = model.actor.decision(local[offset][None, None], slow[offset][None, None],
                    torch.from_numpy(account[:, None]), memory[offset][None, None], hidden[i])
                opinions.append(logits[0, 0].exp())
            probability = torch.stack(opinions).mean(0)
            action = int(probability.argmax())
            columns['p_short'][step], columns['p_flat'][step], columns['p_long'][step] = probability.numpy()
        else:
            action = {'flat': FLAT, 'long': LONG, 'short': SHORT}[baseline]
        _, terminal, info = env.step(np.array([action]))
        columns['action'][step] = action
        for name in ('executed_action', 'contracts', 'log_return', 'equity_btc', 'equity_usd', 'funding_btc', 'fees_btc', 'filled'):
            columns[name][step] = info[name][0]
        columns['mark'][step] = market.prices[env.cursor[0]-1, 3]
        columns['liquidation'][step] = terminal[0]
    frame = pd.DataFrame({'time': pd.date_range(market.timestamp(first), periods=steps, freq='5min'), **columns})
    result = metrics(frame, env.ledger, initial_price)
    ledger = env.ledger.state_dict()
    env.ledger._close(env.ledger.contracts != 0, market.prices[env.cursor-1, 3])
    result.update({'return_btc_after_final_close': float(env.ledger.wallet[0]-1),
                   'final_close_is_proxy': True, 'period': [str(start), str(end)],
                   'mean_probabilities': [float(frame[k].mean()) for k in ('p_short', 'p_flat', 'p_long')]})
    env.ledger.load_state_dict(ledger)
    return result, frame
