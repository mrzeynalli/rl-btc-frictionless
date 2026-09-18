"""Development evaluation only. Test dates are rejected before data access."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import torch

from .config import development_range
from .env import Environment, FLAT, LONG, SHORT


def metrics(frame, ledger, initial_price):
    returns = frame.log_return.to_numpy()
    daily = np.expm1(frame.groupby(frame.time.dt.floor("D")).log_return.sum().to_numpy())
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
    # A realized terminal liquidation is separately reported in the frame;
    # final equity here is marked wealth before discretionary final closure.
    return {"total_return_btc": float(equity[-1] - 1), "total_log_return_btc": float(returns.sum()),
            "total_return_usd": float(frame.equity_usd.iloc[-1] / initial_price - 1),
            "collateral_btc_hold_return_usd": float(frame.mark.iloc[-1] / initial_price - 1),
            "daily_sharpe_btc": float(daily.mean() / daily_std * math.sqrt(365)) if daily_std else 0.0,
            "max_drawdown_btc": float(drawdown.min()), "decisions": len(frame), "days": len(daily),
            "orders": int(ledger.orders[0]), "completed_round_trips": closes,
            "position_changes": int(len(changes)), "median_position_spell_hours": float(np.median(spells) / 12),
            "median_holding_spell_hours": float(np.median(held_spells) / 12) if len(held_spells) else 0.0,
            "side_share": {name: float(np.mean(positions == value)) for name, value in (("short", -1), ("flat", 0), ("long", 1))},
            "fees_btc": float(ledger.fees_total[0]), "funding_btc": float(ledger.funding_total[0]),
            "turnover": float(ledger.turnover_total[0]), "liquidations": int(ledger.liquidations[0]),
            "test_locked": True, "test_performance_accessed": False}


@torch.no_grad()
def evaluate(models, market, scaler, config, start, end, baseline=None, slippage=None):
    start, end = development_range(start, end)
    config = replace(config, slippage=config.slippage if slippage is None else slippage)
    bounds = market.bounds(start, end)
    env = Environment(market, bounds, config, np.random.default_rng(0), training=False)
    first, last = bounds
    steps = (last - first) // 5
    if steps < 1:
        raise ValueError("empty validation interval")
    initial_price = market.prices[first - 1, 3]
    minutes = first + np.arange(steps) * 5
    prefix_minutes, prefix_portfolio = env.initial_prefix()
    all_minutes = np.r_[prefix_minutes[0], minutes]
    history = np.zeros((len(all_minutes), prefix_portfolio.shape[-1]), np.float32)
    history[:config.burn_in] = prefix_portfolio[0]
    cached = []
    if baseline is None:
        if not models:
            raise ValueError("at least one policy required")
        for model in models:
            model.eval()
        # Local market compression is account independent; compute it in
        # batches. Only the decision GRU and head depend on the evolving book.
        cached = [np.empty((len(all_minutes), config.local_width + market.slow.shape[-1]), np.float32) for _ in models]
        for begin in range(0, len(all_minutes), 2048):
            finish = min(len(all_minutes), begin + 2048)
            fast, slow = scaler.observations(market, all_minutes[begin:finish], config.fast_minutes)
            fast = torch.from_numpy(fast).transpose(1, 2)
            for index, model in enumerate(models):
                local = model.actor.local(fast).squeeze(-1).numpy()
                cached[index][begin:finish] = np.concatenate([local, slow], axis=1)
        cached = [torch.from_numpy(x) for x in cached]
    hidden = [None] * len(models)
    rows = []
    for step in range(steps):
        minute, portfolio = env.observation()
        offset = config.burn_in + step
        history[offset] = portfolio[0]
        if baseline is None:
            opinions = []
            for index, model in enumerate(models):
                branch = model.actor
                # Match the finite-prefix refresh used at training rollouts.
                if step % config.rollout == 0:
                    p = torch.from_numpy(history[offset - config.burn_in:offset])
                    z = branch.input(torch.cat([cached[index][offset - config.burn_in:offset], p], dim=-1))
                    _, hidden[index] = branch.gru(z.unsqueeze(0))
                z = branch.input(torch.cat([cached[index][offset], torch.from_numpy(portfolio[0])], dim=-1))
                output, hidden[index] = branch.gru(z.reshape(1, 1, -1), hidden[index])
                opinions.append(torch.softmax(branch.head(output[0, 0]), dim=-1))
            probabilities = torch.stack(opinions).mean(0)
            action = int(probabilities.argmax())
        else:
            action = {"flat": FLAT, "long": LONG, "short": SHORT}[baseline]
        reward, terminal, info = env.step(np.array([action]))
        rows.append({"time": market.timestamp(minute[0]), "action": action,
                     "executed_action": int(info["executed_action"][0]),
                     "contracts": float(info["contracts"][0]), "log_return": float(info["log_return"][0]),
                     "equity_btc": float(info["equity_btc"][0]), "equity_usd": float(info["equity_usd"][0]),
                     "mark": float(market.prices[env.cursor[0] - 1, 3]),
                     "funding_btc": float(info["funding_btc"][0]), "fees_btc": float(info["fees_btc"][0]),
                     "filled": bool(info["filled"][0]), "liquidation": bool(terminal[0])})
    frame = pd.DataFrame(rows)
    result = metrics(frame, env.ledger, initial_price)
    mark = market.prices[env.cursor - 1, 3]
    ledger_state = env.ledger.state_dict()
    env.ledger._close(env.ledger.contracts != 0, mark)
    result["return_btc_after_final_close"] = float(env.ledger.wallet[0] - 1)
    result["final_close_is_proxy"] = True
    env.ledger.load_state_dict(ledger_state)
    result["period"] = [str(start), str(end)]
    return result, frame
