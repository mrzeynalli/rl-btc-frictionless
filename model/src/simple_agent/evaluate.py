"""Run a policy (or a benchmark rule) over one period from a flat account, and score it.

The agent takes its most probable action each hour (argmax). Before the period starts, the week before it is fed
through the LSTM so the memory is warm; no positions are taken during that week.
Scores:
    total log return   = sum_t ln(1 + R_t)                       (what the model works with)
    total return       = exp(total log return) - 1                (the actual gain, e.g. 0.25 = +25%)
    equity from $10k   = 10,000 * exp(total log return)            (USER 2026-09-21: report actual money)
    Sharpe (annual)    = mean(R_t) / std(R_t) * sqrt(8760)            from hourly returns
    max drawdown       = min_t (equity_t / max_{s<=t} equity_s - 1),  equity_t = prod (1 + R_s)
"""
import numpy as np
import torch

from .config import ACTIONS, HOURS_PER_YEAR

POSITIONS = np.array(ACTIONS, dtype=np.float64)


def score(R, positions):
    equity = np.cumprod(1 + R)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    sd = R.std()
    log_return = float(np.log1p(R).sum())
    return {'total_log_return': log_return,
            'total_return_pct': float(np.expm1(log_return) * 100),          # actual gain in percent
            'equity_from_10k': float(10_000 * np.exp(log_return)),           # final equity of a $10,000 account
            'sharpe': float(R.mean() / sd * np.sqrt(HOURS_PER_YEAR)) if sd > 0 else 0.0,
            'max_drawdown': float((equity / peak - 1).min()),
            'position_changes': int((np.diff(np.r_[0.0, positions]) != 0).sum()),
            'long_share': float((positions > 0).mean()), 'short_share': float((positions < 0).mean()),
            'flat_share': float((positions == 0).mean()), 'hours': int(len(R))}


@torch.no_grad()
def run_policy(model, features, returns, period_rows, warmup_hours):
    """Trade the rows `period_rows` (consecutive); returns (score, positions, hourly returns, P(short/flat/long))."""
    first = period_rows[0]
    rows = period_rows[:-1]                     # the period's last row has no next hour inside the period
    state = model.actor.initial_state(1)
    no_reset = torch.zeros(1, 1)
    flat = np.array([0, 1, 0], np.float32)
    for i in range(max(0, first - warmup_hours), first):          # warm the memory, flat, no trading
        obs = torch.from_numpy(np.r_[features[i], flat])[None, None]
        _, state = model.actor(obs, state, no_reset)
    position, positions, probabilities = 0.0, [], []
    for i in rows:
        one_hot = np.eye(3, dtype=np.float32)[int(position) + 1]
        obs = torch.from_numpy(np.r_[features[i], one_hot])[None, None]
        logits, state = model.actor(obs, state, no_reset)
        p = torch.softmax(logits[0, 0], -1).numpy()
        position = POSITIONS[int(p.argmax())]                       # argmax: the most probable position
        positions.append(position)
        probabilities.append(p)
    positions = np.array(positions)
    R = positions * returns[rows]                                   # R_{t+1} = a_t * (P_{t+1}/P_t - 1)
    return score(R, positions), positions, R, np.array(probabilities)


def benchmarks(table, returns, period_rows, seed=0, random_paths=100):
    """Rules on the same rows and the same cost-free account."""
    rows = period_rows[:-1]
    r = returns[rows]
    close = table.close.to_numpy(np.float64)
    # 50-day moving average of hourly closes (1,200 hours): long above it, short below (Brock et al. 1992).
    sma = np.array([close[max(0, i - 1199):i + 1].mean() for i in rows])
    ma = np.sign(close[rows] - sma)
    rng = np.random.default_rng(seed)
    random_scores = [score(p * r, p)['total_log_return']
                     for p in (POSITIONS[rng.integers(0, 3, len(rows))] for _ in range(random_paths))]
    return {'buy_and_hold': score(r, np.ones(len(rows))),
            'sma_50d': score(ma * r, ma),
            'random_uniform_total_log_return': {'median': float(np.median(random_scores)),
                                                'p05': float(np.percentile(random_scores, 5)),
                                                'p95': float(np.percentile(random_scores, 95))}}
