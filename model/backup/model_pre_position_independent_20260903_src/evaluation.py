"""Causal evaluation, performance metrics, benchmarks and matched-turnover nulls.

Evaluation is a single continuous walk through the window.  Weights are frozen;
only the GRU hidden state evolves, and it is warmed over the 240 hours that
precede the window so the state entering it is causal rather than an artificial
zero.  Every policy -- learned or benchmark -- enters the window FLAT and pays
the full cost of its first position, so none of them gets a free entry.

Actions are deterministic argmax at validation and test, as the protocol
requires.  The trunk is run in one fused pass because the GRU never sees the
position: only the head does, so the recurrent stream is action-independent and
can be computed ahead of the decision loop without any leakage.

Sharpe, Sortino and drawdown appear here and only here.  They are measurements,
never terms in the reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from env import ACTION_POSITIONS, ONE_WAY_COST, position_ladder, realised_return
from features import MarketData, SplitData

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class Trajectory:
    """One evaluated path: the action taken every hour and what it earned."""

    positions: np.ndarray
    rewards: np.ndarray  # log equity change, unscaled
    price_return: np.ndarray
    funding_return: np.ndarray
    cost_paid: np.ndarray
    hypothetical_cost: np.ndarray
    equity: np.ndarray
    action_probabilities: np.ndarray | None = None
    action_top_two_margins: np.ndarray | None = None
    action_switch_margins: np.ndarray | None = None


def account_positions(
    positions: np.ndarray,
    market: MarketData,
    first: int,
    last: int,
    cost_rate: float,
    entry_position: float = 0.0,
) -> Trajectory:
    """Price, funding, cost and compounded equity for a given action path.

    The cost rate never enters the state, so the same action path can be
    re-priced at another cost level without re-running the policy.
    """
    index = np.arange(first, last)
    previous = np.concatenate([[entry_position], positions[:-1]])
    price_ratio = market.exit_price[index] / market.entry_price[index]
    price, funding, cost, total = realised_return(
        positions, previous, price_ratio,
        market.funding_on_previous[index], market.funding_on_current[index], cost_rate,
    )
    _, _, hypothetical, _ = realised_return(
        positions, previous, price_ratio,
        market.funding_on_previous[index], market.funding_on_current[index], ONE_WAY_COST,
    )
    growth = np.maximum(1.0 + total, 1e-6)
    equity = np.cumprod(growth)
    return Trajectory(
        positions=positions,
        rewards=np.log(growth),
        price_return=price,
        funding_return=funding,
        cost_paid=cost,
        hypothetical_cost=hypothetical,
        equity=equity,
    )


@torch.no_grad()
def run_policy(
    model: torch.nn.Module,
    split: SplitData,
    first: int,
    last: int,
    cost_rate: float,
    warmup_start: int | None = None,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
    ladder: str = "ternary",
    switch_margin: float = 0.0,
) -> Trajectory:
    """Walk ``[first, last)`` with frozen weights and a causally warmed state."""
    model.eval()
    market = split.market
    warmup_start = split.evaluation_warmup_start if warmup_start is None else warmup_start
    stream_first = min(warmup_start, first)

    fast = torch.from_numpy(split.fast[stream_first:last]).unsqueeze(0)
    hidden = model.initial_hidden(1)
    hourly, _ = model.trunk(fast, hidden)
    hourly = hourly[0]  # [steps, hidden]

    slow = torch.from_numpy(split.slow[stream_first:last])
    steps = last - stream_first
    offset = first - stream_first

    levels = position_ladder(ladder)
    action_count = len(levels)
    flat_index = int(np.searchsorted(levels, 0.0))
    position_index = flat_index
    positions = np.zeros(last - first, dtype=np.float64)
    probabilities = np.zeros((last - first, action_count), dtype=np.float64)
    top_two_margins = np.zeros(last - first, dtype=np.float64)
    switch_margins = np.zeros(last - first, dtype=np.float64)
    one_hot = torch.zeros(action_count)
    for step in range(steps):
        if step == offset:
            position_index = flat_index  # every policy enters the window flat
        one_hot.zero_()
        one_hot[position_index] = 1.0
        joined = torch.cat([hourly[step], slow[step], one_hot]).unsqueeze(0)
        logits = model.actor_output(model.activation(model.actor_hidden(joined)))[0]
        action_probabilities = torch.softmax(logits, dim=-1)
        if deterministic:
            best = int(torch.argmax(logits))
            improvement = float(
                action_probabilities[best] - action_probabilities[position_index]
            )
            action = (
                position_index
                if best != position_index and improvement <= switch_margin
                else best
            )
        else:
            action = int(torch.multinomial(action_probabilities, 1, generator=generator))
        if step >= offset:
            positions[step - offset] = levels[action]
            probabilities[step - offset] = action_probabilities.numpy()
            top = torch.topk(action_probabilities, 2).values
            top_two_margins[step - offset] = float(top[0] - top[1])
            switch_margins[step - offset] = float(
                action_probabilities.max() - action_probabilities[position_index]
            )
        position_index = action

    trajectory = account_positions(positions, market, first, last, cost_rate)
    return Trajectory(
        positions=trajectory.positions,
        rewards=trajectory.rewards,
        price_return=trajectory.price_return,
        funding_return=trajectory.funding_return,
        cost_paid=trajectory.cost_paid,
        hypothetical_cost=trajectory.hypothetical_cost,
        equity=trajectory.equity,
        action_probabilities=probabilities,
        action_top_two_margins=top_two_margins,
        action_switch_margins=switch_margins,
    )


def policy_diagnostics(trajectory: Trajectory) -> dict:
    """Confidence, ambiguity and switching diagnostics for one policy path."""
    probabilities = trajectory.action_probabilities
    if probabilities is None:
        return {}
    top_two = trajectory.action_top_two_margins
    if top_two is None:
        ordered = np.sort(probabilities, axis=1)
        top_two = ordered[:, -1] - ordered[:, -2]
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
    )
    result = {
        "maximum_action_probability_mean": float(probabilities.max(axis=1).mean()),
        "action_probability_entropy_mean": float(entropy.mean()),
        "top_two_margin_mean": float(top_two.mean()),
        "top_two_margin_median": float(np.median(top_two)),
    }
    for threshold in (0.01, 0.05, 0.10):
        result[f"share_top_two_margin_below_{threshold:.2f}"] = float(
            np.mean(top_two < threshold)
        )
    return result


# -- metrics ---------------------------------------------------------------


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak) - 1.0)


def metrics(trajectory: Trajectory, name: str = "") -> dict:
    rewards = trajectory.rewards
    hours = len(rewards)
    years = hours / HOURS_PER_YEAR
    total_return = float(trajectory.equity[-1] - 1.0) if hours else 0.0
    log_total = float(rewards.sum())
    hourly_mean = float(rewards.mean()) if hours else 0.0
    hourly_std = float(rewards.std(ddof=1)) if hours > 1 else 0.0
    downside = rewards[rewards < 0.0]
    downside_std = float(np.sqrt(np.mean(downside**2))) if downside.size else 0.0
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], trajectory.positions]))).sum())
    changes = int(np.count_nonzero(np.diff(np.concatenate([[0.0], trajectory.positions]))))
    non_flat = trajectory.positions != 0.0
    return {
        "name": name,
        "hours": hours,
        "years": years,
        "total_return": total_return,
        "total_log_return": log_total,
        "annualised_return": float(math.exp(log_total / years) - 1.0) if years > 0 else 0.0,
        "annualised_log_return": log_total / years if years > 0 else 0.0,
        "annualised_volatility": hourly_std * math.sqrt(HOURS_PER_YEAR),
        "sharpe": (hourly_mean / hourly_std * math.sqrt(HOURS_PER_YEAR)) if hourly_std > 0 else 0.0,
        "sortino": (hourly_mean / downside_std * math.sqrt(HOURS_PER_YEAR)) if downside_std > 0 else 0.0,
        "max_drawdown": _max_drawdown(trajectory.equity) if hours else 0.0,
        "calmar": (
            float(math.exp(log_total / years) - 1.0) / abs(_max_drawdown(trajectory.equity))
            if hours and _max_drawdown(trajectory.equity) < 0 and years > 0 else 0.0
        ),
        "hit_rate": float(np.mean(rewards[non_flat] > 0)) if non_flat.any() else 0.0,
        "turnover_total": turnover,
        "turnover_per_hour": turnover / hours if hours else 0.0,
        "position_changes": changes,
        "average_holding_hours": hours / changes if changes else float(hours),
        "share_long": float(np.mean(trajectory.positions > 0)) if hours else 0.0,
        "share_flat": float(np.mean(trajectory.positions == 0)) if hours else 0.0,
        "share_short": float(np.mean(trajectory.positions < 0)) if hours else 0.0,
        "price_pnl": float(trajectory.price_return.sum()),
        "funding_pnl": float(trajectory.funding_return.sum()),
        "cost_pnl": -float(trajectory.cost_paid.sum()),
        "hypothetical_cost_pnl": -float(trajectory.hypothetical_cost.sum()),
    }


# -- benchmarks -------------------------------------------------------------


def always_flat(split: SplitData, first: int, last: int, cost_rate: float) -> Trajectory:
    return account_positions(np.zeros(last - first), split.market, first, last, cost_rate)


def always_long(split: SplitData, first: int, last: int, cost_rate: float) -> Trajectory:
    return account_positions(np.ones(last - first), split.market, first, last, cost_rate)


def time_series_momentum(
    split: SplitData, first: int, last: int, cost_rate: float, lookback_hours: int
) -> Trajectory:
    """Sign of the trailing ``lookback_hours`` return, evaluated causally."""
    market = split.market
    index = np.arange(first, last)
    # entry_price[i] is the open at tau_i; the trailing return known at the
    # decision is the one ending at that same open.
    past = market.entry_price[np.maximum(index - lookback_hours, 0)]
    signal = np.sign(market.entry_price[index] / past - 1.0)
    return account_positions(signal.astype(np.float64), market, first, last, cost_rate)


MOMENTUM_LOOKBACKS = (24, 72, 168, 240)


# -- matched-turnover nulls -------------------------------------------------


def circular_shift_null(
    positions: np.ndarray, split: SplitData, first: int, last: int,
    cost_rate: float, draws: int, rng: np.random.Generator,
) -> dict:
    """Rotate the realised action sequence against history.

    A rotation preserves the number of positions of each kind and every holding
    length -- circular turnover is exactly invariant, and the turnover actually
    charged can differ only by the single wrapped transition -- while destroying
    any alignment between the action and the return it was chosen for.
    """
    hours = last - first
    if hours < 32:
        return {"draws": 0}
    offsets = rng.choice(np.arange(1, hours), size=min(draws, hours - 1), replace=False)
    totals = np.array(
        [
            account_positions(np.roll(positions, int(offset)), split.market, first, last, cost_rate)
            .rewards.sum()
            for offset in offsets
        ]
    )
    actual = account_positions(positions, split.market, first, last, cost_rate).rewards.sum()
    return {
        "draws": int(len(totals)),
        "actual_total_log_return": float(actual),
        "null_mean": float(totals.mean()),
        "null_std": float(totals.std(ddof=1)),
        "percentile_of_actual": float((totals < actual).mean() * 100.0),
        "one_sided_p_value": float((totals >= actual).mean()),
    }


def markov_matched_null(
    positions: np.ndarray, split: SplitData, first: int, last: int,
    cost_rate: float, draws: int, rng: np.random.Generator,
    ladder: str = "ternary",
) -> dict:
    """Random policies whose transition matrix matches the evaluated policy.

    Matching the full transition matrix over the action ladder matches the
    turnover rate and the holding-time distribution, not merely the number of
    trades.
    """
    levels = position_ladder(ladder)
    count = len(levels)
    states = np.searchsorted(levels, positions).astype(np.int64)
    transitions = np.ones((count, count)) * 1e-9
    for previous, current in zip(states[:-1], states[1:]):
        transitions[previous, current] += 1.0
    transitions /= transitions.sum(axis=1, keepdims=True)

    hours = last - first
    cumulative = np.cumsum(transitions, axis=1)
    paths = np.empty((draws, hours), dtype=np.int64)
    state = np.full(draws, int(states[0]), dtype=np.int64)
    uniform = rng.random((hours, draws))
    for step in range(hours):
        state = (uniform[step, :, None] > cumulative[state]).sum(axis=1)
        paths[:, step] = state
    sampled = levels[paths]
    totals = np.array(
        [
            account_positions(path, split.market, first, last, cost_rate).rewards.sum()
            for path in sampled
        ]
    )
    actual = account_positions(positions, split.market, first, last, cost_rate).rewards.sum()
    return {
        "draws": int(draws),
        "actual_total_log_return": float(actual),
        "null_mean": float(totals.mean()),
        "null_std": float(totals.std(ddof=1)),
        "percentile_of_actual": float((totals < actual).mean() * 100.0),
        "one_sided_p_value": float((totals >= actual).mean()),
        "matched_turnover_per_hour": float(np.abs(np.diff(sampled, axis=1)).mean()),
        "policy_turnover_per_hour": float(np.abs(np.diff(positions)).mean()),
    }


def benchmark_suite(
    split: SplitData, first: int, last: int, cost_rate: float, momentum_lookback: int
) -> dict:
    return {
        "always_flat": metrics(always_flat(split, first, last, cost_rate), "always_flat"),
        "always_long": metrics(always_long(split, first, last, cost_rate), "always_long"),
        f"momentum_{momentum_lookback}h": metrics(
            time_series_momentum(split, first, last, cost_rate, momentum_lookback),
            f"momentum_{momentum_lookback}h",
        ),
    }
