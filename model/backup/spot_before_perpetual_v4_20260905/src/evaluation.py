"""Causal Spot-policy and equal-weight ensemble evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from env import ACTION_COUNT, ACTION_NAMES, BUY, HOLD, SELL, action_targets, realised_return
from execution import aggregate_market_scores, ensemble_agreement, executable_action_logits
from features import EVALUATION_WARMUP_HOURS, MarketData, SplitData
from model import MarketActorCritic


HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class Trajectory:
    actions: np.ndarray
    positions: np.ndarray
    rewards: np.ndarray
    price_return: np.ndarray
    cost_paid: np.ndarray
    equity: np.ndarray
    segment_resets: np.ndarray
    market_scores: np.ndarray | None = None
    execution_scores: np.ndarray | None = None
    seed_market_scores: np.ndarray | None = None
    initial_position: float = 0.0


def _actions_for_positions(
    positions: np.ndarray, segment_start: np.ndarray, initial_position: float
) -> np.ndarray:
    actions = np.full(len(positions), HOLD, dtype=np.int64)
    current = float(initial_position)
    for step, target in enumerate(positions):
        if target > current:
            actions[step] = BUY
        elif target < current:
            actions[step] = SELL
        current = float(target)
    return actions


def account_actions(
    actions: np.ndarray,
    market: MarketData,
    first: int,
    last: int,
    cost_rate: float,
    initial_position: float = 0.0,
) -> Trajectory:
    positions = np.empty(len(actions), dtype=np.float64)
    price_returns = np.empty(len(actions), dtype=np.float64)
    costs = np.empty(len(actions), dtype=np.float64)
    rewards = np.empty(len(actions), dtype=np.float64)
    segment = market.segment_start[first:last].copy()
    segment[0] = True
    current = float(initial_position)
    for step, (action, index) in enumerate(zip(actions, range(first, last))):
        # Recurrent memory resets at a data gap, but a real account keeps its
        # inventory. Include the passive return between the preceding covered
        # hour and this decision before applying the new order and its cost.
        gap_ratio = 1.0 if step == 0 else market.gap_price_ratio[index]
        gap_return = current * (gap_ratio - 1.0)
        target = float(action_targets(np.array([action]), np.array([current]))[0])
        price, cost, total = realised_return(
            np.array([target]), np.array([current]),
            np.array([market.exit_price[index] / market.entry_price[index]]), cost_rate,
        )
        positions[step] = target
        price_returns[step] = gap_return + price[0]
        costs[step] = cost[0]
        rewards[step] = (
            math.log(max(1.0 + gap_return, 1e-6))
            + math.log(max(1.0 + total[0], 1e-6))
        )
        current = target
    return Trajectory(
        actions=np.asarray(actions, dtype=np.int64),
        positions=positions,
        rewards=rewards,
        price_return=price_returns,
        cost_paid=costs,
        equity=np.exp(np.cumsum(rewards)),
        segment_resets=segment,
        initial_position=float(initial_position),
    )


def account_positions(
    positions: np.ndarray,
    market: MarketData,
    first: int,
    last: int,
    cost_rate: float,
    initial_position: float = 0.0,
) -> Trajectory:
    segment = market.segment_start[first:last].copy()
    segment[0] = True
    actions = _actions_for_positions(positions, segment, initial_position)
    return account_actions(actions, market, first, last, cost_rate, initial_position)


@torch.no_grad()
def _model_scores(
    model: MarketActorCritic,
    split: SplitData,
    first: int,
    last: int,
    warmup_hours: int = EVALUATION_WARMUP_HOURS,
) -> np.ndarray:
    model.eval()
    stream_first = max(0, first - warmup_hours)
    fast = torch.from_numpy(split.fast[stream_first:last]).unsqueeze(0)
    slow = torch.from_numpy(split.slow[stream_first:last]).unsqueeze(0)
    resets = torch.from_numpy(split.market.segment_start[stream_first:last]).unsqueeze(0)
    state = model.initial_state(1)
    scores, _ = model.market_logits(fast, slow, state, resets)
    return scores[0, first - stream_first :].cpu().numpy()


def _execute_scores(
    scores: np.ndarray,
    segment_start: np.ndarray,
    initial_position: float,
    cost_rate: float,
    execution_kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = float(initial_position)
    actions = np.empty(len(scores), dtype=np.int64)
    positions = np.empty(len(scores), dtype=np.float64)
    execution = np.empty_like(scores, dtype=np.float64)
    for step, opinion in enumerate(scores):
        adjusted = executable_action_logits(
            torch.from_numpy(opinion).unsqueeze(0), torch.tensor([current]),
            cost_rate, execution_kappa,
        )[0].numpy()
        action = int(np.argmax(adjusted))
        current = float(action_targets(np.array([action]), np.array([current]))[0])
        actions[step] = action
        positions[step] = current
        execution[step] = adjusted
    return actions, positions, execution


def _attach(
    accounted: Trajectory,
    market_scores: np.ndarray,
    execution_scores: np.ndarray,
    seed_market_scores: np.ndarray | None = None,
) -> Trajectory:
    return Trajectory(
        actions=accounted.actions,
        positions=accounted.positions,
        rewards=accounted.rewards,
        price_return=accounted.price_return,
        cost_paid=accounted.cost_paid,
        equity=accounted.equity,
        segment_resets=accounted.segment_resets,
        market_scores=market_scores,
        execution_scores=execution_scores,
        seed_market_scores=seed_market_scores,
        initial_position=accounted.initial_position,
    )


@torch.no_grad()
def run_policy(
    model: MarketActorCritic,
    split: SplitData,
    first: int,
    last: int,
    cost_rate: float,
    execution_kappa: float = 0.0,
    initial_position: float = 0.0,
) -> Trajectory:
    scores = _model_scores(model, split, first, last)
    segment = split.market.segment_start[first:last].copy()
    segment[0] = True
    actions, _, execution = _execute_scores(
        scores, segment, initial_position, cost_rate, execution_kappa
    )
    accounted = account_actions(actions, split.market, first, last, cost_rate, initial_position)
    return _attach(accounted, scores, execution)


@torch.no_grad()
def run_ensemble(
    models: list[MarketActorCritic],
    split: SplitData,
    first: int,
    last: int,
    cost_rate: float,
    execution_kappa: float = 0.0,
    aggregation: str = "mean",
    initial_position: float = 0.0,
) -> tuple[Trajectory, dict]:
    seed_scores = np.stack([_model_scores(model, split, first, last) for model in models])
    combined = aggregate_market_scores(torch.from_numpy(seed_scores), aggregation).numpy()
    segment = split.market.segment_start[first:last].copy()
    segment[0] = True
    actions, _, execution = _execute_scores(
        combined, segment, initial_position, cost_rate, execution_kappa
    )
    accounted = account_actions(actions, split.market, first, last, cost_rate, initial_position)
    diagnostics = ensemble_agreement(seed_scores)
    preferred = seed_scores.argmax(axis=-1)
    diagnostics["individual_seed_raw_action_frequency"] = [
        (np.bincount(row, minlength=ACTION_COUNT) / len(row)).tolist()
        for row in preferred
    ]
    return _attach(accounted, combined, execution, seed_scores), diagnostics


def _max_drawdown(rewards: np.ndarray) -> float:
    equity = np.concatenate(([1.0], np.exp(np.cumsum(rewards))))
    return float(np.min(equity / np.maximum.accumulate(equity)) - 1.0)


def metrics(trajectory: Trajectory, name: str = "") -> dict:
    rewards = trajectory.rewards
    hours = len(rewards)
    years = hours / HOURS_PER_YEAR
    std = float(rewards.std(ddof=1)) if hours > 1 else 0.0
    downside = np.minimum(rewards, 0.0)
    changes = trajectory.actions != HOLD
    counts = np.bincount(trajectory.actions, minlength=ACTION_COUNT)
    return {
        "name": name,
        "hours": hours,
        "total_return": float(np.exp(rewards.sum()) - 1.0),
        "total_log_return": float(rewards.sum()),
        "mean_hourly_log_return": float(rewards.mean()) if hours else 0.0,
        "annualised_return": float(np.exp(rewards.sum() / years) - 1.0) if years else 0.0,
        "annualised_volatility": std * math.sqrt(HOURS_PER_YEAR),
        "annualised_downside_deviation": float(np.sqrt(np.mean(downside**2))) * math.sqrt(HOURS_PER_YEAR),
        "sharpe": float(rewards.mean() / std * math.sqrt(HOURS_PER_YEAR)) if std else 0.0,
        "max_drawdown": _max_drawdown(rewards),
        "turnover_total": float(changes.sum()),
        "turnover_per_hour": float(changes.mean()) if hours else 0.0,
        "trades": int(changes.sum()),
        "average_hours_between_trades": float(hours / changes.sum()) if changes.any() else float(hours),
        "mean_net_exposure": float(trajectory.positions.mean()) if hours else 0.0,
        "share_btc": float(np.mean(trajectory.positions == 1.0)) if hours else 0.0,
        "share_cash": float(np.mean(trajectory.positions == 0.0)) if hours else 0.0,
        "action_frequency": {
            ACTION_NAMES[index]: float(counts[index] / max(hours, 1))
            for index in range(ACTION_COUNT)
        },
        "price_pnl": float(trajectory.price_return.sum()),
        "cost_pnl": -float(trajectory.cost_paid.sum()),
        "segment_resets": int(trajectory.segment_resets.sum()),
    }


def relative_performance(strategy: np.ndarray, bitcoin: np.ndarray) -> dict:
    """Annualised log alpha, beta, and correlation against unlevered Bitcoin."""
    strategy = np.asarray(strategy, dtype=np.float64)
    bitcoin = np.asarray(bitcoin, dtype=np.float64)
    if strategy.shape != bitcoin.shape:
        raise ValueError("strategy and Bitcoin returns must have the same shape")
    if len(strategy) < 2 or float(bitcoin.var(ddof=1)) <= 1e-16:
        beta = correlation = 0.0
    else:
        covariance = float(np.cov(strategy, bitcoin, ddof=1)[0, 1])
        beta = covariance / float(bitcoin.var(ddof=1))
        correlation = float(np.corrcoef(strategy, bitcoin)[0, 1])
    alpha = (float(strategy.mean()) - beta * float(bitcoin.mean())) * HOURS_PER_YEAR
    return {
        "bitcoin_total_return": float(np.exp(bitcoin.sum()) - 1.0),
        "beta_to_bitcoin": beta,
        "correlation_to_bitcoin": correlation,
        "annualised_log_alpha_to_bitcoin": alpha,
    }


def policy_diagnostics(trajectory: Trajectory) -> dict:
    if trajectory.execution_scores is None:
        return {}
    scores = trajectory.execution_scores
    probabilities = np.zeros_like(scores)
    for step in range(len(scores)):
        finite = np.isfinite(scores[step])
        shifted = scores[step, finite] - np.max(scores[step, finite])
        probabilities[step, finite] = np.exp(shifted) / np.exp(shifted).sum()
    ordered = np.sort(probabilities, axis=-1)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=-1
    )
    return {
        "executable_entropy_mean": float(entropy.mean()),
        "top_two_probability_margin_mean": float((ordered[:, -1] - ordered[:, -2]).mean()),
        "raw_market_score_dispersion": float(trajectory.market_scores.std(axis=-1).mean()),
    }


def _longest_run(values: np.ndarray) -> int:
    if not len(values):
        return 0
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    return int(np.diff(np.concatenate(([0], changes, [len(values)]))).max())


def starting_position_diagnostics(evaluator) -> dict:
    starts = {"CASH": 0.0, "BTC": 1.0}
    paths = {name: evaluator(value) for name, value in starts.items()}
    positions = np.stack([paths[name].positions for name in starts])
    equal = np.all(positions == positions[0], axis=0)
    permanent = np.logical_and.accumulate(equal[::-1])[::-1]
    hits = np.flatnonzero(permanent)
    convergence = int(hits[0]) if len(hits) else None
    reports = {}
    for name, path in paths.items():
        report = metrics(path, name)
        report["longest_uninterrupted_position_hours"] = _longest_run(path.positions)
        prior = np.concatenate(([path.initial_position], path.positions[:-1]))
        transition = np.zeros((2, 2), dtype=np.int64)
        for before, after in zip(prior.astype(int), path.positions.astype(int)):
            transition[before, after] += 1
        report["position_transition_matrix"] = {
            "rows": ["CASH", "BTC"],
            "columns": ["CASH", "BTC"],
            "counts": transition.tolist(),
        }
        if convergence is not None:
            report["log_return_before_convergence"] = float(path.rewards[:convergence].sum())
            report["log_return_after_convergence"] = float(path.rewards[convergence:].sum())
        reports[name] = report
    returns = [item["total_return"] for item in reports.values()]
    return {
        "permanent_convergence_hours": convergence,
        "hourly_position_agreement": float(np.mean(positions[0] == positions[1])),
        "return_spread": float(max(returns) - min(returns)),
        "by_start": reports,
    }


def always_cash(split: SplitData, first: int, last: int, cost_rate: float) -> Trajectory:
    return account_positions(np.zeros(last - first), split.market, first, last, cost_rate)


def always_btc(split: SplitData, first: int, last: int, cost_rate: float) -> Trajectory:
    return account_positions(np.ones(last - first), split.market, first, last, cost_rate)


def time_series_momentum(
    split: SplitData, first: int, last: int, cost_rate: float, lookback_hours: int = 24
) -> Trajectory:
    index = np.arange(first, last)
    past = split.market.entry_price[np.maximum(index - lookback_hours, 0)]
    positions = (split.market.entry_price[index] > past).astype(np.float64)
    return account_positions(positions, split.market, first, last, cost_rate)
