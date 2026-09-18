"""Failure diagnosis for the already-opened July 2026 holdout.

This module does not select or tune a policy.  It evaluates only the diagnostic
variants requested after the locked 0.10-hysteresis result was opened.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from concern_study import SEEDS, load_model, run_ensemble
from env import position_ladder
from evaluation import HOURS_PER_YEAR, Trajectory, account_positions, metrics, policy_diagnostics, run_policy
from features import build_market, build_split
from holdout_july import (
    ARMS,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT,
    HOLDOUT_SPLIT,
    SWITCH_MARGIN,
    locked_cell,
    training_spec,
)
from train import write_json_atomic


OUTPUT_DIR = DEFAULT_OUTPUT
DATASET_DIR = DEFAULT_DATASET
DIAGNOSIS_PATH = OUTPUT_DIR / "failure_diagnosis.json"
MARGIN_SVG = OUTPUT_DIR / "ensemble_top_two_margin_distribution.svg"


def load_locked_models(split):
    selected = locked_cell()
    models = []
    for seed in SEEDS:
        weights = OUTPUT_DIR / training_spec(seed).directory / "model_final.pt"
        models.append(load_model(selected, split, weights))
    return models


@torch.no_grad()
def run_ensemble_with_initial_position(
    models,
    split,
    first: int,
    last: int,
    cost_rate: float,
    switch_margin: float,
    initial_position: float,
):
    warmup_start = split.evaluation_warmup_start
    stream_first = min(warmup_start, first)
    fast = torch.from_numpy(split.fast[stream_first:last]).unsqueeze(0)
    hourly = [
        model.trunk(fast, model.initial_hidden(1))[0][0] for model in models
    ]
    slow = torch.from_numpy(split.slow[stream_first:last])
    levels = position_ladder("ternary")
    action_count = len(levels)
    initial_index = int(np.searchsorted(levels, initial_position))
    position_index = initial_index
    offset = first - stream_first
    length = last - first
    positions = np.zeros(length, dtype=np.float64)
    probabilities = np.zeros((length, action_count), dtype=np.float64)
    top_two = np.zeros(length, dtype=np.float64)
    switch = np.zeros(length, dtype=np.float64)
    aligned_recommendations = np.zeros((length, len(models)), dtype=np.int64)
    one_hot = torch.zeros(action_count)

    for step in range(last - stream_first):
        if step == offset:
            position_index = initial_index
        one_hot.zero_()
        one_hot[position_index] = 1.0
        model_probabilities = []
        model_actions = []
        for model, representation in zip(models, hourly):
            joined = torch.cat([representation[step], slow[step], one_hot]).unsqueeze(0)
            logits = model.actor_output(model.activation(model.actor_hidden(joined)))[0]
            model_probabilities.append(torch.softmax(logits, dim=-1))
            model_actions.append(int(torch.argmax(logits)))
        mean_probability = torch.stack(model_probabilities).mean(dim=0)
        best = int(torch.argmax(mean_probability))
        improvement = float(mean_probability[best] - mean_probability[position_index])
        action = (
            position_index
            if best != position_index and improvement <= switch_margin
            else best
        )
        if step >= offset:
            target = step - offset
            positions[target] = levels[action]
            probabilities[target] = mean_probability.numpy()
            ordered = torch.topk(mean_probability, 2).values
            top_two[target] = float(ordered[0] - ordered[1])
            switch[target] = improvement
            aligned_recommendations[target] = model_actions
        position_index = action

    accounted = account_positions(
        positions,
        split.market,
        first,
        last,
        cost_rate,
        entry_position=initial_position,
    )
    trajectory = Trajectory(
        positions=positions,
        rewards=accounted.rewards,
        price_return=accounted.price_return,
        funding_return=accounted.funding_return,
        cost_paid=accounted.cost_paid,
        hypothetical_cost=accounted.hypothetical_cost,
        equity=accounted.equity,
        action_probabilities=probabilities,
        action_top_two_margins=top_two,
        action_switch_margins=switch,
    )
    return trajectory, aligned_recommendations


def trajectory_summary(
    trajectory: Trajectory, name: str, initial_position: float = 0.0
) -> dict:
    result = metrics(trajectory, name)
    downside = np.minimum(trajectory.rewards, 0.0)
    previous = np.concatenate(([initial_position], trajectory.positions[:-1]))
    changes = trajectory.positions != previous
    result.update({
        "mean_net_exposure": float(trajectory.positions.mean()),
        "annualised_downside_deviation": float(
            np.sqrt(np.mean(downside ** 2)) * math.sqrt(HOURS_PER_YEAR)
        ),
        "total_trading_cost_return_units": float(trajectory.cost_paid.sum()),
        "turnover_total": float(np.abs(trajectory.positions - previous).sum()),
        "turnover_per_hour": float(
            np.abs(trajectory.positions - previous).sum() / len(trajectory.positions)
        ),
        "position_changes": int(changes.sum()),
        "average_holding_hours": (
            float(len(trajectory.positions) / changes.sum())
            if changes.any() else float(len(trajectory.positions))
        ),
        "policy_diagnostics": policy_diagnostics(trajectory),
    })
    return result


def agreement_matrix(actions: np.ndarray) -> list[list[float]]:
    count = actions.shape[1]
    return [
        [float(np.mean(actions[:, left] == actions[:, right])) for right in range(count)]
        for left in range(count)
    ]


def vote_diagnostics(recommendations: np.ndarray) -> dict:
    counts = np.stack([
        np.sum(recommendations == action, axis=1) for action in range(3)
    ], axis=1)
    maximum = counts.max(axis=1)
    return {
        "unanimous_share": float(np.mean(maximum == 5)),
        "four_to_one_share": float(np.mean(maximum == 4)),
        "three_vote_plurality_share": float(np.mean(maximum == 3)),
        "no_three_vote_majority_share": float(np.mean(maximum == 2)),
        "mean_vote_count": {
            "short": float(counts[:, 0].mean()),
            "flat": float(counts[:, 1].mean()),
            "long": float(counts[:, 2].mean()),
        },
        "majority_frequency": {
            "short": float(np.mean(np.argmax(counts, axis=1) == 0)),
            "flat": float(np.mean(np.argmax(counts, axis=1) == 1)),
            "long": float(np.mean(np.argmax(counts, axis=1) == 2)),
        },
    }


def switches_and_segments(
    trajectory: Trajectory,
    execution_time: pd.DatetimeIndex,
    initial_position: float,
) -> tuple[list[dict], list[dict]]:
    previous = np.concatenate(([initial_position], trajectory.positions[:-1]))
    switch_indices = np.flatnonzero(trajectory.positions != previous)
    switches = []
    for index in switch_indices:
        probabilities = trajectory.action_probabilities[index]
        switches.append({
            "execution_time_utc": execution_time[index].isoformat(),
            "from_position": float(previous[index]),
            "to_position": float(trajectory.positions[index]),
            "probability_short": float(probabilities[0]),
            "probability_flat": float(probabilities[1]),
            "probability_long": float(probabilities[2]),
            "top_two_margin": float(trajectory.action_top_two_margins[index]),
            "improvement_over_current": float(trajectory.action_switch_margins[index]),
        })

    boundaries = np.concatenate(([0], switch_indices[switch_indices > 0], [len(trajectory.positions)]))
    boundaries = np.unique(boundaries)
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rewards = trajectory.rewards[start:end]
        segments.append({
            "start_execution_utc": execution_time[start].isoformat(),
            "end_execution_utc_inclusive": execution_time[end - 1].isoformat(),
            "position": float(trajectory.positions[start]),
            "hours": int(end - start),
            "compounded_return": float(np.exp(rewards.sum()) - 1.0),
            "log_return": float(rewards.sum()),
            "price_pnl_return_units": float(trajectory.price_return[start:end].sum()),
            "funding_pnl_return_units": float(trajectory.funding_return[start:end].sum()),
            "trading_cost_return_units": float(trajectory.cost_paid[start:end].sum()),
        })
    return switches, segments


def write_margin_svg(values: np.ndarray, path: Path) -> None:
    bins = np.linspace(0.0, 1.0, 26)
    counts, edges = np.histogram(values, bins=bins)
    shares = counts / max(len(values), 1)
    width, height = 1000, 560
    left, right, top, bottom = 80, 30, 55, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(float(shares.max()), 0.01)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="500" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#1f2933">July holdout: ensemble top-two probability margin</text>',
    ]
    for tick in range(6):
        share = maximum * tick / 5
        y = top + plot_height * (1 - tick / 5)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d7d9dc" stroke-width="1"/>')
        elements.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#52606d">{share*100:.1f}%</text>')
    bar_width = plot_width / len(shares)
    for index, share in enumerate(shares):
        x = left + index * bar_width + 1
        bar_height = plot_height * share / maximum
        y = top + plot_height - bar_height
        elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width-2:.1f}" height="{bar_height:.1f}" fill="#2878b5"/>')
    threshold_x = left + SWITCH_MARGIN * plot_width
    elements.append(f'<line x1="{threshold_x:.1f}" y1="{top}" x2="{threshold_x:.1f}" y2="{top+plot_height}" stroke="#c0392b" stroke-width="3" stroke-dasharray="7,5"/>')
    elements.append(f'<text x="{threshold_x+7:.1f}" y="{top+18}" font-family="sans-serif" font-size="13" fill="#c0392b">0.10 hysteresis</text>')
    for tick in range(0, 11):
        x = left + tick * plot_width / 10
        elements.append(f'<text x="{x:.1f}" y="{top+plot_height+25}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#52606d">{tick/10:.1f}</text>')
    elements.extend([
        f'<text x="{left+plot_width/2:.1f}" y="{height-25}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#323f4b">Top probability minus second-highest probability</text>',
        f'<text x="18" y="{top+plot_height/2:.1f}" transform="rotate(-90 18 {top+plot_height/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#323f4b">Share of hourly decisions</text>',
        '</svg>',
    ])
    path.write_text("\n".join(elements))


def main() -> None:
    opened = OUTPUT_DIR / "results.json"
    if not opened.exists() or not json.loads(opened.read_text()).get("holdout_accessed"):
        raise RuntimeError("locked July holdout must be completed before diagnosis")
    if DIAGNOSIS_PATH.exists():
        existing = json.loads(DIAGNOSIS_PATH.read_text())
        if existing.get("status") == "completed":
            print(json.dumps(existing, indent=2))
            return

    torch.set_num_threads(1)
    market = build_market(DATASET_DIR, funding_features=True)
    split = build_split(market, HOLDOUT_SPLIT)
    first, last = split.evaluation_range
    execution_time = market.execution_time[first:last]
    models = load_locked_models(split)

    individual_trajectories = [
        run_policy(
            model,
            split,
            first,
            last,
            ARMS["economic"],
            ladder="ternary",
            switch_margin=0.0,
        )
        for model in models
    ]
    individual_actions = np.stack(
        [trajectory.positions for trajectory in individual_trajectories], axis=1
    )

    ensemble_zero = run_ensemble(
        models, split, first, last, ARMS["economic"], switch_margin=0.0
    )
    initial_runs = {}
    aligned = {}
    for name, position in (("short", -1.0), ("flat", 0.0), ("long", 1.0)):
        trajectory, recommendations = run_ensemble_with_initial_position(
            models,
            split,
            first,
            last,
            ARMS["economic"],
            SWITCH_MARGIN,
            position,
        )
        initial_runs[name] = trajectory
        aligned[name] = recommendations

    locked = initial_runs["flat"]
    saved = pd.read_csv(OUTPUT_DIR / "hourly_holdout_trajectory.csv")
    if not (
        np.array_equal(saved.position.to_numpy(), locked.positions)
        and np.allclose(saved.log_return.to_numpy(), locked.rewards, atol=1e-14)
    ):
        raise RuntimeError("diagnostic replay differs from the locked holdout trajectory")
    switches, segments = switches_and_segments(locked, execution_time, 0.0)
    margins = locked.action_top_two_margins
    write_margin_svg(margins, MARGIN_SVG)
    quantiles = np.quantile(margins, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    initial_summary = {}
    for name, trajectory in initial_runs.items():
        initial_position = {"short": -1.0, "flat": 0.0, "long": 1.0}[name]
        summary = trajectory_summary(
            trajectory, f"initial_{name}", initial_position=initial_position
        )
        summary["agreement_with_flat_initialisation"] = float(
            np.mean(trajectory.positions == locked.positions)
        )
        summary["final_position"] = float(trajectory.positions[-1])
        initial_summary[name] = summary

    diagnosis = {
        "status": "completed",
        "purpose": "failure diagnosis only; no holdout selection or tuning",
        "hours": last - first,
        "individual_no_hysteresis": {
            str(seed): trajectory_summary(trajectory, f"seed_{seed}")
            for seed, trajectory in zip(SEEDS, individual_trajectories)
        },
        "ensemble_comparison": {
            "margin_0.00": trajectory_summary(ensemble_zero, "ensemble_margin_0.00"),
            "margin_0.10": trajectory_summary(locked, "ensemble_margin_0.10"),
            "position_agreement": float(np.mean(ensemble_zero.positions == locked.positions)),
        },
        "top_two_margin_distribution": {
            "observations": len(margins),
            "mean": float(margins.mean()),
            "std": float(margins.std(ddof=1)),
            "quantiles": {
                name: float(value) for name, value in zip(
                    ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"),
                    quantiles,
                )
            },
            "share_below_0.10": float(np.mean(margins < 0.10)),
            "svg": str(MARGIN_SVG),
        },
        "seed_agreement": {
            "seeds": list(SEEDS),
            "independent_rollout_pairwise_agreement": agreement_matrix(individual_actions),
            "independent_rollout_overall_off_diagonal_mean": float(np.mean([
                np.mean(individual_actions[:, left] == individual_actions[:, right])
                for left in range(len(SEEDS)) for right in range(left + 1, len(SEEDS))
            ])),
            "aligned_on_locked_ensemble_state_pairwise_agreement": agreement_matrix(aligned["flat"]),
            "aligned_on_locked_ensemble_state_overall_off_diagonal_mean": float(np.mean([
                np.mean(aligned["flat"][:, left] == aligned["flat"][:, right])
                for left in range(len(SEEDS)) for right in range(left + 1, len(SEEDS))
            ])),
            "aligned_vote_diagnostics": vote_diagnostics(aligned["flat"]),
        },
        "locked_margin_0.10_switches": switches,
        "locked_margin_0.10_position_segments": segments,
        "initial_position_sensitivity": initial_summary,
    }
    write_json_atomic(DIAGNOSIS_PATH, diagnosis)
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
