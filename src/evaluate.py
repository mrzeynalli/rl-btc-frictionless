"""Open the sealed June 2025 - May 2026 test year exactly once.

Nothing here selects anything.  The configuration, the training budget, the
normaliser and the momentum benchmark's lookback were all fixed on the
walk-forward folds before this module is allowed to run, and it refuses to run
until every final training run reports ``completed``.

Each seed is walked once with frozen weights and deterministic argmax actions,
its hidden state warmed causally over the 240 hours that precede the test year.
Every seed is reported.  The median, the interquartile range and a bootstrap
interval are reported over seeds; the best seed is never singled out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env import ONE_WAY_COST
from evaluation import (
    ACTION_POSITIONS, account_positions, always_flat, always_long,
    circular_shift_null, markov_matched_null, metrics, run_policy,
    time_series_momentum,
)
from env import position_ladder
from features import FINAL_SPLIT, build_market, build_split
from model import GruActorCritic
from train import ARMS, write_json_atomic

NULL_DRAWS = 1000
BOOTSTRAP_DRAWS = 10000
REGIME_LOOKBACK_HOURS = 720
REGIME_THRESHOLD = 0.10


def _aggregate(values: np.ndarray, rng: np.random.Generator) -> dict:
    if values.size == 0:
        return {}
    medians = np.median(rng.choice(values, size=(BOOTSTRAP_DRAWS, values.size)), axis=1)
    return {
        "seeds": int(values.size),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "interquartile_range": [
            float(np.percentile(values, 25)), float(np.percentile(values, 75))
        ],
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "bootstrap_median_95_interval": [
            float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))
        ],
        "share_positive": float(np.mean(values > 0.0)),
    }


def _regimes(split, first: int, last: int) -> dict[str, np.ndarray]:
    """Post-hoc labels, applied to results only -- never to the state."""
    price = split.market.entry_price
    index = np.arange(first, last)
    trailing = np.log(price[index] / price[np.maximum(index - REGIME_LOOKBACK_HOURS, 0)])
    return {
        "bull": trailing > REGIME_THRESHOLD,
        "bear": trailing < -REGIME_THRESHOLD,
        "chop": np.abs(trailing) <= REGIME_THRESHOLD,
    }


def _regime_breakdown(trajectory, regimes: dict[str, np.ndarray]) -> dict:
    output = {}
    for name, mask in regimes.items():
        if not mask.any():
            output[name] = {"hours": 0}
            continue
        rewards = trajectory.rewards[mask]
        output[name] = {
            "hours": int(mask.sum()),
            "total_log_return": float(rewards.sum()),
            "mean_hourly_log_return": float(rewards.mean()),
            "share_long": float(np.mean(trajectory.positions[mask] > 0)),
            "share_flat": float(np.mean(trajectory.positions[mask] == 0)),
            "share_short": float(np.mean(trajectory.positions[mask] < 0)),
        }
    return output


def _quarter_breakdown(trajectory, split, first: int, last: int) -> dict:
    stamps = split.market.execution_time[first:last]
    quarters = pd.PeriodIndex(stamps, freq="Q")
    output = {}
    for quarter in quarters.unique():
        mask = np.asarray(quarters == quarter)
        output[str(quarter)] = {
            "hours": int(mask.sum()),
            "total_log_return": float(trajectory.rewards[mask].sum()),
            "share_long": float(np.mean(trajectory.positions[mask] > 0)),
            "share_short": float(np.mean(trajectory.positions[mask] < 0)),
        }
    return output


def open_sealed_test(output_dir: Path) -> dict:
    selection = json.loads((output_dir / "selection.json").read_text())
    config_id = selection["selected_config_id"]
    chosen = selection["selected_config"]
    ladder = chosen.get("position_ladder", "ternary")
    funding_features = bool(chosen.get("funding_features", True))
    seeds = json.loads((output_dir / "experiment.json").read_text())["protocol"]["final_seeds"]
    lookback = selection["momentum_benchmark"]["selected_lookback_hours"]

    incomplete = []
    for arm in ARMS:
        for seed in seeds:
            path = output_dir / "final" / arm / config_id / f"seed_{seed}" / "training.json"
            if not path.exists() or json.loads(path.read_text()).get("status") != "completed":
                incomplete.append(f"final/{arm}/{config_id}/seed_{seed}")
    if incomplete:
        raise RuntimeError(
            f"the test year stays sealed: {len(incomplete)} final runs are not complete: "
            f"{incomplete[:5]}"
        )

    market = build_market(funding_features=funding_features)
    split = build_split(market, FINAL_SPLIT)
    first, last = split.evaluation_range
    regimes = _regimes(split, first, last)
    trajectory_dir = output_dir / "test_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "opened_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "split": split.as_dict(),
        "selection": {
            "config_id": config_id,
            "budget_updates": selection["selected_budget_updates"],
            "momentum_lookback_hours": lookback,
            "selected_on": "walk-forward validation folds only",
        },
        "benchmarks": {},
        "arms": {},
    }

    for cost_name, cost_rate in ARMS.items():
        benchmarks = {
            "always_flat": always_flat(split, first, last, cost_rate),
            "always_long": always_long(split, first, last, cost_rate),
            f"momentum_{lookback}h": time_series_momentum(
                split, first, last, cost_rate, lookback
            ),
        }
        results["benchmarks"][cost_name] = {
            name: metrics(trajectory, name) for name, trajectory in benchmarks.items()
        }
        for name, trajectory in benchmarks.items():
            pd.DataFrame({
                "execution_time_utc": split.market.execution_time[first:last],
                "position": trajectory.positions,
                "log_reward": trajectory.rewards,
                "equity": trajectory.equity,
            }).to_csv(trajectory_dir / f"benchmark_{name}_{cost_name}.csv", index=False)

    for arm, cost_rate in ARMS.items():
        per_seed = []
        for seed in seeds:
            weights = output_dir / "final" / arm / config_id / f"seed_{seed}" / "model_final.pt"
            model = GruActorCritic(
                slow_features=split.observation_size(len(position_ladder(ladder))),
                action_count=len(position_ladder(ladder)),
            )
            model.load_state_dict(torch.load(weights, weights_only=True))

            own = run_policy(model, split, first, last, cost_rate, ladder=ladder)
            other_arm = "economic" if arm == "zero_cost" else "zero_cost"
            other = account_positions(own.positions, split.market, first, last, ARMS[other_arm])
            rng = np.random.default_rng(seed)
            entry = {
                "seed": seed,
                arm: metrics(own, f"{arm}_seed_{seed}"),
                f"same_policy_at_{other_arm}_cost": metrics(other, other_arm),
                "maximum_action_probability_mean": float(own.action_probabilities.max(axis=1).mean()),
                "action_probability_entropy_mean": float(
                    -(own.action_probabilities * np.log(own.action_probabilities + 1e-12)).sum(axis=1).mean()
                ),
                "circular_shift_null": circular_shift_null(
                    own.positions, split, first, last, cost_rate, NULL_DRAWS, rng
                ),
                "markov_matched_null": markov_matched_null(
                    own.positions, split, first, last, cost_rate, min(NULL_DRAWS, 200), rng,
                    ladder=ladder
                ),
                "regimes": _regime_breakdown(own, regimes),
                "quarters": _quarter_breakdown(own, split, first, last),
            }
            per_seed.append(entry)

            frame = pd.DataFrame({
                "execution_time_utc": split.market.execution_time[first:last],
                "position": own.positions,
                "price_return": own.price_return,
                "funding_return": own.funding_return,
                "cost_paid": own.cost_paid,
                "hypothetical_cost": own.hypothetical_cost,
                "log_reward": own.rewards,
                "equity": own.equity,
                "probability_short": own.action_probabilities[:, 0],
                "probability_flat": own.action_probabilities[:, 1],
                "probability_long": own.action_probabilities[:, 2],
            })
            frame.to_csv(trajectory_dir / f"test_{arm}_seed_{seed}.csv", index=False)

        rng = np.random.default_rng(0)
        aggregate = {}
        for key in ("total_return", "annualised_return", "sharpe", "sortino", "max_drawdown",
                    "turnover_per_hour", "share_long", "share_flat", "share_short",
                    "total_log_return", "position_changes", "average_holding_hours"):
            aggregate[key] = _aggregate(
                np.array([entry[arm][key] for entry in per_seed], dtype=float), rng
            )
        percentiles = np.array([
            entry["circular_shift_null"].get("percentile_of_actual", np.nan) for entry in per_seed
        ], dtype=float)
        aggregate["circular_shift_null_percentile"] = _aggregate(percentiles[np.isfinite(percentiles)], rng)
        aggregate["seeds_above_95th_percentile_of_null"] = int(np.nansum(percentiles > 95.0))
        results["arms"][arm] = {"per_seed": per_seed, "aggregate": aggregate}

    write_json_atomic(output_dir / "test_results.json", results)
    for arm in ARMS:
        for seed_entry in results["arms"][arm]["per_seed"]:
            path = (output_dir / "final" / arm / config_id / f"seed_{seed_entry['seed']}"
                    / "training.json")
            record = json.loads(path.read_text())
            record["test_performance_accessed"] = True
            write_json_atomic(path, record)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = open_sealed_test(args.output_dir)
    for arm, payload in results["arms"].items():
        aggregate = payload["aggregate"]
        print(f"{arm}: median total return {aggregate['total_return']['median']:+.4f} "
              f"sharpe {aggregate['sharpe']['median']:+.3f} "
              f"turnover/h {aggregate['turnover_per_hour']['median']:.3f}")


if __name__ == "__main__":
    main()
