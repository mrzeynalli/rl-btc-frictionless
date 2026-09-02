"""Run the whole study in the order the protocol requires, and only that order.

1. Build and audit the five-year dataset.
2. Record the supervised signal check *before* any RL number exists.
3. Search the predeclared configuration grid on the four walk-forward folds
   with the economic cost model and five seeds.
4. Select one configuration and one common training budget from the median
   validation result.  Nothing after this point changes a modelling choice.
5. Re-run the selected configuration on the folds with ten seeds in both cost
   arms, and evaluate the zero-cost diagnostic gate.
6. Retrain once on June 2021 - May 2025 with the selected budget, ten seeds,
   both arms.  These runs evaluate nothing.
7. Open the sealed June 2025 - May 2026 test year exactly once.

Progress is always valid JSON in ``experiment.json``.  Every stage skips runs
whose ``training.json`` already reports ``completed``, so the study resumes
after an interruption without repeating work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import data as data_module
from features import (
    ALL_SPLITS, FINAL_SPLIT, WALK_FORWARD_SPLITS, build_market, build_split,
)
from ppo import PPOConfig
from train import ARMS, write_json_atomic

SEEDS = (11, 29, 47, 71, 101, 137, 179, 211, 257, 293)
SELECTION_SEEDS = SEEDS[:5]
FINAL_SEEDS = SEEDS

# Everything below was fixed by the ``hourly_v2`` probe
# (artifacts/probe)
# before this grid was written, and none of it is searched here.
#
#   gamma 0.5, lambda 0.5    An action's causal reach is two steps: it sets the
#                            P&L of the interval it opens and the turnover and
#                            funding-on-previous of the one after.  hourly_eq's
#                            0.99/0.95 integrated seventeen, a measured 2.76x
#                            inflation of advantage noise for no signal.
#   adaptive rate, KL 0.02   hourly_eq recorded clip_fraction exactly 0.0000 at
#                            every update of all 320 runs -- the trust region
#                            never engaged once.  The probe puts it at 0.14-0.17
#                            with the KL regulated on target.
#   value clipping off       It constrains a critic that was never the binding
#   return scaling on        constraint, and the target's scale moves with
#                            gamma, so a fixed value coefficient would not mean
#                            the same thing across the grid.
#   funding features on      The position opened at t pays the settlement at
#                            t+1; settlements are a fixed eight-hour UTC
#                            schedule, so the countdown is causal.  Without it
#                            the agent cannot tell which 12.5% of hours carry a
#                            charge, and carry is the single most predictable
#                            quantity in the dataset (IC 0.75 against 0.017).
GAMMA = 0.5
GAE_LAMBDA = 0.5
TARGET_KL = 0.02

# The probe broke the flat-or-uniform bifurcation, so entropy settings below
# 0.10 are live options again rather than guaranteed collapses.
ENTROPY_COEFFICIENTS = (0.01, 0.03, 0.10)
# Three-way against a seven-level ladder.  Mean-variance sizing puts the right
# exposure at a fraction of notional when the edge is of the order of the
# measured IC, and the three-way set cannot express that.
POSITION_LADDERS = ("ternary", "ladder")
# The probe settled the critic multiplier: at 1.0 the value head explained
# 0.118 of return variance against 0.081 at 3.0, so it is fixed here.
CRITIC_MULTIPLIER = 1.0
# The one axis the probe left genuinely open.  Driving the policy to a target
# KL is what cleared the clip-fraction and explained-variance pathologies, but
# it also raised turnover from 0.56 to 0.82 per step and the fold-1 validation
# return fell with it.  Whether a policy that actually moves is worth what its
# movement costs is exactly the kind of question the validation folds exist to
# answer, so both settings go on the grid.
ADAPTIVE_RATES = (False, True)

# Declared before any run: the configuration and budget are the pair whose
# median validation total log return over every (fold, seed) is largest.
SELECTION_METRIC = "total_log_return"
SELECTION_RULE = "median over folds and seeds of the validation " + SELECTION_METRIC
MOMENTUM_LOOKBACKS = (24, 72, 168, 240)

DEFAULT_UPDATES = 600
EVALUATION_INTERVAL = 25


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configuration_grid() -> dict[str, dict]:
    """Twelve cells: entropy x action ladder x fixed-or-adaptive rate."""
    grid: dict[str, dict] = {}
    index = 0
    for entropy in ENTROPY_COEFFICIENTS:
        for ladder in POSITION_LADDERS:
            for adaptive in ADAPTIVE_RATES:
                grid[f"c{index:02d}"] = {
                    "ppo": PPOConfig(
                        gamma=GAMMA,
                        gae_lambda=GAE_LAMBDA,
                        entropy_coefficient=entropy,
                        value_clip_range=None,
                        normalise_returns=True,
                        adaptive_learning_rate=adaptive,
                        target_kl=TARGET_KL,
                        critic_learning_rate_multiplier=CRITIC_MULTIPLIER,
                    ),
                    "position_ladder": ladder,
                    "funding_features": True,
                }
                index += 1
    return grid


def cell_as_dict(cell: dict) -> dict:
    return {
        **cell["ppo"].as_dict(),
        "position_ladder": cell["position_ladder"],
        "funding_features": cell["funding_features"],
    }


class Runner:
    """Dispatch training runs as isolated single-threaded subprocesses."""

    def __init__(self, output_dir: Path, workers: int, updates: int):
        self.output_dir = output_dir
        self.workers = workers
        self.updates = updates
        self.lock = threading.Lock()
        self.completed = 0
        self.failed: list[dict] = []

    def run_path(self, split: str, arm: str, config_id: str, seed: int) -> Path:
        return self.output_dir / split / arm / config_id / f"seed_{seed}" / "training.json"

    def is_complete(self, split: str, arm: str, config_id: str, seed: int) -> bool:
        path = self.run_path(split, arm, config_id, seed)
        if not path.exists():
            return False
        try:
            return json.loads(path.read_text()).get("status") == "completed"
        except json.JSONDecodeError:
            return False

    def _launch(self, job: dict) -> dict:
        command = [
            sys.executable, "-m", "train",
            "--split", job["split"], "--arm", job["arm"],
            "--config-id", job["config_id"], "--seed", str(job["seed"]),
            "--updates", str(job["updates"]),
            "--evaluation-interval", str(job.get("evaluation_interval", EVALUATION_INTERVAL)),
            "--output-dir", str(self.output_dir),
            "--ppo-json", json.dumps(job["ppo"]),
            "--position-ladder", job["position_ladder"],
        ]
        if not job.get("funding_features", True):
            command.append("--no-funding-features")
        if not job.get("evaluate_validation", True):
            command.append("--no-validation")
        environment = dict(os.environ)
        environment.update(
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[0]),
        )
        started = datetime.now(timezone.utc)
        process = subprocess.run(
            command, capture_output=True, text=True, env=environment,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        outcome = {
            **{key: job[key] for key in ("split", "arm", "config_id", "seed")},
            "returncode": process.returncode,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": now(),
        }
        if process.returncode != 0:
            outcome["stderr"] = process.stderr[-4000:]
        return outcome

    def run_all(self, jobs: list[dict], stage: str, progress: dict, manifest_path: Path,
                retry: bool = False) -> None:
        pending = [
            job for job in jobs
            if not self.is_complete(job["split"], job["arm"], job["config_id"], job["seed"])
        ]
        progress["stages"][stage] = {
            "total": len(jobs), "already_complete": len(jobs) - len(pending),
            "pending": len(pending), "finished": 0, "failed": 0, "started_at_utc": now(),
        }
        write_json_atomic(manifest_path, progress)
        print(f"[{now()}] stage {stage}: {len(pending)} runs to do "
              f"({len(jobs) - len(pending)} already complete)", flush=True)
        if not pending:
            progress["stages"][stage]["finished_at_utc"] = now()
            write_json_atomic(manifest_path, progress)
            return

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._launch, job): job for job in pending}
            for future in as_completed(futures):
                outcome = future.result()
                with self.lock:
                    stage_progress = progress["stages"][stage]
                    stage_progress["finished"] += 1
                    if outcome["returncode"] != 0:
                        stage_progress["failed"] += 1
                        self.failed.append(outcome)
                        progress["failures"] = self.failed[-20:]
                    stage_progress["last_finished"] = outcome
                    write_json_atomic(manifest_path, progress)
                    print(f"[{now()}] {stage} {stage_progress['finished']}/{len(pending)} "
                          f"{outcome['split']}/{outcome['arm']}/{outcome['config_id']}/"
                          f"seed_{outcome['seed']} rc={outcome['returncode']}", flush=True)
        still_missing = [
            job for job in pending
            if not self.is_complete(job["split"], job["arm"], job["config_id"], job["seed"])
        ]
        if still_missing and not retry:
            print(f"[{now()}] stage {stage}: retrying {len(still_missing)} incomplete runs",
                  flush=True)
            self.run_all(still_missing, f"{stage}_retry", progress, manifest_path, retry=True)
        progress["stages"][stage]["finished_at_utc"] = now()
        progress["stages"][stage]["incomplete_after_retry"] = len(still_missing) if retry else 0
        write_json_atomic(manifest_path, progress)


# -- selection --------------------------------------------------------------


def read_validation_curve(output_dir: Path, split: str, arm: str, config_id: str, seed: int) -> list[dict]:
    path = output_dir / split / arm / config_id / f"seed_{seed}" / "training.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("validation_curve", [])


def select_configuration(
    output_dir: Path, grid: dict[str, dict], updates: int,
    interval: int = EVALUATION_INTERVAL, seeds: tuple[int, ...] = SELECTION_SEEDS,
) -> dict:
    """One configuration and one budget, from median validation performance."""
    budgets = list(range(interval, updates + 1, interval))
    table: dict[str, dict[int, list[float]]] = {}
    for config_id in grid:
        table[config_id] = {budget: [] for budget in budgets}
        for split in WALK_FORWARD_SPLITS:
            for seed in seeds:
                for point in read_validation_curve(output_dir, split.name, "economic", config_id, seed):
                    budget = int(point["update"])
                    if budget in table[config_id]:
                        table[config_id][budget].append(
                            float(point["economic"][SELECTION_METRIC])
                        )

    scores = []
    for config_id, by_budget in table.items():
        for budget, values in by_budget.items():
            if len(values) < len(WALK_FORWARD_SPLITS) * len(seeds):
                continue
            scores.append({
                "config_id": config_id,
                "budget_updates": budget,
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "interquartile_range": [
                    float(np.percentile(values, 25)), float(np.percentile(values, 75))
                ],
                "runs": len(values),
            })
    if not scores:
        raise RuntimeError("no complete validation curves to select from")
    scores.sort(key=lambda row: row["median"], reverse=True)
    best = scores[0]
    return {
        "rule": SELECTION_RULE,
        "metric": SELECTION_METRIC,
        "seeds_used": list(seeds),
        "folds_used": [split.name for split in WALK_FORWARD_SPLITS],
        "selected_config_id": best["config_id"],
        "selected_config": cell_as_dict(grid[best["config_id"]]),
        "selected_budget_updates": best["budget_updates"],
        "selected_median_validation_log_return": best["median"],
        "ranking_top_20": scores[:20],
        "full_ranking_size": len(scores),
        "selected_at_utc": now(),
    }


def select_momentum_lookback(output_dir: Path) -> dict:
    """Pick the benchmark's lookback on the same validation folds, not on test."""
    from evaluation import time_series_momentum

    market = build_market()
    results = {}
    for lookback in MOMENTUM_LOOKBACKS:
        totals = []
        for split_definition in WALK_FORWARD_SPLITS:
            split = build_split(market, split_definition)
            first, last = split.evaluation_range
            trajectory = time_series_momentum(split, first, last, ARMS["economic"], lookback)
            totals.append(float(trajectory.rewards.sum()))
        results[lookback] = {"per_fold_log_return": totals, "median": float(np.median(totals))}
    best = max(results, key=lambda key: results[key]["median"])
    return {"candidates": results, "selected_lookback_hours": int(best)}


# -- the zero-cost diagnostic gate -----------------------------------------


def zero_cost_gate(output_dir: Path, selection: dict, momentum: dict) -> dict:
    """Does the zero-cost policy beat matched-turnover noise and the simple rules?"""
    from evaluation import (
        always_flat, always_long, circular_shift_null, metrics, run_policy,
        time_series_momentum,
    )
    import torch

    from model import GruActorCritic

    from env import position_ladder

    chosen = selection["selected_config"]
    ladder = chosen.get("position_ladder", "ternary")
    action_count = len(position_ladder(ladder))
    market = build_market(funding_features=bool(chosen.get("funding_features", True)))
    config_id = selection["selected_config_id"]
    budget = selection["selected_budget_updates"]
    lookback = momentum["selected_lookback_hours"]

    per_run = []
    for split_definition in WALK_FORWARD_SPLITS:
        split = build_split(market, split_definition)
        first, last = split.evaluation_range
        benchmark = {
            "always_flat": float(always_flat(split, first, last, 0.0).rewards.sum()),
            "always_long": float(always_long(split, first, last, 0.0).rewards.sum()),
            f"momentum_{lookback}h": float(
                time_series_momentum(split, first, last, 0.0, lookback).rewards.sum()
            ),
        }
        for seed in FINAL_SEEDS:
            directory = output_dir / split_definition.name / "zero_cost" / config_id / f"seed_{seed}"
            curve = read_validation_curve(output_dir, split_definition.name, "zero_cost", config_id, seed)
            at_budget = [point for point in curve if int(point["update"]) == budget]
            if not at_budget:
                continue
            weights = directory / "checkpoints" / f"update_{budget:04d}.pt"
            null = {}
            if weights.exists():
                model = GruActorCritic(
                    slow_features=split.observation_size(action_count),
                    action_count=action_count,
                )
                model.load_state_dict(torch.load(weights, weights_only=True))
                trajectory = run_policy(model, split, first, last, 0.0, ladder=ladder)
                null = circular_shift_null(
                    trajectory.positions, split, first, last, 0.0, 500,
                    np.random.default_rng(seed),
                )
            per_run.append({
                "fold": split_definition.name,
                "seed": seed,
                "policy_log_return_at_budget": float(at_budget[0]["zero_cost"][SELECTION_METRIC]),
                "benchmarks": benchmark,
                "matched_turnover_null": null,
            })

    if not per_run:
        return {"status": "not_evaluated", "reason": "no zero-cost runs found"}
    policy = np.array([row["policy_log_return_at_budget"] for row in per_run])
    beats_null = [
        row["matched_turnover_null"].get("percentile_of_actual", 0.0) > 95.0
        for row in per_run if row["matched_turnover_null"]
    ]
    best_benchmark = np.array([max(row["benchmarks"].values()) for row in per_run])
    return {
        "status": "evaluated",
        "config_id": config_id,
        "budget_updates": budget,
        "runs": len(per_run),
        "policy_median_log_return": float(np.median(policy)),
        "policy_interquartile_range": [
            float(np.percentile(policy, 25)), float(np.percentile(policy, 75))
        ],
        "best_benchmark_median_log_return": float(np.median(best_benchmark)),
        "share_of_runs_beating_best_benchmark": float(np.mean(policy > best_benchmark)),
        "share_of_runs_above_95th_percentile_of_matched_turnover_null": (
            float(np.mean(beats_null)) if beats_null else None
        ),
        "gate_passed": bool(
            np.median(policy) > np.median(best_benchmark)
            and np.median(policy) > 0.0
            and (np.mean(beats_null) > 0.5 if beats_null else False)
        ),
        "criteria": (
            "median zero-cost validation log return exceeds zero, exceeds the best "
            "simple benchmark, and more than half the runs sit above the 95th "
            "percentile of their own matched-turnover circular-shift null"
        ),
        "per_run": per_run,
        "evaluated_at_utc": now(),
    }


# -- stages -----------------------------------------------------------------


def jobs_for(splits, arms, config_ids, seeds, grid, updates,
             evaluate_validation=True, interval=EVALUATION_INTERVAL) -> list[dict]:
    return [
        {
            "split": split.name, "arm": arm, "config_id": config_id, "seed": seed,
            "updates": updates, "ppo": grid[config_id]["ppo"].as_dict(),
            "position_ladder": grid[config_id]["position_ladder"],
            "funding_features": grid[config_id]["funding_features"],
            "evaluate_validation": evaluate_validation, "evaluation_interval": interval,
        }
        for split in splits for arm in arms for config_id in config_ids for seed in seeds
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--evaluation-interval", type=int, default=EVALUATION_INTERVAL)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stop-if-gate-fails", action="store_true")
    parser.add_argument("--skip-signal-check", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "experiment.json"

    grid = configuration_grid()
    selection_seeds, final_seeds = list(SELECTION_SEEDS), list(FINAL_SEEDS)
    updates = args.updates
    interval = args.evaluation_interval
    if args.smoke_test:
        grid = {key: grid[key] for key in list(grid)[:2]}
        selection_seeds, final_seeds = [SEEDS[0]], [SEEDS[0], SEEDS[1]]
        updates, interval = 4, 2

    print(f"[{now()}] building dataset", flush=True)
    dataset_manifest = data_module.build()
    market = build_market()

    progress = {
        "status": "running",
        "started_at_utc": now(),
        "output_dir": str(output_dir),
        "protocol": {
            "symbol": "BTCUSDT USD-M perpetual futures",
            "cross_asset_data_used": False,
            "study_window": "2021-06-01 inclusive to 2026-06-01 exclusive",
            "decision_interval_hours": 1,
            "observation": ("12 five-minute bars x 4 variables + 7 slow variables + "
                            "3 causal funding-schedule variables + position one-hot"),
            "actions": "three-way or seven-level ladder, chosen on validation",
            "exposure": "1x notional at the ladder's extremes",
            "credit_horizon": "gamma 0.5, lambda 0.5, matched to the two-step causal reach",
            "reward": "log change in realised portfolio equity, constant multiplier only",
            "cost_arms": {name: rate for name, rate in ARMS.items()},
            "funding": "applied at exact Binance settlement timestamps to the position open before them",
            "selection_rule": SELECTION_RULE,
            "selection_seeds": selection_seeds,
            "final_seeds": final_seeds,
            "maximum_updates": updates,
            "evaluation_interval": interval,
            "configuration_grid": {key: cell_as_dict(value) for key, value in grid.items()},
            "splits": [split.as_dict() for split in ALL_SPLITS],
        },
        "dataset": {
            key: dataset_manifest[key]
            for key in ("total_bars", "study_bars", "warmup_days", "hourly_decisions_in_study",
                        "funding_settlements_total", "audit_passed")
        },
        "market": market.audit,
        "stages": {},
    }
    write_json_atomic(manifest_path, progress)

    if not args.skip_signal_check:
        from signal_check import run_signal_check
        print(f"[{now()}] supervised signal check", flush=True)
        progress["signal_check"] = run_signal_check(market)
        write_json_atomic(manifest_path, progress)

    runner = Runner(output_dir, args.workers, updates)

    # Stage A1 -- the configuration search, economic costs, five seeds.
    runner.run_all(
        jobs_for(WALK_FORWARD_SPLITS, ["economic"], list(grid), selection_seeds, grid, updates,
                 interval=interval),
        "a1_configuration_search", progress, manifest_path,
    )

    selection = select_configuration(output_dir, grid, updates, interval, tuple(selection_seeds))
    momentum = select_momentum_lookback(output_dir)
    progress["selection"] = selection
    progress["momentum_benchmark"] = momentum
    write_json_atomic(manifest_path, progress)
    write_json_atomic(output_dir / "selection.json", {**selection, "momentum_benchmark": momentum})
    print(f"[{now()}] selected {selection['selected_config_id']} at "
          f"{selection['selected_budget_updates']} updates", flush=True)

    # Stage A2 -- the selected configuration, both arms, ten seeds, same folds.
    chosen = [selection["selected_config_id"]]
    runner.run_all(
        jobs_for(WALK_FORWARD_SPLITS, ["economic", "zero_cost"], chosen, final_seeds, grid, updates,
                 interval=interval),
        "a2_selected_configuration_both_arms", progress, manifest_path,
    )

    gate = zero_cost_gate(output_dir, selection, momentum)
    progress["zero_cost_gate"] = gate
    write_json_atomic(manifest_path, progress)
    write_json_atomic(output_dir / "zero_cost_gate.json", gate)
    print(f"[{now()}] zero-cost gate passed={gate.get('gate_passed')}", flush=True)
    if args.stop_if_gate_fails and not gate.get("gate_passed"):
        progress["status"] = "stopped_at_zero_cost_gate"
        progress["finished_at_utc"] = now()
        write_json_atomic(manifest_path, progress)
        print("zero-cost gate failed and --stop-if-gate-fails was set; test stays sealed")
        return

    # Stage B -- retrain once on 2021-06..2025-05.  These runs evaluate nothing.
    runner.run_all(
        jobs_for([FINAL_SPLIT], ["economic", "zero_cost"], chosen, final_seeds, grid,
                 selection["selected_budget_updates"], evaluate_validation=False,
                 interval=interval),
        "b_final_training", progress, manifest_path,
    )

    progress["status"] = "training_complete"
    progress["finished_training_at_utc"] = now()
    write_json_atomic(manifest_path, progress)

    from evaluate import open_sealed_test
    print(f"[{now()}] opening the sealed test year", flush=True)
    test = open_sealed_test(output_dir)
    progress["status"] = "completed"
    progress["test_summary"] = {
        arm: test["arms"][arm]["aggregate"] for arm in test["arms"]
    }
    progress["finished_at_utc"] = now()
    write_json_atomic(manifest_path, progress)

    from report import write_report
    write_report(output_dir)
    print(f"[{now()}] study complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
