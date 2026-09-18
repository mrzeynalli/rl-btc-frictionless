"""Train one fixed five-seed one-minute Spot ensemble, then evaluate 2026."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env import ACTION_NAMES, ONE_WAY_COST
from evaluation import (
    always_btc,
    always_cash,
    metrics,
    relative_performance,
    run_ensemble,
    run_policy,
    starting_position_diagnostics,
)
from features import LOCKED_TEST_SPLIT, build_market, build_split
from model import MarketActorCritic
from train import write_json_atomic


MODEL_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = MODEL_ROOT / "results" / "spot_v3_1m"
RUNS_ROOT = RESULTS_ROOT / "runs"
LOGS_ROOT = RESULTS_ROOT / "logs"
CHARTS_ROOT = RESULTS_ROOT / "charts"
STATUS_PATH = RESULTS_ROOT / "experiment_status.json"
SEEDS = (11, 29, 47, 71, 101)
COSTS_BP = (4, 6, 8, 10)
UPDATES = 225
CONFIG_ID = "spot_v3_1m_g095_l095_detached_adaptive"
PPO = {
    "environments": 32,
    "rollout_steps": 128,
    "minibatches": 2,
    "actor_epochs": 4,
    "extra_critic_epochs": 4,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "value_coefficient": 1.0,
    "entropy_coefficient": 0.02,
    "entropy_final_coefficient": 0.005,
    "entropy_anneal_updates": UPDATES,
    "adaptive_entropy": True,
    "entropy_target_fraction": 0.60,
    "entropy_adaptation_rate": 0.05,
    "entropy_min_coefficient": 0.005,
    "entropy_max_coefficient": 0.05,
    "actor_learning_rate": 0.0003,
    "critic_learning_rate": 0.0001,
    "burn_in_hours": 240,
    "execution_kappa": 0.0,
    "measure_gradient_conflict": False,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_directory(seed: int) -> Path:
    return RUNS_ROOT / "train_pre_2026" / "locked_2026" / CONFIG_ID / f"seed_{seed}"


def completed(seed: int) -> bool:
    path = run_directory(seed) / "training.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def launch(seed: int) -> dict:
    command = [
        sys.executable,
        "-m",
        "train",
        "--split",
        "locked_2026",
        "--config-id",
        CONFIG_ID,
        "--seed",
        str(seed),
        "--updates",
        str(UPDATES),
        "--output-dir",
        str(RUNS_ROOT),
        "--output-group",
        "train_pre_2026",
        "--evaluation-interval",
        str(UPDATES),
        "--no-validation",
        "--encoder-type",
        "gru",
        "--critic-mode",
        "detached",
        "--ppo-json",
        json.dumps(PPO),
    ]
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(MODEL_ROOT / "src"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    started = now()
    process = subprocess.run(
        command,
        cwd=MODEL_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    return {
        "seed": seed,
        "started_at_utc": started,
        "finished_at_utc": now(),
        "returncode": process.returncode,
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-4000:],
    }


def load_model(seed: int) -> MarketActorCritic:
    model = MarketActorCritic(encoder_type="gru", critic_mode="detached")
    model.load_state_dict(
        torch.load(run_directory(seed) / "model_final.pt", weights_only=True)
    )
    model.eval()
    return model


def svg_line(values: np.ndarray, path: Path, title: str) -> None:
    width, height, padding = 1000, 360, 40
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 1e-12)
    x = np.linspace(padding, width - padding, len(values))
    y = height - padding - (values - low) / span * (height - 2 * padding)
    points = " ".join(f"{left:.2f},{top:.2f}" for left, top in zip(x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{padding}" y="24" font-family="sans-serif" font-size="16">{title}</text>'
        f'<polyline fill="none" stroke="#1769aa" stroke-width="2" points="{points}"/>'
        f'<text x="{padding}" y="{height-8}" font-family="sans-serif" font-size="12">'
        f'min {low:.3f} / max {high:.3f}</text></svg>'
    )


def evaluate(models: list[MarketActorCritic]) -> dict:
    market = build_market()
    split = build_split(market, LOCKED_TEST_SPLIT)
    first, last = split.evaluation_range
    bitcoin = always_btc(split, first, last, 0.0).rewards
    by_cost = {}
    selected = None
    agreement = None
    for bp in COSTS_BP:
        trajectory, diagnostics = run_ensemble(
            models,
            split,
            first,
            last,
            bp / 10_000.0,
            aggregation="mean",
            initial_position=0.0,
        )
        report = metrics(trajectory, f"ensemble_{bp}bp")
        report.update(relative_performance(trajectory.rewards, bitcoin))
        by_cost[str(bp)] = report
        if bp == 4:
            selected = trajectory
            agreement = diagnostics
    assert selected is not None
    individual = []
    for seed, model in zip(SEEDS, models):
        trajectory = run_policy(model, split, first, last, ONE_WAY_COST)
        report = metrics(trajectory, f"seed_{seed}")
        report.update(relative_performance(trajectory.rewards, bitcoin))
        individual.append(report)
    starts = starting_position_diagnostics(
        lambda initial: run_ensemble(
            models,
            split,
            first,
            last,
            ONE_WAY_COST,
            aggregation="mean",
            initial_position=initial,
        )[0]
    )
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "execution_time": market.execution_time[first:last],
        "log_return": selected.rewards,
        "equity": selected.equity,
        "position": selected.positions,
        "action": [ACTION_NAMES[item] for item in selected.actions],
    })
    frame.to_csv(RESULTS_ROOT / "test_2026_equity.csv", index=False)
    svg_line(selected.equity, CHARTS_ROOT / "test_2026_equity.svg", "Spot-v3 1m ensemble: 2026 test equity")
    drawdown = selected.equity / np.maximum.accumulate(selected.equity) - 1.0
    svg_line(drawdown, CHARTS_ROOT / "test_2026_drawdown.svg", "Spot-v3 1m ensemble: 2026 test drawdown")
    return {
        "status": "complete",
        "created_at_utc": now(),
        "test_label": "held out from Spot-v3 training; previously exposed at project level",
        "period": LOCKED_TEST_SPLIT.as_dict(),
        "configuration_id": CONFIG_ID,
        "cost_sensitivity": by_cost,
        "individual_seeds_4bp": individual,
        "ensemble_agreement": agreement,
        "starting_position_diagnostics": starts,
        "baselines_4bp": {
            "always_cash": metrics(always_cash(split, first, last, ONE_WAY_COST), "always_cash"),
            "buy_and_hold_btc": metrics(always_btc(split, first, last, ONE_WAY_COST), "buy_and_hold_btc"),
        },
    }


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    performance_log = LOGS_ROOT / "jobs.jsonl"
    market = build_market()
    split = build_split(market, LOCKED_TEST_SPLIT)
    status = {
        "status": "training",
        "started_at_utc": now(),
        "market_audit": market.audit,
        "split": split.as_dict(),
        "configuration_id": CONFIG_ID,
        "configuration": PPO,
        "seeds": list(SEEDS),
        "updates_per_seed": UPDATES,
        "workers": 3,
        "locked_2026_accessed": False,
        "training": {
            "total": len(SEEDS),
            "already_complete": sum(completed(seed) for seed in SEEDS),
            "finished_this_run": 0,
            "failed": 0,
        },
    }
    write_json_atomic(STATUS_PATH, status)
    pending = [seed for seed in SEEDS if not completed(seed)]
    lock = threading.Lock()
    print(f"[{now()}] training {len(pending)} pending seeds", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(launch, seed): seed for seed in pending}
        for future in as_completed(futures):
            outcome = future.result()
            with lock:
                status["training"]["finished_this_run"] += 1
                status["training"]["failed"] += int(outcome["returncode"] != 0)
                status["training"]["last_finished"] = outcome
                with performance_log.open("a") as stream:
                    stream.write(json.dumps(outcome) + "\n")
                write_json_atomic(STATUS_PATH, status)
            print(
                f"[{now()}] seed {outcome['seed']} rc={outcome['returncode']}",
                flush=True,
            )
    incomplete = [seed for seed in SEEDS if not completed(seed)]
    if incomplete:
        status["status"] = "failed"
        status["incomplete_seeds"] = incomplete
        write_json_atomic(STATUS_PATH, status)
        raise RuntimeError(f"incomplete seeds: {incomplete}")
    status["status"] = "testing"
    status["locked_2026_accessed"] = True
    status["test_accessed_at_utc"] = now()
    write_json_atomic(STATUS_PATH, status)
    report = evaluate([load_model(seed) for seed in SEEDS])
    write_json_atomic(RESULTS_ROOT / "test_2026_report.json", report)
    status["status"] = "complete"
    status["finished_at_utc"] = now()
    write_json_atomic(STATUS_PATH, status)
    print(f"[{now()}] complete", flush=True)


if __name__ == "__main__":
    main()
