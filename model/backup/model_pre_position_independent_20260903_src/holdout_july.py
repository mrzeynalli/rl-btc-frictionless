"""One-time evaluation of the locked five-seed ensemble on July 2026.

The July archive remains outside the ordinary data loader.  This module is an
explicit, auditable boundary: it verifies the sealed hashes, trains the fixed
configuration only on observations before July, and evaluates exactly one
deployment rule (probability ensemble with 0.10 switching hysteresis).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data as data_module
from concern_study import SEEDS, cell, load_model, run_ensemble
from evaluation import (
    HOURS_PER_YEAR,
    account_positions,
    always_flat,
    always_long,
    metrics,
    policy_diagnostics,
    time_series_momentum,
)
from features import Split, build_market, build_split
from train import ARMS, RunSpec, train_run, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
SEALED_DIR = REPO_ROOT / "data" / "UNTOUCHED_2026_07"
MAIN_MINUTE = REPO_ROOT / "data" / "1MIN" / "BTCUSDT.parquet"
MAIN_FUNDING = REPO_ROOT / "data" / "FUNDING" / "BTCUSDT.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "july_2026_holdout"
DEFAULT_DATASET = Path("/tmp/cryptorl_july_2026_dataset")

TRAIN_START = pd.Timestamp("2021-06-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-07-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-08-01", tz="UTC")
DATA_START = pd.Timestamp("2021-05-01", tz="UTC")
SWITCH_MARGIN = 0.10
UPDATES = 225
CONFIG_ID = "locked_ensemble_g090_e010_b240_detached"
MOMENTUM_LOOKBACK = 72
HOLDOUT_SPLIT = Split(
    "july_2026_holdout",
    "holdout",
    TRAIN_START,
    HOLDOUT_START,
    HOLDOUT_START,
    HOLDOUT_END,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_cell() -> dict:
    return cell(
        gamma=0.90,
        burn_in_hours=240,
        entropy=0.10,
        critic_encoder="detached",
    )


def verify_seal() -> dict:
    manifest = json.loads((SEALED_DIR / "sealed_manifest.json").read_text())
    expected = manifest["records"]["BTCUSDT"]
    price = SEALED_DIR / "1MIN" / "BTCUSDT.parquet"
    funding = SEALED_DIR / "FUNDING" / "BTCUSDT.parquet"
    actual = {"price_sha256": sha256(price), "funding_sha256": sha256(funding)}
    for key, value in actual.items():
        if value != expected[key]:
            raise ValueError(f"sealed BTCUSDT {key} mismatch")
    return {
        "sealed_at_utc": manifest["sealed_at_utc"],
        "period": manifest["period"],
        **actual,
    }


def prepare_dataset(dataset_dir: Path) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    seal = verify_seal()

    minute_columns = [
        "open_time", "symbol", "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    frames = [
        pd.read_parquet(MAIN_MINUTE, columns=minute_columns),
        pd.read_parquet(SEALED_DIR / "1MIN" / "BTCUSDT.parquet", columns=minute_columns),
    ]
    minute = pd.concat(frames, ignore_index=True)
    minute = minute[minute.symbol.eq("BTCUSDT")].drop(columns="symbol")
    minute = minute[(minute.open_time >= DATA_START) & (minute.open_time < HOLDOUT_END)]
    minute = minute.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    expected_grid = pd.date_range(DATA_START, HOLDOUT_END, freq="1min", inclusive="left")
    if len(minute) != len(expected_grid) or not minute.open_time.equals(pd.Series(expected_grid)):
        missing = expected_grid.difference(pd.DatetimeIndex(minute.open_time))
        raise ValueError(f"combined one-minute grid is incomplete: missing={len(missing)}")

    bars = data_module._aggregate_five_minute(minute)
    funding = pd.concat([
        pd.read_parquet(MAIN_FUNDING),
        pd.read_parquet(SEALED_DIR / "FUNDING" / "BTCUSDT.parquet"),
    ], ignore_index=True)
    funding = funding[funding.symbol.eq("BTCUSDT")].drop(columns="symbol")
    funding = funding[funding.funding_time < HOLDOUT_END]
    funding = funding.drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True)
    gaps = funding.loc[funding.funding_time >= DATA_START, "funding_time"].diff().dropna()
    if not (gaps == pd.Timedelta(hours=8)).all():
        raise ValueError("combined funding grid is not exactly eight-hourly")

    bars_path = dataset_dir / "BTCUSDT-5m-bars.parquet"
    funding_path = dataset_dir / "BTCUSDT-funding-settlements.parquet"
    bars.to_parquet(bars_path, index=False)
    funding.to_parquet(funding_path, index=False)
    audit = {
        "seal": seal,
        "minute_rows": len(minute),
        "five_minute_rows": len(bars),
        "funding_rows": len(funding),
        "first_minute": minute.open_time.iloc[0].isoformat(),
        "last_minute": minute.open_time.iloc[-1].isoformat(),
        "bars_sha256": sha256(bars_path),
        "combined_funding_sha256": sha256(funding_path),
    }
    write_json_atomic(dataset_dir / "audit.json", audit)
    return audit


def training_spec(seed: int) -> RunSpec:
    selected = locked_cell()
    return RunSpec(
        split_name=HOLDOUT_SPLIT.name,
        arm="economic",
        config_id=CONFIG_ID,
        seed=seed,
        updates=UPDATES,
        ppo=selected["ppo"],
        evaluation_interval=25,
        evaluate_validation=False,
        position_ladder=selected["position_ladder"],
        funding_features=selected["funding_features"],
    )


def train_seed(seed: int, output_dir: Path, dataset_dir: Path) -> None:
    market = build_market(dataset_dir, funding_features=True)
    train_run(
        training_spec(seed),
        output_dir,
        market=market,
        split_definition=HOLDOUT_SPLIT,
    )


def completed(seed: int, output_dir: Path) -> bool:
    path = output_dir / training_spec(seed).directory / "training.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "completed"
    except json.JSONDecodeError:
        return False


def launch_seed(seed: int, output_dir: Path, dataset_dir: Path) -> dict:
    started = now()
    command = [
        sys.executable,
        "-m",
        "holdout_july",
        "--worker-seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--dataset-dir",
        str(dataset_dir),
    ]
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(REPO_ROOT / "src"),
    )
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    return {
        "seed": seed,
        "started_at_utc": started,
        "finished_at_utc": now(),
        "returncode": process.returncode,
        "stderr": process.stderr[-4000:] if process.returncode else "",
    }


def continuous_max_drawdown(log_returns: np.ndarray) -> float:
    equity = np.exp(np.cumsum(log_returns))
    with_start = np.concatenate(([1.0], equity))
    peak = np.maximum.accumulate(with_start)
    return float(np.min(with_start / peak - 1.0))


def additional_metrics(trajectory, btc_log_returns: np.ndarray) -> dict:
    rewards = trajectory.rewards
    hourly_std = float(rewards.std(ddof=1))
    downside = np.minimum(rewards, 0.0)
    beta = float(
        np.cov(rewards, btc_log_returns, ddof=1)[0, 1]
        / np.var(btc_log_returns, ddof=1)
    )
    alpha_log = (
        float(rewards.mean()) - beta * float(btc_log_returns.mean())
    ) * HOURS_PER_YEAR
    return {
        "continuous_max_drawdown": continuous_max_drawdown(rewards),
        "annualised_downside_deviation": float(
            np.sqrt(np.mean(downside ** 2)) * math.sqrt(HOURS_PER_YEAR)
        ),
        "beta_to_btc": beta,
        "correlation_to_btc": float(np.corrcoef(rewards, btc_log_returns)[0, 1]),
        "annualised_alpha": float(np.exp(alpha_log) - 1.0),
        "mean_net_exposure": float(trajectory.positions.mean()),
        "total_trading_cost_return_units": float(trajectory.cost_paid.sum()),
        "action_frequency": {
            "long": float(np.mean(trajectory.positions > 0)),
            "flat": float(np.mean(trajectory.positions == 0)),
            "short": float(np.mean(trajectory.positions < 0)),
        },
        "policy_diagnostics": policy_diagnostics(trajectory),
        "hourly_return_mean": float(rewards.mean()),
        "hourly_return_std": hourly_std,
    }


def evaluate_once(output_dir: Path, dataset_dir: Path) -> dict:
    market = build_market(dataset_dir, funding_features=True)
    split = build_split(market, HOLDOUT_SPLIT)
    models = []
    selected = locked_cell()
    for seed in SEEDS:
        weights = output_dir / training_spec(seed).directory / "model_final.pt"
        models.append(load_model(selected, split, weights))

    first, last = split.evaluation_range
    trajectory = run_ensemble(
        models,
        split,
        first,
        last,
        ARMS["economic"],
        SWITCH_MARGIN,
    )
    btc_log_returns = np.log(
        market.exit_price[first:last] / market.entry_price[first:last]
    )
    benchmark_paths = {
        "buy_and_hold": always_long(split, first, last, ARMS["economic"]),
        "short_and_hold": account_positions(
            -np.ones(last - first), market, first, last, ARMS["economic"]
        ),
        "momentum_72h": time_series_momentum(
            split, first, last, ARMS["economic"], MOMENTUM_LOOKBACK
        ),
        "always_flat": always_flat(split, first, last, ARMS["economic"]),
    }

    frame = pd.DataFrame({
        "execution_time": market.execution_time[first:last],
        "position": trajectory.positions,
        "log_return": trajectory.rewards,
        "equity": trajectory.equity,
        "price_return": trajectory.price_return,
        "funding_return": trajectory.funding_return,
        "cost_paid": trajectory.cost_paid,
        "probability_short": trajectory.action_probabilities[:, 0],
        "probability_flat": trajectory.action_probabilities[:, 1],
        "probability_long": trajectory.action_probabilities[:, 2],
    })
    frame.to_csv(output_dir / "hourly_holdout_trajectory.csv", index=False)

    return {
        "status": "completed",
        "evaluated_at_utc": now(),
        "holdout_accessed": True,
        "configuration": {
            **selected["ppo"].as_dict(),
            "position_ladder": selected["position_ladder"],
            "funding_features": selected["funding_features"],
            "seeds": list(SEEDS),
            "updates": UPDATES,
            "switch_margin": SWITCH_MARGIN,
            "cost_bps": ARMS["economic"] * 10_000,
        },
        "split": split.as_dict(),
        "actual_evaluation": {
            "first_execution_utc": market.execution_time[first].isoformat(),
            "last_execution_utc": market.execution_time[last - 1].isoformat(),
            "hours": last - first,
            "excluded_terminal_hour_reason": (
                "The archive ends at 2026-07-31 23:59 UTC, so the 23:00 position "
                "cannot be closed at an unavailable 2026-08-01 00:00 price."
            ),
        },
        "ensemble": {
            **metrics(trajectory, "five_seed_ensemble_margin_0.10"),
            **additional_metrics(trajectory, btc_log_returns),
        },
        "benchmarks": {
            name: metrics(path, name) for name, path in benchmark_paths.items()
        },
        "bitcoin_underlying_total_return": float(np.exp(btc_log_returns.sum()) - 1.0),
    }


def run(output_dir: Path, dataset_dir: Path, workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_result = output_dir / "results.json"
    if completed_result.exists():
        existing = json.loads(completed_result.read_text())
        if existing.get("status") == "completed":
            print(json.dumps(existing, indent=2), flush=True)
            return
    selection_path = REPO_ROOT / "artifacts" / "concern_study" / "results.json"
    protocol = {
        "status": "preparing",
        "started_at_utc": now(),
        "locked_before_holdout": {
            "source_selection_sha256": sha256(selection_path),
            "configuration": {
                **locked_cell()["ppo"].as_dict(),
                "seeds": list(SEEDS),
                "updates": UPDATES,
                "switch_margin": SWITCH_MARGIN,
                "cost_bps": ARMS["economic"] * 10_000,
            },
        },
        "training_end_exclusive": HOLDOUT_START.isoformat(),
        "holdout_start_inclusive": HOLDOUT_START.isoformat(),
        "holdout_end_exclusive": HOLDOUT_END.isoformat(),
        "holdout_performance_accessed": False,
        "sealed_manifest_rule_deviation": (
            "The sealed manifest named median chronological-validation subperiod "
            "Sharpe; the user explicitly locked the validation-selected 0.10 "
            "ensemble instead. No July return was inspected when this was recorded."
        ),
        "runs": [],
    }
    write_json_atomic(output_dir / "protocol.json", protocol)
    protocol["dataset_audit"] = prepare_dataset(dataset_dir)
    protocol["status"] = "training"
    write_json_atomic(output_dir / "protocol.json", protocol)

    pending = [seed for seed in SEEDS if not completed(seed, output_dir)]
    if pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {
                pool.submit(launch_seed, seed, output_dir, dataset_dir): seed
                for seed in pending
            }
            for future in as_completed(futures):
                outcome = future.result()
                protocol["runs"].append(outcome)
                write_json_atomic(output_dir / "protocol.json", protocol)
                print(
                    f"[{now()}] seed {outcome['seed']} rc={outcome['returncode']}",
                    flush=True,
                )
    incomplete = [seed for seed in SEEDS if not completed(seed, output_dir)]
    if incomplete:
        protocol["status"] = "failed"
        protocol["incomplete_seeds"] = incomplete
        write_json_atomic(output_dir / "protocol.json", protocol)
        raise RuntimeError(f"holdout training incomplete for seeds {incomplete}")

    protocol["status"] = "evaluating_locked_rule"
    protocol["holdout_performance_accessed"] = True
    protocol["holdout_opened_at_utc"] = now()
    write_json_atomic(output_dir / "protocol.json", protocol)
    result = evaluate_once(output_dir, dataset_dir)
    write_json_atomic(output_dir / "results.json", result)
    protocol["status"] = "completed"
    protocol["finished_at_utc"] = now()
    write_json_atomic(output_dir / "protocol.json", protocol)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--worker-seed", type=int)
    args = parser.parse_args()
    if args.worker_seed is not None:
        train_seed(args.worker_seed, args.output_dir, args.dataset_dir)
    else:
        run(args.output_dir, args.dataset_dir, args.workers)


if __name__ == "__main__":
    main()
