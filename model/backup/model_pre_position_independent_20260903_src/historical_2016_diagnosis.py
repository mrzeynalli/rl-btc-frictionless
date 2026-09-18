"""Out-of-domain 2016 initialization-sensitivity audit for the locked ensemble.

BTC perpetual futures did not exist in 2016.  This diagnostic therefore uses
Coinbase BTC-USD five-minute spot candles, zero funding, a neutral taker-side
imbalance, and approximate quote volume.  It must not be used for model
selection or presented as a like-for-like futures backtest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from concern_study import SEEDS, load_model
from evaluation import HOURS_PER_YEAR, metrics
from features import (
    FAST_FEATURE_NAMES,
    EVALUATION_WARMUP_HOURS,
    MarketData,
    Normaliser,
    Split,
    SplitData,
    build_market,
)
from holdout_diagnosis import run_ensemble_with_initial_position
from holdout_july import ARMS, DEFAULT_OUTPUT, HOLDOUT_SPLIT, SWITCH_MARGIN, locked_cell, training_spec
from train import write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_DATA = Path("/tmp/cryptorl_btc_2016_coinbase")
REFERENCE_TRAINING = (
    DEFAULT_OUTPUT / training_spec(SEEDS[0]).directory / "training.json"
)
RESULT_DIR = REPO_ROOT / "artifacts" / "2016_initialisation_diagnosis"
RESULT_PATH = RESULT_DIR / "results.json"
MONTHLY_CSV = RESULT_DIR / "monthly_results.csv"

PRODUCT = "BTC-USD"
API = f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles"
DATA_START = pd.Timestamp("2015-12-01T00:00:00Z")
DATA_END = pd.Timestamp("2017-01-02T00:00:00Z")
EVALUATION_START = pd.Timestamp("2016-01-01T00:00:00Z")
EVALUATION_END = pd.Timestamp("2017-01-01T00:00:00Z")
GRANULARITY_SECONDS = 300
REQUEST_HOURS = 24
STARTING_POSITIONS = {"SHORT": -1.0, "FLAT": 0.0, "LONG": 1.0}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_z(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_candles(start: pd.Timestamp, end: pd.Timestamp) -> list[list[float]]:
    query = urllib.parse.urlencode({
        "granularity": GRANULARITY_SECONDS,
        "start": _iso_z(start),
        "end": _iso_z(end),
    })
    request = urllib.request.Request(
        f"{API}?{query}",
        headers={"User-Agent": "CryptoRL-research/1.0", "Accept": "application/json"},
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
            if not isinstance(payload, list):
                raise RuntimeError(f"unexpected Coinbase response: {payload!r}")
            return payload
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt == 7:
                raise RuntimeError(
                    f"Coinbase request failed for {start} to {end}"
                ) from error
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def download_raw(output_dir: Path, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Download daily chunks and cache the untouched exchange observations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "coinbase-btc-usd-5m-raw.parquet"
    audit_path = output_dir / "download-audit.json"
    if raw_path.exists() and audit_path.exists() and not force:
        return pd.read_parquet(raw_path), json.loads(audit_path.read_text())

    rows: list[list[float]] = []
    cursor = DATA_START
    requests = 0
    while cursor < DATA_END:
        chunk_end = min(cursor + pd.Timedelta(hours=REQUEST_HOURS), DATA_END)
        rows.extend(_request_candles(cursor, chunk_end))
        requests += 1
        cursor = chunk_end
        # The public Exchange REST limit is shared.  This stays deliberately
        # below ten historical requests per second.
        time.sleep(0.12)

    raw = pd.DataFrame(rows, columns=["unix", "low", "high", "open", "close", "volume"])
    for column in ("unix", "low", "high", "open", "close", "volume"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    raw["open_time"] = pd.to_datetime(raw.pop("unix"), unit="s", utc=True)
    raw = raw[(raw.open_time >= DATA_START) & (raw.open_time < DATA_END)]
    duplicates = int(raw.open_time.duplicated().sum())
    raw = raw.drop_duplicates("open_time", keep="last").sort_values("open_time")
    raw = raw[["open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    raw.to_parquet(raw_path, index=False)
    expected = pd.date_range(DATA_START, DATA_END, freq="5min", inclusive="left")
    missing = expected.difference(pd.DatetimeIndex(raw.open_time))
    audit = {
        "downloaded_at_utc": now(),
        "source": "Coinbase Exchange REST API product candles",
        "endpoint": API,
        "product": PRODUCT,
        "granularity_seconds": GRANULARITY_SECONDS,
        "start_inclusive_utc": DATA_START.isoformat(),
        "end_exclusive_utc": DATA_END.isoformat(),
        "requests": requests,
        "rows_returned_before_deduplication": len(rows),
        "duplicate_timestamps_removed": duplicates,
        "raw_unique_rows": len(raw),
        "expected_grid_rows": len(expected),
        "missing_no_tick_intervals": len(missing),
        "raw_path": str(raw_path),
        "raw_sha256": sha256(raw_path),
    }
    write_json_atomic(audit_path, audit)
    return raw, audit


def build_proxy_dataset(output_dir: Path, force_download: bool = False) -> dict:
    raw, download_audit = download_raw(output_dir, force_download)
    expected = pd.date_range(DATA_START, DATA_END, freq="5min", inclusive="left")
    indexed = raw.set_index("open_time").reindex(expected)
    missing = indexed.close.isna()
    if bool(missing.iloc[0]):
        raise ValueError("the first Coinbase interval is missing; causal fill is impossible")

    prior_close = indexed.close.ffill()
    for column in ("open", "high", "low", "close"):
        indexed.loc[missing, column] = prior_close.loc[missing]
    indexed.loc[missing, "volume"] = 0.0
    indexed.index.name = "open_time"
    bars = indexed.reset_index()
    typical_price = bars[["open", "high", "low", "close"]].mean(axis=1)
    bars["close_time"] = bars.open_time + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
    bars["quote_volume"] = bars.volume * typical_price
    bars["trades"] = 0
    bars["taker_buy_base"] = bars.volume / 2.0
    bars["taker_buy_quote"] = bars.quote_volume / 2.0
    bars = bars[[
        "open_time", "close_time", "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
    ]]

    # Synthetic timestamps preserve the known eight-hour settlement clock.
    # Rates are zero because this is spot data and no perpetual existed.
    funding_times = pd.date_range(
        DATA_START - pd.Timedelta(hours=8), DATA_END + pd.Timedelta(hours=8), freq="8h"
    )
    funding = pd.DataFrame({
        "funding_time": funding_times,
        "funding_rate": np.zeros(len(funding_times), dtype=np.float64),
        "funding_interval_hours": np.full(len(funding_times), 8.0),
    })
    bars_path = output_dir / "BTCUSDT-5m-bars.parquet"
    funding_path = output_dir / "BTCUSDT-funding-settlements.parquet"
    bars.to_parquet(bars_path, index=False)
    funding.to_parquet(funding_path, index=False)
    audit = {
        "download": download_audit,
        "proxy_market": "Coinbase BTC-USD spot; NOT Binance BTCUSDT perpetual",
        "filled_no_tick_intervals": int(missing.sum()),
        "gap_fill": "prior close for OHLC and zero base/quote volume",
        "quote_volume": "approximated as Coinbase base volume times OHLC mean",
        "taker_imbalance": "unavailable; both taker-buy volumes set to 50% (imbalance zero)",
        "trades": "unavailable; set to zero and not used by model features",
        "funding": "unavailable/not applicable; zero rates on synthetic 8-hour schedule",
        "five_minute_rows": len(bars),
        "bars_sha256": sha256(bars_path),
        "funding_sha256": sha256(funding_path),
        "built_at_utc": now(),
    }
    write_json_atomic(output_dir / "proxy-audit.json", audit)
    return audit


def locked_normaliser(reference_training: Path, market: MarketData) -> Normaliser:
    payload = json.loads(reference_training.read_text())["split"]["normaliser"]
    if tuple(payload["fast"]) != FAST_FEATURE_NAMES:
        raise ValueError("locked fast-feature order differs from current code")
    if tuple(payload["slow"]) != market.slow_names:
        raise ValueError("locked slow-feature order differs from 2016 proxy")
    return Normaliser(
        fast_mean=np.array([payload["fast"][name]["mean"] for name in FAST_FEATURE_NAMES]),
        fast_std=np.array([payload["fast"][name]["std"] for name in FAST_FEATURE_NAMES]),
        slow_mean=np.array([payload["slow"][name]["mean"] for name in market.slow_names]),
        slow_std=np.array([payload["slow"][name]["std"] for name in market.slow_names]),
        slow_names=market.slow_names,
        clip=float(payload["clip_sigma"]),
        fitted_on=payload["fitted_on"],
    )


def split_data(market: MarketData, normaliser: Normaliser) -> SplitData:
    fast, slow = normaliser.apply(market.fast, market.slow)
    evaluation_range = market.index_range(EVALUATION_START, EVALUATION_END)
    definition = Split(
        "coinbase_spot_2016_domain_shift",
        "diagnostic",
        HOLDOUT_SPLIT.train_start,
        HOLDOUT_SPLIT.train_end,
        EVALUATION_START,
        EVALUATION_END,
    )
    return SplitData(
        split=definition,
        market=market,
        normaliser=normaliser,
        fast=fast,
        slow=slow,
        train_range=(0, 0),
        evaluation_range=evaluation_range,
        evaluation_warmup_start=max(0, evaluation_range[0] - EVALUATION_WARMUP_HOURS),
    )


def path_summary(trajectory, initial_position: float) -> dict:
    result = metrics(trajectory)
    previous = np.concatenate(([initial_position], trajectory.positions[:-1]))
    turnover = np.abs(trajectory.positions - previous)
    changes = trajectory.positions != previous
    result.update({
        "mean_net_exposure": float(trajectory.positions.mean()),
        "turnover_total_from_preexisting_start": float(turnover.sum()),
        "position_changes_from_preexisting_start": int(changes.sum()),
        "trading_cost_return_units": float(trajectory.cost_paid.sum()),
        "action_frequency": {
            "short": float(np.mean(trajectory.positions < 0)),
            "flat": float(np.mean(trajectory.positions == 0)),
            "long": float(np.mean(trajectory.positions > 0)),
        },
    })
    return result


def btc_month_return(market: MarketData, first: int, last: int) -> float:
    ratios = market.exit_price[first:last] / market.entry_price[first:last]
    return float(np.prod(ratios) - 1.0)


def convergence_hours(paths: dict[str, np.ndarray]) -> int | None:
    stacked = np.stack([paths[name] for name in STARTING_POSITIONS])
    all_equal = np.all(stacked == stacked[0], axis=0)
    # Deterministic paths cannot separate after reaching the same position, but
    # checking the suffix makes that property explicit in the recorded result.
    suffix_equal = np.logical_and.accumulate(all_equal[::-1])[::-1]
    indices = np.flatnonzero(suffix_equal)
    return int(indices[0]) if len(indices) else None


def pairwise_agreement(paths: dict[str, np.ndarray]) -> dict:
    names = list(STARTING_POSITIONS)
    return {
        f"{left}-{right}": float(np.mean(paths[left] == paths[right]))
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def evaluate(output_dir: Path, result_dir: Path) -> dict:
    torch.set_num_threads(1)
    market = build_market(output_dir, funding_features=True)
    normaliser = locked_normaliser(REFERENCE_TRAINING, market)
    base_split = split_data(market, normaliser)
    models = [
        load_model(
            locked_cell(),
            base_split,
            DEFAULT_OUTPUT / training_spec(seed).directory / "model_final.pt",
        )
        for seed in SEEDS
    ]

    month_starts = pd.date_range(EVALUATION_START, EVALUATION_END, freq="MS", inclusive="left")
    monthly: list[dict] = []
    concatenated_rewards: dict[str, list[np.ndarray]] = {name: [] for name in STARTING_POSITIONS}
    concatenated_positions: dict[str, list[np.ndarray]] = {name: [] for name in STARTING_POSITIONS}
    concatenated_costs: dict[str, list[np.ndarray]] = {name: [] for name in STARTING_POSITIONS}
    for start in month_starts:
        end = start + pd.offsets.MonthBegin(1)
        first, last = market.index_range(start, end)
        month_split = replace(
            base_split,
            evaluation_range=(first, last),
            evaluation_warmup_start=max(0, first - EVALUATION_WARMUP_HOURS),
        )
        paths: dict[str, np.ndarray] = {}
        summaries: dict[str, dict] = {}
        for name, initial in STARTING_POSITIONS.items():
            trajectory, _ = run_ensemble_with_initial_position(
                models,
                month_split,
                first,
                last,
                ARMS["economic"],
                SWITCH_MARGIN,
                initial,
            )
            paths[name] = trajectory.positions
            summaries[name] = path_summary(trajectory, initial)
            concatenated_rewards[name].append(trajectory.rewards)
            concatenated_positions[name].append(trajectory.positions)
            concatenated_costs[name].append(trajectory.cost_paid)
        returns = [summaries[name]["total_return"] for name in STARTING_POSITIONS]
        monthly.append({
            "month": start.strftime("%Y-%m"),
            "hours": last - first,
            "btc_spot_return": btc_month_return(market, first, last),
            "by_start": summaries,
            "return_range": float(max(returns) - min(returns)),
            "pairwise_action_agreement": pairwise_agreement(paths),
            "hours_until_all_paths_permanently_converge": convergence_hours(paths),
            "same_terminal_position": bool(len({float(path[-1]) for path in paths.values()}) == 1),
        })

    aggregate: dict[str, dict] = {}
    for name in STARTING_POSITIONS:
        rewards = np.concatenate(concatenated_rewards[name])
        positions = np.concatenate(concatenated_positions[name])
        costs = np.concatenate(concatenated_costs[name])
        monthly_returns = np.array([row["by_start"][name]["total_return"] for row in monthly])
        std = rewards.std(ddof=1)
        aggregate[name] = {
            "average_monthly_return": float(monthly_returns.mean()),
            "median_monthly_return": float(np.median(monthly_returns)),
            "positive_months": int(np.sum(monthly_returns > 0)),
            "monthly_reset_compounded_return": float(np.prod(1.0 + monthly_returns) - 1.0),
            "pooled_hourly_sharpe": float(rewards.mean() / std * math.sqrt(HOURS_PER_YEAR)) if std > 0 else 0.0,
            "mean_net_exposure": float(positions.mean()),
            "action_frequency": {
                "short": float(np.mean(positions < 0)),
                "flat": float(np.mean(positions == 0)),
                "long": float(np.mean(positions > 0)),
            },
            "total_trading_cost_return_units": float(costs.sum()),
            "average_monthly_turnover": float(np.mean([
                row["by_start"][name]["turnover_total_from_preexisting_start"] for row in monthly
            ])),
            "total_position_changes": int(np.sum([
                row["by_start"][name]["position_changes_from_preexisting_start"] for row in monthly
            ])),
        }

    convergence = [row["hours_until_all_paths_permanently_converge"] for row in monthly]
    return_ranges = np.array([row["return_range"] for row in monthly])
    result = {
        "status": "completed",
        "completed_at_utc": now(),
        "purpose": "initial-position sensitivity only; no model selection or tuning",
        "configuration": {
            "model": "locked July 2026 five-seed probability ensemble",
            "seeds": list(SEEDS),
            "switch_hysteresis": SWITCH_MARGIN,
            "one_way_cost": ARMS["economic"],
            "starting_positions": STARTING_POSITIONS,
            "initial_position_accounting": "preexisting at each month boundary; no artificial entry cost",
            "gru_warmup_hours_before_each_month": EVALUATION_WARMUP_HOURS,
            "normaliser": normaliser.as_dict(),
        },
        "data_audit": json.loads((output_dir / "proxy-audit.json").read_text()),
        "aggregate_by_start": aggregate,
        "dependence_summary": {
            "average_monthly_return_range": float(return_ranges.mean()),
            "median_monthly_return_range": float(np.median(return_ranges)),
            "maximum_monthly_return_range": float(return_ranges.max()),
            "months_all_paths_converged": int(sum(value is not None for value in convergence)),
            "months_never_converged": int(sum(value is None for value in convergence)),
            "median_hours_to_convergence_when_converged": (
                float(np.median([value for value in convergence if value is not None]))
                if any(value is not None for value in convergence) else None
            ),
            "months_with_same_terminal_position": int(sum(row["same_terminal_position"] for row in monthly)),
        },
        "monthly": monthly,
        "limitations": [
            "BTC perpetual futures and their funding/taker fields did not exist for this model's market in 2016.",
            "Coinbase BTC-USD spot prices and base volume replace Binance BTCUSDT perpetual inputs.",
            "Quote volume is approximated; taker imbalance is neutral; funding rates are zero.",
            "The model and normaliser were trained on the much later 2021-2026 regime.",
            "Monthly resets are counterfactual diagnostics, not one continuous tradable strategy.",
        ],
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_dir / "results.json", result)
    csv_rows = []
    for row in monthly:
        for name, summary in row["by_start"].items():
            csv_rows.append({
                "month": row["month"],
                "start": name,
                "total_return": summary["total_return"],
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "mean_net_exposure": summary["mean_net_exposure"],
                "turnover": summary["turnover_total_from_preexisting_start"],
                "position_changes": summary["position_changes_from_preexisting_start"],
                **{f"frequency_{key}": value for key, value in summary["action_frequency"].items()},
                "return_range_across_starts": row["return_range"],
                "hours_to_convergence": row["hours_until_all_paths_permanently_converge"],
            })
    pd.DataFrame(csv_rows).to_csv(result_dir / "monthly_results.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-data", type=Path, default=TEMP_DATA)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    audit = build_proxy_dataset(args.temp_data, args.force_download)
    print(json.dumps({"download": audit["download"], "stage": "evaluating"}, indent=2), flush=True)
    result = evaluate(args.temp_data, args.result_dir)
    print(json.dumps({
        "status": result["status"],
        "result": str(args.result_dir / "results.json"),
        "summary": result["dependence_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
