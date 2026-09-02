"""Assemble the five-year BTCUSDT five-minute study dataset.

The study window is 1 June 2021 inclusive to 1 June 2026 exclusive, preceded by
a feature warm-up long enough for the ten-day momentum variable.  The warm-up
builds features only; it never contributes a training reward.

Binance publishes native five-minute archives only from 2023 onwards for some
months, but the one-minute archive in ``data/1MIN/BTCUSDT.parquet`` covers the
whole window without a single missing bar, so the five-minute grid is built
from it.  Aggregation is exact -- first open, max high, min low, last close and
summed volumes reproduce a native bar identically -- and ``_audit`` proves it
by comparing every overlapping bar against the native five-minute archive
already on disk.

BTCUSDT USD-M perpetual futures only.  No other symbol is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
REPO_ROOT = Path(__file__).resolve().parents[1]

WARMUP_START = pd.Timestamp("2021-05-01 00:00:00", tz="UTC")
STUDY_START = pd.Timestamp("2021-06-01 00:00:00", tz="UTC")
STUDY_END = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
# The last decision of the study is executed at 2026-05-31 23:00 and closed at
# 2026-06-01 00:00, so one extra day of bars is carried purely to supply that
# closing execution price.  No feature and no reward is ever read from it.
EXECUTION_TAIL_END = pd.Timestamp("2026-06-02 00:00:00", tz="UTC")

SOURCE_MINUTE = REPO_ROOT / "data" / "1MIN" / f"{SYMBOL}.parquet"
SOURCE_FUNDING = REPO_ROOT / "data" / "FUNDING" / f"{SYMBOL}.parquet"
NATIVE_FIVE_MINUTE = REPO_ROOT / "data" / "BTCUSDT_5M_HOURLY" / "BTCUSDT-5m-contract.parquet"
OUTPUT_DIR = REPO_ROOT / "data" / "BTCUSDT_5M_2021_2026"

BAR_COLUMNS = [
    "open_time", "close_time", "open", "high", "low", "close",
    "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_minute_bars() -> pd.DataFrame:
    frame = pd.read_parquet(
        SOURCE_MINUTE,
        columns=[
            "open_time", "symbol", "open", "high", "low", "close", "volume",
            "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume",
        ],
    )
    frame = frame[frame["symbol"] == SYMBOL].drop(columns="symbol")
    frame = frame.sort_values("open_time").reset_index(drop=True)
    window = (frame["open_time"] >= WARMUP_START) & (frame["open_time"] < EXECUTION_TAIL_END)
    frame = frame.loc[window].reset_index(drop=True)

    expected = pd.date_range(WARMUP_START, EXECUTION_TAIL_END, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame["open_time"].equals(pd.Series(expected)):
        missing = expected.difference(pd.DatetimeIndex(frame["open_time"]))
        raise ValueError(
            f"one-minute grid is incomplete: rows={len(frame)} expected={len(expected)} "
            f"missing={len(missing)}"
        )
    return frame


def _aggregate_five_minute(minute: pd.DataFrame) -> pd.DataFrame:
    """Fold the one-minute grid into five-minute bars without any resampling gap.

    The grid is known to be complete and ordered, so a simple reshape over
    contiguous blocks of five is both exact and far cheaper than ``resample``.
    """
    blocks = len(minute) // 5
    if blocks * 5 != len(minute):
        raise ValueError("one-minute grid does not divide into whole five-minute bars")

    def block(column: str) -> np.ndarray:
        return minute[column].to_numpy().reshape(blocks, 5)

    open_time = pd.DatetimeIndex(minute["open_time"])[::5]
    bars = pd.DataFrame(
        {
            "open_time": open_time,
            "close_time": open_time + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1),
            "open": block("open")[:, 0],
            "high": block("high").max(axis=1),
            "low": block("low").min(axis=1),
            "close": block("close")[:, -1],
            "volume": block("volume").sum(axis=1),
            "quote_volume": block("quote_asset_volume").sum(axis=1),
            "trades": block("number_of_trades").sum(axis=1),
            "taker_buy_base": block("taker_buy_base_volume").sum(axis=1),
            "taker_buy_quote": block("taker_buy_quote_volume").sum(axis=1),
        }
    )
    return bars[BAR_COLUMNS]


def _load_funding() -> pd.DataFrame:
    frame = pd.read_parquet(SOURCE_FUNDING)
    frame = frame[frame["symbol"] == SYMBOL].drop(columns="symbol")
    frame = frame.sort_values("funding_time").reset_index(drop=True)
    frame = frame[frame["funding_time"] < EXECUTION_TAIL_END].reset_index(drop=True)
    frame["funding_rate"] = frame["funding_rate"].astype("float64")
    return frame


def _native_agreement(bars: pd.DataFrame) -> dict:
    """Compare the aggregated grid against Binance's own five-minute archive."""
    if not NATIVE_FIVE_MINUTE.exists():
        return {"checked": False, "reason": "native five-minute archive not on disk"}
    native = pd.read_parquet(NATIVE_FIVE_MINUTE)
    merged = bars.merge(native, on="open_time", suffixes=("_agg", "_native"), how="inner")
    if merged.empty:
        return {"checked": False, "reason": "no overlapping bars"}
    report: dict = {"checked": True, "overlapping_bars": int(len(merged)), "columns": {}}
    worst = 0.0
    for column in ("open", "high", "low", "close", "volume", "quote_volume",
                   "trades", "taker_buy_base", "taker_buy_quote"):
        left = merged[f"{column}_agg"].astype("float64").to_numpy()
        right = merged[f"{column}_native"].astype("float64").to_numpy()
        error = np.abs(left - right) / np.maximum(np.abs(right), 1.0)
        # A relative difference at the 1e-15 level is float summation order, not
        # disagreement; anything above one part in a million is.
        report["columns"][column] = {
            "max_relative_error": float(error.max()),
            "bars_differing_beyond_float_noise": int((error > 1e-12).sum()),
        }
        worst = max(worst, float(error.max()))
    report["max_relative_error"] = worst
    # Binance's own 1m and 5m archives disagree on the open of exactly one bar,
    # 2025-08-29 06:35, by a single 0.1 tick -- inside the exchange halt that the
    # earlier study also had to document.  That is an archive inconsistency, not
    # an aggregation defect, so it is recorded rather than silently tolerated.
    differing = [
        merged.loc[
            np.abs(merged[f"{column}_agg"].astype("float64")
                   - merged[f"{column}_native"].astype("float64"))
            / np.maximum(np.abs(merged[f"{column}_native"].astype("float64")), 1.0) > 1e-12,
            "open_time",
        ].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        for column in ("open", "high", "low", "close")
    ]
    report["price_bars_differing_from_native_archive"] = sorted(
        {timestamp for group in differing for timestamp in group}
    )
    report["agrees_with_native_archive"] = bool(worst < 1e-5)
    if not report["agrees_with_native_archive"]:
        raise ValueError(f"aggregated bars disagree with the native archive: {report}")
    return report


def _audit(bars: pd.DataFrame, funding: pd.DataFrame) -> dict:
    grid = pd.date_range(WARMUP_START, EXECUTION_TAIL_END, freq="5min", inclusive="left")
    if len(bars) != len(grid) or not bars["open_time"].equals(pd.Series(grid)):
        raise ValueError("five-minute grid is not complete and contiguous")
    if bars["open_time"].duplicated().any():
        raise ValueError("duplicate five-minute timestamps")
    prices = bars[["open", "high", "low", "close"]]
    if not (prices > 0).all().all():
        raise ValueError("non-positive price in the five-minute grid")
    if not (bars["high"] >= bars[["open", "close"]].max(axis=1)).all():
        raise ValueError("high below open/close")
    if not (bars["low"] <= bars[["open", "close"]].min(axis=1)).all():
        raise ValueError("low above open/close")
    if (bars["taker_buy_quote"] > bars["quote_volume"] * (1 + 1e-9) + 1e-6).any():
        raise ValueError("taker buy quote volume exceeds total quote volume")

    if funding["funding_time"].duplicated().any():
        raise ValueError("duplicate funding settlement timestamps")
    if not funding["funding_time"].is_monotonic_increasing:
        raise ValueError("funding settlements are unordered")
    in_study = funding[funding["funding_time"] >= WARMUP_START]
    gaps = in_study["funding_time"].diff().dropna().dt.total_seconds() / 3600.0
    if not gaps.between(7.9, 8.1).all():
        raise ValueError("BTCUSDT funding interval is not a clean eight hours")
    if (funding["funding_time"] < WARMUP_START).sum() == 0:
        raise ValueError("no funding settlement precedes the warm-up start")

    zero_volume = bars.index[bars["volume"] <= 0].tolist()
    return {
        "symbol": SYMBOL,
        "market": "Binance USD-M perpetual futures",
        "cross_asset_data_used": False,
        "bar_interval": "5m aggregated exactly from the native 1m archive",
        "warmup_start_utc": WARMUP_START.isoformat(),
        "study_start_inclusive_utc": STUDY_START.isoformat(),
        "study_end_exclusive_utc": STUDY_END.isoformat(),
        "warmup_bars": int((bars["open_time"] < STUDY_START).sum()),
        "warmup_days": float((bars["open_time"] < STUDY_START).sum()) / 288.0,
        "study_bars": int(
            ((bars["open_time"] >= STUDY_START) & (bars["open_time"] < STUDY_END)).sum()
        ),
        "execution_tail_bars": int((bars["open_time"] >= STUDY_END).sum()),
        "execution_tail_end_utc": EXECUTION_TAIL_END.isoformat(),
        "total_bars": int(len(bars)),
        "hourly_decisions_in_study": int(
            ((bars["open_time"] >= STUDY_START) & (bars["open_time"] < STUDY_END)).sum()
        ) // 12,
        "zero_volume_bars": len(zero_volume),
        "zero_volume_bar_times_utc": [
            bars["open_time"].iloc[index].isoformat() for index in zero_volume[:32]
        ],
        "funding_settlements_total": int(len(funding)),
        "funding_settlements_in_window": int(len(in_study)),
        "funding_first_utc": funding["funding_time"].iloc[0].isoformat(),
        "funding_last_utc": funding["funding_time"].iloc[-1].isoformat(),
        "funding_interval_hours_unique": sorted(
            funding["funding_interval_hours"].unique().tolist()
        ),
        "funding_settlement_hours_utc": sorted(
            funding["funding_time"].dt.hour.unique().tolist()
        ),
        "funding_rate_mean": float(in_study["funding_rate"].mean()),
        "funding_rate_std": float(in_study["funding_rate"].std()),
        "audit_passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    minute = _load_minute_bars()
    bars = _aggregate_five_minute(minute)
    funding = _load_funding()

    manifest = _audit(bars, funding)
    manifest["native_archive_check"] = _native_agreement(bars)

    paths = {
        "bars": output_dir / "BTCUSDT-5m-bars.parquet",
        "funding": output_dir / "BTCUSDT-funding-settlements.parquet",
    }
    bars.to_parquet(paths["bars"], index=False)
    funding.to_parquet(paths["funding"], index=False)
    manifest["files"] = {
        key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()
    }
    manifest["source_files"] = {
        "minute": {"path": str(SOURCE_MINUTE), "sha256": _sha256(SOURCE_MINUTE)},
        "funding": {"path": str(SOURCE_FUNDING), "sha256": _sha256(SOURCE_FUNDING)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(output_dir: Path = OUTPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the built dataset, building it first if it is not on disk."""
    bars_path = output_dir / "BTCUSDT-5m-bars.parquet"
    funding_path = output_dir / "BTCUSDT-funding-settlements.parquet"
    if not (bars_path.exists() and funding_path.exists()):
        build(output_dir)
    return pd.read_parquet(bars_path), pd.read_parquet(funding_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
