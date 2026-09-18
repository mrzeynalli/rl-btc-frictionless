"""Load the audited Binance BTCUSDT Spot one-minute dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "binance_spot_btcusdt_1m_2017_2026_07_21"
STUDY_START = pd.Timestamp("2017-08-17T04:00:00Z")
STUDY_END = pd.Timestamp("2026-07-22T00:00:00Z")


def load(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Return native Spot bars after validating the published audit manifest."""
    bars_path = output_dir / "BTCUSDT-1m-spot.parquet"
    manifest_path = output_dir / "manifest.json"
    if not (bars_path.exists() and manifest_path.exists()):
        raise FileNotFoundError(f"Spot dataset is incomplete under {output_dir}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("audit_passed"):
        raise ValueError("refusing to load a dataset without a passed audit")
    if manifest.get("market") != "BTCUSDT Spot":
        raise ValueError("dataset manifest is not BTCUSDT Spot")
    if manifest.get("interval") != "1m":
        raise ValueError("dataset manifest is not one-minute data")
    if manifest.get("funding_data_present"):
        raise ValueError("Spot dataset unexpectedly contains funding data")
    bars = pd.read_parquet(bars_path)
    required = {
        "open_time", "close_time", "open", "high", "low", "close",
        "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
    }
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Spot table is missing columns: {sorted(missing)}")
    if bars.open_time.duplicated().any() or not bars.open_time.is_monotonic_increasing:
        raise ValueError("Spot timestamps are duplicated or unordered")
    return bars
