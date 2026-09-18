"""Fixed specification for the BTCUSDT Spot one-minute research dataset."""

from pathlib import Path

import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = MODEL_ROOT / "data" / "binance_spot_btcusdt_1m_2017_2026_07_21"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
REQUESTED_START = pd.Timestamp("2016-07-22T00:00:00Z")
END_EXCLUSIVE = pd.Timestamp("2026-07-22T00:00:00Z")
ARCHIVE_TEMPLATE = (
    "https://data.binance.vision/data/spot/monthly/klines/"
    "{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
)
RAW_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
)
OUTPUT_COLUMNS = (
    "open_time", "close_time", "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
)
