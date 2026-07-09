"""Configuration for market data collection."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]

INTERVAL = "1m"
INTERVAL_DIR = "1MIN"

# Inclusive month range: (year, month)
START_MONTH = (2021, 1)
END_MONTH = (2026, 6)

# Column order as returned by both the Binance Vision CSV files and the
# REST /api/v3/klines endpoint.
RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# Final schema written to parquet.
OUTPUT_COLUMNS = [
    "open_time",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def month_range(start: tuple[int, int], end: tuple[int, int]):
    """Yield (year, month) tuples from start to end, inclusive."""
    year, month = start
    while (year, month) <= end:
        yield year, month
        month += 1
        if month > 12:
            year, month = year + 1, 1


def parquet_path(symbol: str) -> Path:
    """data/1MIN/<symbol>.parquet — one time-ordered dataset per symbol."""
    return DATA_DIR / INTERVAL_DIR / f"{symbol}.parquet"
