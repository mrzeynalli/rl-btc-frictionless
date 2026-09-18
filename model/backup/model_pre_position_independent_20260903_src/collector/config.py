"""Configuration for market data collection.

Market: Binance USD-M perpetual futures (the <COIN>USDT contracts on
fapi/`futures/um`), not spot. Perpetuals are what carry funding rates.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

SYMBOLS = ["BTCUSDT"]

# "um" = USD-margined perpetual futures on data.binance.vision / fapi.
MARKET = "um"

INTERVAL = "5m"
KLINES_DIR = "5MIN"
FUNDING_DIR = "FUNDING"

# Inclusive month range: (year, month)
START_MONTH = (2025, 5)
END_MONTH = (2026, 5)

# Column order as returned by both the Binance Vision CSV files and the
# fapi /fapi/v1/klines endpoint.
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

# Final kline schema written to parquet.
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

# Funding rate archives ship these three columns.
FUNDING_RAW_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

FUNDING_OUTPUT_COLUMNS = [
    "funding_time",
    "symbol",
    "funding_rate",
    "funding_interval_hours",
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
    """data/1MIN/<symbol>.parquet — one time-ordered kline dataset per symbol."""
    return DATA_DIR / KLINES_DIR / f"{symbol}.parquet"


def funding_path(symbol: str) -> Path:
    """data/FUNDING/<symbol>.parquet — one time-ordered funding series per symbol."""
    return DATA_DIR / FUNDING_DIR / f"{symbol}.parquet"
