# CryptoRL

Reinforcement learning on crypto markets. Currently: Binance 1-minute kline
collection into monthly parquet files.

## Project layout

```
CryptoRL/
├── src/cryptorl/          # importable package (installed with pip install -e .)
│   └── data/
│       ├── config.py      # symbols, date range, schema, output paths
│       ├── binance.py     # bulk-archive downloader + REST API fallback
│       └── collect.py     # CLI entry point
├── data/                  # parquet output (gitignored)
│   └── 1MIN/<SYMBOL>/<YYYY>/<MM>/<SYMBOL>-1m-<YYYY>-<MM>.parquet
├── pyproject.toml
├── requirements.txt
└── .env                   # BINANCE_API_KEY / BINANCE_API_SECRET (gitignored)
```

Future components (feature engineering, environments, agents, backtesting)
belong in sibling packages: `src/cryptorl/features/`, `src/cryptorl/envs/`,
`src/cryptorl/agents/`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Collecting data

```bash
# Full configured range (BTC/ETH/SOL/XRP/BNB, 2021-01 .. 2026-06)
.venv/bin/python -m cryptorl.data.collect

# Subset
.venv/bin/python -m cryptorl.data.collect --symbols BTCUSDT ETHUSDT --start 2024-01 --end 2024-12

# Re-download existing files
.venv/bin/python -m cryptorl.data.collect --force
```

Existing files are skipped, so re-running only fetches what is missing —
extend `END_MONTH` in `config.py` (or pass `--end`) to append new months.

## Data source and rate limits

Historical months come from Binance's official bulk archive
(`data.binance.vision`), which has no request-weight limits. Months not yet
published there (typically only the current month) fall back to the REST
`/api/v3/klines` endpoint, throttled against the 6000/min request-weight cap.
The API key from `.env` is attached to REST requests when present; market
data endpoints work without it.

Timestamps are UTC, stored as `datetime64[us, UTC]`. Binance switched its
archive files from millisecond to microsecond timestamps on 2025-01-01; the
collector detects the unit and normalizes both eras to one schema.

## Schema

| column | type |
|---|---|
| open_time | datetime64[us, UTC] |
| symbol | str |
| open, high, low, close | float64 |
| volume, quote_asset_volume | float64 |
| close_time | datetime64[us, UTC] |
| number_of_trades | int64 |
| taker_buy_base_volume, taker_buy_quote_volume | float64 |
