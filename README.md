# CryptoRL

Reinforcement learning on crypto markets. Currently: Binance **USD-M
perpetual futures** 1-minute klines and funding rates, stored as one
parquet per symbol.

## Project layout

```
CryptoRL/
├── src/cryptorl/          # importable package (installed with pip install -e .)
│   └── data/
│       ├── config.py      # symbols, date range, schema, output paths
│       ├── binance.py     # bulk-archive downloader + REST API fallback
│       └── collect.py     # CLI entry point
├── data/                  # parquet output (gitignored)
│   ├── 1MIN/<SYMBOL>.parquet      # 1-minute klines
│   └── FUNDING/<SYMBOL>.parquet   # funding rate settlements
├── pyproject.toml
├── requirements.txt
└── .env                   # BINANCE_API_KEY / BINANCE_API_SECRET (gitignored)
```

Future components (feature engineering, environments, agents, backtesting)
belong in sibling packages: `src/cryptorl/features/`, `src/cryptorl/envs/`,
`src/cryptorl/agents/`.

## Market

Data is the **USD-M perpetual futures** contract (`<COIN>USDT` on
`fapi` / `futures/um`), *not* spot. Perpetuals are the tradable venue for
leveraged long/short RL policies and are what carry funding rates; their
prices differ from spot by the basis (e.g. BTC 2021-01-01 00:00 opens at
28948.19 on perp vs 28923.63 on spot).

Change `MARKET` / `SYMBOLS` / `START_MONTH` / `END_MONTH` in `config.py` to
retarget the collection.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Collecting data

```bash
# Full configured range: klines + funding (BTC/ETH/SOL/XRP/BNB, 2021-01 .. 2026-06)
.venv/bin/python -m cryptorl.data.collect

# One kind only
.venv/bin/python -m cryptorl.data.collect --what funding

# Subset
.venv/bin/python -m cryptorl.data.collect --symbols BTCUSDT ETHUSDT --start 2024-01 --end 2024-12

# Rebuild existing files (per-month progress with --verbose)
.venv/bin/python -m cryptorl.data.collect --force --verbose
```

Each symbol is written as a single parquet covering the whole range, sorted
by time. Symbols whose file already exists are skipped; to extend the date
range, pass the new `--end` together with `--force` to rebuild (months are
downloaded in parallel, a full rebuild takes ~3 minutes).

## Data source and rate limits

Historical months come from Binance's official bulk archive
(`data.binance.vision/data/futures/um/monthly/...`), which has no
request-weight limits. Months not yet published there (typically only the
current month) fall back to the REST `fapi/v1/klines` and
`fapi/v1/fundingRate` endpoints, throttled against the 2400/min
request-weight cap. The API key from `.env` is attached to REST requests
when present; market data endpoints work without it.

Binance sometimes publishes a **truncated monthly archive** — SOLUSDT and
XRPUSDT `2022-02` stop on the 25th, and their `2022-04` starts on the 3rd,
while BTCUSDT's are complete for the same months. The collector counts the
minutes in every elapsed day and tops up short days from the daily archives
(or REST if a day is absent there too), so a silently short month cannot
reach the dataset.

Timestamps are UTC, stored as `datetime64[us, UTC]`. Binance switched its
spot archives from millisecond to microsecond timestamps on 2025-01-01
while futures archives stayed in milliseconds, and newer files added a CSV
header row; the collector detects both and normalizes to one schema.

## Schema

### `data/1MIN/<SYMBOL>.parquet`

| column | type |
|---|---|
| open_time | datetime64[us, UTC] |
| symbol | str |
| open, high, low, close | float64 |
| volume, quote_asset_volume | float64 |
| close_time | datetime64[us, UTC] |
| number_of_trades | int64 |
| taker_buy_base_volume, taker_buy_quote_volume | float64 |

### `data/FUNDING/<SYMBOL>.parquet`

| column | type |
|---|---|
| funding_time | datetime64[us, UTC] |
| symbol | str |
| funding_rate | float64 |
| funding_interval_hours | int64 |

Funding normally settles every 8 hours (00:00 / 08:00 / 16:00 UTC), but
the interval is **not constant**: during the FTX collapse Binance shortened
SOLUSDT's to 4h and then 2h (2022-11-09 → 2022-11-18, 101 settlements) while
the rate sat pinned at the -0.02 clamp. Always weight a settlement by its
own `funding_interval_hours` rather than assuming three per day, or funding
PnL across that window comes out roughly 4x too small.

The archive's raw `calc_time` carries a few milliseconds of settlement
latency, so it is rounded to the minute — `funding_time` joins directly
against kline `open_time`:

```python
klines.merge(funding, left_on=["open_time", "symbol"],
             right_on=["funding_time", "symbol"], how="left")
```

`funding_rate` is the rate applied at that settlement (longs pay shorts
when positive). To carry it across the minutes between settlements, use
`.ffill()` — but note the rate is only *known* at settlement, so shift
before using it as a feature to avoid lookahead.

## Completeness

As collected (2021-01-01 .. 2026-06-30, verified against a full minute grid):

| symbol | kline rows | missing minutes | funding rows | missing settlements |
|---|---|---|---|---|
| BTCUSDT | 2,890,080 | 0 | 6,021 | 0 |
| ETHUSDT | 2,890,080 | 0 | 6,021 | 0 |
| SOLUSDT | 2,890,080 | 0 | 6,096 | 0 |
| XRPUSDT | 2,890,080 | 0 | 6,021 | 0 |
| BNBUSDT | 2,890,080 | 0 | 6,021 | 0 |

2,890,080 = 2007 days x 1440. SOLUSDT carries 75 extra funding rows from the
shortened-interval window described above. Note the perpetual series is
*more* complete than spot, which had ~1,000 minutes of exchange downtime
per symbol over the same range (including the 2021-04-25 outage).
