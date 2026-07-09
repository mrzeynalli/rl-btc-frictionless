"""Collect 1-minute Binance klines into one time-ordered parquet per symbol.

Usage:
    python -m cryptorl.data.collect                # full configured range
    python -m cryptorl.data.collect --symbols BTCUSDT --start 2021-01 --end 2021-03
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from dotenv import load_dotenv

from .binance import fetch_month
from .config import (
    END_MONTH,
    INTERVAL,
    PROJECT_ROOT,
    START_MONTH,
    SYMBOLS,
    month_range,
    parquet_path,
)

log = logging.getLogger("collect")


def collect_symbol(
    session: requests.Session,
    symbol: str,
    months: list[tuple[int, int]],
    api_key: str | None,
    force: bool,
    workers: int,
) -> dict:
    """Download every month for one symbol and write a single sorted parquet."""
    path = parquet_path(symbol)
    if path.exists() and not force:
        return {"symbol": symbol, "status": "skipped"}

    frames: list[pd.DataFrame] = []
    sources: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_month, session, symbol, INTERVAL, y, m, api_key): (y, m)
            for y, m in months
        }
        for future in as_completed(futures):
            year, month = futures[future]
            df, source = future.result()  # any failure aborts the symbol
            frames.append(df)
            sources.add(source)
            log.info("%s %d-%02d: %d rows via %s", symbol, year, month, len(df), source)

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, engine="pyarrow", index=False)
    return {
        "symbol": symbol,
        "status": "ok",
        "rows": len(merged),
        "sources": sorted(sources),
        "first": merged["open_time"].iloc[0],
        "last": merged["open_time"].iloc[-1],
    }


def parse_month(value: str) -> tuple[int, int]:
    year, month = value.split("-")
    return int(year), int(month)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start", type=parse_month, default=START_MONTH, metavar="YYYY-MM")
    parser.add_argument("--end", type=parse_month, default=END_MONTH, metavar="YYYY-MM")
    parser.add_argument("--workers", type=int, default=6, help="parallel month downloads")
    parser.add_argument("--force", action="store_true", help="rebuild existing files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("BINANCE_API_KEY")

    months = list(month_range(args.start, args.end))
    log.info("collecting %d months x %d symbols", len(months), len(args.symbols))

    session = requests.Session()
    written, skipped, failed = 0, 0, 0
    for symbol in args.symbols:
        try:
            res = collect_symbol(session, symbol, months, api_key, args.force, args.workers)
        except Exception as exc:
            log.error("%s FAILED: %s", symbol, exc)
            failed += 1
            continue
        if res["status"] == "skipped":
            log.info("%s: already exists, skipped (use --force to rebuild)", symbol)
            skipped += 1
        else:
            log.info(
                "%s: wrote %d rows (%s .. %s)",
                symbol, res["rows"], res["first"], res["last"],
            )
            written += 1

    log.info("finished: %d written, %d skipped, %d failed", written, skipped, failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
