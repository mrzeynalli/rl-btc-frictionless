"""Collect Binance USD-M perpetual futures data into per-symbol parquet files.

Writes one time-ordered file per symbol:
    data/1MIN/<SYMBOL>.parquet      1-minute klines
    data/FUNDING/<SYMBOL>.parquet   funding rate settlements

Usage:
    python -m cryptorl.data.collect                      # klines + funding
    python -m cryptorl.data.collect --what funding
    python -m cryptorl.data.collect --symbols BTCUSDT --start 2021-01 --end 2021-03
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from dotenv import load_dotenv

from .binance import fetch_funding_month, fetch_month
from .config import (
    END_MONTH,
    INTERVAL,
    PROJECT_ROOT,
    START_MONTH,
    SYMBOLS,
    funding_path,
    month_range,
    parquet_path,
)

log = logging.getLogger("collect")


def _collect(
    symbol: str,
    months: list[tuple[int, int]],
    path: Path,
    fetcher: Callable[[int, int], tuple[pd.DataFrame, str]],
    time_column: str,
    force: bool,
    workers: int,
    kind: str,
) -> dict:
    """Download every month for one symbol and write a single sorted parquet."""
    if path.exists() and not force:
        return {"status": "skipped"}

    frames: list[pd.DataFrame] = []
    sources: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetcher, y, m): (y, m) for y, m in months}
        for future in as_completed(futures):
            year, month = futures[future]
            df, source = future.result()  # any failure aborts this symbol
            frames.append(df)
            sources.update(source.split("+"))  # a month may mix bulk/daily/rest
            log.debug("%s %s %d-%02d: %d rows via %s", symbol, kind, year, month, len(df), source)

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=time_column)
        .sort_values(time_column)
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, engine="pyarrow", index=False)
    return {
        "status": "ok",
        "rows": len(merged),
        "sources": sorted(sources),
        "first": merged[time_column].iloc[0],
        "last": merged[time_column].iloc[-1],
    }


def parse_month(value: str) -> tuple[int, int]:
    year, month = value.split("-")
    return int(year), int(month)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start", type=parse_month, default=START_MONTH, metavar="YYYY-MM")
    parser.add_argument("--end", type=parse_month, default=END_MONTH, metavar="YYYY-MM")
    parser.add_argument("--what", choices=["both", "klines", "funding"], default="both")
    parser.add_argument("--workers", type=int, default=6, help="parallel month downloads")
    parser.add_argument("--force", action="store_true", help="rebuild existing files")
    parser.add_argument("--verbose", action="store_true", help="log every month")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("BINANCE_API_KEY")

    months = list(month_range(args.start, args.end))
    kinds = ["klines", "funding"] if args.what == "both" else [args.what]
    log.info(
        "collecting %s for %d symbols x %d months (USD-M perpetual futures)",
        " + ".join(kinds), len(args.symbols), len(months),
    )

    session = requests.Session()
    written, skipped, failed = 0, 0, 0
    for symbol in args.symbols:
        for kind in kinds:
            if kind == "klines":
                path, time_column = parquet_path(symbol), "open_time"

                def fetcher(y, m, _s=symbol):
                    return fetch_month(session, _s, INTERVAL, y, m, api_key)
            else:
                path, time_column = funding_path(symbol), "funding_time"

                def fetcher(y, m, _s=symbol):
                    return fetch_funding_month(session, _s, y, m, api_key)

            try:
                res = _collect(
                    symbol, months, path, fetcher, time_column,
                    args.force, args.workers, kind,
                )
            except Exception as exc:
                log.error("%s %s FAILED: %s", symbol, kind, exc)
                failed += 1
                continue

            if res["status"] == "skipped":
                log.info("%s %s: exists, skipped (use --force to rebuild)", symbol, kind)
                skipped += 1
            else:
                log.info(
                    "%s %s: wrote %d rows (%s .. %s) via %s",
                    symbol, kind, res["rows"], res["first"], res["last"],
                    "+".join(res["sources"]),
                )
                written += 1

    log.info("finished: %d written, %d skipped, %d failed", written, skipped, failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
