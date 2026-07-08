"""Collect 1-minute Binance klines and store them as monthly parquet files.

Usage:
    python -m cryptorl.data.collect                # full configured range
    python -m cryptorl.data.collect --symbols BTCUSDT --start 2021-01 --end 2021-03
"""

from __future__ import annotations

import argparse
import calendar
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def collect_one(
    session: requests.Session,
    symbol: str,
    year: int,
    month: int,
    api_key: str | None,
    force: bool,
) -> dict:
    path = parquet_path(symbol, year, month)
    if path.exists() and not force:
        return {"symbol": symbol, "year": year, "month": month, "status": "skipped"}

    df, source = fetch_month(session, symbol, INTERVAL, year, month, api_key)
    expected = calendar.monthrange(year, month)[1] * 1440
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return {
        "symbol": symbol,
        "year": year,
        "month": month,
        "status": "ok",
        "source": source,
        "rows": len(df),
        "expected": expected,
    }


def parse_month(value: str) -> tuple[int, int]:
    year, month = value.split("-")
    return int(year), int(month)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start", type=parse_month, default=START_MONTH, metavar="YYYY-MM")
    parser.add_argument("--end", type=parse_month, default=END_MONTH, metavar="YYYY-MM")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("BINANCE_API_KEY")

    tasks = [
        (symbol, year, month)
        for symbol in args.symbols
        for year, month in month_range(args.start, args.end)
    ]
    log.info("collecting %d symbol-months with %d workers", len(tasks), args.workers)

    session = requests.Session()
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(collect_one, session, s, y, m, api_key, args.force): (s, y, m)
            for s, y, m in tasks
        }
        for future in as_completed(futures):
            symbol, year, month = futures[future]
            try:
                res = future.result()
            except Exception as exc:
                log.error("%s %d-%02d FAILED: %s", symbol, year, month, exc)
                failures.append((symbol, year, month, str(exc)))
                continue
            results.append(res)
            if res["status"] == "ok":
                gap = res["expected"] - res["rows"]
                log.info(
                    "%s %d-%02d: %d rows via %s%s",
                    symbol, year, month, res["rows"], res["source"],
                    f" ({gap} minutes missing)" if gap else "",
                )
            else:
                log.info("%s %d-%02d: already exists, skipped", symbol, year, month)

    done = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    total_rows = sum(r.get("rows", 0) for r in results)
    log.info(
        "finished: %d written, %d skipped, %d failed, %d rows total",
        done, skipped, len(failures), total_rows,
    )
    if failures:
        for symbol, year, month, err in failures:
            log.error("failed: %s %d-%02d (%s)", symbol, year, month, err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
