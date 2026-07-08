"""Binance 1-minute kline downloader.

Primary source: monthly zip archives from data.binance.vision (Binance's
official bulk-data CDN, no rate limits). Fallback: the REST
/api/v3/klines endpoint, throttled against the exchange request-weight
limit, for months not yet published as archives.
"""

from __future__ import annotations

import calendar
import io
import logging
import threading
import time
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import FLOAT_COLUMNS, OUTPUT_COLUMNS, RAW_COLUMNS

log = logging.getLogger(__name__)

BULK_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/"
    "{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
)
REST_URL = "https://api.binance.com/api/v3/klines"

REST_LIMIT = 1000  # klines per REST request
# Spot request-weight cap is 6000/min; stay well below it.
REST_WEIGHT_SOFT_CAP = 4500

# The REST fallback is serialized across worker threads so the weight
# accounting stays correct.
_rest_lock = threading.Lock()


def month_bounds_ms(year: int, month: int) -> tuple[int, int]:
    """Return [start, end] of the month in epoch milliseconds, inclusive."""
    days = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end_ms = int(start.timestamp() * 1000) + days * 86_400_000 - 1
    return int(start.timestamp() * 1000), end_ms


def _epoch_to_utc(series: pd.Series) -> pd.Series:
    """Convert epoch timestamps to UTC datetimes.

    Binance Vision files switched from milliseconds to microseconds on
    2025-01-01; the REST API returns milliseconds. Detect the unit from
    the magnitude (µs timestamps are ~1e15, ms are ~1e12).
    """
    unit = "us" if int(series.iloc[0]) > 10**14 else "ms"
    return pd.to_datetime(series, unit=unit, utc=True)


def normalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Cast raw kline rows to the output schema."""
    df = raw.copy()
    # Standardize on microsecond precision so every file shares one schema
    # regardless of the source era (ms before 2025, µs after).
    df["open_time"] = _epoch_to_utc(df["open_time"]).astype("datetime64[us, UTC]")
    df["close_time"] = _epoch_to_utc(df["close_time"]).astype("datetime64[us, UTC]")
    for col in FLOAT_COLUMNS:
        df[col] = df[col].astype("float64")
    df["number_of_trades"] = df["number_of_trades"].astype("int64")
    df["symbol"] = symbol
    df = df[OUTPUT_COLUMNS]
    df = df.drop_duplicates(subset="open_time").sort_values("open_time")
    return df.reset_index(drop=True)


def download_month_bulk(
    session: requests.Session,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    retries: int = 4,
) -> pd.DataFrame | None:
    """Download one monthly kline archive. Returns None if not published."""
    url = BULK_URL.format(symbol=symbol, interval=interval, year=year, month=month)
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=120)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                member = zf.namelist()[0]
                with zf.open(member) as fh:
                    first = fh.readline().decode()
                # Newer archives ship a header row, older ones do not.
                header = 0 if first.split(",")[0] == "open_time" else None
                with zf.open(member) as fh:
                    raw = pd.read_csv(fh, header=header, names=RAW_COLUMNS)
            return raw
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            log.warning(
                "%s %s-%02d: attempt %d failed (%s), retrying in %ds",
                symbol, year, month, attempt, exc, wait,
            )
            time.sleep(wait)
    return None


def fetch_month_rest(
    session: requests.Session,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Page one month of klines through the REST API, honoring rate limits."""
    start_ms, end_ms = month_bounds_ms(year, month)
    headers = {"X-MBX-APIKEY": api_key} if api_key else {}
    rows: list[list] = []
    cursor = start_ms
    with _rest_lock:
        while cursor <= end_ms:
            resp = session.get(
                REST_URL,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": REST_LIMIT,
                },
                headers=headers,
                timeout=30,
            )
            if resp.status_code in (429, 418):
                wait = int(resp.headers.get("Retry-After", 60))
                log.warning("REST rate limited (%d), sleeping %ds", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            rows.extend(batch)
            cursor = batch[-1][0] + 60_000  # next minute after last open_time
            used = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", 0))
            if used > REST_WEIGHT_SOFT_CAP:
                log.info("REST weight %d near cap, pausing 30s", used)
                time.sleep(30)
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def fetch_month(
    session: requests.Session,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch one month of klines, preferring the bulk archive.

    Returns (normalized dataframe, source) where source is "bulk" or "rest".
    """
    raw = download_month_bulk(session, symbol, interval, year, month)
    if raw is not None:
        return normalize(raw, symbol), "bulk"
    raw = fetch_month_rest(session, symbol, interval, year, month, api_key)
    return normalize(raw, symbol), "rest"
