"""Binance USD-M perpetual futures downloader (klines + funding rates).

Primary source: monthly zip archives from data.binance.vision (Binance's
official bulk-data CDN, no rate limits). Fallback: the fapi REST
endpoints, throttled against the exchange request-weight limit, for
months not yet published as archives.
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

from .config import (
    FLOAT_COLUMNS,
    FUNDING_OUTPUT_COLUMNS,
    FUNDING_RAW_COLUMNS,
    MARKET,
    OUTPUT_COLUMNS,
    RAW_COLUMNS,
)

log = logging.getLogger(__name__)

BULK_KLINES_URL = (
    "https://data.binance.vision/data/futures/{market}/monthly/klines/"
    "{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
)
BULK_DAILY_KLINES_URL = (
    "https://data.binance.vision/data/futures/{market}/daily/klines/"
    "{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
)
BULK_FUNDING_URL = (
    "https://data.binance.vision/data/futures/{market}/monthly/fundingRate/"
    "{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
)
REST_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
REST_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

REST_LIMIT = 1000  # rows per REST request
# USD-M futures caps request weight at 2400/min; stay well below it.
REST_WEIGHT_SOFT_CAP = 1800

# The REST fallback is serialized across worker threads so the weight
# accounting stays correct.
_rest_lock = threading.Lock()


def month_bounds_ms(year: int, month: int) -> tuple[int, int]:
    """Return [start, end] of the month in epoch milliseconds, inclusive."""
    days = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    return start_ms, start_ms + days * 86_400_000 - 1


def _epoch_to_utc(series: pd.Series) -> pd.Series:
    """Convert epoch timestamps to UTC datetimes at microsecond precision.

    Binance Vision spot files switched from milliseconds to microseconds
    on 2025-01-01 while futures files stayed in milliseconds; the REST
    API returns milliseconds. Detect the unit from the magnitude (µs
    timestamps are ~1e15, ms are ~1e12) so every era lands on one schema.
    """
    if series.empty:
        return pd.Series(dtype="datetime64[us, UTC]")
    unit = "us" if int(series.iloc[0]) > 10**14 else "ms"
    return pd.to_datetime(series, unit=unit, utc=True).astype("datetime64[us, UTC]")


def _read_csv_zip(content: bytes, columns: list[str]) -> pd.DataFrame:
    """Read the single CSV inside a Binance archive, header optional.

    Older archives ship no header row; newer ones do.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as fh:
            first = fh.readline().decode()
        header = 0 if first.split(",")[0] == columns[0] else None
        with zf.open(member) as fh:
            return pd.read_csv(fh, header=header, names=columns)


def _get_bulk(
    session: requests.Session, url: str, columns: list[str], label: str, retries: int = 4
) -> pd.DataFrame | None:
    """Download and parse one monthly archive. None if not published (404)."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=120)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return _read_csv_zip(resp.content, columns)
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            log.warning("%s: attempt %d failed (%s), retrying in %ds", label, attempt, exc, wait)
            time.sleep(wait)
    return None


# --------------------------------------------------------------------------
# Klines
# --------------------------------------------------------------------------


def normalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Cast raw kline rows to the output schema."""
    df = raw.copy()
    df["open_time"] = _epoch_to_utc(df["open_time"])
    df["close_time"] = _epoch_to_utc(df["close_time"])
    for col in FLOAT_COLUMNS:
        df[col] = df[col].astype("float64")
    df["number_of_trades"] = df["number_of_trades"].astype("int64")
    df["symbol"] = symbol
    df = df[OUTPUT_COLUMNS]
    df = df.drop_duplicates(subset="open_time").sort_values("open_time")
    return df.reset_index(drop=True)


def fetch_range_rest(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Page a [start, end] window of klines through the REST API."""
    headers = {"X-MBX-APIKEY": api_key} if api_key else {}
    rows: list[list] = []
    cursor = start_ms
    with _rest_lock:
        while cursor <= end_ms:
            resp = session.get(
                REST_KLINES_URL,
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


def fetch_month_rest(
    session: requests.Session,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Page one month of klines through the REST API."""
    start_ms, end_ms = month_bounds_ms(year, month)
    return fetch_range_rest(session, symbol, interval, start_ms, end_ms, api_key)


def _incomplete_days(df: pd.DataFrame, year: int, month: int) -> list[pd.Timestamp]:
    """Days of the month holding fewer than 1440 minutes.

    Only days that have fully elapsed are reported, so an in-progress
    month is not flagged for its future days.
    """
    counts = df["open_time"].dt.floor("D").value_counts()
    now = pd.Timestamp.now(tz="UTC")
    days = calendar.monthrange(year, month)[1]
    out = []
    for day in range(1, days + 1):
        start = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
        if start + pd.Timedelta(days=1) > now:
            break
        if counts.get(start, 0) < 1440:
            out.append(start)
    return out


def _fill_day_gaps(
    session: requests.Session,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    df: pd.DataFrame,
    api_key: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Backfill days the monthly archive left short.

    Binance occasionally publishes truncated monthly archives (e.g.
    SOLUSDT 2022-02 stops at the 25th) while the daily archives for the
    same dates are complete. Try daily archives first, then REST.
    """
    gaps = _incomplete_days(df, year, month)
    if not gaps:
        return df, []

    extra, sources = [], set()
    for day in gaps:
        date = day.strftime("%Y-%m-%d")
        url = BULK_DAILY_KLINES_URL.format(
            market=MARKET, symbol=symbol, interval=interval, date=date
        )
        raw = _get_bulk(session, url, RAW_COLUMNS, f"{symbol} {date} klines")
        if raw is not None:
            sources.add("daily")
        else:
            start_ms = int(day.timestamp() * 1000)
            raw = fetch_range_rest(
                session, symbol, interval, start_ms, start_ms + 86_400_000 - 1, api_key
            )
            sources.add("rest")
        if len(raw):
            extra.append(normalize(raw, symbol))

    if not extra:
        log.warning("%s %d-%02d: %d day(s) short and unrecoverable", symbol, year, month, len(gaps))
        return df, []

    log.info(
        "%s %d-%02d: backfilled %d day(s) missing from the monthly archive",
        symbol, year, month, len(gaps),
    )
    merged = pd.concat([df, *extra], ignore_index=True)
    merged = merged.drop_duplicates(subset="open_time").sort_values("open_time")
    return merged.reset_index(drop=True), sorted(sources)


def fetch_month(
    session: requests.Session,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch one month of klines, preferring the bulk archive.

    Returns (normalized dataframe, source). Monthly archives that are
    published short get topped up from the daily archives or REST.
    """
    url = BULK_KLINES_URL.format(
        market=MARKET, symbol=symbol, interval=interval, year=year, month=month
    )
    raw = _get_bulk(session, url, RAW_COLUMNS, f"{symbol} {year}-{month:02d} klines")
    if raw is None:
        raw = fetch_month_rest(session, symbol, interval, year, month, api_key)
        return normalize(raw, symbol), "rest"

    df, filled = _fill_day_gaps(
        session, symbol, interval, year, month, normalize(raw, symbol), api_key
    )
    return df, "+".join(["bulk", *filled])


# --------------------------------------------------------------------------
# Funding rates
# --------------------------------------------------------------------------


def normalize_funding(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Cast raw funding rows while preserving recorded settlement times.

    Binance archive ``calc_time`` values can be a few milliseconds after the
    nominal boundary.  That latency matters for causal replay: a rate recorded
    just after 00:00 is not observable at 23:59:59.999.  Do not round it.
    """
    df = raw.copy()
    df["funding_time"] = _epoch_to_utc(df["calc_time"])
    df["funding_rate"] = df["last_funding_rate"].astype("float64")
    df["funding_interval_hours"] = df["funding_interval_hours"].astype("int64")
    df["symbol"] = symbol
    df = df[FUNDING_OUTPUT_COLUMNS]
    df = df.drop_duplicates(subset="funding_time").sort_values("funding_time")
    return df.reset_index(drop=True)


def fetch_funding_month_rest(
    session: requests.Session,
    symbol: str,
    year: int,
    month: int,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Page one month of funding rates through the REST API."""
    start_ms, end_ms = month_bounds_ms(year, month)
    headers = {"X-MBX-APIKEY": api_key} if api_key else {}
    rows: list[dict] = []
    cursor = start_ms
    with _rest_lock:
        while cursor <= end_ms:
            resp = session.get(
                REST_FUNDING_URL,
                params={
                    "symbol": symbol,
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
            cursor = int(batch[-1]["fundingTime"]) + 1
            used = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", 0))
            if used > REST_WEIGHT_SOFT_CAP:
                log.info("REST weight %d near cap, pausing 30s", used)
                time.sleep(30)

    if not rows:
        return pd.DataFrame(columns=FUNDING_RAW_COLUMNS)

    # The REST payload has no interval field; derive it from the spacing
    # between settlements (8h for most symbols, 4h for some since 2025).
    df = pd.DataFrame(rows)
    times = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    hours = times.diff().dt.total_seconds().div(3600).round()
    return pd.DataFrame(
        {
            "calc_time": df["fundingTime"].astype("int64"),
            "funding_interval_hours": hours.bfill().ffill().fillna(8).astype("int64"),
            "last_funding_rate": df["fundingRate"].astype("float64"),
        }
    )


def fetch_funding_month(
    session: requests.Session,
    symbol: str,
    year: int,
    month: int,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch one month of funding rates, preferring the bulk archive."""
    url = BULK_FUNDING_URL.format(market=MARKET, symbol=symbol, year=year, month=month)
    raw = _get_bulk(session, url, FUNDING_RAW_COLUMNS, f"{symbol} {year}-{month:02d} funding")
    if raw is not None:
        return normalize_funding(raw, symbol), "bulk"
    raw = fetch_funding_month_rest(session, symbol, year, month, api_key)
    return normalize_funding(raw, symbol), "rest"
