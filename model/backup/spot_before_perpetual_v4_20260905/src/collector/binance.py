"""Official Binance Vision downloader for Spot klines only.

There is deliberately no futures endpoint and no funding-rate code in this
collector. A missing month before BTCUSDT was listed is recorded rather than
fabricated.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from collector.config import ARCHIVE_TEMPLATE, RAW_COLUMNS


def _epoch_to_utc(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype("int64")
    # Binance Spot archives moved from millisecond to microsecond timestamps.
    unit = "us" if int(numeric.abs().max()) >= 10**15 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True)


def download_month(
    session: requests.Session,
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> pd.DataFrame | None:
    url = ARCHIVE_TEMPLATE.format(
        symbol=symbol, interval=interval, year=year, month=month
    )
    response = session.get(url, timeout=90)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"unexpected archive layout at {url}")
        with archive.open(members[0]) as stream:
            raw = pd.read_csv(stream, header=None, names=RAW_COLUMNS)
    raw["open_time"] = _epoch_to_utc(raw["open_time"])
    raw["close_time"] = _epoch_to_utc(raw["close_time"])
    for name in (
        "open", "high", "low", "close", "volume", "quote_volume",
        "taker_buy_base", "taker_buy_quote",
    ):
        raw[name] = pd.to_numeric(raw[name], errors="raise").astype("float64")
    raw["trades"] = pd.to_numeric(raw["trades"], errors="raise").astype("int64")
    return raw.drop(columns="ignore")
