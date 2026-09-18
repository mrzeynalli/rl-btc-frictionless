"""Binance USD-M BTCUSDT perpetual: one-minute klines and funding, with an audit.

Sources, in order of preference:
- data.binance.vision monthly archives (with the published SHA-256 of every zip);
- daily archives for the current, unfinished month;
- the public REST API (fapi/v1/klines, fapi/v1/fundingRate) for the months the
  archive does not cover: the contract launched on 2019-09-08, the monthly kline
  archive starts in 2020-01.

The output mirrors the BitMEX dataset: one kline parquet with UTC minute starts and
no forward fill, one funding parquet, and a manifest with row counts, gaps and the
SHA-256 of both files. A minute whose kline reports no trades carries no prices,
the policy the BitMEX collector applies to minutes without regular trades.

    PYTHONPATH=src .venv/bin/python -m collector.binance_um --output data/binance_um_btcusdt_1m
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = 'BTCUSDT'
LAUNCH = pd.Timestamp('2019-09-08', tz='UTC')
ARCHIVE = 'https://data.binance.vision/data/futures/um'
REST = 'https://fapi.binance.com/fapi/v1'
KLINE_COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume',
                 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']


def fetch(url, attempts=6):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)


def archive_csv(path):
    """One archive zip, verified against its published SHA-256; None when it does not exist."""
    payload = fetch(f'{ARCHIVE}/{path}')
    if payload is None:
        return None
    published = fetch(f'{ARCHIVE}/{path}.CHECKSUM').decode().split()[0]
    if hashlib.sha256(payload).hexdigest() != published:
        raise ValueError(f'checksum mismatch for {path}')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        raw = archive.read(archive.namelist()[0])
    first = raw.split(b'\n', 1)[0]
    # Older files have no header row; newer ones do.
    header = 0 if first[:1].isalpha() else None
    return pd.read_csv(io.BytesIO(raw), header=header)


def as_klines(frame):
    frame = frame.iloc[:, :12].copy()
    frame.columns = KLINE_COLUMNS
    stamp = frame.open_time.astype('int64')
    # Guard against a switch to microsecond timestamps.
    stamp = np.where(stamp > 10 ** 14, stamp // 1000, stamp)
    out = pd.DataFrame({'open_time': pd.to_datetime(stamp, unit='ms', utc=True)})
    for name in ('open', 'high', 'low', 'close', 'volume', 'quote_volume'):
        out[name] = frame[name].astype('float64')
    out['trades'] = frame['count'].astype('int64')
    return out


def rest_klines(start, end):
    rows, cursor = [], int(start.value // 10 ** 6)
    stop = int(end.value // 10 ** 6)
    while cursor < stop:
        payload = json.loads(fetch(f'{REST}/klines?symbol={SYMBOL}&interval=1m&startTime={cursor}'
                                   f'&endTime={stop - 1}&limit=1500'))
        if not payload:
            break
        rows.extend(payload)
        cursor = payload[-1][0] + 60_000
        time.sleep(0.25)
    return as_klines(pd.DataFrame(rows)) if rows else None


def rest_funding(start, end):
    rows, cursor = [], int(start.value // 10 ** 6)
    stop = int(end.value // 10 ** 6)
    while cursor < stop:
        payload = json.loads(fetch(f'{REST}/fundingRate?symbol={SYMBOL}&startTime={cursor}'
                                   f'&endTime={stop - 1}&limit=1000'))
        if not payload:
            break
        rows.extend(payload)
        cursor = payload[-1]['fundingTime'] + 1
        time.sleep(0.25)
    frame = pd.DataFrame(rows)
    return pd.DataFrame({'timestamp': pd.to_datetime(frame.fundingTime.astype('int64'), unit='ms', utc=True),
                         'fundingRate': frame.fundingRate.astype('float64')})


def months(start, end):
    return pd.period_range(start.tz_localize(None), end.tz_localize(None), freq='M')


def collect(output, end):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    first_month = pd.Timestamp('2020-01-01', tz='UTC')
    parts, sources = [rest_klines(LAUNCH, first_month)], {'rest_klines': [str(LAUNCH), str(first_month)]}
    archived, daily = [], []
    current = pd.Timestamp(end.year, end.month, 1, tz='UTC')
    for month in months(first_month, current - pd.Timedelta(days=1)):
        frame = archive_csv(f'monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{month}.zip')
        if frame is None:
            raise ValueError(f'monthly kline archive missing for {month}')
        parts.append(as_klines(frame))
        archived.append(str(month))
        print('klines', month, len(frame), flush=True)
    for day in pd.date_range(current, end - pd.Timedelta(days=1), freq='D'):
        frame = archive_csv(f'daily/klines/{SYMBOL}/1m/{SYMBOL}-1m-{day.date()}.zip')
        if frame is None:
            break
        parts.append(as_klines(frame))
        daily.append(str(day.date()))
    sources.update({'monthly_kline_archives': archived, 'daily_kline_archives': daily})
    bars = pd.concat(parts, ignore_index=True).drop_duplicates('open_time').sort_values('open_time')
    bars = bars[bars.open_time < end].reset_index(drop=True)
    # A minute with no trades carries no prices (the BitMEX dataset's policy).
    silent = bars.trades == 0
    bars.loc[silent, ['open', 'high', 'low', 'close']] = np.nan

    funding_parts = [rest_funding(LAUNCH, first_month)]
    for month in months(first_month, current - pd.Timedelta(days=1)):
        frame = archive_csv(f'monthly/fundingRate/{SYMBOL}/{SYMBOL}-fundingRate-{month}.zip')
        if frame is None:
            raise ValueError(f'monthly funding archive missing for {month}')
        frame = frame.iloc[:, [0, 2]]
        frame.columns = ['calc_time', 'rate']
        funding_parts.append(pd.DataFrame({
            'timestamp': pd.to_datetime(frame.calc_time.astype('int64'), unit='ms', utc=True),
            'fundingRate': frame.rate.astype('float64')}))
    funding_parts.append(rest_funding(current, end))
    funding = pd.concat(funding_parts, ignore_index=True)
    # Settlement stamps can carry a few milliseconds; settlements are on the minute.
    funding['timestamp'] = funding.timestamp.dt.floor('min')
    funding = funding.drop_duplicates('timestamp').sort_values('timestamp')
    funding = funding[funding.timestamp < end].reset_index(drop=True)

    kline_path, funding_path = output / f'{SYMBOL}-1m-OHLCV.parquet', output / f'{SYMBOL}-funding.parquet'
    bars.to_parquet(kline_path, index=False, compression='zstd')
    funding.to_parquet(funding_path, index=False, compression='zstd')
    grid = pd.date_range(bars.open_time.iloc[0], end, freq='min', inclusive='left')
    spacing = funding.timestamp.diff().dt.total_seconds().div(3600).round(2).value_counts()
    manifest = {
        'exchange': 'Binance USD-M futures', 'symbol': SYMBOL, 'instrument': 'linear perpetual, USDT-margined',
        'interval': '1m', 'first_open_time_utc': str(bars.open_time.iloc[0]),
        'last_open_time_utc': str(bars.open_time.iloc[-1]), 'end_exclusive_utc': str(end),
        'rows': int(len(bars)), 'expected_minutes': int(len(grid)),
        'missing_minutes': int(len(grid) - len(bars)), 'minutes_without_trades': int(silent.sum()),
        'duplicate_timestamps': 0, 'volume_units': 'volume in BTC; quote_volume in USDT',
        'timestamp_policy': 'UTC minute start; no forward fill; minutes with zero trades carry no prices',
        'sources': sources,
        'funding': {'rows': int(len(funding)), 'first': str(funding.timestamp.iloc[0]),
                    'last': str(funding.timestamp.iloc[-1]),
                    'spacing_hours_counts': {str(k): int(v) for k, v in spacing.items()},
                    'sha256': hashlib.sha256(funding_path.read_bytes()).hexdigest()},
        'sha256': hashlib.sha256(kline_path.read_bytes()).hexdigest(),
        'collected_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'sources'}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/binance_um_btcusdt_1m'))
    parser.add_argument('--end', default=None, help='exclusive UTC end; default the start of today')
    args = parser.parse_args()
    end = pd.Timestamp(args.end, tz='UTC') if args.end else pd.Timestamp.now(tz='UTC').floor('D')
    collect(args.output, end)


if __name__ == '__main__':
    main()
