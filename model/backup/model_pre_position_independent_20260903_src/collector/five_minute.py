"""Download and audit the fixed BTCUSDT five-minute study dataset.

Only Binance USD-M perpetual-futures data are used.  Contract klines and mark
price klines are native 5-minute archives; they are never resampled from the
old 1-minute collection.  Funding settlement timestamps retain the exact
recorded millisecond latency from Binance Vision.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .binance import _epoch_to_utc, _read_csv_zip
from .config import FUNDING_RAW_COLUMNS, RAW_COLUMNS


SYMBOL = "BTCUSDT"
INTERVAL = "5m"
WARMUP_START = pd.Timestamp("2025-05-20 00:00:00", tz="UTC")
STUDY_START = pd.Timestamp("2025-06-01 00:00:00", tz="UTC")
STUDY_END = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "BTCUSDT_5M_2025_2026"

URLS = {
    "contract": (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
    ),
    "mark": (
        "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/"
        "{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
    ),
    "funding": (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        "{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
    ),
}


def _months() -> list[tuple[int, int]]:
    output, current = [], WARMUP_START.normalize().replace(day=1)
    last = (STUDY_END - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1)
    while current <= last:
        output.append((current.year, current.month))
        current += pd.offsets.MonthBegin(1)
    return output


def _download(session: requests.Session, kind: str, year: int, month: int) -> pd.DataFrame:
    url = URLS[kind].format(
        symbol=SYMBOL, interval=INTERVAL, year=year, month=month
    )
    response = session.get(url, timeout=120)
    response.raise_for_status()
    columns = FUNDING_RAW_COLUMNS if kind == "funding" else RAW_COLUMNS
    try:
        return _read_csv_zip(response.content, columns)
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"invalid Binance archive: {url}") from error


def _contract_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["open_time"] = _epoch_to_utc(frame["open_time"])
    frame["close_time"] = _epoch_to_utc(frame["close_time"])
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = frame[name].astype("float64")
    return frame[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


def _mark_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _contract_frame(raw)
    return frame.rename(
        columns={
            "open": "mark_open", "high": "mark_high", "low": "mark_low",
            "close": "mark_close", "volume": "mark_volume",
        }
    )


def _funding_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "funding_time": _epoch_to_utc(raw["calc_time"]),
            "funding_rate": raw["last_funding_rate"].astype("float64"),
            "funding_interval_hours": raw["funding_interval_hours"].astype("int64"),
        }
    )
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit(contract: pd.DataFrame, mark: pd.DataFrame, funding: pd.DataFrame) -> dict:
    expected_rows = int((STUDY_END - WARMUP_START) / pd.Timedelta(minutes=5))
    expected_grid = pd.date_range(
        WARMUP_START, STUDY_END, freq="5min", inclusive="left"
    )
    missing = expected_grid.difference(contract.open_time)
    extra = pd.DatetimeIndex(contract.open_time).difference(expected_grid)
    if len(contract) != expected_rows or len(missing) or len(extra):
        raise ValueError(
            f"contract grid failed: rows={len(contract)} expected={expected_rows} "
            f"missing={len(missing)} extra={len(extra)}"
        )
    if not contract.open_time.equals(mark.open_time):
        raise ValueError("contract and mark-price grids differ")
    if contract.open_time.duplicated().any() or not contract.open_time.is_monotonic_increasing:
        raise ValueError("contract timestamps are duplicated or unordered")
    if funding.funding_time.duplicated().any() or not funding.funding_time.is_monotonic_increasing:
        raise ValueError("funding timestamps are duplicated or unordered")
    if not (
        (contract[["open", "high", "low", "close"]] > 0).all().all()
        and (contract.high >= contract[["open", "close"]].max(axis=1)).all()
        and (contract.low <= contract[["open", "close"]].min(axis=1)).all()
    ):
        raise ValueError("invalid OHLC relationship")

    # Verify every recorded settlement lies in a unique native 5-minute bar.
    event_bar = funding.funding_time.dt.floor("5min")
    unmatched = event_bar[~event_bar.isin(contract.open_time)]
    if len(unmatched):
        raise ValueError(f"{len(unmatched)} funding timestamps do not map to a 5-minute bar")
    nominal = funding.funding_time.dt.floor("1h")
    latency_ms = (funding.funding_time - nominal).dt.total_seconds() * 1000.0
    if not funding.funding_time.dt.hour.isin([0, 8, 16]).all():
        raise ValueError("unexpected BTC funding settlement hour")
    if not funding.funding_interval_hours.eq(8).all():
        raise ValueError("unexpected BTC funding interval in the fixed study")
    if not latency_ms.between(0, 5000).all():
        raise ValueError("funding settlement latency exceeds five seconds")

    study_rows = int(((contract.open_time >= STUDY_START) & (contract.open_time < STUDY_END)).sum())
    warmup_rows = int((contract.open_time < STUDY_START).sum())
    return {
        "source": "Binance Vision USD-M perpetual futures monthly archives",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "contract_bars_are_native_not_resampled": True,
        "warmup_start_utc": WARMUP_START.isoformat(),
        "study_start_inclusive_utc": STUDY_START.isoformat(),
        "study_end_exclusive_utc": STUDY_END.isoformat(),
        "warmup_rows": warmup_rows,
        "warmup_days": warmup_rows / 288.0,
        "study_rows": study_rows,
        "contract_rows_total": len(contract),
        "mark_price_rows_total": len(mark),
        "missing_contract_bars": len(missing),
        "funding_settlements": len(funding),
        "funding_first_recorded_utc": funding.funding_time.iloc[0].isoformat(),
        "funding_last_recorded_utc": funding.funding_time.iloc[-1].isoformat(),
        "funding_interval_hours_unique": sorted(funding.funding_interval_hours.unique().tolist()),
        "funding_recorded_latency_ms": {
            "minimum": float(latency_ms.min()),
            "median": float(latency_ms.median()),
            "maximum": float(latency_ms.max()),
        },
        "funding_timestamps_preserved_without_rounding": True,
        "audit_passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def collect(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in URLS}
    session = requests.Session()
    jobs = [(kind, year, month) for kind in URLS for year, month in _months()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_download, session, kind, year, month): (kind, year, month)
            for kind, year, month in jobs
        }
        for future in as_completed(futures):
            kind, year, month = futures[future]
            frames[kind].append(future.result())
            print(f"downloaded {kind} {year}-{month:02d}", flush=True)

    contract = _contract_frame(pd.concat(frames["contract"], ignore_index=True))
    mark = _mark_frame(pd.concat(frames["mark"], ignore_index=True))
    funding = _funding_frame(pd.concat(frames["funding"], ignore_index=True))
    contract = contract.drop_duplicates("open_time").sort_values("open_time")
    mark = mark.drop_duplicates("open_time").sort_values("open_time")
    funding = funding.drop_duplicates("funding_time").sort_values("funding_time")
    contract = contract[(contract.open_time >= WARMUP_START) & (contract.open_time < STUDY_END)].reset_index(drop=True)
    mark = mark[(mark.open_time >= WARMUP_START) & (mark.open_time < STUDY_END)].reset_index(drop=True)
    # Retain one pre-warmup funding settlement so the first warm-up observation
    # also has a genuinely last-known rate.
    funding = funding[funding.funding_time < STUDY_END].reset_index(drop=True)
    audit = _audit(contract, mark, funding[funding.funding_time >= WARMUP_START].reset_index(drop=True))

    paths = {
        "contract": output_dir / "BTCUSDT-5m-contract.parquet",
        "mark": output_dir / "BTCUSDT-5m-mark-price.parquet",
        "funding": output_dir / "BTCUSDT-funding-settlements.parquet",
    }
    contract.to_parquet(paths["contract"], index=False)
    mark.to_parquet(paths["mark"], index=False)
    funding.to_parquet(paths["funding"], index=False)
    audit["files"] = {
        key: {"path": str(path), "sha256": _sha256(path)}
        for key, path in paths.items()
    }
    (output_dir / "manifest.json").write_text(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(collect(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
