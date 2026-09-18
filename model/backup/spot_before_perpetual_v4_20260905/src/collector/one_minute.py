"""Rebuild the fixed BTCUSDT Spot one-minute dataset from official archives."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import requests

from collector.binance import download_month
from collector.config import (
    END_EXCLUSIVE,
    INTERVAL,
    OUTPUT_COLUMNS,
    OUTPUT_DIR,
    REQUESTED_START,
    SYMBOL,
)


def _audit(frame: pd.DataFrame, unavailable: list[str]) -> dict:
    timestamps = pd.DatetimeIndex(frame.open_time)
    aligned = timestamps == timestamps.floor("1min")
    canonical = timestamps[aligned]
    expected = pd.date_range(canonical[0], canonical[-1], freq="1min")
    missing = expected.difference(canonical)
    return {
        "source": "Binance Vision official Spot archives",
        "market": f"{SYMBOL} Spot",
        "interval": INTERVAL,
        "requested_start_inclusive_utc": REQUESTED_START.isoformat(),
        "requested_end_exclusive_utc": END_EXCLUSIVE.isoformat(),
        "actual_start_inclusive_utc": timestamps[0].isoformat(),
        "actual_end_inclusive_utc": timestamps[-1].isoformat(),
        "rows": len(frame),
        "duplicate_open_times": int(timestamps.duplicated().sum()),
        "off_grid_rows": int((~aligned).sum()),
        "missing_standard_grid_bars": len(missing),
        "unavailable_month_archives": unavailable,
        "funding_data_present": False,
        "timeline_policy": (
            "Preserve official native timestamps. Feature construction skips "
            "windows crossing missing intervals and never forward-fills."
        ),
        "audit_passed": bool(
            len(frame) > 0
            and timestamps.is_monotonic_increasing
            and not timestamps.duplicated().any()
            and timestamps[-1] < END_EXCLUSIVE
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def collect(force: bool = False) -> None:
    output = OUTPUT_DIR / f"{SYMBOL}-{INTERVAL}-spot.parquet"
    manifest_path = OUTPUT_DIR / "manifest.json"
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to rebuild it")
    months = pd.period_range(
        REQUESTED_START.tz_localize(None).to_period("M"),
        (END_EXCLUSIVE - pd.Timedelta(days=1)).tz_localize(None).to_period("M"),
        freq="M",
    )
    frames: list[pd.DataFrame] = []
    unavailable: list[str] = []
    with requests.Session() as session:
        for period in months:
            frame = download_month(session, SYMBOL, INTERVAL, period.year, period.month)
            if frame is None:
                unavailable.append(str(period))
            else:
                frames.append(frame)
                print(f"downloaded {period}: {len(frame):,} rows", flush=True)
    if not frames:
        raise RuntimeError("no Binance Spot archives were available")
    data = pd.concat(frames, ignore_index=True)
    data = data[
        (data.open_time >= REQUESTED_START) & (data.open_time < END_EXCLUSIVE)
    ]
    data = data.sort_values("open_time").drop_duplicates("open_time", keep="last")
    data = data.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)
    manifest = _audit(data, unavailable)
    if not manifest["audit_passed"]:
        raise ValueError("Spot dataset failed ordering/coverage audit")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    collect(force=args.force)


if __name__ == "__main__":
    main()
