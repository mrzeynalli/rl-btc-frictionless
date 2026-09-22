"""OHLCV-only BitMEX import: resume existing bars; never retain raw ticks.

The only retained market-data file is XBTUSD-1m-OHLCV.parquet. Full daily
source archives exist only temporarily while their candles are constructed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from collector import bitmex as source

COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
FILENAME = "XBTUSD-1m-OHLCV.parquet"
REPORT = source.ROOT / "results/bitmex_data_import/ohlcv_cleanup.json"
PRUNED_DATA_NAMES = ("trades", "reference", "download_cache", "provenance/rest_trade_pages")


def compact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={"volume_contracts": "volume"}).loc[:, COLUMNS].copy()
    # Pandas/Arrow may infer ns for older shards and us for newer ISO inputs.
    # One explicit storage unit keeps every Parquet row group schema identical.
    frame["open_time"] = pd.to_datetime(frame.open_time, utc=True).astype("datetime64[us, UTC]")
    time_index = pd.DatetimeIndex(frame.open_time)
    if time_index.has_duplicates or not time_index.is_monotonic_increasing:
        raise ValueError("Duplicate or unordered OHLCV timestamps")
    if not (time_index == time_index.floor("min")).all():
        raise ValueError("Off-grid OHLCV timestamps")
    if not np.isfinite(frame[COLUMNS[1:]]).all().all():
        raise ValueError("Nonfinite OHLCV")
    if not ((frame.low > 0) & (frame.low <= frame.open) & (frame.open <= frame.high)
            & (frame.low <= frame.close) & (frame.close <= frame.high) & (frame.volume > 0)).all():
        raise ValueError("Invalid OHLCV values")
    if not np.equal(frame.volume, np.floor(frame.volume)).all():
        raise ValueError("Fractional contract volume")
    return frame


def validate_day(record: dict, output: Path) -> pd.DataFrame | None:
    name = f"bars_1m/{record['day'].replace('-', '')}.parquet"
    if record["bars"] == 0:
        if record["regular_inverse_trades"] != 0:
            raise ValueError("Trades without corresponding bars")
        return None
    if name not in record["outputs"] or source.sha256(output / name) != record["outputs"][name]:
        raise ValueError(f"Daily bar checksum mismatch: {name}")
    frame = pd.read_parquet(output / name)
    if len(frame) != record["bars"] or int(frame.trades.sum()) != record["regular_inverse_trades"]:
        raise ValueError(f"Daily counts mismatch: {name}")
    if not np.array_equal(frame.volume_contracts, frame.taker_buy_contracts + frame.taker_sell_contracts):
        raise ValueError(f"Volume conservation failed: {name}")
    if not ((frame.first_trade_time >= frame.open_time) & (frame.last_trade_time < frame.close_time)).all():
        raise ValueError(f"Noncausal trade aggregation: {name}")
    day = pd.Timestamp(record["day"], tz="UTC")
    if not ((frame.open_time >= max(day, source.LAUNCH)) &
            (frame.open_time < min(day + pd.Timedelta(days=1), source.END))).all():
        raise ValueError(f"Date-boundary failure: {name}")
    return compact_frame(frame)


def remove_exact(target: Path, allowed: list[Path], report_path: Path = REPORT) -> int:
    """Permanent removal, never trash; all targets explicitly allow-listed."""
    if target not in allowed or target.is_symlink() or target.resolve() != target:
        raise ValueError(f"Unsafe cleanup target: {target}")
    if not target.exists():
        return 0
    paths = list(target.rglob("*")) if target.is_dir() else [target]
    if any(path.is_symlink() for path in paths):
        raise ValueError(f"Unexpected symlink inside cleanup target: {target}")
    size = sum(path.stat().st_size for path in paths if path.is_file())
    report = json.loads(report_path.read_text()) if report_path.exists() else {"removed": []}
    record = {"path": str(target), "bytes": size, "files": sum(path.is_file() for path in paths),
              "started_at": source.now(), "method": "permanent deletion; not moved to trash"}
    report["removed"].append(record)
    source.write_json(report_path, report)
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    record["completed_at"] = source.now()
    source.write_json(report_path, report)
    print(f"Removed {target}: {size / 1e9:.3f} GB", flush=True)
    return size


def prune_extras(output: Path, final: bool = False) -> None:
    if output != source.DEFAULT_OUTPUT.resolve():
        raise ValueError("Cleanup is restricted to the production dataset")
    allowed = [output / name for name in PRUNED_DATA_NAMES]
    allowed.append(source.ROOT / "results/bitmex_data_import/smoke")
    if final:
        allowed.extend([output / "bars_1m", output / "provenance"])
    for target in allowed:
        remove_exact(target, allowed)


def consolidate(output: Path, records: list[dict]) -> dict:
    target = output / FILENAME
    temporary = target.with_suffix(".parquet.tmp")
    writer = None
    times = []
    buffer = []
    buffered_rows = 0
    rows = 0
    yearly = {}
    try:
        for record in sorted(records, key=lambda value: value["day"]):
            frame = validate_day(record, output)
            if frame is None:
                continue
            times.append(pd.DatetimeIndex(frame.open_time))
            rows += len(frame)
            year = record["day"][:4]
            yearly[year] = yearly.get(year, 0) + len(frame)
            buffer.append(frame)
            buffered_rows += len(frame)
            if buffered_rows >= 130_000:
                table = pa.Table.from_pandas(pd.concat(buffer, ignore_index=True), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd",
                                             compression_level=9, use_dictionary=False,
                                             use_byte_stream_split=True)
                writer.write_table(table, row_group_size=131_072)
                buffer, buffered_rows = [], 0
        if buffer:
            table = pa.Table.from_pandas(pd.concat(buffer, ignore_index=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd",
                                         compression_level=9, use_dictionary=False,
                                         use_byte_stream_split=True)
            writer.write_table(table, row_group_size=131_072)
    finally:
        if writer is not None:
            writer.close()
    index = times[0].append(times[1:])
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Consolidated date-order/duplicate failure")
    expected = pd.date_range(source.LAUNCH, source.END, inclusive="left", freq="min")
    if not index.isin(expected).all():
        raise ValueError("Consolidated range mismatch")
    stored = pq.ParquetFile(temporary)
    if stored.metadata.num_rows != rows or stored.schema_arrow.names != COLUMNS:
        raise ValueError("Stored Parquet schema/count mismatch")
    temporary.replace(target)
    manifest = {
        "status": "compacted", "audit_passed": True, "completed_at_utc": source.now(),
        "exchange": "BitMEX", "symbol": "XBTUSD", "instrument": "inverse perpetual",
        "interval": "1m", "columns": COLUMNS, "format": "Parquet", "compression": "Zstandard level 9",
        "requested_start_inclusive_utc": str(source.START), "requested_end_exclusive_utc": str(source.END),
        "contract_launch_utc": str(source.LAUNCH), "first_open_time_utc": str(index[0]),
        "last_open_time_utc": str(index[-1]), "rows": rows, "rows_by_year": yearly,
        "expected_minutes_since_launch": len(expected), "minutes_without_regular_trades": len(expected) - rows,
        "duplicate_timestamps": 0, "verified_source_days": len(records),
        "missing_daily_sources": [], "rest_recovered_archive_days": [r["day"] for r in records if r.get("source") == "public_rest_trade_fallback"],
        "volume_units": "USD 1 inverse contracts (not BTC units)",
        "timestamp_policy": "UTC minute start; OHLCV becomes observable one minute later. Open/close are first/last regular trades in [open_time,open_time+1min). No forward-fill or invented candles for minutes without trades.",
        "prelaunch_policy": "Same-symbol legacy records before 2016-05-13 12:00 UTC excluded.",
        "trade_filter": "Regular and UnspecifiedLegacy only; SpreadLeg and other special prints excluded from both prices and volume. This differs from BitMEX bucketed volume, which can include spread legs.",
        "equal_timestamp_policy": "First/last follow stable source archive row order for tied trade timestamps; these can differ from REST bucket close at ties. REST bucket open is the previous close, not the first actual trade used here.",
        "reused_match_identifier_days": {r["day"]: r["reused_match_identifiers"] for r in records if r.get("reused_match_identifiers")},
        "repeated_regular_record_days": {r["day"]: r["repeated_regular_records"] for r in records if r.get("repeated_regular_records")},
        "deduplication_policy": "No trade-row deduplication: identical published fields can be distinct fills. Preserving July 15 2026 repeated records matches native exchange bucket trade counts/volume; deduplication does not. Final minute timestamps remain unique.",
        "file": FILENAME, "file_bytes": target.stat().st_size, "sha256": source.sha256(target),
        "retained_market_data": "OHLCV only; no raw trades, funding, reference/index data, or duplicate daily shards",
        "source_archive": source.ARCHIVE, "agent_changed": False}
    source.write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prune-extras", action="store_true")
    args = parser.parse_args()
    output = source.DEFAULT_OUTPUT.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "import.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source.STOP.clear()
    started = time.monotonic()
    try:
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get("file") == FILENAME and manifest.get("audit_passed"):
            if source.sha256(output / FILENAME) != manifest["sha256"]:
                raise ValueError("Existing completed OHLCV checksum mismatch")
        else:
            files = source.catalog(output, source.START, source.END)
            records, pending = [], []
            for item in files:
                path = output / f"provenance/days/{item['day'].replace('-', '')}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    validate_day(record, output)
                    records.append(record)
                else:
                    pending.append(item)
            print(f"Verified {len(records)}/{len(files)} existing days; {len(pending)} remaining", flush=True)
            if args.prune_extras:
                prune_extras(output)
            source.write_json(output / "status.json", {"status": "ohlcv_download", "completed_days": len(records),
                                                       "total_days": len(files), "updated_at": source.now()})
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(source.collect_day, item, output, source.START, source.END, keep_raw=False): item for item in pending}
                try:
                    for future in as_completed(futures):
                        record = future.result()
                        validate_day(record, output)
                        records.append(record)
                        elapsed = time.monotonic() - started
                        source.write_json(output / "status.json", {"status": "ohlcv_download", "completed_days": len(records),
                            "total_days": len(files), "last_day": record["day"], "elapsed_seconds": elapsed, "updated_at": source.now()})
                        if len(records) % 50 == 0 or len(records) == len(files):
                            print(f"OHLCV {len(records)}/{len(files)}: {record['day']} ({elapsed/60:.1f} min)", flush=True)
                except Exception:
                    source.STOP.set()
                    for future in futures:
                        future.cancel()
                    raise
            if len(records) != len(files):
                raise ValueError("Incomplete daily-source coverage")
            source.write_json(output / "status.json", {"status": "consolidating", "completed_days": len(records), "total_days": len(files)})
            manifest = consolidate(output, records)
        if args.prune_extras:
            prune_extras(output, final=True)
        manifest["status"] = "complete"
        manifest["cleanup_completed"] = args.prune_extras
        source.write_json(manifest_path, manifest)
        source.write_json(output / "status.json", {"status": "complete", "rows": manifest["rows"],
            "file_bytes": manifest["file_bytes"], "cleanup_completed": args.prune_extras, "updated_at": source.now()})
        print(f"COMPLETE: {manifest['rows']:,} OHLCV rows; {manifest['file_bytes']/1e6:.1f} MB", flush=True)
    except Exception as error:
        source.write_json(output / "status.json", {"status": "failed", "error": repr(error), "updated_at": source.now()})
        raise


if __name__ == "__main__":
    main()
