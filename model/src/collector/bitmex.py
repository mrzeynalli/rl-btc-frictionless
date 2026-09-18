"""Resumable public XBTUSD archive import; independent of the Spot agent.

The CLI delegates to collector.bitmex_ohlcv (OHLCV-only storage).
Extended import functions are legacy helpers, not the active job.
No credentials. All dates and stored timestamps are UTC. See data README.
"""

from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import threading
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/bitmex_xbtusd_1m_2016_2026_08_01"
ARCHIVE = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/"
API = "https://www.bitmex.com/api/v1/"
UDF = "https://www.bitmex.com/api/udf/history"
LAUNCH = pd.Timestamp("2016-05-13T12:00:00Z")
START = pd.Timestamp("2016-05-01T00:00:00Z")
END = pd.Timestamp("2026-08-02T00:00:00Z")  # August 1 inclusive
INDEX_SWITCH = pd.Timestamp("2017-03-31T12:00:00Z")
FIELDS = ["timestamp", "symbol", "side", "size", "price", "tickDirection",
          "trdMatchID", "grossValue", "homeNotional", "foreignNotional", "trdType", "pool"]
SUM_FIELDS = ["volume_contracts", "volume_btc", "volume_usd", "trades",
              "taker_buy_contracts", "taker_sell_contracts", "taker_buy_btc",
              "taker_sell_btc", "turnover_satoshis"]
API_LOCK = threading.Lock()
LAST_API_REQUEST = 0.0
STOP = threading.Event()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    global LAST_API_REQUEST
    for attempt in range(8):
        if STOP.is_set():
            raise RuntimeError("Import cancelled after another task failed")
        try:
            if url.startswith(API) or url.startswith(UDF):
                with API_LOCK:
                    delay = max(0, LAST_API_REQUEST + 2.2 - time.monotonic())
                    time.sleep(delay)
                    LAST_API_REQUEST = time.monotonic()
            response = session.get(url, timeout=(20, 120), **kwargs)
            if response.status_code in (429, 500, 502, 503, 504):
                delay = max(float(response.headers.get("Retry-After", 0)),
                            min(45, 2 ** attempt))
                response.close()
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(min(45, 2 ** attempt))
    raise RuntimeError(f"Retries exhausted: {url}")


def with_missing_days(files: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    expected = set(pd.date_range(start.floor("D"), end, inclusive="left", freq="D").strftime("%Y-%m-%d"))
    missing = sorted(expected - {item["day"] for item in files})
    return sorted(files + [{"day": day, "key": None, "bytes": 0, "etag": None,
                            "source": "public_rest_trade_fallback"} for day in missing], key=lambda item: item["day"])


def catalog(output: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    path = output / "provenance/archive_catalog.json"
    if path.exists():
        value = json.loads(path.read_text())
        if value["start"] == start.isoformat() and value["end"] == end.isoformat():
            return with_missing_days(value["files"], start, end)
        raise ValueError("Output directory belongs to a different date range")
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    files = []
    with requests.Session() as session:
        for year in range(start.year, end.year + 1):
            root = ET.fromstring(get(session, ARCHIVE, params={
                "prefix": f"data/trade/{year}", "max-keys": 1000}).content)
            if root.findtext("s:IsTruncated", namespaces=ns) == "true":
                raise ValueError("Unexpected truncated annual archive catalog")
            for item in root.findall("s:Contents", ns):
                key = item.findtext("s:Key", namespaces=ns)
                if not key.endswith(".csv.gz"):
                    continue
                day = pd.to_datetime(Path(key).name[:8], format="%Y%m%d", utc=True)
                if start.floor("D") <= day < end:
                    files.append({"day": day.strftime("%Y-%m-%d"), "key": key,
                                  "bytes": int(item.findtext("s:Size", namespaces=ns)),
                                  "etag": item.findtext("s:ETag", namespaces=ns).strip('"')})
    files.sort(key=lambda item: item["day"])
    expected = set(pd.date_range(start.floor("D"), end, inclusive="left", freq="D").strftime("%Y-%m-%d"))
    missing = sorted(expected - {item["day"] for item in files})
    write_json(path, {"start": start.isoformat(), "end": end.isoformat(),
                      "fetched_at": now(), "files": files, "missing_days": missing,
                      "total_compressed_bytes": sum(item["bytes"] for item in files)})
    if missing:
        print(f"Missing archives will be recovered from public REST trades: {missing}", flush=True)
    return with_missing_days(files, start, end)


def rest_trade_archive(day: pd.Timestamp, output: Path, target: Path) -> None:
    """Recover missing daily archives; offsets preserve trades with tied timestamps."""
    frames = []
    with requests.Session() as session:
        offset = 0
        while True:
            path = output / f"provenance/rest_trade_pages/{day.strftime('%Y%m%d')}/{offset:09d}.json"
            if path.exists():
                page = json.loads(path.read_text())
            else:
                page = get(session, API + "trade", params={"symbol": "XBTUSD", "count": 1000,
                    "start": offset, "reverse": "false", "startTime": day.isoformat(),
                    "endTime": (day + pd.Timedelta(days=1)).isoformat()}).json()
                if not isinstance(page, list):
                    raise ValueError(f"Unexpected REST trade page: {page}")
                write_json(path, page)
            if page:
                frame = pd.DataFrame(page)
                timestamp = pd.to_datetime(frame.timestamp, format="ISO8601", utc=True)
                frame = frame.loc[(timestamp >= day) & (timestamp < day + pd.Timedelta(days=1))]
                frames.append(frame.loc[:, [name for name in FIELDS if name in frame]])
            if len(page) < 1000:
                break
            offset += len(page)
            if offset % 10000 == 0:
                print(f"REST recovery {day.date()}: {offset:,} trades", flush=True)
    if not frames:
        raise ValueError(f"No trades found for missing archive {day.date()}")
    frame = pd.concat(frames, ignore_index=True)
    temporary = target.with_suffix(".gz.part")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(target)
    print(f"REST recovered {day.date()}: {len(frame):,} trades", flush=True)


def normalize_trades(frame: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    frame = frame.loc[frame.symbol.eq("XBTUSD")].copy()
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(
        frame.timestamp.str.replace("D", "T", regex=False), format="ISO8601", utc=True)
    if not ((frame.timestamp >= day) & (frame.timestamp < day + pd.Timedelta(days=1))).all():
        raise ValueError("Trade outside its archive UTC day")
    inverse = frame.timestamp >= LAUNCH
    if not frame.loc[inverse, "side"].isin(["Buy", "Sell"]).all():
        raise ValueError("Unknown aggressor side")
    if not np.isfinite(frame.loc[inverse, ["size", "price", "grossValue", "homeNotional", "foreignNotional"]]).all().all():
        raise ValueError("Nonfinite trade fields")
    if (frame.loc[inverse, "price"] <= 0).any() or (frame.loc[inverse, "size"] <= 0).any():
        raise ValueError("Nonpositive price or trade size")
    if "trdType" not in frame:
        frame["trdType"] = "UnspecifiedLegacy"
    frame["trdType"] = frame.trdType.fillna("UnspecifiedLegacy")
    if not np.allclose(frame.loc[inverse, "foreignNotional"], frame.loc[inverse, "size"], rtol=1e-7, atol=1e-5):
        raise ValueError("Post-launch USD notional does not match $1 inverse contracts")
    frame["contract_regime"] = np.where(inverse, "inverse_perpetual", "pre_launch_legacy")
    return frame


def minute_parts(trades: pd.DataFrame) -> pd.DataFrame:
    """Real trade OHLC in [open_time, close_time); no synthetic empty candles."""
    frame = trades.sort_values(["timestamp", "source_row"], kind="stable").copy()
    frame["open_time"] = frame.timestamp.dt.floor("min")
    frame["taker_buy_contracts"] = frame["size"].where(frame.side.eq("Buy"), 0)
    frame["taker_sell_contracts"] = frame["size"].where(frame.side.eq("Sell"), 0)
    frame["taker_buy_btc"] = frame.homeNotional.where(frame.side.eq("Buy"), 0)
    frame["taker_sell_btc"] = frame.homeNotional.where(frame.side.eq("Sell"), 0)
    return frame.groupby("open_time", sort=True).agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"),
        close=("price", "last"), first_trade_time=("timestamp", "first"),
        last_trade_time=("timestamp", "last"), first_source_row=("source_row", "first"),
        last_source_row=("source_row", "last"), volume_contracts=("size", "sum"),
        volume_btc=("homeNotional", "sum"), volume_usd=("foreignNotional", "sum"),
        trades=("price", "size"), taker_buy_contracts=("taker_buy_contracts", "sum"),
        taker_sell_contracts=("taker_sell_contracts", "sum"),
        taker_buy_btc=("taker_buy_btc", "sum"), taker_sell_btc=("taker_sell_btc", "sum"),
        turnover_satoshis=("grossValue", "sum")).reset_index()


def combine_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    sums = frame.groupby("open_time", sort=True)[SUM_FIELDS].sum()
    first = frame.sort_values(["first_trade_time", "first_source_row"], kind="stable").groupby("open_time").first()
    last = frame.sort_values(["last_trade_time", "last_source_row"], kind="stable").groupby("open_time").last()
    sums["open"] = first.open
    sums["close"] = last.close
    sums["high"] = frame.groupby("open_time").high.max()
    sums["low"] = frame.groupby("open_time").low.min()
    sums["first_trade_time"] = first.first_trade_time
    sums["last_trade_time"] = last.last_trade_time
    sums["vwap_usd_per_btc"] = sums.volume_usd / sums.volume_btc
    result = sums.reset_index()
    result["close_time"] = result.open_time + pd.Timedelta(minutes=1)
    return result


def collect_day(item: dict, output: Path, start: pd.Timestamp, end: pd.Timestamp,
                *, keep_raw: bool = True) -> dict:
    day = pd.Timestamp(item["day"], tz="UTC")
    stamp = day.strftime("%Y%m%d")
    receipt = output / f"provenance/days/{stamp}.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        required = record["outputs"] if keep_raw else {path: digest for path, digest in record["outputs"].items() if path.startswith("bars_1m/")}
        if all((output / path).exists() for path in required):
            return record
    cache = output / f"download_cache/{stamp}.csv.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if item.get("source") == "public_rest_trade_fallback":
        if not cache.exists():
            rest_trade_archive(day, output, cache)
    elif not cache.exists() or cache.stat().st_size != item["bytes"]:
        temporary = cache.with_suffix(".gz.part")
        with requests.Session() as session, get(session, ARCHIVE + item["key"], stream=True) as response:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        if temporary.stat().st_size != item["bytes"]:
            raise ValueError(f"Archive size mismatch: {stamp}")
        temporary.replace(cache)
    digest = hashlib.md5()  # S3 single-part ETag integrity check, not security
    with cache.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if item["etag"] and "-" not in item["etag"] and digest.hexdigest() != item["etag"]:
        cache.unlink()  # exact corrupt download, never a user dataset
        raise ValueError(f"Archive ETag mismatch: {stamp}")
    source_sha = sha256(cache)
    raw_path = output / f"trades/{stamp}.parquet"
    if keep_raw:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(".parquet.tmp")
    writer = None
    parts, id_hashes, regular_row_hashes = [], [], []
    total = selected = excluded_legacy = nonregular = 0
    first_time = last_time = None
    try:
        for chunk in pd.read_csv(cache, compression="gzip", chunksize=150_000,
                                 usecols=lambda name: name in FIELDS,
                                 dtype={"size": "int64", "price": "float64", "grossValue": "int64",
                                        "homeNotional": "float64", "foreignNotional": "float64"}):
            chunk["source_row"] = np.arange(total, total + len(chunk), dtype=np.int64)
            total += len(chunk)
            frame = normalize_trades(chunk, day)
            frame = frame.loc[(frame.timestamp >= start) & (frame.timestamp < end)] if not frame.empty else frame
            if frame.empty:
                continue
            selected += len(frame)
            id_hashes.append(pd.util.hash_pandas_object(frame.trdMatchID, index=False).to_numpy())
            first_time = min(first_time, frame.timestamp.min()) if first_time is not None else frame.timestamp.min()
            last_time = max(last_time, frame.timestamp.max()) if last_time is not None else frame.timestamp.max()
            if keep_raw:
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
            eligible = frame.timestamp >= LAUNCH
            excluded_legacy += int((~eligible).sum())
            # Block/other special prints are retained raw but not executable regular-market bars.
            regular = frame.trdType.isin(["Regular", "UnspecifiedLegacy"])
            nonregular += int((eligible & ~regular).sum())
            valid = frame.loc[eligible & regular]
            if not valid.empty:
                # Recent spread executions reuse a match ID across distinct fills.
                # Audit repeats, but do not assume these are globally unique events.
                regular_row_hashes.append(pd.util.hash_pandas_object(
                    valid.loc[:, [name for name in FIELDS if name in valid]], index=False).to_numpy())
                parts.append(minute_parts(valid))
    finally:
        if writer is not None:
            writer.close()
    reused_ids = 0
    if id_hashes:
        ids = np.concatenate(id_hashes)
        reused_ids = len(ids) - len(np.unique(ids))
        if reused_ids:
            print(f"{stamp}: {reused_ids} reused match IDs; preserving distinct regular fills", flush=True)
    repeated_regular_records = 0
    if regular_row_hashes:
        rows = np.concatenate(regular_row_hashes)
        repeated_regular_records = len(rows) - len(np.unique(rows))
        if repeated_regular_records:
            # 2026-07-15 13:02 UTC proves identical published fields may represent
            # distinct fills: native bucket volume/count include both records.
            # Preserve source multiplicity; deduplication would undercount volume.
            print(f"{stamp}: {repeated_regular_records} repeated regular records; preserving native source multiplicity", flush=True)
    outputs = {}
    if writer is not None:
        temporary.replace(raw_path)
        outputs[str(raw_path.relative_to(output))] = sha256(raw_path)
    bars = combine_parts(parts)
    if not bars.empty:
        bar_path = output / f"bars_1m/{stamp}.parquet"
        write_parquet(bar_path, bars)
        outputs[str(bar_path.relative_to(output))] = sha256(bar_path)
    record = {**item, "download_sha256": source_sha, "outputs": outputs,
              "reused_match_identifiers": reused_ids,
              "repeated_regular_records": repeated_regular_records,
              "all_symbol_source_rows": total, "xbtusd_trade_rows": selected,
              "excluded_prelaunch_trades": excluded_legacy, "excluded_special_trades": nonregular,
              "regular_inverse_trades": int(bars.trades.sum()) if not bars.empty else 0,
              "bars": len(bars), "first_trade": first_time, "last_trade": last_time,
              "completed_at": now()}
    write_json(receipt, record)
    cache.unlink()  # verified XBTUSD parquet replaces temporary all-symbol source
    return record


def collect_funding(output: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    folder = output / "funding"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    cursor = max(start, LAUNCH)
    with requests.Session() as session:
        while cursor < end:
            path = folder / f"pages/{cursor.value}.json"
            if path.exists():
                page = json.loads(path.read_text())
            else:
                page = get(session, API + "funding", params={
                    "symbol": "XBTUSD", "count": 500, "reverse": "false",
                    "startTime": cursor.isoformat(), "endTime": end.isoformat()}).json()
                if not isinstance(page, list):
                    raise ValueError(f"Unexpected funding response: {page}")
                write_json(path, page)
                time.sleep(2.1)
            if not page:
                break
            rows.extend(page)
            next_cursor = pd.Timestamp(page[-1]["timestamp"]) + pd.Timedelta(milliseconds=1)
            if next_cursor <= cursor:
                raise ValueError("Funding cursor did not advance")
            cursor = next_cursor
            if len(page) < 500:
                break
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame = frame.loc[(frame.timestamp >= max(start, LAUNCH)) & (frame.timestamp < end)].sort_values("timestamp")
    if frame.timestamp.duplicated().any() or not frame.symbol.eq("XBTUSD").all():
        raise ValueError("Funding identity/duplicate audit failed")
    if not np.isfinite(frame.fundingRate).all():
        raise ValueError("Nonfinite funding")
    # API encodes durations relative to 2000-01-01, including the original daily schedule.
    frame["funding_interval_hours"] = (
        pd.to_datetime(frame.fundingInterval, utc=True) - pd.Timestamp("2000-01-01", tz="UTC")
    ).dt.total_seconds() / 3600
    actual = frame.timestamp.diff().dt.total_seconds() / 3600
    frame["hours_since_previous_settlement"] = actual
    if not np.allclose(actual.iloc[1:], frame.funding_interval_hours.iloc[1:]):
        raise ValueError("Unexplained gap/change in the funding settlement sequence")
    if frame.timestamp.iloc[-1] + pd.Timedelta(hours=float(frame.funding_interval_hours.iloc[-1])) < end:
        raise ValueError("Funding stops before the requested end")
    write_parquet(folder / "XBTUSD-funding.parquet", frame)
    report = {"rows": len(frame), "first": frame.timestamp.iloc[0], "last": frame.timestamp.iloc[-1],
              "interval_hours_counts": frame.funding_interval_hours.value_counts().to_dict(),
              "actual_spacing_hours_counts": actual.value_counts().to_dict(),
              "sha256": sha256(folder / "XBTUSD-funding.parquet")}
    write_json(folder / "audit.json", report)
    pages = folder / "pages"
    if pages.exists():
        shutil.rmtree(pages)  # temporary resumable API cache; compact Parquet is authoritative
    return report


def udf_frame(payload: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """UDF shifts documented bucket END timestamps back 60s; tested against REST."""
    if payload.get("s") == "no_data":
        return pd.DataFrame()
    if payload.get("s") != "ok":
        raise ValueError(f"Bad chart response: {payload}")
    frame = pd.DataFrame({"open_time": pd.to_datetime(payload["t"], unit="s", utc=True),
                          "open": payload["o"], "high": payload["h"], "low": payload["l"],
                          "close": payload["c"], "volume": payload["v"]})
    frame = frame.loc[(frame.open_time >= start) & (frame.open_time < end)].copy()
    frame["available_at"] = frame.open_time + pd.Timedelta(minutes=1)
    if frame.open_time.duplicated().any() or not frame.open_time.is_monotonic_increasing:
        raise ValueError("Unordered/duplicate reference bars")
    if not np.isfinite(frame[["open", "high", "low", "close"]]).all().all():
        raise ValueError("Nonfinite reference data")
    return frame


def collect_references(output: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    reports = {}
    with requests.Session() as session:
        snapshot_path = output / "reference/instrument_snapshot.json"
        if not snapshot_path.exists():
            write_json(snapshot_path, {"fetched_at": now(), "historical": False,
                                      "data": get(session, API + "instrument", params={"symbol": "XBTUSD", "count": 1}).json()})
            time.sleep(2.1)
        for symbol, label in [(".BXBT", "index"), (".XBTUSDPI", "premium"), (".XBT", "legacy_index")]:
            frames, empty_windows = [], []
            series_end = min(end, INDEX_SWITCH) if label == "legacy_index" else end
            for window_number, left in enumerate(pd.date_range(max(start, LAUNCH).floor("D"), series_end, freq="7D", inclusive="left")):
                right = min(left + pd.Timedelta(days=7), series_end)
                path = output / f"reference/{label}/pages/{left.strftime('%Y%m%d')}.json"
                if path.exists():
                    payload = json.loads(path.read_text())
                else:
                    payload = get(session, UDF, params={"symbol": symbol, "resolution": "1",
                        "from": int(left.timestamp()), "to": int(right.timestamp())}).json()
                    # Validate before caching; a transient error must not become permanent history.
                    udf_frame(payload, max(left, start, LAUNCH), right)
                    write_json(path, payload)
                    time.sleep(2.1)
                frame = udf_frame(payload, max(left, start, LAUNCH), right)
                if frame.empty:
                    empty_windows.append([left.isoformat(), right.isoformat()])
                else:
                    frames.append(frame)
                if window_number % 25 == 0:
                    print(f"reference {label} through {right.date()}", flush=True)
            data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not data.empty:
                if data.open_time.duplicated().any():
                    raise ValueError("Reference paging overlaps")
                path = output / f"reference/{label}/{symbol}-1m.parquet"
                write_parquet(path, data)
                expected = pd.date_range(max(start, LAUNCH), series_end, freq="min", inclusive="left")
                missing = expected.difference(pd.DatetimeIndex(data.open_time))
                write_parquet(output / f"reference/{label}/missing_minutes.parquet", pd.DataFrame({"open_time": missing}))
                reports[label] = {"symbol": symbol, "rows": len(data), "first": data.open_time.iloc[0],
                                  "last": data.open_time.iloc[-1], "empty_windows": empty_windows,
                                  "missing_minutes_in_requested_series_window": len(missing),
                                  "sha256": sha256(path), "timestamp_convention": "open_time; available_at=open_time+1min"}
            else:
                reports[label] = {"symbol": symbol, "rows": 0, "empty_windows": empty_windows}
            write_json(output / f"reference/{label}/audit.json", reports[label])
    index_parts = []
    for label, symbol in [("legacy_index", ".XBT"), ("index", ".BXBT")]:
        path = output / f"reference/{label}/{symbol}-1m.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        keep = frame.available_at < INDEX_SWITCH if label == "legacy_index" else frame.available_at >= INDEX_SWITCH
        frame = frame.loc[keep].copy()
        frame["reference_symbol"] = symbol
        index_parts.append(frame)
    if index_parts:
        combined = pd.concat(index_parts, ignore_index=True).sort_values("available_at")
        if combined.available_at.duplicated().any():
            raise ValueError("Duplicate timestamp at historical index switch")
        path = output / "reference/XBTUSD-historical-reference-index-1m.parquet"
        write_parquet(path, combined)
        reports["historical_reference_index"] = {"rows": len(combined), "switch_at": INDEX_SWITCH,
            "before": ".XBT", "after": ".BXBT", "sha256": sha256(path),
            "source": "https://www.bitmex.com/blog/notice-regarding-bitcoinusd-products-index-tick-size"}
    reports["historical_mark_price"] = {
        "available": False, "reason": "No direct historical mark series found in the public trade archive or UDF .XBTUSD symbol. Index/premium are NOT substituted for marks; snapshot is current metadata only."}
    write_json(output / "reference/audit.json", reports)
    return reports


def finalize(output: Path, files: list[dict], start: pd.Timestamp, end: pd.Timestamp,
             funding: dict, reference: dict) -> dict:
    records = [json.loads((output / f"provenance/days/{pd.Timestamp(item['day']).strftime('%Y%m%d')}.json").read_text()) for item in files]
    for record in records:
        for name, expected_sha in record["outputs"].items():
            if sha256(output / name) != expected_sha:
                raise ValueError(f"Final file checksum failed: {name}")
    bar_files = sorted((output / "bars_1m").glob("*.parquet"))
    target = output / "XBTUSD-1m-inverse-perpetual.parquet"
    temporary = target.with_suffix(".parquet.tmp")
    writer = None
    timestamps = []
    rows = 0
    try:
        for path in bar_files:
            frame = pd.read_parquet(path)
            if not ((frame.low <= frame.open) & (frame.open <= frame.high) &
                    (frame.low <= frame.close) & (frame.close <= frame.high)).all():
                raise ValueError(f"OHLC audit failed: {path}")
            if not np.array_equal(frame.volume_contracts, frame.taker_buy_contracts + frame.taker_sell_contracts):
                raise ValueError("Taker volume conservation failed")
            if not ((frame.first_trade_time >= frame.open_time) & (frame.last_trade_time < frame.close_time)).all():
                raise ValueError("Trade timestamp causality audit failed")
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            timestamps.append(pd.DatetimeIndex(frame.open_time))
            rows += len(frame)
    finally:
        if writer is not None:
            writer.close()
    index = timestamps[0].append(timestamps[1:])
    if index.duplicated().any() or not index.is_monotonic_increasing:
        raise ValueError("Final duplicate/order audit failed")
    expected = pd.date_range(max(start, LAUNCH), end, freq="min", inclusive="left")
    missing = expected.difference(index)
    missing_frame = pd.DataFrame({"open_time": missing, "reason": "no_regular_trade_in_complete_daily_archive"})
    write_parquet(output / "audit/no_trade_minutes.parquet", missing_frame)
    if index[0] < max(start, LAUNCH) or index[-1] >= end:
        raise ValueError("Final date audit failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target)
    manifest = {"status": "complete", "audit_passed": True, "completed_at": now(),
        "exchange": "BitMEX", "symbol": "XBTUSD", "market": "USD-quoted BTC-settled inverse perpetual",
        "interval": "1m", "requested_start_inclusive_utc": start, "requested_end_exclusive_utc": end,
        "inverse_contract_launch_utc": LAUNCH, "first_bar_open_utc": index[0], "last_bar_open_utc": index[-1],
        "bars": rows, "expected_minutes_since_launch": len(expected), "minutes_without_regular_trades": len(missing),
        "daily_sources": len(records), "daily_archives": sum(item["key"] is not None for item in files),
        "rest_recovered_archive_days": [item["day"] for item in files if item.get("source") == "public_rest_trade_fallback"],
        "source_compressed_bytes": sum(item["bytes"] for item in files),
        "stored_xbtusd_trades": sum(record["xbtusd_trade_rows"] for record in records),
        "regular_inverse_trades": sum(record["regular_inverse_trades"] for record in records),
        "prelaunch_legacy_trades_segregated_by_contract_regime": sum(record["excluded_prelaunch_trades"] for record in records),
        "special_trades_excluded_from_bars": sum(record["excluded_special_trades"] for record in records),
        "bar_sha256": sha256(target), "funding": funding, "reference": reference,
        "causality": "Trades aggregated in [open_time, close_time). OHLC available only at close_time. Open is first actual trade, not previous close. No forward-fill or artificial empty bars. Funding timestamp is settlement, not announcement; do not use realized future funding as an observation.",
        "agent_converted": False}
    write_json(output / "manifest.json", manifest)
    return manifest


def remove_retired_spot(output: Path) -> None:
    """Explicit user-authorized targets only; never touch results or model code."""
    manifest = json.loads((output / "manifest.json").read_text())
    if output != DEFAULT_OUTPUT.resolve() or not manifest.get("audit_passed") or manifest.get("status") != "complete":
        raise ValueError("Spot removal requires the complete audited production replacement")
    if (not manifest.get("reference") or manifest["requested_end_exclusive_utc"] != str(END)
            or manifest["requested_start_inclusive_utc"] != str(START)):
        raise ValueError("Incomplete production replacement; refusing Spot removal")
    targets = [ROOT / "data/binance_spot_btcusdt_1m_2017_2026_07_21",
               ROOT / "data/binance_spot_btcusdt_2017_2026_07_21"]
    receipt_path = output / "provenance/retired_spot_data.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"datasets": [], "authorized_request": "Bulk download and store the data (remove the old spot data)."}
    for target in targets:
        if target.is_symlink() or target.resolve().parent != (ROOT / "data").resolve():
            raise ValueError(f"Unsafe retirement target: {target}")
        if not target.exists():
            continue
        record = {"path": str(target), "files": [], "started_at": now(),
                  "former_manifest": json.loads((target / "manifest.json").read_text())}
        for path in sorted(target.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in Spot dataset: {path}")
            if path.is_file():
                record["files"].append({"path": str(path.relative_to(target)), "bytes": path.stat().st_size,
                                        "sha256": sha256(path)})
        receipt["datasets"].append(record)
        write_json(receipt_path, receipt)
        shutil.rmtree(target)
        record["removed_at"] = now()
        write_json(receipt_path, receipt)
        print(f"Removed retired Spot data: {target}", flush=True)


def legacy_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default=START.isoformat())
    parser.add_argument("--end-exclusive", default=END.isoformat())
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-references", action="store_true", help="Smoke tests only")
    parser.add_argument("--remove-retired-spot", action="store_true", help="Remove the two exact retired Spot datasets ONLY after the full production import audit succeeds")
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end_exclusive)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Dates must be timezone-aware and start < end")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "import.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    write_json(output / "status.json", {"status": "catalog", "started_at": now()})
    try:
        files = catalog(output, start, end)
        print(f"catalog: {len(files)} days, {sum(item['bytes'] for item in files)/1e9:.2f} GB compressed", flush=True)
        with ThreadPoolExecutor(max_workers=1) as background:
            def auxiliary():
                funding = collect_funding(output, start, end)
                reference = {} if args.skip_references else collect_references(output, start, end)
                return funding, reference
            aux = background.submit(auxiliary)
            completed = byte_count = 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(collect_day, item, output, start, end): item for item in files}
                for future in as_completed(futures):
                    try:
                        record = future.result()
                        if aux.done() and aux.exception() is not None:
                            raise aux.exception()
                    except Exception as error:
                        print(f"FAILED at {futures[future]['day']}: {error!r}", flush=True)
                        STOP.set()
                        for pending in futures:
                            pending.cancel()
                        raise
                    completed += 1
                    byte_count += record["bytes"]
                    elapsed = time.monotonic() - started
                    write_json(output / "status.json", {"status": "trades", "completed_days": completed,
                        "total_days": len(files), "verified_source_bytes": byte_count,
                        "total_source_bytes": sum(item["bytes"] for item in files), "last_day": record["day"],
                        "elapsed_seconds": elapsed, "updated_at": now()})
                    if completed % 20 == 0 or completed == len(files):
                        print(f"trades {completed}/{len(files)} ({record['day']}); {byte_count/1e9:.2f} GB; {elapsed/60:.1f} min", flush=True)
            print("trades complete; waiting for funding/reference history", flush=True)
            write_json(output / "status.json", {"status": "references", "completed_days": completed, "total_days": len(files), "updated_at": now()})
            funding, reference = aux.result()
        print("verifying stored hashes and consolidating one-minute bars", flush=True)
        manifest = finalize(output, files, start, end, funding, reference)
        if args.remove_retired_spot:
            remove_retired_spot(output)
        write_json(output / "status.json", {"status": "complete", "bars": manifest["bars"],
                                            "elapsed_seconds": time.monotonic() - started, "updated_at": now()})
        print(f"COMPLETE: {manifest['bars']:,} bars, {manifest['stored_xbtusd_trades']:,} raw XBTUSD trades", flush=True)
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error), "updated_at": now()})
        raise


if __name__ == "__main__":
    from collector.bitmex_ohlcv import main
    main()
