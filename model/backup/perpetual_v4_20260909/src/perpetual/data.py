from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import DATA, OUTPUT, TEST_START, TRAIN_START, development_range, write_json

FAST_NAMES = ("minute_log_return", "log_high_low", "log_close_open", "log_volume_change",
              "regular_trade_present", "log_stale_minutes")
SLOW_NAMES = tuple(f"momentum_{x}m" for x in (15, 60, 180, 720, 1440, 4320, 14400)) + (
    "volatility_1h", "volatility_24h", "relative_volume_1h", "last_settled_funding",
    "hours_since_funding", "last_known_funding_interval_hours")
CACHE_VERSION = 1


@dataclass
class Market:
    # Each row is the minute starting at epoch_start + i minutes.
    prices: np.ndarray  # regular open, high, low, carried close
    present: np.ndarray
    funding: np.ndarray
    fast: np.ndarray
    slow: np.ndarray  # causal state at minute boundary: uses rows strictly before it
    epoch_start: int  # UTC unix minutes
    audit: dict

    def __len__(self):
        return len(self.prices)

    def index(self, stamp):
        return int(pd.Timestamp(stamp).value // 60_000_000_000 - self.epoch_start)

    def timestamp(self, index):
        return pd.Timestamp((self.epoch_start + int(index)) * 60, unit="s", tz="UTC")

    def bounds(self, start, end, warmup=14401):
        start, end = development_range(start, end)
        first = max(self.index(start), warmup)
        # Align to UTC clock boundaries, even when the first actual trade
        # starts the cache at (for example) 00:08 rather than midnight.
        first += (-(self.epoch_start + first)) % 5
        last = min(self.index(end), len(self))
        if last - first < 5:
            raise ValueError("insufficient development coverage")
        return first, last


def causal_arrays(bars, funding, start, end):
    """No future backfills. Event-rate observations shift past settlement."""
    start, end = development_range(start, end)
    bars = bars[(bars.open_time >= start) & (bars.open_time < end)].copy()
    if bars.empty or bars.open_time.duplicated().any():
        raise ValueError("empty or duplicate minute bars")
    bars = bars.sort_values("open_time")
    # Start at an actual print, never backward fill early missing prices.
    start = max(start, bars.open_time.iloc[0])
    grid = pd.date_range(start, end, freq="min", inclusive="left")
    frame = bars.set_index("open_time").reindex(grid)
    present = frame.close.notna().to_numpy()
    close = frame.close.ffill().to_numpy(dtype=np.float64)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("invalid trade price")
    volume = frame.volume.fillna(0).to_numpy(dtype=np.float64)
    opening = frame.open.to_numpy(dtype=np.float64)
    high = frame.high.to_numpy(dtype=np.float64)
    low = frame.low.to_numpy(dtype=np.float64)
    prices = np.stack([opening, high, low, close], axis=1)
    r = np.r_[0.0, np.diff(np.log(close))]
    stale = np.arange(len(grid)) - np.maximum.accumulate(np.where(present, np.arange(len(grid)), 0))
    fast = np.stack([r, np.where(present, np.log(high / low), 0.0),
                     np.where(present, np.log(close / opening), 0.0),
                     np.r_[0.0, np.diff(np.log1p(volume))], present.astype(float),
                     np.log1p(stale)], axis=1).astype(np.float32)
    slow = []
    s = pd.Series(np.log(close))
    for width in (15, 60, 180, 720, 1440, 4320, 14400):
        slow.append((s - s.shift(width)).fillna(0).to_numpy())
    squared = pd.Series(r * r)
    slow.extend([np.sqrt(squared.rolling(w, min_periods=1).sum()).to_numpy() for w in (60, 1440)])
    v = pd.Series(volume)
    slow.append(np.log((v.rolling(60, min_periods=1).sum().to_numpy() + 1) /
                       (v.rolling(1440, min_periods=1).mean().to_numpy() * 60 + 1)))
    events = np.zeros(len(grid), dtype=np.float64)
    known_rate = np.full(len(grid), np.nan)
    interval = np.full(len(grid), np.nan)
    event_minute = np.full(len(grid), np.nan)
    epoch = int(grid[0].value // 60_000_000_000)
    funding = funding[(funding.timestamp >= start) & (funding.timestamp < end)].sort_values("timestamp")
    if funding.timestamp.duplicated().any():
        raise ValueError("duplicate funding settlements")
    if not funding.empty:
        if not (funding.timestamp == funding.timestamp.dt.floor("min")).all():
            raise ValueError("sub-minute funding event needs explicit handling")
        # Normalize Arrow/pandas timestamp units explicitly.
        idx = pd.DatetimeIndex(funding.timestamp).as_unit("ns").asi8 // 60_000_000_000 - epoch
        events[idx] = funding.fundingRate.to_numpy()
        known_rate[idx] = funding.fundingRate.to_numpy()
        interval[idx] = funding.funding_interval_hours.to_numpy()
        event_minute[idx] = idx
    rate_series = pd.Series(known_rate).ffill().fillna(0).to_numpy()
    interval_series = pd.Series(interval).ffill().fillna(0).to_numpy()
    last_event = pd.Series(event_minute).ffill().to_numpy()
    elapsed = np.where(np.isfinite(last_event), (np.arange(len(grid)) - last_event) / 60, 0)
    slow.extend([rate_series, elapsed, interval_series])
    # At boundary t the actor only has minute t-1 and settlements strictly < t.
    slow = np.vstack([np.zeros(len(SLOW_NAMES)), np.stack(slow, axis=1)[:-1]]).astype(np.float32)
    if not np.isfinite(fast).all() or not np.isfinite(slow).all():
        raise ValueError("nonfinite causal observation")
    audit = {"start": str(grid[0]), "end_exclusive": str(end), "minutes": len(grid),
             "regular_trade_bars": int(present.sum()), "funding_events": len(funding),
             "test_locked": True, "test_performance_accessed": False,
             "fast_names": list(FAST_NAMES), "slow_names": list(SLOW_NAMES)}
    return Market(prices, present, events, fast, slow, epoch, audit)


def prepare(cache=OUTPUT / "cache"):
    cache = Path(cache)
    manifest_path = cache / "manifest.json"
    source = json.loads((DATA / "manifest.json").read_text())
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if (meta["cache_version"] != CACHE_VERSION or meta["source_sha256"] != source["sha256"]
                or meta["source_funding_sha256"] != source["funding"]["sha256"]):
            raise ValueError("cache provenance changed; use a new output directory")
        development_range(meta["start"], meta["end_exclusive"])
        return load_market(cache)
    # Predicate pushdown: never materialize test rows for feature/scaler fitting.
    start = pd.Timestamp("2016-12-01", tz="UTC")
    bars = pd.read_parquet(DATA / "XBTUSD-1m-OHLCV.parquet",
                           filters=[("open_time", ">=", start), ("open_time", "<", TEST_START)])
    funding = pd.read_parquet(DATA / "funding/XBTUSD-funding.parquet",
                              filters=[("timestamp", ">=", start), ("timestamp", "<", TEST_START)])
    market = causal_arrays(bars, funding, start, TEST_START)
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("prices", "present", "funding", "fast", "slow"):
        np.save(cache / f"{name}.npy", getattr(market, name))
    write_json(manifest_path, {**market.audit, "epoch_start": market.epoch_start,
                               "cache_version": CACHE_VERSION, "source_sha256": source["sha256"],
                               "source_funding_sha256": source["funding"]["sha256"]})
    return load_market(cache)


def load_market(cache=OUTPUT / "cache"):
    cache = Path(cache)
    meta = json.loads((cache / "manifest.json").read_text())
    development_range(meta["start"], meta["end_exclusive"])
    values = {name: np.load(cache / f"{name}.npy", mmap_mode="r")
              for name in ("prices", "present", "funding", "fast", "slow")}
    expected = int(pd.Timestamp(meta["end_exclusive"]).value // 60_000_000_000 - meta["epoch_start"])
    if any(len(a) != expected for a in values.values()):
        raise ValueError("cache length differs from locked development interval")
    return Market(**values, epoch_start=meta["epoch_start"], audit=meta)


@dataclass
class Scaler:
    fast_mean: np.ndarray
    fast_std: np.ndarray
    slow_mean: np.ndarray
    slow_std: np.ndarray
    fitted_end: str

    @classmethod
    def fit(cls, market, end):
        first, last = market.bounds(TRAIN_START, end)
        fast = market.fast[first:last]
        slow = market.slow[first:last:5]
        return cls(fast.mean(0, dtype=np.float64).astype(np.float32),
                   np.maximum(fast.std(0, dtype=np.float64), 1e-6).astype(np.float32),
                   slow.mean(0, dtype=np.float64).astype(np.float32),
                   np.maximum(slow.std(0, dtype=np.float64), 1e-6).astype(np.float32), str(end))

    def observations(self, market, minutes, fast_minutes):
        minutes = np.asarray(minutes, dtype=np.int64)
        if minutes.min() < fast_minutes or minutes.max() >= len(market):
            raise ValueError("observation window out of development bounds")
        fast = market.fast[minutes[..., None] - np.arange(fast_minutes, 0, -1)]
        fast = np.clip((fast - self.fast_mean) / self.fast_std, -8, 8).astype(np.float32)
        slow = np.clip((market.slow[minutes] - self.slow_mean) / self.slow_std, -8, 8).astype(np.float32)
        return fast, slow

    def as_dict(self):
        return {name: getattr(self, name).tolist() for name in
                ("fast_mean", "fast_std", "slow_mean", "slow_std")} | {"fitted_end": self.fitted_end}

    @classmethod
    def from_dict(cls, item):
        return cls(**{k: np.asarray(v, dtype=np.float32) if k != "fitted_end" else v for k, v in item.items()})
