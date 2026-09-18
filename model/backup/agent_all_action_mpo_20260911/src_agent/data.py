"""Causal market arrays and the exogenous half of the observation.

The observation follows the split Dietterich, Trimponias & Chen (ICML 2018)
formalise between exogenous and endogenous state. Everything in this file is
exogenous - a one-contract trader does not move the XBTUSD price - so it is
computed once, cached, and shared by every account and every sampling seed. The
endogenous half, the account, is in ledger.py.

Every feature is dimensionless. BTC's annualised volatility ran from 119% in 2017
to 44% in 2023 in this sample, so a raw return means something different in each
year; divided by the ex-ante volatility it does not. That is the construction of
Moskowitz, Ooi & Pedersen (2012), used as a reinforcement-learning state by Lim,
Zohren & Roberts (2019) and Zhang, Zohren & Roberts (2020).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import CACHE, DATA, TEST_START, TRAIN_START, development_range, write_json

FEATURE_NAMES = (
    # The latest five-minute bar, in units of its ex-ante standard deviation.
    "bar_return", "bar_range", "bar_volume", "bar_coverage",
    # Time-series momentum at the formation horizons Liu & Tsyvinski (RFS 2021)
    # find predictive for Bitcoin: one day and one week.
    "momentum_1d", "momentum_1w",
    # Regimes: the latest day's volatility against the last four weeks, and the
    # latest week's volume against the four weeks before it.
    "volatility_regime", "volume_regime",
    # Perpetual funding: the last settled rate and the position in its cycle.
    "funding_rate", "funding_phase_sin", "funding_phase_cos",
    # Intraday periodicity (Andersen & Bollerslev 1997).
    "hour_sin", "hour_cos")
CACHE_VERSION = 2
BAR_MINUTES = 5
DAY_MINUTES = 1440
DAY_BARS = DAY_MINUTES // BAR_MINUTES
WEEK_MINUTES = 7 * DAY_MINUTES
# The volume regime compares the latest week with the four weeks before it, so
# the longest look-back is five weeks plus the bar being formed.
HISTORY_MINUTES = 5 * WEEK_MINUTES + BAR_MINUTES


@dataclass
class Market:
    # Each row is the minute starting at epoch_start + i minutes.
    prices: np.ndarray            # regular open, high, low, carried close
    present: np.ndarray           # a regular trade printed in the minute
    funding: np.ndarray           # funding rate settled at the minute, else zero
    features: np.ndarray          # exogenous observation at each minute boundary, from rows strictly before it
    daily_volatility: np.ndarray  # the same ex-ante estimate, as a one-day log-return standard deviation
    epoch_start: int              # UTC unix minutes
    audit: dict
    history_minutes: int = HISTORY_MINUTES

    def __len__(self):
        return len(self.prices)

    def index(self, stamp):
        return int(pd.Timestamp(stamp).value // 60_000_000_000 - self.epoch_start)

    def timestamp(self, index):
        return pd.Timestamp((self.epoch_start + int(index)) * 60, unit="s", tz="UTC")

    def bounds(self, start, end):
        start, end = development_range(start, end)
        first = max(self.index(start), self.history_minutes)
        # Align to UTC clock boundaries, even when the first trade starts the
        # cache at (for example) 00:08 rather than midnight.
        first += (-(self.epoch_start + first)) % BAR_MINUTES
        last = min(self.index(end), len(self))
        if last - first < BAR_MINUTES:
            raise ValueError("insufficient development coverage")
        return first, last


def trailing_sum(values, width):
    """Sum over the last ``width`` rows ending at each row, inclusive."""
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    index = np.arange(len(values))
    return cumulative[index + 1] - cumulative[np.maximum(index + 1 - width, 0)]


def trailing_count(length, width):
    return np.minimum(np.arange(length) + 1, width).astype(np.float64)


def strided_mean(values, bars):
    """Mean of values[m], values[m-5], ... over ``bars`` non-overlapping bars."""
    out = np.zeros(len(values), dtype=np.float64)
    for residue in range(BAR_MINUTES):
        sequence = values[residue::BAR_MINUTES]
        out[residue::BAR_MINUTES] = trailing_sum(sequence, bars) / trailing_count(len(sequence), bars)
    return out


def lagged_difference(values, lag):
    """values[m] - values[m - lag], or the change since the first row when shorter."""
    out = values - values[0]
    out[lag:] = values[lag:] - values[:-lag]
    return out


def trailing_extreme(values, width, reducer, fill):
    padded = np.concatenate([np.full(width - 1, fill), values])
    return reducer(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)


def causal_arrays(bars, funding, start, end):
    """No future backfills: the features at boundary t use minutes before t only."""
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
    size = len(grid)
    index = np.arange(size)
    epoch = int(grid[0].value // 60_000_000_000)
    log_close = np.log(close)

    # Ex-ante variance of a five-minute return: the realised variance of the last
    # day's 288 five-minute returns. Five-minute rather than one-minute returns,
    # because microstructure noise biases finer sampling (Liu, Patton & Sheppard
    # 2015); a whole day, because intraday periodicity dominates high-frequency
    # volatility and a shorter window would mix it into the level (Andersen &
    # Bollerslev 1997).
    bar_return = lagged_difference(log_close, BAR_MINUTES)
    bar_variance = strided_mean(bar_return ** 2, DAY_BARS)
    month_variance = strided_mean(bar_return ** 2, 28 * DAY_BARS)
    bar_sd = np.sqrt(bar_variance)
    usable = bar_sd > 0
    safe_sd = np.where(usable, bar_sd, 1.0)

    def standardised(values, bars):
        return np.where(usable, values / (safe_sd * np.sqrt(bars)), 0.0)

    # The bar's high-low range, the extreme-value volatility proxy (Parkinson 1980).
    bar_high = trailing_extreme(np.where(present, high, -np.inf), BAR_MINUTES, np.max, -np.inf)
    bar_low = trailing_extreme(np.where(present, low, np.inf), BAR_MINUTES, np.min, np.inf)
    traded = np.isfinite(bar_high) & np.isfinite(bar_low)
    bar_range = np.zeros(size)
    bar_range[traded] = np.log(bar_high[traded] / bar_low[traded])

    # Abnormal volume as the attention proxy (Barber & Odean 2008), constructed
    # as Liu & Tsyvinski construct theirs: the latest week against the average
    # of the four weeks before it.
    per_bar_day = trailing_sum(volume, DAY_MINUTES) / trailing_count(size, DAY_MINUTES) * BAR_MINUTES
    week = trailing_sum(volume, WEEK_MINUTES)
    week_minutes = trailing_count(size, WEEK_MINUTES)
    before = trailing_sum(volume, 5 * WEEK_MINUTES) - week
    before_minutes = trailing_count(size, 5 * WEEK_MINUTES) - week_minutes
    volume_regime = np.where(before_minutes > 0,
                             np.log1p(week / week_minutes * DAY_MINUTES)
                             - np.log1p(before / np.maximum(before_minutes, 1) * DAY_MINUTES), 0.0)
    volatility_regime = np.where(usable & (month_variance > 0),
                                 0.5 * np.log(np.where(usable, bar_variance, 1.0)
                                              / np.where(month_variance > 0, month_variance, 1.0)), 0.0)

    events = np.zeros(size, dtype=np.float64)
    known_rate = np.full(size, np.nan)
    cycle_minutes = np.full(size, np.nan)
    event_minute = np.full(size, np.nan)
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
        cycle_minutes[idx] = funding.funding_interval_hours.to_numpy() * 60
        event_minute[idx] = idx
    rate = pd.Series(known_rate).ffill().fillna(0).to_numpy()
    cycle = pd.Series(cycle_minutes).ffill().to_numpy()
    last_event = pd.Series(event_minute).ffill().to_numpy()
    known = np.isfinite(last_event) & np.isfinite(cycle) & (np.nan_to_num(cycle) > 0)
    # Minutes from the last settlement at or before row m to the boundary m + 1.
    phase = np.where(known, (index + 1 - np.nan_to_num(last_event)) / np.where(known, cycle, 1.0), 0.0)

    ending = np.stack([
        standardised(bar_return, 1),
        standardised(bar_range, 1),
        np.log1p(trailing_sum(volume, BAR_MINUTES)) - np.log1p(per_bar_day),
        trailing_sum(present.astype(np.float64), BAR_MINUTES) / trailing_count(size, BAR_MINUTES),
        standardised(lagged_difference(log_close, DAY_MINUTES), DAY_BARS),
        standardised(lagged_difference(log_close, WEEK_MINUTES), 7 * DAY_BARS),
        volatility_regime,
        volume_regime,
        rate,
        np.sin(2 * np.pi * phase),
        np.cos(2 * np.pi * phase)], axis=1)
    # At boundary t the actor has only minute t-1 and settlements strictly before t.
    features = np.zeros((size, len(FEATURE_NAMES)), dtype=np.float32)
    features[1:, :ending.shape[1]] = ending[:-1]
    clock = ((epoch + index) % DAY_MINUTES) / DAY_MINUTES
    features[:, FEATURE_NAMES.index("hour_sin")] = np.sin(2 * np.pi * clock)
    features[:, FEATURE_NAMES.index("hour_cos")] = np.cos(2 * np.pi * clock)
    daily_volatility = np.zeros(size, dtype=np.float32)
    daily_volatility[1:] = (bar_sd * np.sqrt(DAY_BARS))[:-1]
    if not np.isfinite(features).all() or not np.isfinite(daily_volatility).all():
        raise ValueError("nonfinite causal observation")
    audit = {"start": str(grid[0]), "end_exclusive": str(end), "minutes": size,
             "regular_trade_bars": int(present.sum()), "funding_events": len(funding),
             "test_locked": True, "test_performance_accessed": False,
             "feature_names": list(FEATURE_NAMES), "history_minutes": HISTORY_MINUTES}
    return Market(prices, present, events, features, daily_volatility, epoch, audit)


ARRAYS = ("prices", "present", "funding", "features", "daily_volatility")


def prepare(cache=CACHE):
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
    # Enough history before the first training minute for the longest look-back.
    # Predicate pushdown: test rows are never materialised.
    start = (TRAIN_START - pd.Timedelta(minutes=HISTORY_MINUTES)).floor("D")
    bars = pd.read_parquet(DATA / "XBTUSD-1m-OHLCV.parquet",
                           filters=[("open_time", ">=", start), ("open_time", "<", TEST_START)])
    funding = pd.read_parquet(DATA / "funding/XBTUSD-funding.parquet",
                              filters=[("timestamp", ">=", start), ("timestamp", "<", TEST_START)])
    market = causal_arrays(bars, funding, start, TEST_START)
    cache.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        np.save(cache / f"{name}.npy", getattr(market, name))
    write_json(manifest_path, {**market.audit, "epoch_start": market.epoch_start,
                               "cache_version": CACHE_VERSION, "source_sha256": source["sha256"],
                               "source_funding_sha256": source["funding"]["sha256"]})
    return load_market(cache)


def load_market(cache=CACHE):
    cache = Path(cache)
    meta = json.loads((cache / "manifest.json").read_text())
    development_range(meta["start"], meta["end_exclusive"])
    values = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
    expected = int(pd.Timestamp(meta["end_exclusive"]).value // 60_000_000_000 - meta["epoch_start"])
    if any(len(a) != expected for a in values.values()):
        raise ValueError("cache length differs from locked development interval")
    return Market(**values, epoch_start=meta["epoch_start"], audit=meta)


@dataclass
class Scaler:
    """Standardisation fitted on the training span only.

    Andrychowicz et al. (ICLR 2021): observation normalisation "is crucial for
    good performance". Clipped at the configured bound after standardising.
    """
    mean: np.ndarray
    std: np.ndarray
    fitted_end: str

    @classmethod
    def fit(cls, market, end):
        first, last = market.bounds(TRAIN_START, end)
        rows = np.asarray(market.features[first:last:BAR_MINUTES], dtype=np.float64)
        return cls(rows.mean(0).astype(np.float32), np.maximum(rows.std(0), 1e-6).astype(np.float32), str(end))

    def observations(self, market, minutes, clip):
        minutes = np.asarray(minutes, dtype=np.int64)
        if minutes.min() < 1 or minutes.max() >= len(market):
            raise ValueError("observation outside development bounds")
        values = (np.asarray(market.features[minutes]) - self.mean) / self.std
        return np.clip(values, -clip, clip).astype(np.float32)

    def as_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "fitted_end": self.fitted_end}

    @classmethod
    def from_dict(cls, item):
        return cls(np.asarray(item["mean"], dtype=np.float32), np.asarray(item["std"], dtype=np.float32),
                   item["fitted_end"])
