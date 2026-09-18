"""Causal market arrays and the exogenous half of the observation.

The observation follows the split Dietterich, Trimponias & Chen (ICML 2018)
formalise between exogenous and endogenous state. Everything in this file is
exogenous - a one-contract trader does not move the XBTUSD price - so it is
computed once, cached, and shared by every account and every sampling seed. The
endogenous half, the account, is in ledger.py.

The market state describes the regime the market is in, not a forecast of the
next price. Six dimensionless variables survived a measurement on the training
years alone (scripts/regime_state_analytics.py, ARCHITECTURE.md §4.2): each
conditions a property of the following day or week - realised volatility, the
payoff of holding the prevailing trend, the drift of the next week - and each
adds information the others do not carry. Candidates that failed were removed:
single-bar values, one-day trend (its only increment is next-day reversal, a
return forecast), a second and third trend horizon (rank correlations of 0.81 to
0.99 with the kept ones), the variance ratio (its persistence changed sign across
years), and clock and funding-cycle encodings.

Everything is computed from five-minute bars, which carry the trend and
volatility information at a finer resolution than hourly bars; nothing here looks
for patterns inside a bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import BAR_MINUTES, CACHE, DATA, TEST_START, TRAIN_START, development_range, write_json

FEATURE_NAMES = (
    # Trend. The signed efficiency of the last week's path - its net move over the
    # sum of its absolute five-minute moves, in [-1, 1] - and the drawdown from the
    # four-week high, in units of daily volatility.
    "trend_efficiency_1w", "drawdown_4w",
    # Risk. The four-week level of daily volatility, and the last day against it.
    "volatility_level", "volatility_regime",
    # Participation and carry. The last week's volume against the four weeks
    # before it, and the last settled funding rate.
    "volume_regime", "funding_rate")
CACHE_VERSION = 3
DAY_MINUTES = 1440
DAY_BARS = DAY_MINUTES // BAR_MINUTES
WEEK_MINUTES = 7 * DAY_MINUTES
WEEK_BARS = WEEK_MINUTES // BAR_MINUTES
MONTH_BARS = 4 * WEEK_BARS
# The volume regime compares the latest week with the four weeks before it, so
# the longest look-back is five weeks plus the bar being formed.
HISTORY_MINUTES = 5 * WEEK_MINUTES + BAR_MINUTES
# Numerical only: a mean five-minute variance below this (0.01 bp of standard
# deviation) is treated as no movement. Differences of cumulative sums cancel to
# about 1e-16, which would otherwise turn a flat stretch into a huge log ratio.
NO_MOVEMENT = 1e-12


@dataclass
class Market:
    # Each row is the minute starting at epoch_start + i minutes.
    prices: np.ndarray            # regular open, high, low, carried close
    present: np.ndarray           # a regular trade printed in the minute
    funding: np.ndarray           # funding rate settled at the minute, else zero
    features: np.ndarray          # exogenous observation at each minute boundary, from rows strictly before it
    daily_volatility: np.ndarray  # four-week estimate of a one-day log-return standard deviation, same timing
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


def lagged_difference(values, lag):
    """values[m] - values[m - lag], or the change since the first row when shorter."""
    out = values - values[0]
    out[lag:] = values[lag:] - values[:-lag]
    return out


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
    prices = np.stack([frame.open.to_numpy(dtype=np.float64), frame.high.to_numpy(dtype=np.float64),
                       frame.low.to_numpy(dtype=np.float64), close], axis=1)
    size = len(grid)
    index = np.arange(size)
    epoch = int(grid[0].value // 60_000_000_000)

    # The five-minute grid. Boundary b lies on a UTC multiple of five minutes; the
    # bar ending there covers minutes b-5 to b-1, and what is computed at b uses
    # those minutes and earlier ones only.
    boundary = index[((epoch + index) % BAR_MINUTES == 0) & (index >= 1)]
    count = len(boundary)
    if not count:
        raise ValueError("no five-minute boundary in the interval")
    bar_close = np.log(close[boundary - 1])
    bar_return = np.r_[0.0, np.diff(bar_close)]
    cumulative_volume = np.r_[0.0, np.cumsum(volume)]
    bar_volume = cumulative_volume[boundary] - cumulative_volume[np.maximum(boundary - BAR_MINUTES, 0)]

    # Realised variance from five-minute returns, which microstructure noise does
    # not bias the way finer sampling does (Liu, Patton & Sheppard 2015), over a day
    # and over four weeks: the daily and monthly components of Corsi's (2009) HAR
    # model. Their level and their ratio are nearly uncorrelated (rank correlation
    # -0.07 on 2017-2023), and both forecast the next day's volatility.
    squared = bar_return ** 2
    day_variance = trailing_sum(squared, DAY_BARS) / trailing_count(count, DAY_BARS)
    month_variance = trailing_sum(squared, MONTH_BARS) / trailing_count(count, MONTH_BARS)
    usable = month_variance > NO_MOVEMENT
    month_safe = np.where(usable, month_variance, 1.0)
    daily_sd = np.where(usable, np.sqrt(month_safe * DAY_BARS), 0.0)
    volatility_level = np.where(usable, np.log(np.where(usable, daily_sd, 1.0)), 0.0)
    moving_day = usable & (day_variance > NO_MOVEMENT)
    volatility_regime = np.where(moving_day, 0.5 * np.log(np.where(moving_day, day_variance, 1.0) / month_safe), 0.0)

    # Kaufman's (1995) efficiency ratio, signed: how much of the week's movement
    # went in one direction. Bounded by construction, so it needs no volatility
    # estimate.
    path = trailing_sum(np.abs(bar_return), WEEK_BARS)
    moved = path > np.sqrt(NO_MOVEMENT)
    trend_efficiency = np.where(moved, lagged_difference(bar_close, WEEK_BARS) / np.where(moved, path, 1.0), 0.0)
    trend_efficiency = np.clip(trend_efficiency, -1.0, 1.0)

    # Distance below the running peak, the quantity bull and bear phases are dated
    # by (Pagan & Sossounov 2003; Lunde & Timmermann 2004).
    peak = pd.Series(bar_close).rolling(MONTH_BARS, min_periods=1).max().to_numpy()
    drawdown = np.where(usable, (bar_close - peak) / np.where(usable, daily_sd, 1.0), 0.0)

    # Abnormal volume as the attention proxy (Barber & Odean 2008), constructed as
    # Liu & Tsyvinski (2021) construct theirs: the latest week against the average
    # of the four weeks before it.
    week_volume = trailing_sum(bar_volume, WEEK_BARS)
    week_bars = trailing_count(count, WEEK_BARS)
    before_volume = np.maximum(trailing_sum(bar_volume, 5 * WEEK_BARS) - week_volume, 0.0)
    before_bars = trailing_count(count, 5 * WEEK_BARS) - week_bars
    volume_regime = np.where(before_bars > 0,
                             np.log1p(week_volume / week_bars * DAY_BARS)
                             - np.log1p(before_volume / np.maximum(before_bars, 1) * DAY_BARS), 0.0)

    events = np.zeros(size, dtype=np.float64)
    known_rate = np.full(size, np.nan)
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
    # The last rate settled strictly before each boundary: the carry an open
    # position is paying, and the crowding of the side that pays it.
    settled = pd.Series(known_rate).ffill().fillna(0).to_numpy()
    funding_rate = settled[boundary - 1]

    columns = {"trend_efficiency_1w": trend_efficiency, "drawdown_4w": drawdown,
               "volatility_level": volatility_level, "volatility_regime": volatility_regime,
               "volume_regime": volume_regime, "funding_rate": funding_rate}
    at_boundary = np.stack([columns[name] for name in FEATURE_NAMES], axis=1)
    if not np.isfinite(at_boundary).all() or not np.isfinite(daily_sd).all():
        raise ValueError("nonfinite causal observation")
    # Between boundaries a minute carries the observation of the last boundary at
    # or before it, which uses nothing later than that boundary.
    owner = np.searchsorted(boundary, index, side="right") - 1
    known = owner >= 0
    features = np.zeros((size, len(FEATURE_NAMES)), dtype=np.float32)
    features[known] = at_boundary.astype(np.float32)[owner[known]]
    daily_volatility = np.zeros(size, dtype=np.float32)
    daily_volatility[known] = daily_sd.astype(np.float32)[owner[known]]
    audit = {"start": str(grid[0]), "end_exclusive": str(end), "minutes": size,
             "regular_trade_bars": int(present.sum()), "funding_events": len(funding),
             "test_locked": True, "test_performance_accessed": False,
             "feature_names": list(FEATURE_NAMES), "history_minutes": HISTORY_MINUTES,
             "bar_minutes": BAR_MINUTES}
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


def training_decisions(market, end, config):
    """How many decisions one pass over the training span holds."""
    first, last = market.bounds(TRAIN_START, end)
    return (last - first) // (config.decision_bars * BAR_MINUTES)


def bar_return_sd(market, first, last):
    """Standard deviation of five-minute log returns between the boundaries in [first, last)."""
    closes = np.log(np.asarray(market.prices[first - 1:last - 1:BAR_MINUTES, 3], dtype=np.float64))
    return float(np.diff(closes).std())


@dataclass
class Scaler:
    """Everything measured on the training span only, and nothing after it.

    Observation standardisation - Andrychowicz et al. (ICLR 2021): "Always use observation
    normalization" - clipped at the configured bound; and the five-minute return
    volatility that sets the reward's prior risk and the cost break-even.
    """
    mean: np.ndarray
    std: np.ndarray
    fitted_end: str
    bar_return_sd: float | None = None

    @classmethod
    def fit(cls, market, end):
        first, last = market.bounds(TRAIN_START, end)
        rows = np.asarray(market.features[first:last:BAR_MINUTES], dtype=np.float64)
        return cls(rows.mean(0).astype(np.float32), np.maximum(rows.std(0), 1e-6).astype(np.float32), str(end),
                   bar_return_sd(market, first, last))

    def observations(self, market, minutes, clip):
        minutes = np.asarray(minutes, dtype=np.int64)
        if minutes.min() < 1 or minutes.max() >= len(market):
            raise ValueError("observation outside development bounds")
        values = (np.asarray(market.features[minutes]) - self.mean) / self.std
        return np.clip(values, -clip, clip).astype(np.float32)

    def as_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "fitted_end": self.fitted_end,
                "bar_return_sd": self.bar_return_sd}

    @classmethod
    def from_dict(cls, item):
        return cls(np.asarray(item["mean"], dtype=np.float32), np.asarray(item["std"], dtype=np.float32),
                   item["fitted_end"], item.get("bar_return_sd"))
