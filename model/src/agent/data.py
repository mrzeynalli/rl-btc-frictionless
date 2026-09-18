"""Causal market arrays and the exogenous half of the observation.

The observation follows the split Dietterich, Trimponias & Chen (ICML 2018) formalise
between exogenous and endogenous state. Everything in this file is exogenous - a
one-contract trader does not move the price - so it is computed once, cached, and
shared by every account. The endogenous half, the account, is in ledger.py.

Version 2 (2026-09-15). Prices, volume and funding come from one venue, BitMEX XBTUSD,
from 2016 (USER: one consistent dataset rather than BitMEX spliced to Binance, whose
funding histories disagree). The ledger settles them as a USDT-margined linear contract,
the account the user trades (ledger.py).

The market state carries returns over several horizons, normalised by volatility as
time-series momentum studies normalise them (Moskowitz, Ooi & Pedersen 2012; Zhang,
Zohren & Roberts 2020), beside the v1 regime features. Every candidate is computed and
cached; FEATURE_NAMES is the subset that survived the pre-registered redundancy rule.

Everything is computed from five-minute bars; the policy sees one snapshot per decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .config import BAR_MINUTES, BITMEX, CACHE, TEST_START, TRAIN_START, development_range, write_json

DAY_MINUTES = 1440
DAY_BARS = DAY_MINUTES // BAR_MINUTES
WEEK_MINUTES = 7 * DAY_MINUTES
WEEK_BARS = WEEK_MINUTES // BAR_MINUTES
MONTH_BARS = 4 * WEEK_BARS
HOUR_BARS = 60 // BAR_MINUTES

# Horizons of the volatility-normalised returns, in hours, in the pre-registered
# precedence order (results/agent_v2/PREREGISTRATION.md §2).
RETURN_HOURS = {"return_1h": 1, "return_1d": 24, "return_1w": 168, "return_4w": 672, "return_13w": 2184,
                "return_26w": 4368, "return_3h": 3, "return_3d": 72, "return_8w": 1344}
REGIME_FEATURES = ("volatility_level", "volatility_regime", "volume_regime", "funding_rate", "drawdown_4w",
                   "trend_efficiency_1w")
CANDIDATE_FEATURES = tuple(RETURN_HOURS) + REGIME_FEATURES
# The survivors of the pre-registered redundancy rule on 2017-2023, in candidate order
# (ARCHITECTURE.md §4.2; results/agent_v2/audit/feature_redundancy.json): drawdown_4w and
# trend_efficiency_1w were redundant with the four-week and one-week returns (0.82, 0.985).
FEATURE_NAMES = ("return_1h", "return_1d", "return_1w", "return_4w", "return_13w", "return_26w", "return_3h",
                 "return_3d", "return_8w", "volatility_level", "volatility_regime", "volume_regime", "funding_rate")
CACHE_VERSION = 4
# The longest look-back is the 26-week return plus the bar being formed.
HISTORY_MINUTES = 26 * WEEK_MINUTES + BAR_MINUTES
# Numerical only: a mean five-minute variance below this (0.01 bp of standard
# deviation) is treated as no movement.
NO_MOVEMENT = 1e-12


@dataclass
class Market:
    # Each row is the minute starting at epoch_start + i minutes.
    prices: np.ndarray            # regular open, high, low, carried close (USDT)
    present: np.ndarray           # a regular trade printed in the minute
    funding: np.ndarray           # funding rate settled at the minute, else zero
    features: np.ndarray          # every candidate feature at each minute, from rows strictly before it
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
        # Align to UTC clock boundaries.
        first += (-(self.epoch_start + first)) % BAR_MINUTES
        last = min(self.index(end), len(self))
        if last - first < BAR_MINUTES:
            raise ValueError("insufficient development coverage")
        return first, last


def feature_columns(names=None):
    names = FEATURE_NAMES if names is None else names
    return np.array([CANDIDATE_FEATURES.index(name) for name in names], dtype=np.int64)


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
    """No future backfills: the features at boundary t use minutes before t only.

    ``bars``: open_time, open, high, low, close, volume (one row per traded minute);
    ``funding``: timestamp, fundingRate.
    """
    start, end = development_range(start, end)
    bars = bars[(bars.open_time >= start) & (bars.open_time < end)].copy()
    if bars.empty or bars.open_time.duplicated().any():
        raise ValueError("empty or duplicate minute bars")
    bars = bars.sort_values("open_time")
    # Start at an actual print, never backward fill early missing prices.
    first_print = bars.open_time[bars.close.notna()].iloc[0]
    start = max(start, first_print)
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

    # The five-minute grid. Boundary b lies on a UTC multiple of five minutes; the bar
    # ending there covers minutes b-5 to b-1, and what is computed at b uses those
    # minutes and earlier ones only.
    boundary = index[((epoch + index) % BAR_MINUTES == 0) & (index >= 1)]
    count = len(boundary)
    if not count:
        raise ValueError("no five-minute boundary in the interval")
    bar_close = np.log(close[boundary - 1])
    bar_return = np.r_[0.0, np.diff(bar_close)]
    cumulative_volume = np.r_[0.0, np.cumsum(volume)]
    bar_volume = cumulative_volume[boundary] - cumulative_volume[np.maximum(boundary - BAR_MINUTES, 0)]

    # Realised variance from five-minute returns (Liu, Patton & Sheppard 2015) over a day
    # and over four weeks: the daily and monthly components of Corsi's (2009) HAR model.
    squared = bar_return ** 2
    day_variance = trailing_sum(squared, DAY_BARS) / trailing_count(count, DAY_BARS)
    month_variance = trailing_sum(squared, MONTH_BARS) / trailing_count(count, MONTH_BARS)
    usable = month_variance > NO_MOVEMENT
    month_safe = np.where(usable, month_variance, 1.0)
    daily_sd = np.where(usable, np.sqrt(month_safe * DAY_BARS), 0.0)
    columns = {}
    columns["volatility_level"] = np.where(usable, np.log(np.where(usable, daily_sd, 1.0)), 0.0)
    moving_day = usable & (day_variance > NO_MOVEMENT)
    columns["volatility_regime"] = np.where(
        moving_day, 0.5 * np.log(np.where(moving_day, day_variance, 1.0) / month_safe), 0.0)

    # Returns over several horizons in units of their own volatility, sigma_d sqrt(h / 24):
    # the normalisation of Moskowitz, Ooi & Pedersen (2012) and Zhang, Zohren & Roberts
    # (2020), with the four-week realised daily volatility as sigma_d.
    for name, hours in RETURN_HOURS.items():
        scale = np.where(usable, daily_sd * np.sqrt(hours / 24), 1.0)
        columns[name] = np.where(usable, lagged_difference(bar_close, hours * HOUR_BARS) / scale, 0.0)

    # Kaufman's (1995) efficiency ratio, signed.
    path = trailing_sum(np.abs(bar_return), WEEK_BARS)
    moved = path > np.sqrt(NO_MOVEMENT)
    columns["trend_efficiency_1w"] = np.clip(
        np.where(moved, lagged_difference(bar_close, WEEK_BARS) / np.where(moved, path, 1.0), 0.0), -1.0, 1.0)

    # Distance below the four-week peak, in daily volatilities (Pagan & Sossounov 2003).
    peak = pd.Series(bar_close).rolling(MONTH_BARS, min_periods=1).max().to_numpy()
    columns["drawdown_4w"] = np.where(usable, (bar_close - peak) / np.where(usable, daily_sd, 1.0), 0.0)

    # Abnormal volume as attention (Barber & Odean 2008; Liu & Tsyvinski 2021): the
    # latest week against the average of the four weeks before it.
    week_volume = trailing_sum(bar_volume, WEEK_BARS)
    week_bars = trailing_count(count, WEEK_BARS)
    before_volume = np.maximum(trailing_sum(bar_volume, 5 * WEEK_BARS) - week_volume, 0.0)
    before_bars = trailing_count(count, 5 * WEEK_BARS) - week_bars
    columns["volume_regime"] = np.where(before_bars > 0,
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
        idx = pd.DatetimeIndex(funding.timestamp).as_unit("ns").asi8 // 60_000_000_000 - epoch
        events[idx] = funding.fundingRate.to_numpy()
        known_rate[idx] = funding.fundingRate.to_numpy()
    # The last rate settled strictly before each boundary.
    settled = pd.Series(known_rate).ffill().fillna(0).to_numpy()
    columns["funding_rate"] = settled[boundary - 1]

    at_boundary = np.stack([columns[name] for name in CANDIDATE_FEATURES], axis=1)
    if not np.isfinite(at_boundary).all() or not np.isfinite(daily_sd).all():
        raise ValueError("nonfinite causal observation")
    # Between boundaries a minute carries the observation of the last boundary at or
    # before it, which uses nothing later than that boundary.
    owner = np.searchsorted(boundary, index, side="right") - 1
    known = owner >= 0
    features = np.zeros((size, len(CANDIDATE_FEATURES)), dtype=np.float32)
    features[known] = at_boundary.astype(np.float32)[owner[known]]
    daily_volatility = np.zeros(size, dtype=np.float32)
    daily_volatility[known] = daily_sd.astype(np.float32)[owner[known]]
    audit = {"start": str(grid[0]), "end_exclusive": str(end), "minutes": size,
             "regular_trade_bars": int(present.sum()), "funding_events": int(len(funding)),
             "test_locked": True, "test_performance_accessed": False,
             "candidate_features": list(CANDIDATE_FEATURES), "history_minutes": HISTORY_MINUTES,
             "bar_minutes": BAR_MINUTES}
    return Market(prices, present, events, features, daily_volatility, epoch, audit)


ARRAYS = ("prices", "present", "funding", "features", "daily_volatility")


def load_sources(end, start=None):
    """BitMEX XBTUSD one-minute bars and funding, from the first look-back minute to ``end``.

    One venue for prices, volume and funding (USER, 2026-09-15): mixing Binance and BitMEX
    would mix two funding histories (1.84 against 3.51 bp a day over 2020-2025). Rows at or
    after ``end`` are never read.
    """
    history_start = (TRAIN_START - pd.Timedelta(minutes=HISTORY_MINUTES)).floor("D") if start is None else start
    bars = pd.read_parquet(BITMEX / "XBTUSD-1m-OHLCV.parquet",
                           filters=[("open_time", ">=", history_start), ("open_time", "<", end)])
    funding = pd.read_parquet(BITMEX / "funding/XBTUSD-funding.parquet",
                              filters=[("timestamp", ">=", history_start), ("timestamp", "<", end)])
    return (bars[["open_time", "open", "high", "low", "close", "volume"]],
            funding[["timestamp", "fundingRate"]], {"venue": "BitMEX XBTUSD", "volume_units": "USD contracts"})


def source_digest():
    manifest = json.loads((BITMEX / "manifest.json").read_text())
    return {"bitmex_sha256": manifest["sha256"], "bitmex_funding_sha256": manifest["funding"]["sha256"]}


def prepare(cache=CACHE):
    cache = Path(cache)
    manifest_path = cache / "manifest.json"
    digest = source_digest()
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if meta["cache_version"] != CACHE_VERSION or any(meta.get(k) != v for k, v in digest.items()):
            raise ValueError("cache provenance changed; use a new output directory")
        development_range(meta["start"], meta["end_exclusive"])
        return load_market(cache)
    # Predicate pushdown: test rows are never materialised.
    start = (TRAIN_START - pd.Timedelta(minutes=HISTORY_MINUTES)).floor("D")
    bars, funding, sources = load_sources(TEST_START)
    market = causal_arrays(bars, funding, start, TEST_START)
    cache.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        np.save(cache / f"{name}.npy", getattr(market, name))
    write_json(manifest_path, {**market.audit, "epoch_start": market.epoch_start, "cache_version": CACHE_VERSION,
                               **digest, "sources": sources})
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


def training_decisions(market, end, config, start=TRAIN_START):
    """How many decisions one pass over the training span holds."""
    first, last = market.bounds(start, end)
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
    volatility that sets the reward's prior risk and the cost break-even. Fitted on the
    selected features only, and stores their names so a checkpoint cannot be read with
    another state.
    """
    mean: np.ndarray
    std: np.ndarray
    fitted_end: str
    bar_return_sd: float | None = None
    names: tuple = FEATURE_NAMES

    @classmethod
    def fit(cls, market, end, start=TRAIN_START):
        first, last = market.bounds(start, end)
        rows = np.asarray(market.features[first:last:BAR_MINUTES][:, feature_columns()], dtype=np.float64)
        return cls(rows.mean(0).astype(np.float32), np.maximum(rows.std(0), 1e-6).astype(np.float32), str(end),
                   bar_return_sd(market, first, last), tuple(FEATURE_NAMES))

    def observations(self, market, minutes, clip):
        minutes = np.asarray(minutes, dtype=np.int64)
        if minutes.min() < 1 or minutes.max() >= len(market):
            raise ValueError("observation outside development bounds")
        if tuple(self.names) != tuple(FEATURE_NAMES):
            raise ValueError("scaler was fitted on a different market state")
        values = (np.asarray(market.features[minutes][..., feature_columns()]) - self.mean) / self.std
        return np.clip(values, -clip, clip).astype(np.float32)

    def as_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "fitted_end": self.fitted_end,
                "bar_return_sd": self.bar_return_sd, "names": list(self.names)}

    @classmethod
    def from_dict(cls, item):
        return cls(np.asarray(item["mean"], dtype=np.float32), np.asarray(item["std"], dtype=np.float32),
                   item["fitted_end"], item.get("bar_return_sd"), tuple(item.get("names", FEATURE_NAMES)))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
