"""Causal Spot-market features, walk-forward folds, and train-only scaling.

Each decision observes sixty completed one-minute bars plus ten slower Spot
features.  It is made after an hour closes, fills at the next hour's open, and
earns the following open-to-open return.  Native Binance gaps are never filled;
feature windows that touch a gap are removed and recurrent state is reset at
the resulting segment boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import STUDY_END, STUDY_START, load


BARS_PER_HOUR = 60
BARS_PER_DAY = 1440
FAST_FEATURE_NAMES = (
    "log_return_1m",
    "log_high_low_range",
    "log_quote_volume_change",
    "taker_buy_imbalance",
)
SLOW_FEATURE_NAMES = (
    "momentum_3h",
    "momentum_5h",
    "momentum_12h",
    "momentum_1d",
    "momentum_3d",
    "momentum_10d",
    "realised_volatility_6h",
    "realised_volatility_24h",
    "quote_volume_ratio_24h",
    "taker_buy_imbalance_1h",
)
WARMUP_HOURS_REQUIRED = 240
EVALUATION_WARMUP_HOURS = 240
VOLUME_EPSILON = 1.0
NORMALISED_CLIP = 8.0


@dataclass(frozen=True)
class Split:
    name: str
    kind: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "train_start_inclusive": self.train_start.isoformat(),
            "train_end_exclusive": self.train_end.isoformat(),
            "evaluation_start_inclusive": self.evaluation_start.isoformat(),
            "evaluation_end_exclusive": self.evaluation_end.isoformat(),
        }


def _stamp(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


WALK_FORWARD_SPLITS = tuple(
    Split(
        f"fold_{year}", "walk_forward", STUDY_START, _stamp(f"{year}-01-01"),
        _stamp(f"{year}-01-01"), _stamp(f"{year + 1}-01-01"),
    )
    for year in range(2020, 2026)
)
LOCKED_TEST_SPLIT = Split(
    "locked_2026", "locked_previously_exposed", STUDY_START, _stamp("2026-01-01"),
    _stamp("2026-01-01"), STUDY_END,
)
ALL_SPLITS = WALK_FORWARD_SPLITS + (LOCKED_TEST_SPLIT,)


@dataclass(frozen=True)
class MarketData:
    execution_time: pd.DatetimeIndex
    decision_time: pd.DatetimeIndex
    fast: np.ndarray
    slow: np.ndarray
    entry_price: np.ndarray
    exit_price: np.ndarray
    gap_price_ratio: np.ndarray
    segment_start: np.ndarray
    slow_names: tuple[str, ...]
    audit: dict

    def __len__(self) -> int:
        return int(self.fast.shape[0])

    def index_range(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
        return (
            int(self.execution_time.searchsorted(start, "left")),
            int(self.execution_time.searchsorted(end, "left")),
        )


def _rolling_sum(values: np.ndarray, end: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return cumulative[end] - cumulative[end - width]


def build_market(output_dir: Path | None = None) -> MarketData:
    bars = load() if output_dir is None else load(output_dir)
    timestamps = pd.DatetimeIndex(bars.open_time)
    aligned = timestamps == timestamps.floor("1min")
    canonical = bars.loc[aligned].set_index("open_time")
    first = timestamps[aligned][0]
    last = timestamps[aligned][-1]
    grid_index = pd.date_range(first, last, freq="1min")
    grid = canonical.reindex(grid_index)
    if len(grid) % BARS_PER_HOUR:
        raise ValueError("canonical Spot grid does not divide into whole hours")

    close = grid.close.to_numpy(dtype=np.float64)
    high = grid.high.to_numpy(dtype=np.float64)
    low = grid.low.to_numpy(dtype=np.float64)
    open_price = grid.open.to_numpy(dtype=np.float64)
    quote = grid.quote_volume.to_numpy(dtype=np.float64)
    taker = grid.taker_buy_quote.to_numpy(dtype=np.float64)
    present = np.isfinite(np.stack([open_price, high, low, close, quote, taker])).all(axis=0)

    log_return = np.full(len(grid), np.nan, dtype=np.float64)
    valid_pair = present[1:] & present[:-1]
    log_return[1:][valid_pair] = np.log(close[1:][valid_pair] / close[:-1][valid_pair])
    log_range = np.where(present, np.log(high / low), np.nan)
    log_quote_change = np.full(len(grid), np.nan, dtype=np.float64)
    log_quote_change[1:][valid_pair] = np.log(
        (quote[1:][valid_pair] + VOLUME_EPSILON)
        / (quote[:-1][valid_pair] + VOLUME_EPSILON)
    )
    imbalance = np.full(len(grid), np.nan, dtype=np.float64)
    imbalance[present] = np.where(
        quote[present] > 0,
        (2.0 * taker[present] - quote[present])
        / np.maximum(quote[present], VOLUME_EPSILON),
        0.0,
    )
    bar_features = np.stack(
        [log_return, log_range, log_quote_change, imbalance], axis=1
    )

    hours_total = len(grid) // BARS_PER_HOUR
    hourly_features = bar_features.reshape(hours_total, BARS_PER_HOUR, -1)
    hourly_present = present.reshape(hours_total, BARS_PER_HOUR).all(axis=1)
    hourly_close = close.reshape(hours_total, BARS_PER_HOUR)[:, -1]
    hourly_open = open_price.reshape(hours_total, BARS_PER_HOUR)[:, 0]
    hourly_quote = np.nansum(quote.reshape(hours_total, BARS_PER_HOUR), axis=1)
    hourly_taker = np.nansum(taker.reshape(hours_total, BARS_PER_HOUR), axis=1)

    candidates = np.arange(WARMUP_HOURS_REQUIRED, hours_total - 2, dtype=np.int64)
    complete_prefix = np.concatenate(([0], np.cumsum(hourly_present.astype(np.int64))))
    complete_history = (
        complete_prefix[candidates + 1]
        - complete_prefix[candidates - WARMUP_HOURS_REQUIRED]
        == WARMUP_HOURS_REQUIRED + 1
    )
    execution_available = (
        np.isfinite(hourly_open[candidates + 1])
        & np.isfinite(hourly_open[candidates + 2])
    )
    valid = complete_history & execution_available
    hours = candidates[valid]
    if not len(hours):
        raise ValueError("no complete causal Spot decision windows")

    squared = np.nan_to_num(log_return * log_return, nan=0.0)
    bar_end = (hours + 1) * BARS_PER_HOUR

    def momentum(lag: int) -> np.ndarray:
        return np.log(hourly_close[hours] / hourly_close[hours - lag])

    quote_cumulative = np.concatenate(([0.0], np.cumsum(hourly_quote)))
    quote_mean_24h = (
        quote_cumulative[hours + 1] - quote_cumulative[hours - 23]
    ) / 24.0
    taker_imbalance = np.where(
        hourly_quote[hours] > 0,
        (2.0 * hourly_taker[hours] - hourly_quote[hours]) / hourly_quote[hours],
        0.0,
    )
    slow = np.stack([
        momentum(3), momentum(5), momentum(12), momentum(24), momentum(72),
        momentum(240),
        np.sqrt(_rolling_sum(squared, bar_end, 6 * BARS_PER_HOUR)),
        np.sqrt(_rolling_sum(squared, bar_end, 24 * BARS_PER_HOUR)),
        np.log((hourly_quote[hours] + VOLUME_EPSILON) / (quote_mean_24h + VOLUME_EPSILON)),
        taker_imbalance,
    ], axis=1)
    fast = hourly_features[hours]
    execution_time = pd.DatetimeIndex(grid_index[(hours + 1) * BARS_PER_HOUR])
    close_times = pd.DatetimeIndex(grid.close_time)
    decision_time = close_times[hours * BARS_PER_HOUR + BARS_PER_HOUR - 1]
    entry_price = hourly_open[hours + 1]
    exit_price = hourly_open[hours + 2]
    segment_start = np.r_[
        True,
        (execution_time[1:] - execution_time[:-1]) != pd.Timedelta(hours=1),
    ]
    gap_price_ratio = np.ones(len(hours), dtype=np.float64)
    gap_price_ratio[1:] = entry_price[1:] / exit_price[:-1]

    if not np.isfinite(fast).all() or not np.isfinite(slow).all():
        raise ValueError("non-finite Spot feature produced")
    if not (np.isfinite(entry_price).all() and np.isfinite(exit_price).all()):
        raise ValueError("non-finite execution price")
    if not np.all(decision_time < execution_time):
        raise ValueError("a feature timestamp reaches or exceeds execution")

    audit = {
        "market": "BTCUSDT Spot",
        "funding_features": False,
        "decisions_total": int(len(hours)),
        "first_execution_utc": execution_time[0].isoformat(),
        "last_execution_utc": execution_time[-1].isoformat(),
        "fast_feature_names": list(FAST_FEATURE_NAMES),
        "slow_feature_names": list(SLOW_FEATURE_NAMES),
        "position_is_separate_from_market_features": True,
        "observation_bars_per_decision": BARS_PER_HOUR,
        "native_off_grid_bars_excluded": int((~aligned).sum()),
        "canonical_missing_bars": int((~present).sum()),
        "decision_segments": int(segment_start.sum()),
        "gap_policy": (
            "drop feature windows touching gaps and reset recurrent state; "
            "evaluation carries inventory and accounts for passive gap return"
        ),
        "features_end_before_execution_price_starts": True,
        "warmup_hours_required": WARMUP_HOURS_REQUIRED,
    }
    return MarketData(
        execution_time=execution_time,
        decision_time=decision_time,
        fast=fast,
        slow=slow,
        entry_price=entry_price,
        exit_price=exit_price,
        gap_price_ratio=gap_price_ratio,
        segment_start=segment_start.astype(bool),
        slow_names=SLOW_FEATURE_NAMES,
        audit=audit,
    )


@dataclass(frozen=True)
class Normaliser:
    fast_mean: np.ndarray
    fast_std: np.ndarray
    slow_mean: np.ndarray
    slow_std: np.ndarray
    slow_names: tuple[str, ...]
    clip: float
    fitted_on: dict

    def apply(self, fast: np.ndarray, slow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.clip((fast - self.fast_mean) / self.fast_std, -self.clip, self.clip).astype(np.float32),
            np.clip((slow - self.slow_mean) / self.slow_std, -self.clip, self.clip).astype(np.float32),
        )

    def as_dict(self) -> dict:
        return {
            "fitted_on": self.fitted_on,
            "clip_sigma": self.clip,
            "fast": {
                name: {"mean": float(mean), "std": float(std)}
                for name, mean, std in zip(FAST_FEATURE_NAMES, self.fast_mean, self.fast_std)
            },
            "slow": {
                name: {"mean": float(mean), "std": float(std)}
                for name, mean, std in zip(self.slow_names, self.slow_mean, self.slow_std)
            },
        }


def fit_normaliser(market: MarketData, split: Split) -> Normaliser:
    first, last = market.index_range(split.train_start, split.train_end)
    if last - first < 24 * 30:
        raise ValueError(f"training window {split.name} is too short to normalise")
    fast = market.fast[first:last].reshape(-1, market.fast.shape[-1])
    slow = market.slow[first:last]
    fast_std = fast.std(axis=0)
    slow_std = slow.std(axis=0)
    if (fast_std <= 0).any() or (slow_std <= 0).any():
        raise ValueError("a feature is constant across the training window")
    return Normaliser(
        fast.mean(axis=0), fast_std, slow.mean(axis=0), slow_std,
        market.slow_names, NORMALISED_CLIP,
        {
            "split": split.name,
            "start_inclusive": split.train_start.isoformat(),
            "end_exclusive": split.train_end.isoformat(),
            "decisions": int(last - first),
        },
    )


@dataclass(frozen=True)
class SplitData:
    split: Split
    market: MarketData
    normaliser: Normaliser
    fast: np.ndarray
    slow: np.ndarray
    train_range: tuple[int, int]
    evaluation_range: tuple[int, int]
    evaluation_warmup_start: int

    def as_dict(self) -> dict:
        train_first, train_last = self.train_range
        eval_first, eval_last = self.evaluation_range
        return {
            **self.split.as_dict(),
            "train_decisions": int(train_last - train_first),
            "evaluation_decisions": int(eval_last - eval_first),
            "evaluation_warmup_hours": int(eval_first - self.evaluation_warmup_start),
            "normaliser": self.normaliser.as_dict(),
        }


def build_split(market: MarketData, split: Split) -> SplitData:
    normaliser = fit_normaliser(market, split)
    fast, slow = normaliser.apply(market.fast, market.slow)
    train_range = market.index_range(split.train_start, split.train_end)
    evaluation_range = market.index_range(split.evaluation_start, split.evaluation_end)
    warmup_start = max(0, evaluation_range[0] - EVALUATION_WARMUP_HOURS)
    return SplitData(
        split, market, normaliser, fast, slow, train_range, evaluation_range, warmup_start
    )


def split_by_name(name: str) -> Split:
    for split in ALL_SPLITS:
        if split.name == name:
            return split
    raise KeyError(f"unknown split {name!r}")


def main() -> None:
    market = build_market()
    print(json.dumps({
        "market": market.audit,
        "splits": [build_split(market, split).as_dict() for split in ALL_SPLITS],
    }, indent=2))


if __name__ == "__main__":
    main()
