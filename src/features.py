"""State construction, walk-forward folds and the frozen normaliser.

The state is exactly the one the architecture specifies and nothing else.

Twelve five-minute bars, four variables each:

* five-minute log return
* log high-low range
* log change in quote volume
* taker-buy imbalance

Seven slow variables read once at the hourly decision:

* 3-hour, 5-hour, 1-day, 3-day and 10-day momentum
* 24-hour realised volatility
* the latest funding rate that had *already settled*

plus a three-way one-hot of the position currently held.  There is no raw BTC
price, no technical indicator, no regime label, no funding countdown, no
unrealised P&L, no calendar variable and no exposure-to-equity variable.

Timing.  Decision ``d`` for clock hour ``h`` is taken once the twelfth bar of
hour ``h`` has closed.  The order is executed at the next five-minute open,
which is the top of hour ``h+1``, and the resulting position is held until the
top of hour ``h+2``.  Features therefore end at the close of hour ``h`` and
prices begin at the open of hour ``h+1``; the two never overlap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import STUDY_END, STUDY_START, load

BARS_PER_HOUR = 12
BARS_PER_DAY = 288
FAST_FEATURE_NAMES = (
    "log_return_5m",
    "log_high_low_range",
    "log_quote_volume_change",
    "taker_buy_imbalance",
)
SLOW_FEATURE_NAMES = (
    "momentum_3h",
    "momentum_5h",
    "momentum_1d",
    "momentum_3d",
    "momentum_10d",
    "realised_volatility_24h",
    "last_settled_funding_rate",
)

# The position taken at decision t pays the settlement stamped at tau_{t+1}.
# Settlements are on a fixed eight-hour UTC schedule, so *whether* the next
# interval carries one is known from the clock alone -- no rate is read from
# the future.  ``expected_funding_charge`` multiplies that clock indicator by
# the last rate that had already settled, which is the causal analogue of the
# ``-kappa_t f_t`` term whose gradient lets Borrageiro's agent harvest carry.
# Without these the agent sees a charge arriving in 12.5% of hours and has no
# way to tell which hour it is in.
FUNDING_FEATURE_NAMES = (
    "settlement_at_next_execution",
    "hours_to_next_settlement",
    "expected_funding_charge",
)


def slow_feature_names(funding_features: bool) -> tuple[str, ...]:
    return SLOW_FEATURE_NAMES + (FUNDING_FEATURE_NAMES if funding_features else ())
POSITION_NAMES = ("short", "flat", "long")
POSITION_LEVELS = np.array([-1.0, 0.0, 1.0], dtype=np.float64)

# Ten-day momentum is the longest lookback; the dataset carries 31 days of
# warm-up so the first study decision is fully formed.
WARMUP_HOURS_REQUIRED = 240
# Hours of causal GRU burn-in run before any evaluation window so the hidden
# state entering the window is not an artificial zero.
EVALUATION_WARMUP_HOURS = 240

VOLUME_EPSILON = 1.0  # one US dollar, against the 55 genuinely empty bars
NORMALISED_CLIP = 8.0


@dataclass(frozen=True)
class Split:
    """One chronological training window and the window evaluated after it."""

    name: str
    kind: str  # "walk_forward" or "final"
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


WALK_FORWARD_SPLITS = (
    Split("fold_1", "walk_forward", _stamp("2021-06-01"), _stamp("2023-06-01"),
          _stamp("2023-06-01"), _stamp("2023-12-01")),
    Split("fold_2", "walk_forward", _stamp("2021-06-01"), _stamp("2023-12-01"),
          _stamp("2023-12-01"), _stamp("2024-06-01")),
    Split("fold_3", "walk_forward", _stamp("2021-06-01"), _stamp("2024-06-01"),
          _stamp("2024-06-01"), _stamp("2024-12-01")),
    Split("fold_4", "walk_forward", _stamp("2021-06-01"), _stamp("2024-12-01"),
          _stamp("2024-12-01"), _stamp("2025-06-01")),
)
FINAL_SPLIT = Split(
    "final", "final", _stamp("2021-06-01"), _stamp("2025-06-01"),
    _stamp("2025-06-01"), _stamp("2026-06-01"),
)
ALL_SPLITS = WALK_FORWARD_SPLITS + (FINAL_SPLIT,)


@dataclass(frozen=True)
class MarketData:
    """Every hourly decision in the study, unnormalised.

    ``fast`` is ``[decisions, 12, 4]`` and ``slow`` is ``[decisions, 7]``.
    ``entry_price``/``exit_price`` are the five-minute opens at the start and
    end of the holding interval.  Funding is split by which position owns it:
    ``funding_on_previous`` is settled at the very instant the new order
    executes and therefore belongs to the position held before it.
    """

    execution_time: pd.DatetimeIndex  # tau_t, the top of the hour the order fills
    decision_time: pd.DatetimeIndex  # the close of the twelfth bar informing it
    fast: np.ndarray
    slow: np.ndarray
    entry_price: np.ndarray
    exit_price: np.ndarray
    funding_on_previous: np.ndarray
    funding_on_current: np.ndarray
    slow_names: tuple[str, ...]
    audit: dict

    def __len__(self) -> int:
        return int(self.fast.shape[0])

    def index_range(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
        """Half-open decision index range whose holding interval starts in ``[start, end)``."""
        first = int(self.execution_time.searchsorted(start, "left"))
        last = int(self.execution_time.searchsorted(end, "left"))
        return first, last


def _hourly_from_bars(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = bars["close"].to_numpy(dtype=np.float64)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    quote = bars["quote_volume"].to_numpy(dtype=np.float64)
    taker = bars["taker_buy_quote"].to_numpy(dtype=np.float64)

    log_return = np.zeros_like(close)
    log_return[1:] = np.log(close[1:] / close[:-1])
    log_range = np.log(high / low)
    padded_quote = quote + VOLUME_EPSILON
    log_quote_change = np.zeros_like(close)
    log_quote_change[1:] = np.log(padded_quote[1:] / padded_quote[:-1])
    imbalance = np.where(quote > 0.0, (2.0 * taker - quote) / np.maximum(quote, VOLUME_EPSILON), 0.0)

    bar_features = np.stack([log_return, log_range, log_quote_change, imbalance], axis=1)
    return bar_features, close, log_return


def build_market(output_dir: Path | None = None, funding_features: bool = True) -> MarketData:
    bars, funding = load() if output_dir is None else load(output_dir)
    bar_features, close, log_return = _hourly_from_bars(bars)
    open_price = bars["open"].to_numpy(dtype=np.float64)
    bar_time = pd.DatetimeIndex(bars["open_time"])

    total_hours = len(bars) // BARS_PER_HOUR
    if total_hours * BARS_PER_HOUR != len(bars):
        raise ValueError("bar grid does not divide into whole hours")

    # Hour h owns bars [12h, 12h+12); its close is the last of them.
    hourly_close = close[BARS_PER_HOUR - 1 :: BARS_PER_HOUR]
    squared_return = log_return * log_return
    cumulative = np.concatenate([[0.0], np.cumsum(squared_return)])

    # A decision on hour h needs the open of hour h+2, so h runs to total-3.
    first_hour = WARMUP_HOURS_REQUIRED
    last_hour = total_hours - 3
    hours = np.arange(first_hour, last_hour + 1, dtype=np.int64)

    bar_index = hours[:, None] * BARS_PER_HOUR + np.arange(BARS_PER_HOUR)[None, :]
    fast = bar_features[bar_index].astype(np.float64)

    def momentum(lag_hours: int) -> np.ndarray:
        return np.log(hourly_close[hours] / hourly_close[hours - lag_hours])

    end_bar = hours * BARS_PER_HOUR + BARS_PER_HOUR  # exclusive
    start_bar = end_bar - BARS_PER_DAY
    realised_volatility = np.sqrt(cumulative[end_bar] - cumulative[start_bar])

    execution_time = bar_time[(hours + 1) * BARS_PER_HOUR]
    decision_time = pd.DatetimeIndex(bars["close_time"])[hours * BARS_PER_HOUR + BARS_PER_HOUR - 1]

    settlement_time = pd.DatetimeIndex(funding["funding_time"]).values
    settlement_rate = funding["funding_rate"].to_numpy(dtype=np.float64)
    execution_values = execution_time.values

    # The latest rate that had already settled strictly before the decision is
    # taken.  A settlement stamped at the execution instant has not happened yet
    # when the decision is made, so the comparison is strict.
    previous = np.searchsorted(settlement_time, execution_values, "left") - 1
    if previous.min() < 0:
        raise ValueError("a decision precedes the first known funding settlement")
    last_settled = settlement_rate[previous]

    columns = [
        momentum(3), momentum(5), momentum(24), momentum(72), momentum(240),
        realised_volatility, last_settled,
    ]

    # Funding owned by the position held *before* the order executes: any
    # settlement stamped exactly at tau_t.  Funding owned by the new position:
    # any settlement strictly inside (tau_t, tau_{t+1}).
    next_execution = np.empty_like(execution_values)
    next_execution[:-1] = execution_values[1:]
    next_execution[-1] = execution_values[-1] + np.timedelta64(1, "h")
    at_or_after = np.searchsorted(settlement_time, execution_values, "left")
    after = np.searchsorted(settlement_time, execution_values, "right")
    before_next = np.searchsorted(settlement_time, next_execution, "left")
    cumulative_rate = np.concatenate([[0.0], np.cumsum(settlement_rate)])
    funding_on_previous = cumulative_rate[after] - cumulative_rate[at_or_after]
    funding_on_current = cumulative_rate[before_next] - cumulative_rate[after]

    if funding_features:
        # A settlement stamped exactly at tau_{t+1} is charged to the position
        # opened at tau_t.  Only the *timing* is read here, and the timing is a
        # fixed eight-hour UTC schedule known long in advance.
        next_at_or_after = np.searchsorted(settlement_time, next_execution, "left")
        next_after = np.searchsorted(settlement_time, next_execution, "right")
        settlement_next = (next_after > next_at_or_after).astype(np.float64)
        # Hours from execution to the first settlement strictly after it,
        # expressed in units of the eight-hour cycle.  Decisions past the last
        # known settlement have no "next" one in the table; they are given a
        # full cycle, which is what the fixed schedule implies anyway.
        upcoming = np.minimum(after, len(settlement_time) - 1)
        delta = (settlement_time[upcoming] - execution_values).astype("timedelta64[m]")
        hours_to_settlement = delta.astype(np.float64) / 60.0 / 8.0
        hours_to_settlement[after >= len(settlement_time)] = 1.0
        if not ((hours_to_settlement > 0.0) & (hours_to_settlement <= 1.0)).all():
            raise ValueError("settlement countdown outside one eight-hour cycle")
        columns.extend([settlement_next, hours_to_settlement, last_settled * settlement_next])

    slow = np.stack(columns, axis=1)
    names = slow_feature_names(funding_features)

    entry_price = open_price[(hours + 1) * BARS_PER_HOUR]
    exit_price = open_price[(hours + 2) * BARS_PER_HOUR]

    study_first = int(execution_time.searchsorted(STUDY_START, "left"))
    study_last = int(execution_time.searchsorted(STUDY_END, "left"))
    audit = {
        "decisions_total": int(len(hours)),
        "decisions_in_study_window": int(study_last - study_first),
        "first_execution_utc": execution_time[0].isoformat(),
        "last_execution_utc": execution_time[-1].isoformat(),
        "fast_feature_names": list(FAST_FEATURE_NAMES),
        "slow_feature_names": list(names),
        "position_one_hot": list(POSITION_NAMES),
        "observation_bars_per_decision": BARS_PER_HOUR,
        "settlements_charged_to_previous_position": int(
            np.count_nonzero(funding_on_previous)
        ),
        "settlements_charged_to_new_position": int(np.count_nonzero(funding_on_current)),
        "settlement_rate_total_attributed": float(
            funding_on_previous.sum() + funding_on_current.sum()
        ),
        "features_end_before_execution_price_starts": True,
        "warmup_hours_required": WARMUP_HOURS_REQUIRED,
    }
    if not np.isfinite(fast).all() or not np.isfinite(slow).all():
        raise ValueError("non-finite feature produced")
    if not (np.isfinite(entry_price).all() and np.isfinite(exit_price).all()):
        raise ValueError("non-finite execution price")

    return MarketData(
        execution_time=execution_time,
        decision_time=decision_time,
        fast=fast,
        slow=slow,
        entry_price=entry_price,
        exit_price=exit_price,
        funding_on_previous=funding_on_previous,
        funding_on_current=funding_on_current,
        slow_names=names,
        audit=audit,
    )


@dataclass(frozen=True)
class Normaliser:
    """Per-feature z-score fitted on one training window and then frozen."""

    fast_mean: np.ndarray
    fast_std: np.ndarray
    slow_mean: np.ndarray
    slow_std: np.ndarray
    slow_names: tuple[str, ...]
    clip: float
    fitted_on: dict

    def apply(self, fast: np.ndarray, slow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalised_fast = np.clip(
            (fast - self.fast_mean) / self.fast_std, -self.clip, self.clip
        ).astype(np.float32)
        normalised_slow = np.clip(
            (slow - self.slow_mean) / self.slow_std, -self.clip, self.clip
        ).astype(np.float32)
        return normalised_fast, normalised_slow

    def as_dict(self) -> dict:
        return {
            "fitted_on": self.fitted_on,
            "clip_sigma": self.clip,
            "fast": {
                name: {"mean": float(m), "std": float(s)}
                for name, m, s in zip(FAST_FEATURE_NAMES, self.fast_mean, self.fast_std)
            },
            "slow": {
                name: {"mean": float(m), "std": float(s)}
                for name, m, s in zip(self.slow_names, self.slow_mean, self.slow_std)
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
        fast_mean=fast.mean(axis=0),
        fast_std=fast_std,
        slow_mean=slow.mean(axis=0),
        slow_std=slow_std,
        slow_names=market.slow_names,
        clip=NORMALISED_CLIP,
        fitted_on={
            "split": split.name,
            "start_inclusive": split.train_start.isoformat(),
            "end_exclusive": split.train_end.isoformat(),
            "decisions": int(last - first),
        },
    )


@dataclass(frozen=True)
class SplitData:
    """One split's normalised tensors plus the index ranges that define it."""

    split: Split
    market: MarketData
    normaliser: Normaliser
    fast: np.ndarray
    slow: np.ndarray
    train_range: tuple[int, int]
    evaluation_range: tuple[int, int]
    evaluation_warmup_start: int

    def observation_size(self, action_count: int = len(POSITION_NAMES)) -> int:
        """Width of the head input: slow features plus the position one-hot."""
        return int(self.slow.shape[1] + action_count)

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
    if warmup_start < train_range[0]:
        raise ValueError("evaluation warm-up would start before the training window")
    if evaluation_range[0] != train_range[1]:
        raise ValueError("evaluation window does not begin where training ends")
    return SplitData(
        split=split,
        market=market,
        normaliser=normaliser,
        fast=fast,
        slow=slow,
        train_range=train_range,
        evaluation_range=evaluation_range,
        evaluation_warmup_start=warmup_start,
    )


def split_by_name(name: str) -> Split:
    for split in ALL_SPLITS:
        if split.name == name:
            return split
    raise KeyError(f"unknown split {name!r}")


def main() -> None:
    market = build_market()
    report = {"market": market.audit, "splits": []}
    for split in ALL_SPLITS:
        report["splits"].append(build_split(market, split).as_dict())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
