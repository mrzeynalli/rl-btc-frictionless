"""Causal long-only BTC Spot environment with BUY, SELL, and HOLD orders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEE_RATE = 0.0003
SLIPPAGE_RATE = 0.0001
ONE_WAY_COST = FEE_RATE + SLIPPAGE_RATE
BUY, SELL, HOLD = 0, 1, 2
ACTION_NAMES = ("BUY", "SELL", "HOLD")
ACTION_COUNT = len(ACTION_NAMES)
PORTFOLIO_FEATURE_NAMES = (
    "btc_position",
    "open_trade_log_return",
    "accumulated_open_loss",
    "open_trade_drawdown",
    "log_time_held",
    "recent_downside_deviation",
    "recent_turnover",
)
PORTFOLIO_WIDTH = len(PORTFOLIO_FEATURE_NAMES)
REWARD_SCALE = 100.0
EQUITY_FLOOR = 1e-6


def valid_action_mask(position: np.ndarray) -> np.ndarray:
    """Cash permits BUY/HOLD; BTC permits SELL/HOLD."""
    held = np.asarray(position, dtype=np.float64)
    if not np.isin(held, (0.0, 1.0)).all():
        raise ValueError("Spot position must be cash (0) or BTC (1)")
    mask = np.zeros(held.shape + (ACTION_COUNT,), dtype=bool)
    mask[..., BUY] = held == 0.0
    mask[..., SELL] = held == 1.0
    mask[..., HOLD] = True
    return mask


def action_targets(actions: np.ndarray, previous: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    previous = np.asarray(previous, dtype=np.float64)
    if actions.shape != previous.shape:
        raise ValueError("actions and positions must have the same shape")
    mask = valid_action_mask(previous)
    if not mask[np.arange(len(actions)), actions].all():
        raise ValueError("invalid Spot order for current inventory")
    return np.where(actions == BUY, 1.0, np.where(actions == SELL, 0.0, previous))


def realised_return(
    position: np.ndarray,
    previous_position: np.ndarray,
    price_ratio: np.ndarray,
    cost_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return price P&L, transaction cost, and their net simple return."""
    price = position * (price_ratio - 1.0)
    cost = float(cost_rate) * np.abs(position - previous_position)
    return price, cost, price - cost


@dataclass(frozen=True)
class EnvironmentConfig:
    episode_hour_options: tuple[int, ...] = (720, 1440, 2160, 2880, 3600)
    cost_rate: float = ONE_WAY_COST
    reward_scale: float = REWARD_SCALE
    burn_in_hours: int = 240
    randomize_starting_position: bool = True
    regime_balanced_sampling: bool = True
    downside_decay: float = 0.95
    turnover_decay: float = 0.95


class VectorHourlyEnvironment:
    """Parallel, randomly located, causal Spot episodes.

    Episode lengths range from one to five months where the native data permit.
    Starts are balanced over ex-post trend/volatility strata formed only inside
    the training window.  These strata select training blocks and are never
    supplied to the policy.
    """

    def __init__(
        self,
        fast: np.ndarray,
        slow: np.ndarray,
        entry_price: np.ndarray,
        exit_price: np.ndarray,
        segment_start: np.ndarray,
        index_range: tuple[int, int],
        environments: int,
        config: EnvironmentConfig,
        rng: np.random.Generator,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.segment_start = np.asarray(segment_start, dtype=bool)
        self.first, self.last = index_range
        self.environments = int(environments)
        self.config = config
        self.rng = rng
        self.burn_in_hours = int(config.burn_in_hours)
        if not 0 <= self.burn_in_hours < self.last - self.first:
            raise ValueError("invalid burn-in for training window")
        self.slow_width = int(slow.shape[1])
        self.action_count = ACTION_COUNT

        self._segment_ids = np.cumsum(self.segment_start) - 1
        self._candidate_pools = self._build_candidate_pools()
        self.available_episode_hours = tuple(sorted(self._candidate_pools))
        if not self.available_episode_hours:
            raise ValueError("no contiguous training segment can hold an episode")

        size = self.environments
        self.cursor = np.zeros(size, dtype=np.int64)
        self.position = np.zeros(size, dtype=np.float64)
        self.equity = np.ones(size, dtype=np.float64)
        self.step_in_episode = np.zeros(size, dtype=np.int64)
        self.episode_target = np.zeros(size, dtype=np.int64)
        self.open_trade_log_return = np.zeros(size, dtype=np.float64)
        self.open_trade_peak = np.zeros(size, dtype=np.float64)
        self.time_held = np.zeros(size, dtype=np.int64)
        self.downside_variance = np.zeros(size, dtype=np.float64)
        self.turnover_ewma = np.zeros(size, dtype=np.float64)
        self.turnover = np.zeros(size, dtype=np.float64)
        self.start_position_counts = np.zeros(2, dtype=np.int64)
        self.episode_length_counts = {str(item): 0 for item in self.available_episode_hours}
        self.regime_sample_counts = np.zeros(6, dtype=np.int64)
        self.reset_all()

    def _build_candidate_pools(self) -> dict[int, tuple[np.ndarray, ...]]:
        pools: dict[int, tuple[np.ndarray, ...]] = {}
        indices = np.arange(self.first + self.burn_in_hours, self.last, dtype=np.int64)
        for length in sorted(set(int(item) for item in self.config.episode_hour_options)):
            if length <= 0:
                continue
            end = indices + length
            valid = end < self.last
            safe_end = np.minimum(end, self.last - 1)
            valid &= (
                self._segment_ids[indices - self.burn_in_hours]
                == self._segment_ids[safe_end]
            )
            starts = indices[valid]
            if not len(starts):
                continue
            if not self.config.regime_balanced_sampling:
                pools[length] = tuple(starts.copy() for _ in range(6))
                continue
            block_return = np.log(self.exit_price[starts + length - 1] / self.entry_price[starts])
            first_price = self.entry_price[starts]
            midpoint = self.entry_price[starts + length // 2]
            variation = np.abs(np.log(midpoint / first_price)) + np.abs(
                np.log(self.exit_price[starts + length - 1] / midpoint)
            )
            lower, upper = np.quantile(block_return, [1 / 3, 2 / 3])
            volatility_cut = float(np.median(variation))
            trend = np.where(block_return <= lower, 0, np.where(block_return >= upper, 2, 1))
            volatility = (variation > volatility_cut).astype(np.int64)
            strata = trend * 2 + volatility
            grouped = tuple(starts[strata == group] for group in range(6))
            if all(len(group) for group in grouped):
                pools[length] = grouped
        return pools

    def _sample_episodes(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.asarray(self.available_episode_hours, dtype=np.int64)
        selected_lengths = self.rng.choice(lengths, size=count, replace=True)
        starts = np.empty(count, dtype=np.int64)
        for index, length in enumerate(selected_lengths):
            if self.config.regime_balanced_sampling:
                least = np.flatnonzero(self.regime_sample_counts == self.regime_sample_counts.min())
                stratum = int(self.rng.choice(least))
            else:
                stratum = int(self.rng.integers(0, 6))
            pool = self._candidate_pools[int(length)][stratum]
            starts[index] = int(self.rng.choice(pool))
            self.regime_sample_counts[stratum] += 1
            self.episode_length_counts[str(int(length))] += 1
        return starts, selected_lengths

    def _balanced_positions(self, count: int) -> np.ndarray:
        if not self.config.randomize_starting_position:
            return np.zeros(count, dtype=np.float64)
        positions = np.arange(count, dtype=np.int64) % 2
        offset = int(self.rng.integers(0, 2))
        positions = (positions + offset) % 2
        self.rng.shuffle(positions)
        self.start_position_counts += np.bincount(positions, minlength=2)
        return positions.astype(np.float64)

    def burn_in_fast(self, mask: np.ndarray | None = None) -> np.ndarray:
        offsets = np.arange(self.burn_in_hours, 0, -1, dtype=np.int64)
        cursor = self.cursor if mask is None else self.cursor[np.asarray(mask, dtype=bool)]
        return self.fast[cursor[:, None] - offsets[None, :]]

    def _reset_state(self, mask: np.ndarray) -> None:
        count = int(mask.sum())
        if not count:
            return
        starts, lengths = self._sample_episodes(count)
        self.cursor[mask] = starts
        self.episode_target[mask] = lengths
        self.position[mask] = self._balanced_positions(count)
        self.equity[mask] = 1.0
        self.step_in_episode[mask] = 0
        self.open_trade_log_return[mask] = 0.0
        self.open_trade_peak[mask] = 0.0
        self.time_held[mask] = 0
        self.downside_variance[mask] = 0.0
        self.turnover_ewma[mask] = 0.0
        self.turnover[mask] = 0.0

    def reset_all(self) -> None:
        self._reset_state(np.ones(self.environments, dtype=bool))

    def portfolio_features(self) -> np.ndarray:
        loss = np.maximum(-self.open_trade_log_return, 0.0)
        drawdown = np.maximum(self.open_trade_peak - self.open_trade_log_return, 0.0)
        longest = max(self.config.episode_hour_options)
        return np.stack([
            self.position,
            np.clip(self.open_trade_log_return, -1.0, 1.0),
            np.clip(loss, 0.0, 1.0),
            np.clip(drawdown, 0.0, 1.0),
            np.log1p(self.time_held) / np.log1p(longest),
            np.clip(np.sqrt(self.downside_variance) * 100.0, 0.0, 5.0),
            np.clip(self.turnover_ewma, 0.0, 1.0),
        ], axis=1).astype(np.float32)

    def observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.fast[self.cursor],
            self.slow[self.cursor],
            self.portfolio_features(),
            valid_action_mask(self.position),
        )

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        previous = self.position.copy()
        target = action_targets(actions, previous)
        index = self.cursor.copy()
        price_ratio = self.exit_price[index] / self.entry_price[index]
        price, cost, total = realised_return(
            target, previous, price_ratio, self.config.cost_rate
        )
        growth = np.maximum(1.0 + total, EQUITY_FLOOR)
        base_log_return = np.log(growth)

        continuing = target == 1.0
        newly_opened = continuing & (previous == 0.0)
        continued = continuing & (previous == 1.0)
        next_trade_return = np.zeros(self.environments, dtype=np.float64)
        next_trade_return[newly_opened] = base_log_return[newly_opened]
        next_trade_return[continued] = (
            self.open_trade_log_return[continued] + base_log_return[continued]
        )
        # The learning reward is exactly the realised net economic return.
        # ``base_log_return`` includes c * |p_t - p_(t-1)| through ``total``;
        # HOLD therefore pays no cost, while BUY and SELL each pay one leg.
        reward = base_log_return * self.config.reward_scale

        next_peak = np.zeros(self.environments, dtype=np.float64)
        next_peak[newly_opened] = np.maximum(0.0, next_trade_return[newly_opened])
        next_peak[continued] = np.maximum(
            self.open_trade_peak[continued], next_trade_return[continued]
        )
        self.open_trade_log_return = next_trade_return
        self.open_trade_peak = next_peak
        self.time_held = np.where(continuing, self.time_held + 1, 0)
        negative = np.minimum(base_log_return, 0.0)
        self.downside_variance = (
            self.config.downside_decay * self.downside_variance
            + (1.0 - self.config.downside_decay) * negative * negative
        )
        turnover = np.abs(target - previous)
        self.turnover_ewma = (
            self.config.turnover_decay * self.turnover_ewma
            + (1.0 - self.config.turnover_decay) * turnover
        )
        self.equity *= growth
        self.position = target
        self.turnover += turnover
        self.cursor += 1
        self.step_in_episode += 1
        done = self.step_in_episode >= self.episode_target

        continuation = self.observation()
        info = {
            "price_return": price,
            "cost_paid": cost,
            "economic_log_return": base_log_return,
            "turnover": turnover,
            "equity": self.equity.copy(),
            "position": self.position.copy(),
            "previous_position": previous,
            "continuation_fast": continuation[0],
            "continuation_slow": continuation[1],
            "continuation_portfolio": continuation[2],
            "continuation_mask": continuation[3],
        }
        self._reset_state(done)
        return reward.astype(np.float64), done.copy(), info
