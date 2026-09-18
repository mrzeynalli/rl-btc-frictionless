"""The hourly trading environment: three target positions, honest equity.

One decision per hour.  The action *is* the target position, so ``FLAT`` is the
close action and no separate close exists.  Exposure is fixed at 1x notional --
no leverage choice, no partial position, no conviction multiplier, no
volatility targeting.

Over the holding interval ``[tau_t, tau_{t+1})`` the account earns the price
P&L of the position it holds, pays or receives every funding settlement that
falls in the interval attributed to whichever position was open immediately
before it, and pays ``c |q_t - q_{t-1}|`` on executed turnover.  Equity
compounds by that realised fraction and the reward is the log of the equity
ratio.  A single constant multiplier is applied to every reward alike, purely
so the critic sees numbers of order one.

Nothing else touches the reward: no differential Sharpe, no local-volatility
division, no reward EWMA, no clipping beyond a numerical floor that has never
fired, no turnover penalty on top of the real cost, no edge gate, no conviction
anchor, no shadow portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 3 bp fee plus 1 bp slippage, charged one way on executed turnover.
FEE_RATE = 0.0003
SLIPPAGE_RATE = 0.0001
ONE_WAY_COST = FEE_RATE + SLIPPAGE_RATE

ACTION_NAMES = ("SHORT", "FLAT", "LONG")
ACTION_POSITIONS = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
FLAT_ACTION = 1

# A finer ladder of target exposures.  Mean-variance sizing puts the optimal
# position at a small fraction of notional when the edge is of the order of the
# measured 1-hour IC, and the three-way action set cannot express that: it
# offers nothing between flat and a full 1x.  A ladder lets the agent hold a
# small position and, just as importantly, lets it *adjust* one for a fraction
# of the turnover a full reversal costs.
LADDER_POSITIONS = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0], dtype=np.float64)


def position_ladder(name: str) -> np.ndarray:
    if name == "ternary":
        return ACTION_POSITIONS
    if name == "ladder":
        return LADDER_POSITIONS
    raise KeyError(f"unknown position ladder {name!r}")

# Rewards are hourly log-equity changes of order 1e-3.  One fixed multiplier,
# identical at every step and in every arm, keeps the value targets near unit
# scale without changing a single relative preference.
REWARD_SCALE = 100.0

# Equity is compounded arithmetically -- E_{t+1} = E_t (1 + q r + F - C) with r
# the simple price return -- because that is the P&L a 1x notional position
# actually realises.  The log-return form in the specification agrees with it
# exactly for LONG and to second order for SHORT.
EQUITY_ACCOUNTING = "arithmetic"
EQUITY_FLOOR = 1e-6


@dataclass(frozen=True)
class EnvironmentConfig:
    episode_hours: int = 512
    cost_rate: float = ONE_WAY_COST
    reward_scale: float = REWARD_SCALE
    gamma_free_terminal: bool = True  # episode ends are truncations, not failures
    position_ladder: str = "ternary"  # "ternary" or "ladder"
    burn_in_hours: int = 0


def realised_return(
    position: np.ndarray,
    previous_position: np.ndarray,
    price_ratio: np.ndarray,
    funding_on_previous: np.ndarray,
    funding_on_current: np.ndarray,
    cost_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose one holding interval into price, funding, cost and total.

    ``price_ratio`` is ``P_{tau_{t+1}} / P_{tau_t}``.  ``funding_on_previous``
    is the total rate settled at the very instant the order executes, which the
    position held before the order pays; ``funding_on_current`` is everything
    settled strictly inside the interval, which the new position pays.  A
    positive rate with a long position is a cash outflow.
    """
    price = position * (price_ratio - 1.0)
    funding = -(previous_position * funding_on_previous + position * funding_on_current)
    cost = cost_rate * np.abs(position - previous_position)
    return price, funding, cost, price + funding - cost


class VectorHourlyEnvironment:
    """Independent hourly trading episodes at random offsets, stepped together.

    Every environment holds its own equity, position and cursor into the same
    decision table.  Episodes are a fixed number of hours long and start at a
    uniformly random offset inside the training window, so a rollout is not one
    long walk down a single path through history.
    """

    def __init__(
        self,
        fast: np.ndarray,
        slow: np.ndarray,
        entry_price: np.ndarray,
        exit_price: np.ndarray,
        funding_on_previous: np.ndarray,
        funding_on_current: np.ndarray,
        index_range: tuple[int, int],
        environments: int,
        config: EnvironmentConfig,
        rng: np.random.Generator,
    ):
        self.fast = fast
        self.slow = slow
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.funding_on_previous = funding_on_previous
        self.funding_on_current = funding_on_current
        self.first, self.last = index_range
        self.environments = int(environments)
        self.config = config
        self.rng = rng

        self.episode_hours = int(config.episode_hours)
        self.burn_in_hours = int(config.burn_in_hours)
        if self.burn_in_hours < 0:
            raise ValueError("burn_in_hours cannot be negative")
        if self.last - self.first <= self.episode_hours + self.burn_in_hours:
            raise ValueError("training window is shorter than one episode")
        # The observation handed to the heads is the slow market block plus the
        # three-way one-hot of the position currently held.
        self.positions = position_ladder(config.position_ladder)
        self.action_count = int(len(self.positions))
        self.slow_width = int(slow.shape[1]) + self.action_count

        self.cursor = np.zeros(self.environments, dtype=np.int64)
        self.position = np.zeros(self.environments, dtype=np.float64)
        self.equity = np.ones(self.environments, dtype=np.float64)
        self.step_in_episode = np.zeros(self.environments, dtype=np.int64)
        self.turnover = np.zeros(self.environments, dtype=np.float64)
        self.equity_floor_events = 0
        self.reset_all()

    # -- episode management -------------------------------------------------

    def _sample_offsets(self, count: int) -> np.ndarray:
        high = self.last - self.episode_hours
        low = self.first + self.burn_in_hours
        return self.rng.integers(low, high, size=count, dtype=np.int64)

    def burn_in_fast(self) -> np.ndarray:
        """Fast observations immediately preceding each current cursor."""
        if self.burn_in_hours == 0:
            return np.empty(
                (self.environments, 0, self.fast.shape[1], self.fast.shape[2]),
                dtype=self.fast.dtype,
            )
        offsets = np.arange(self.burn_in_hours, 0, -1, dtype=np.int64)
        indices = self.cursor[:, None] - offsets[None, :]
        if indices.min() < self.first:
            raise RuntimeError("burn-in reached outside the training window")
        return self.fast[indices]

    def reset_all(self) -> None:
        self.cursor[:] = self._sample_offsets(self.environments)
        self.position[:] = 0.0
        self.equity[:] = 1.0
        self.step_in_episode[:] = 0
        self.turnover[:] = 0.0

    def _reset_where(self, mask: np.ndarray) -> None:
        count = int(mask.sum())
        if count == 0:
            return
        self.cursor[mask] = self._sample_offsets(count)
        self.position[mask] = 0.0
        self.equity[mask] = 1.0
        self.step_in_episode[mask] = 0
        self.turnover[mask] = 0.0

    # -- observation --------------------------------------------------------

    def observation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(fast[K, 12, 4], slow_with_position[K, slow + 3])``."""
        cursor = np.minimum(self.cursor, len(self.fast) - 1)
        fast = self.fast[cursor]
        slow = self.slow[cursor]
        one_hot = np.zeros((self.environments, self.action_count), dtype=np.float32)
        # The held position is always one of the ladder's levels exactly, so
        # searchsorted on the sorted ladder recovers its index without a
        # tolerance.
        held = np.searchsorted(self.positions, self.position)
        one_hot[np.arange(self.environments), held] = 1.0
        return fast, np.concatenate([slow, one_hot], axis=1)

    # -- transition ---------------------------------------------------------

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        target = self.positions[actions]
        previous = self.position.copy()
        index = self.cursor
        price_ratio = self.exit_price[index] / self.entry_price[index]

        price, funding, cost, total = realised_return(
            target, previous, price_ratio,
            self.funding_on_previous[index], self.funding_on_current[index],
            self.config.cost_rate,
        )
        _, _, hypothetical_cost, _ = realised_return(
            target, previous, price_ratio,
            self.funding_on_previous[index], self.funding_on_current[index],
            ONE_WAY_COST,
        )

        growth = 1.0 + total
        floored = np.maximum(growth, EQUITY_FLOOR)
        self.equity_floor_events += int((growth < EQUITY_FLOOR).sum())
        reward = np.log(floored) * self.config.reward_scale

        self.equity *= floored
        self.position = target
        self.turnover += np.abs(target - previous)
        self.cursor = index + 1
        self.step_in_episode += 1

        done = self.step_in_episode >= self.episode_hours
        done |= self.cursor >= self.last
        # The observation the episode would have continued into, captured before
        # any reset so a truncated episode can still be bootstrapped honestly.
        continuation_fast, continuation_slow = self.observation()
        info = {
            "price_return": price,
            "funding_return": funding,
            "cost_paid": cost,
            "hypothetical_cost": hypothetical_cost,
            "total_return": total,
            "turnover": np.abs(target - previous),
            "equity": self.equity.copy(),
            "position": self.position.copy(),
            "continuation_fast": continuation_fast,
            "continuation_slow": continuation_slow,
        }
        self._reset_where(done)
        return reward.astype(np.float64), done.copy(), info
