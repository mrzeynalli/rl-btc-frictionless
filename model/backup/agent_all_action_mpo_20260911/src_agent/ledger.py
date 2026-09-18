"""Vector inverse-contract ledger and the endogenous half of the observation.

Orders last for one decision interval and fill at its first regular-trade open.
Funding at a minute boundary is booked before that minute's order. Portfolio
marking uses the minute close; OHLC adverse extremes enforce a conservative
margin approximation, not a claim to reconstruct BitMEX's liquidation engine.

The account observation is the smallest set that makes reward and termination
Markov given the market (an information state in the sense of Subramanian et al.,
JMLR 2022): which side is held, how far the open trade has moved, the exposure
that decides liquidation, and the two Sharpe statistics the reward depends on.

Because a one-contract trader does not move the market, the outcome of every
action on a bar can be computed exactly on copies of an account: Huang (2018)
calls this action augmentation and rests it on the "zero market impact
hypothesis". `advance` is the single piece of accounting used both by the
environment and by those counterfactual branches.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .reward import differential_sharpe, observation_features

SHORT, FLAT, LONG = 0, 1, 2
ACTION_NAMES = ("SHORT", "FLAT", "LONG")
SIDES = np.array([-1, 0, 1])
PORTFOLIO_NAMES = ("short", "flat", "long",
                   # Open trade's log return, in units of the one-day ex-ante volatility.
                   "trade_return",
                   # log(1 + |contracts| / price / equity). Liquidation is exactly the
                   # event that this exceeds log(1 + 1 / (maintenance + fee)).
                   "exposure",
                   "running_sharpe", "relative_volatility")
PORTFOLIO_WIDTH = len(PORTFOLIO_NAMES)
EPS = 1e-12


class Ledger:
    def __init__(self, size, config=Config()):
        self.size, self.config = size, config
        for name in ("wallet", "contracts", "entry", "equity", "trade_start", "age",
                     "fees_total", "funding_total", "turnover_total", "dsr_mean", "dsr_second"):
            setattr(self, name, np.zeros(size, dtype=np.float64))
        self.terminated = np.zeros(size, dtype=bool)
        self.orders = np.zeros(size, dtype=np.int64)
        self.liquidations = np.zeros(size, dtype=np.int64)
        self.reset(np.ones(size, bool))

    def reset(self, mask):
        for name in ("wallet", "equity", "trade_start"):
            getattr(self, name)[mask] = 1.0
        for name in ("contracts", "entry", "age", "fees_total", "funding_total", "turnover_total",
                     "orders", "liquidations", "dsr_mean"):
            getattr(self, name)[mask] = 0
        self.terminated[mask] = False
        # A fresh account starts with no measured edge and the asset's own
        # five-minute variance, so the ratio is defined from the first bar.
        self.dsr_second[mask] = self.config.dsr_initial_second_moment

    def value(self, mark):
        safe_entry = np.where(self.contracts != 0, self.entry, mark)
        return self.wallet + self.contracts * (1 / safe_entry - 1 / mark)

    def funding_event(self, rate, mark):
        payment = np.where(self.terminated, 0.0, -self.contracts / mark * rate)
        self.wallet += payment
        self.funding_total += payment
        return payment

    def _close(self, mask, price):
        quantity = self.contracts[mask]
        if not len(quantity):
            return
        # Close a short with a buy; close a long with a sell.
        fill = price[mask] * (1 - np.sign(quantity) * self.config.slippage)
        fee = np.abs(quantity) / fill * self.config.fee
        self.wallet[mask] += quantity * (1 / self.entry[mask] - 1 / fill) - fee
        self.fees_total[mask] += fee
        # Notional relative to the equity the trade started with. The account's
        # current value is not used: at a liquidation it is near zero, and the
        # predecessor booked turnover of up to 2.6e12 per event dividing by it.
        self.turnover_total[mask] += np.abs(quantity) / fill / np.maximum(self.trade_start[mask], EPS)
        self.contracts[mask] = 0
        self.entry[mask] = 0
        self.orders[mask] += 1

    def execute(self, actions, price, eligible):
        target = SIDES[np.asarray(actions, dtype=np.int64)]
        previous = np.sign(self.contracts)
        change = eligible & ~self.terminated & (target != previous)
        self._close(change & (previous != 0), price)
        opening = change & (target != 0) & (self.wallet > EPS)
        if opening.any():
            fill = price[opening] * (1 + target[opening] * self.config.slippage)
            # Fixed USD contract quantity at entry; no uncharged rebalancing.
            quantity = target[opening] * np.floor(self.wallet[opening] * price[opening] * self.config.entry_leverage)
            fee = np.abs(quantity) / fill * self.config.fee
            self.trade_start[opening] = self.wallet[opening]
            self.wallet[opening] -= fee
            self.contracts[opening] = quantity
            self.entry[opening] = fill
            self.age[opening] = 0
            self.fees_total[opening] += fee
            self.turnover_total[opening] += np.abs(quantity) / fill / self.trade_start[opening]
            self.orders[opening] += (quantity != 0)
        closed = change & (target == 0)
        self.age[closed] = 0
        self.trade_start[closed] = np.maximum(self.wallet[closed], EPS)
        return change

    def mark(self, price, adverse=None, elapsed=True):
        active = ~self.terminated
        worst = price if adverse is None else adverse
        worst_value = self.value(worst)
        requirement = np.abs(self.contracts) / worst * (self.config.maintenance + self.config.fee)
        breached = active & ((worst_value <= requirement) | (worst_value <= EPS))
        if breached.any():
            self._close(breached & (self.contracts != 0), worst)
            self.wallet[breached] = np.maximum(self.wallet[breached], EPS)
            self.terminated[breached] = True
            self.liquidations[breached] += 1
        self.equity = np.maximum(self.value(price), EPS)
        self.age[self.contracts != 0] += int(elapsed)

    def portfolio(self, mark, daily_volatility):
        """Endogenous observation: side, open-trade move, exposure, reward state."""
        equity = np.maximum(self.value(mark), EPS)
        held = self.contracts != 0
        trade = np.where(held, np.log(equity / np.maximum(self.trade_start, EPS)), 0.0)
        volatility = np.broadcast_to(np.asarray(daily_volatility, dtype=np.float64), trade.shape)
        trade = np.where(volatility > 0, trade / np.where(volatility > 0, volatility, 1.0), 0.0)
        exposure = np.log1p(np.abs(self.contracts) / mark / equity)
        continuous = np.stack([trade, exposure, *observation_features(self.dsr_mean, self.dsr_second, self.config)],
                              axis=1)
        continuous = np.clip(continuous, -self.config.observation_clip, self.config.observation_clip)
        side = np.sign(self.contracts).astype(np.int64) + 1
        return np.concatenate([np.eye(3)[side], continuous], axis=1).astype(np.float32)

    def state_dict(self):
        return {name: value.copy() for name, value in vars(self).items() if isinstance(value, np.ndarray)}

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value.copy())


def advance(ledger, market, cursor, actions, config, bound):
    """One five-minute decision for any vector of accounts at the given cursors."""
    actions = np.asarray(actions, dtype=np.int64)
    cursor = np.asarray(cursor, dtype=np.int64)
    if actions.shape != cursor.shape or not np.isin(actions, [SHORT, FLAT, LONG]).all():
        raise ValueError("invalid actions")
    if (cursor + 5 > bound).any():
        raise ValueError("step would cross development boundary")
    before = ledger.value(market.prices[cursor - 1, 3])
    fees_before, funding_before = ledger.fees_total.copy(), ledger.funding_total.copy()
    turnover_before = ledger.turnover_total.copy()
    was_terminal = ledger.terminated.copy()
    filled = np.zeros(len(cursor), bool)
    for offset in range(5):
        minute = cursor + offset
        prior_mark = market.prices[minute - 1, 3]
        ledger.funding_event(market.funding[minute], prior_mark)
        # Funding alone can breach margin before a pending entry/reversal.
        ledger.mark(prior_mark, elapsed=False)
        eligible = market.present[minute] & ~filled & ~ledger.terminated
        price = np.where(market.present[minute], market.prices[minute, 0], prior_mark)
        ledger.execute(actions, price, eligible)
        filled |= eligible
        close = market.prices[minute, 3]
        low = np.where(market.present[minute], market.prices[minute, 2], close)
        high = np.where(market.present[minute], market.prices[minute, 1], close)
        adverse = np.where(ledger.contracts > 0, low, high)
        ledger.mark(close, adverse)
    log_return = np.log(ledger.equity / np.maximum(before, EPS))
    log_return[was_terminal] = 0.0
    # The raw log return is kept alongside the reward so the plain log-equity
    # objective can be recomputed from saved logs without retraining.
    sharpe, ledger.dsr_mean, ledger.dsr_second = differential_sharpe(
        log_return, ledger.dsr_mean, ledger.dsr_second, config, active=~was_terminal)
    reward = (sharpe * config.reward_scale).astype(np.float32)
    # The resulting side when an order found no trade to fill against.
    executed = np.where(filled, actions, np.sign(ledger.contracts).astype(int) + 1)
    terminal = ledger.terminated & ~was_terminal
    return reward, terminal, {
        "log_return": log_return, "differential_sharpe": sharpe, "equity_btc": ledger.equity.copy(),
        "contracts": ledger.contracts.copy(), "filled": filled,
        "fees_btc": ledger.fees_total - fees_before, "funding_btc": ledger.funding_total - funding_before,
        "turnover": ledger.turnover_total - turnover_before,
        "active": ~was_terminal, "executed_action": executed}


def replicate(ledger, rows, copies):
    """A new ledger holding ``copies`` consecutive copies of each selected account."""
    rows = np.asarray(rows, dtype=np.int64)
    state = ledger.state_dict()
    copy = Ledger(len(rows) * copies, ledger.config)
    copy.load_state_dict({name: np.repeat(values[rows], copies, axis=0) for name, values in state.items()})
    return copy


def branch_every_action(ledger, market, cursor, config, bound):
    """Every account stepped under each action on the same bar.

    Returns the branched ledger, ordered [account 0 x SHORT, FLAT, LONG, account 1
    x ...], and the rewards and terminations with shape (accounts, actions).
    """
    actions = len(ACTION_NAMES)
    branches = replicate(ledger, np.arange(ledger.size), actions)
    reward, terminal, _ = advance(branches, market, np.repeat(np.asarray(cursor, np.int64), actions),
                                  np.tile(np.arange(actions), ledger.size), config, bound)
    return branches, reward.reshape(ledger.size, actions), terminal.reshape(ledger.size, actions)


class Environment:
    def __init__(self, market, bounds, config, rng, training=True):
        self.market, self.bounds, self.config, self.rng = market, bounds, config, rng
        self.training = training
        self.size = config.environments if training else 1
        self.ledger = Ledger(self.size, config)
        self.cursor = np.zeros(self.size, dtype=np.int64)
        self.steps = np.zeros(self.size, dtype=np.int64)
        self.resets = 0
        self.reset(np.ones(self.size, bool))

    def reset(self, mask):
        count = int(mask.sum())
        first, last = self.bounds
        if self.training:
            first = max(first, 1)
            upper = last - (self.config.episode_steps + 1) * 5
            if upper <= first:
                raise ValueError("training interval too short for contiguous episodes")
            epoch = self.market.epoch_start
            starts = self.rng.integers((first + epoch + 4) // 5, (upper + epoch) // 5, size=count) * 5 - epoch
        else:
            starts = np.full(count, first)
        self.cursor[mask] = starts
        self.steps[mask] = 0
        self.ledger.reset(mask)
        self.resets += count

    def observation(self):
        mark = self.market.prices[self.cursor - 1, 3]
        return self.cursor.copy(), self.ledger.portfolio(mark, self.market.daily_volatility[self.cursor])

    def initial_prefix(self, mask=None):
        """Market minutes before the cursor that rebuild the recurrent state."""
        mask = np.ones(self.size, bool) if mask is None else mask
        return self.cursor[mask, None] - np.arange(self.config.burn_in, 0, -1)[None, :] * 5

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.size,):
            raise ValueError("invalid actions")
        reward, terminal, info = advance(self.ledger, self.market, self.cursor, actions, self.config, self.bounds[1])
        self.cursor += 5
        self.steps += 1
        info["equity_usd"] = info["equity_btc"] * self.market.prices[self.cursor - 1, 3]
        return reward, terminal, info

    def state_dict(self):
        return {"cursor": self.cursor.copy(), "steps": self.steps.copy(), "resets": self.resets,
                "ledger": self.ledger.state_dict(), "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.cursor, self.steps = state["cursor"].copy(), state["steps"].copy()
        self.resets = state["resets"]
        self.ledger.load_state_dict(state["ledger"])
        self.rng.bit_generator.state = state["rng"]


class TrainingEnvironment(Environment):
    """Quarter-long episodes from random points, starting flat, short or long.

    Exploring starts (Sutton & Barto 2018, §5.3): a policy that only ever starts
    flat seldom practises an exit. A short prefix with real fees, funding and
    entry prices builds the starting position before the episode proper begins;
    one third of accounts stay flat and the rest enter a uniformly chosen side at
    a uniformly chosen step of the prefix.

    The account before the prefix's last bar is kept, so that the three accounts
    reachable over that bar - the siblings the trainer compares actions on - can
    be rebuilt exactly at every episode start.

    The first episode of each environment is shortened by a random whole number
    of rollouts, so resets are spread across updates instead of replacing all
    eight market windows at once. Parallel actors exist to decorrelate updates
    (Mnih et al., ICML 2016); synchronised resets would undo that every quarter.
    """

    def __init__(self, market, bounds, config, rng):
        super().__init__(market, bounds, config, rng, training=True)
        episodes = config.episode_steps // config.rollout
        self.steps[:] = config.rollout * rng.integers(0, episodes, self.size)

    def reset(self, mask):
        # Leave room for the prefix as well as the episode proper.
        original_bounds = self.bounds
        self.bounds = (self.bounds[0], self.bounds[1] - self.config.burn_in * 5)
        try:
            super().reset(mask)
        finally:
            self.bounds = original_bounds
        if not hasattr(self, "warm_minutes"):
            self.warm_minutes = np.zeros((self.size, self.config.burn_in), np.int64)
            self.pre_ledger = Ledger(self.size, self.config)
            self.pre_action = np.full(self.size, FLAT, np.int64)
        shadow = object.__new__(Environment)
        shadow.size = int(mask.sum())
        shadow.market, shadow.bounds, shadow.config = self.market, self.bounds, self.config
        shadow.training, shadow.resets, shadow.rng = False, 0, self.rng
        shadow.cursor, shadow.steps = self.cursor[mask].copy(), np.zeros(shadow.size, np.int64)
        shadow.ledger = Ledger(shadow.size, self.config)
        untouched_flat = self.rng.random(shadow.size) < self.config.flat_start_fraction
        side = np.where(self.rng.random(shadow.size) < 0.5, SHORT, LONG)
        entry_step = self.rng.integers(0, self.config.burn_in, shadow.size)
        for step in range(self.config.burn_in):
            self.warm_minutes[mask, step] = shadow.cursor
            action = np.where(untouched_flat | (step < entry_step), FLAT, side)
            if step == self.config.burn_in - 1:
                for name, values in shadow.ledger.state_dict().items():
                    getattr(self.pre_ledger, name)[mask] = values
                self.pre_action[mask] = action
            shadow.step(action)
        self.cursor[mask] = shadow.cursor
        for name, values in shadow.ledger.state_dict().items():
            getattr(self.ledger, name)[mask] = values
        # A rare prefix liquidation starts a fresh account that stands flat
        # through the previous bar, so its siblings over that bar stay exact.
        bad = self.ledger.terminated & mask
        if bad.any():
            rows = np.flatnonzero(bad)
            self.pre_ledger.reset(bad)
            self.pre_action[bad] = FLAT
            fresh = replicate(self.pre_ledger, rows, 1)
            advance(fresh, self.market, self.cursor[rows] - 5, np.full(len(rows), FLAT), self.config, self.bounds[1])
            for name, values in fresh.state_dict().items():
                getattr(self.ledger, name)[bad] = values
            self.warm_minutes[bad] = self.cursor[bad, None] - np.arange(self.config.burn_in, 0, -1)[None] * 5

    def starting_siblings(self, mask):
        """The three accounts reachable over the last bar before each masked episode start."""
        rows = np.flatnonzero(mask)
        base = replicate(self.pre_ledger, rows, 1)
        branches, _, _ = branch_every_action(base, self.market, self.cursor[rows] - 5, self.config, self.bounds[1])
        return branches, self.pre_action[rows].copy()

    def initial_prefix(self, mask=None):
        mask = np.ones(self.size, bool) if mask is None else mask
        return self.warm_minutes[mask].copy()

    def state_dict(self):
        return {**super().state_dict(), "warm_minutes": self.warm_minutes.copy(),
                "pre_ledger": self.pre_ledger.state_dict(), "pre_action": self.pre_action.copy()}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.warm_minutes = state["warm_minutes"].copy()
        self.pre_ledger.load_state_dict(state["pre_ledger"])
        self.pre_action = state["pre_action"].copy()
