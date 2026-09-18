"""Vector inverse-contract ledger. Prices are explicit trade-mark proxies.

Orders last for one decision interval and fill at its first regular-trade open.
Funding at a minute boundary is booked before that minute's order. Portfolio
marking uses the minute close; OHLC adverse extremes enforce a conservative
margin approximation, not a claim to reconstruct BitMEX's liquidation engine.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .reward import differential_sharpe, observation_features

SHORT, FLAT, LONG = 0, 1, 2
ACTION_NAMES = ("SHORT", "FLAT", "LONG")
SIDES = np.array([-1, 0, 1])
PORTFOLIO_NAMES = ("contract_side", "contract_exposure", "unrealized_return",
                   "log_position_age", "open_trade_net_log_return", "favorable_excursion",
                   "adverse_excursion", "open_trade_drawdown", "account_drawdown",
                   "maintenance_headroom", "open_trade_fees", "open_trade_funding",
                   "previous_reward",
                   # The reward is a function of the running Sharpe statistics,
                   # so the policy and critic must be able to see them. Without
                   # these two the same action in the same market state pays
                   # differently depending on unobserved recent performance.
                   "running_sharpe", "relative_volatility",
                   "previous_short", "previous_flat", "previous_long")
PORTFOLIO_WIDTH = len(PORTFOLIO_NAMES)
EPS = 1e-12


class Ledger:
    def __init__(self, size, config=Config()):
        self.size, self.config = size, config
        for name in ("wallet", "contracts", "entry", "equity", "peak", "trade_start",
                     "age", "trade_peak", "trade_trough", "trade_fees", "trade_funding",
                     "last_reward", "fees_total", "funding_total", "turnover_total",
                     "dsr_mean", "dsr_second"):
            setattr(self, name, np.zeros(size, dtype=np.float64))
        self.previous_action = np.full(size, FLAT, dtype=np.int64)
        self.terminated = np.zeros(size, dtype=bool)
        self.orders = np.zeros(size, dtype=np.int64)
        self.liquidations = np.zeros(size, dtype=np.int64)
        self.reset(np.ones(size, bool))

    def reset(self, mask):
        for name in ("wallet", "equity", "peak", "trade_start"):
            getattr(self, name)[mask] = 1.0
        for name in ("contracts", "entry", "age", "trade_peak", "trade_trough", "trade_fees",
                     "trade_funding", "last_reward", "fees_total", "funding_total",
                     "turnover_total", "orders", "liquidations"):
            getattr(self, name)[mask] = 0
        self.previous_action[mask] = FLAT
        self.terminated[mask] = False
        # A fresh account starts with no measured edge and the asset's own
        # typical five-minute variance, so the ratio's denominator is sane from
        # the first bar instead of dividing by zero. The prior washes out within
        # one 1/eta window.
        self.dsr_mean[mask] = 0.0
        self.dsr_second[mask] = self.config.dsr_initial_second_moment

    def value(self, mark):
        safe_entry = np.where(self.contracts != 0, self.entry, mark)
        return self.wallet + self.contracts * (1 / safe_entry - 1 / mark)

    def funding_event(self, rate, mark):
        payment = np.where(self.terminated, 0.0, -self.contracts / mark * rate)
        self.wallet += payment
        self.trade_funding += payment
        self.funding_total += payment
        return payment

    def _close(self, mask, price):
        quantity = self.contracts[mask]
        if not len(quantity):
            return
        # Close a short with a buy; close a long with a sell.
        fill = price[mask] * (1 - np.sign(quantity) * self.config.slippage)
        fee = np.abs(quantity) / fill * self.config.fee
        value_before = np.maximum(self.value(price)[mask], EPS)
        self.wallet[mask] += quantity * (1 / self.entry[mask] - 1 / fill) - fee
        self.fees_total[mask] += fee
        self.trade_fees[mask] += fee
        self.turnover_total[mask] += np.abs(quantity) / price[mask] / value_before
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
            quantity = np.floor(self.wallet[opening] * price[opening] * self.config.entry_leverage)
            quantity = target[opening] * quantity
            fee = np.abs(quantity) / fill * self.config.fee
            self.trade_start[opening] = self.wallet[opening]
            self.wallet[opening] -= fee
            self.contracts[opening] = quantity
            self.entry[opening] = fill
            self.trade_fees[opening] = fee
            self.trade_funding[opening] = 0
            self.trade_peak[opening] = 0
            self.trade_trough[opening] = 0
            self.age[opening] = 0
            self.fees_total[opening] += fee
            self.turnover_total[opening] += np.abs(quantity) / price[opening] / self.trade_start[opening]
            self.orders[opening] += (quantity != 0)
        closed = change & (target == 0)
        for name in ("trade_fees", "trade_funding", "trade_peak", "trade_trough", "age"):
            getattr(self, name)[closed] = 0
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
        self.peak = np.maximum(self.peak, self.equity)
        held = self.contracts != 0
        self.age[held] += int(elapsed)
        trade_return = np.log(self.equity / np.maximum(self.trade_start, EPS))
        self.trade_peak[held] = np.maximum(self.trade_peak[held], trade_return[held])
        self.trade_trough[held] = np.minimum(self.trade_trough[held], trade_return[held])

    def portfolio(self, mark):
        eq = np.maximum(self.value(mark), EPS)
        held = self.contracts != 0
        entry = np.where(held, self.entry, mark)
        unrealized = self.contracts * (1 / entry - 1 / mark) / eq
        trade = np.where(held, np.log(eq / np.maximum(self.trade_start, EPS)), 0)
        exposure = self.contracts / mark / eq
        common = np.stack([
            np.sign(self.contracts), np.clip(exposure, -5, 5), np.clip(10 * unrealized, -5, 5),
            np.log1p(self.age) / np.log1p(60 * 24 * 14), np.clip(10 * trade, -5, 5),
            np.clip(10 * self.trade_peak, 0, 5), np.clip(10 * self.trade_trough, -5, 0),
            np.clip(10 * (self.trade_peak - trade) * held, 0, 5),
            np.clip(10 * (1 - eq / np.maximum(self.peak, EPS)), 0, 10),
            np.clip(1 - np.abs(exposure) * self.config.maintenance, 0, 1),
            np.clip(self.trade_fees / np.maximum(self.trade_start, EPS) * 100, 0, 5),
            np.clip(self.trade_funding / np.maximum(self.trade_start, EPS) * 100, -5, 5),
            np.clip(self.last_reward, -5, 5),
            *observation_features(self.dsr_mean, self.dsr_second, self.config)], axis=1)
        return np.concatenate([common, np.eye(3)[self.previous_action]], axis=1).astype(np.float32)

    def state_dict(self):
        return {name: value.copy() for name, value in vars(self).items() if isinstance(value, np.ndarray)}

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value.copy())


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
            first = max(first, self.config.burn_in * 5 + self.config.fast_minutes)
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
        return self.cursor.copy(), self.ledger.portfolio(self.market.prices[self.cursor - 1, 3])

    def initial_prefix(self, mask=None):
        if mask is None:
            mask = np.ones(self.size, bool)
        minutes = self.cursor[mask, None] - np.arange(self.config.burn_in, 0, -1)[None, :] * 5
        blank = Ledger(int(mask.sum()), self.config).portfolio(self.market.prices[self.cursor[mask] - 1, 3])
        portfolio = np.repeat(blank[:, None, :], self.config.burn_in, axis=1)
        return minutes, portfolio

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.size,) or not np.isin(actions, [SHORT, FLAT, LONG]).all():
            raise ValueError("invalid actions")
        if (self.cursor + 5 > self.bounds[1]).any():
            raise ValueError("step would cross development boundary")
        before = self.ledger.value(self.market.prices[self.cursor - 1, 3])
        fees_before, funding_before = self.ledger.fees_total.copy(), self.ledger.funding_total.copy()
        turnover_before = self.ledger.turnover_total.copy()
        was_terminal = self.ledger.terminated.copy()
        filled = np.zeros(self.size, bool)
        active = ~was_terminal
        for offset in range(5):
            minute = self.cursor + offset
            prior_mark = self.market.prices[minute - 1, 3]
            self.ledger.funding_event(self.market.funding[minute], prior_mark)
            # Funding alone can breach margin before a pending entry/reversal.
            self.ledger.mark(prior_mark, elapsed=False)
            eligible = self.market.present[minute] & ~filled & ~self.ledger.terminated
            price = np.where(self.market.present[minute], self.market.prices[minute, 0], prior_mark)
            self.ledger.execute(actions, price, eligible)
            filled |= eligible
            close = self.market.prices[minute, 3]
            low = np.where(self.market.present[minute], self.market.prices[minute, 2], close)
            high = np.where(self.market.present[minute], self.market.prices[minute, 1], close)
            adverse = np.where(self.ledger.contracts > 0, low, high)
            self.ledger.mark(close, adverse)
        log_return = np.log(self.ledger.equity / np.maximum(before, EPS))
        log_return[was_terminal] = 0.0
        # The raw log return is kept alongside the reward so that the plain
        # log-equity objective can be recomputed from saved logs without
        # retraining anything.
        sharpe, self.ledger.dsr_mean, self.ledger.dsr_second = differential_sharpe(
            log_return, self.ledger.dsr_mean, self.ledger.dsr_second, self.config, active=~was_terminal)
        reward = sharpe * self.config.reward_scale
        self.ledger.last_reward = reward.copy()
        # Interaction input records actual resulting side when an order had no fill.
        executed = np.where(filled, actions, np.sign(self.ledger.contracts).astype(int) + 1)
        self.ledger.previous_action = executed
        self.cursor += 5
        self.steps += 1
        terminal = self.ledger.terminated & ~was_terminal
        return reward.astype(np.float32), terminal, {
            "log_return": log_return, "differential_sharpe": sharpe,
            "equity_btc": self.ledger.equity.copy(),
            "equity_usd": self.ledger.equity * self.market.prices[self.cursor - 1, 3],
            "contracts": self.ledger.contracts.copy(), "filled": filled,
            "fees_btc": self.ledger.fees_total - fees_before,
            "funding_btc": self.ledger.funding_total - funding_before,
            "turnover": self.ledger.turnover_total - turnover_before,
            "active": active, "executed_action": executed}

    def state_dict(self):
        return {"cursor": self.cursor.copy(), "steps": self.steps.copy(), "resets": self.resets,
                "ledger": self.ledger.state_dict(), "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.cursor, self.steps = state["cursor"].copy(), state["steps"].copy()
        self.resets = state["resets"]
        self.ledger.load_state_dict(state["ledger"])
        self.rng.bit_generator.state = state["rng"]


class TrainingEnvironment(Environment):
    """Episodes begin from valid flat AND open-position histories.

    A policy that only ever sees flat starting states never learns when to exit,
    so a persistent random-action prefix is run first to build real positions,
    with real fees, funding and ages, before the episode proper begins.

    The prefix does NOT reuse ``initial_persistence``. That constant is the
    policy's starting belief about how long to hold, near 24 hours; driving the
    prefix with it would change side on only a third of accounts across a
    128-step prefix and start roughly three quarters of all episodes flat. The
    prefix instead draws a per-account holding time spanning hours to a week, so
    starting sides and position ages are both varied.
    """

    def reset(self, mask):
        # Leave room for the exploratory prefix as well as the actual episode.
        original_bounds = self.bounds
        self.bounds = (self.bounds[0], self.bounds[1] - self.config.burn_in * 5)
        try:
            super().reset(mask)
        finally:
            self.bounds = original_bounds
        if not hasattr(self, "warm_minutes"):
            self.warm_minutes = np.zeros((self.size, self.config.burn_in), np.int64)
            self.warm_account = np.zeros((self.size, self.config.burn_in, PORTFOLIO_WIDTH), np.float32)
        shadow = object.__new__(Environment)
        shadow.size = int(mask.sum())
        shadow.market, shadow.bounds, shadow.config = self.market, self.bounds, self.config
        shadow.training, shadow.resets, shadow.rng = False, 0, self.rng
        shadow.cursor, shadow.steps = self.cursor[mask].copy(), np.zeros(shadow.size, np.int64)
        shadow.ledger = Ledger(shadow.size, self.config)
        # Choose the starting side explicitly rather than hoping a random walk
        # wanders into one. Drawing a holding time and letting persistence decide
        # leaves most accounts flat: a one-week mean hold changes side within a
        # 128-step prefix only six per cent of the time, which produced 88 per
        # cent flat starts and not a single short. Here one third stay flat and
        # the rest enter a uniformly chosen side at a uniformly chosen moment, so
        # sides are balanced and position ages span the whole prefix.
        untouched_flat = self.rng.random(shadow.size) < self.config.flat_start_fraction
        side = np.where(self.rng.random(shadow.size) < 0.5, SHORT, LONG)
        entry_step = self.rng.integers(0, self.config.burn_in, shadow.size)
        for step in range(self.config.burn_in):
            minute, account = shadow.observation()
            self.warm_minutes[mask, step], self.warm_account[mask, step] = minute, account
            action = np.where(untouched_flat | (step < entry_step), FLAT, side)
            shadow.step(action)
        self.cursor[mask] = shadow.cursor
        for name, values in shadow.ledger.state_dict().items():
            getattr(self.ledger, name)[mask] = values
        # These are starting-state experiences, not fake on-policy reward rows.
        # A rare prefix liquidation starts a fresh flat account at that boundary.
        bad = self.ledger.terminated & mask
        if bad.any():
            self.ledger.reset(bad)
            minute = self.cursor[bad, None] - np.arange(self.config.burn_in, 0, -1)[None] * 5
            blank = Ledger(int(bad.sum()), self.config).portfolio(self.market.prices[self.cursor[bad] - 1, 3])
            self.warm_minutes[bad] = minute
            self.warm_account[bad] = blank[:, None]

    def initial_prefix(self, mask=None):
        mask = np.ones(self.size, bool) if mask is None else mask
        return self.warm_minutes[mask].copy(), self.warm_account[mask].copy()

    def state_dict(self):
        return {**super().state_dict(), "warm_minutes": self.warm_minutes.copy(),
                "warm_account": self.warm_account.copy()}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.warm_minutes, self.warm_account = state["warm_minutes"].copy(), state["warm_account"].copy()
