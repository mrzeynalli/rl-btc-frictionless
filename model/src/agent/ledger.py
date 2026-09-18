"""Vector linear-contract ledger and the endogenous half of the observation.

Version 2 (2026-09-15): a USDT-margined linear perpetual (Binance USD-M BTCUSDT). The
wallet is USDT, the position is a signed BTC quantity, profit and loss are settled in
USDT, and a flat account holds cash with no BTC exposure. v1 modelled BitMEX's inverse
contract with BTC collateral, on which flat was long BTC in dollars and a 1x short an
exact dollar hedge - not the account the user trades.

A decision holds for ``decision_bars`` five-minute bars and its order fills at the
first regular-trade open inside that interval. Funding at a minute boundary is
booked before that minute's order. Portfolio marking uses the minute close; OHLC
adverse extremes enforce a conservative margin approximation, not a claim to
reconstruct BitMEX's liquidation engine.

The account observation is what a decision needs beyond the market. Side,
exposure and the two reward statistics make reward and termination Markov given
the market (an information state in the sense of Subramanian et al., JMLR 2022).
The open trade's return and age are not needed for that. They are there because
the regime is not observed: bull and bear phases show duration dependence (Lunde &
Timmermann, JBES 2004), so whether the move a trade was opened on is still in
force depends on how long it has run and how far - and the recurrent memory,
rebuilt from a burn-in prefix at every rollout, cannot count a holding period
longer than its own span.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .data import BAR_MINUTES
from .reward import differential_ratio, observation_features

SHORT, FLAT, LONG = 0, 1, 2
ACTION_NAMES = ("SHORT", "FLAT", "LONG")
SIDES = np.array([-1, 0, 1])
PORTFOLIO_NAMES = ("short", "flat", "long",
                   # Open trade's log return, in units of the four-week daily volatility.
                   "trade_return",
                   # log(1 + bars held / credit window): zero at entry, log 2 after one
                   # credit window, and still graded after weeks.
                   "trade_age",
                   # log(1 + |BTC position| x price / equity): the position's notional over
                   # the account's equity.
                   "exposure",
                   # The reward's own statistics: the running ratio, and the log of its
                   # risk second moment against the prior.
                   "running_ratio", "relative_risk")
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
        # five-minute risk, so the ratio is defined from the first bar.
        self.dsr_second[mask] = self.config.initial_reward_second_moment

    def value(self, mark):
        """USDT equity: the wallet plus the open position's unrealised profit."""
        safe_entry = np.where(self.contracts != 0, self.entry, mark)
        return self.wallet + self.contracts * (mark - safe_entry)

    def funding_event(self, rate, mark):
        # Binance: funding = position notional x rate; a long pays a positive rate.
        payment = np.where(self.terminated, 0.0, -self.contracts * mark * rate)
        self.wallet += payment
        self.funding_total += payment
        return payment

    def _close(self, mask, price):
        quantity = self.contracts[mask]
        if not len(quantity):
            return
        # Close a short with a buy; close a long with a sell.
        fill = price[mask] * (1 - np.sign(quantity) * self.config.slippage)
        fee = np.abs(quantity) * fill * self.config.fee
        self.wallet[mask] += quantity * (fill - self.entry[mask]) - fee
        self.fees_total[mask] += fee
        # Notional relative to the equity the trade started with. The account's
        # current value is not used: at a liquidation it is near zero, and the
        # predecessor booked turnover of up to 2.6e12 per event dividing by it.
        self.turnover_total[mask] += np.abs(quantity) * fill / np.maximum(self.trade_start[mask], EPS)
        self.contracts[mask] = 0
        self.entry[mask] = 0
        self.orders[mask] += 1

    def execute(self, actions, price, eligible, sizes=None):
        """Move each eligible account to its target side.

        ``sizes`` (benchmarks only; the agent never passes it) is a notional per unit of equity for
        each account. It replaces `entry_leverage` on opening and, for an account already on its
        target side, resizes the open position to that size, charging fee and slippage on the
        traded difference.
        """
        target = SIDES[np.asarray(actions, dtype=np.int64)]
        previous = np.sign(self.contracts)
        change = eligible & ~self.terminated & (target != previous)
        self._close(change & (previous != 0), price)
        opening = change & (target != 0) & (self.wallet > EPS)
        if opening.any():
            fill = price[opening] * (1 + target[opening] * self.config.slippage)
            # A BTC quantity worth entry_leverage x the wallet at the fill's reference price,
            # fixed while the position is held: no uncharged rebalancing.
            leverage = self.config.entry_leverage if sizes is None else np.asarray(sizes, dtype=np.float64)[opening]
            quantity = target[opening] * self.wallet[opening] * leverage / price[opening]
            fee = np.abs(quantity) * fill * self.config.fee
            self.trade_start[opening] = self.wallet[opening]
            self.wallet[opening] -= fee
            self.contracts[opening] = quantity
            self.entry[opening] = fill
            self.age[opening] = 0
            self.fees_total[opening] += fee
            self.turnover_total[opening] += np.abs(quantity) * fill / self.trade_start[opening]
            self.orders[opening] += (quantity != 0)
        closed = change & (target == 0)
        self.age[closed] = 0
        self.trade_start[closed] = np.maximum(self.wallet[closed], EPS)
        if sizes is not None:
            self._resize(eligible & ~self.terminated & ~change & (target != 0), target,
                         np.asarray(sizes, dtype=np.float64), price)
        return change

    def _resize(self, mask, target, sizes, price):
        """Trade an open position to target x size x equity / price on the same side.

        A reduction realises profit on the part closed at the entry price; an increase buys at the fill
        and moves the entry to the quantity-weighted average. Both pay fee and slippage on the difference.
        """
        if not mask.any():
            return
        desired = np.where(mask, target * sizes * self.value(price) / price, self.contracts)
        delta = desired - self.contracts
        reduce = mask & (np.abs(desired) < np.abs(self.contracts)) & (delta != 0)
        increase = mask & (np.abs(desired) > np.abs(self.contracts)) & (delta != 0)
        if reduce.any():
            part = self.contracts[reduce] - desired[reduce]
            fill = price[reduce] * (1 - np.sign(part) * self.config.slippage)
            fee = np.abs(part) * fill * self.config.fee
            self.wallet[reduce] += part * (fill - self.entry[reduce]) - fee
            self.fees_total[reduce] += fee
            self.turnover_total[reduce] += np.abs(part) * fill / np.maximum(self.trade_start[reduce], EPS)
            self.contracts[reduce] = desired[reduce]
            self.orders[reduce] += 1
        if increase.any():
            part = delta[increase]
            fill = price[increase] * (1 + np.sign(part) * self.config.slippage)
            fee = np.abs(part) * fill * self.config.fee
            self.wallet[increase] -= fee
            self.entry[increase] = (self.contracts[increase] * self.entry[increase] + part * fill) / desired[increase]
            self.fees_total[increase] += fee
            self.turnover_total[increase] += np.abs(part) * fill / np.maximum(self.trade_start[increase], EPS)
            self.contracts[increase] = desired[increase]
            self.orders[increase] += 1

    def mark(self, price, adverse=None, elapsed=True):
        active = ~self.terminated
        worst = price if adverse is None else adverse
        worst_value = self.value(worst)
        # Maintenance margin on the notional at the adverse price, plus the fee to close.
        requirement = np.abs(self.contracts) * worst * (self.config.maintenance + self.config.fee)
        breached = active & ((worst_value <= requirement) | (worst_value <= EPS))
        if breached.any():
            self._close(breached & (self.contracts != 0), worst)
            self.wallet[breached] = np.maximum(self.wallet[breached], EPS)
            self.terminated[breached] = True
            self.liquidations[breached] += 1
        self.equity = np.maximum(self.value(price), EPS)
        # Minutes the open position has been held.
        self.age[self.contracts != 0] += int(elapsed)

    def portfolio(self, mark, daily_volatility):
        """Endogenous observation: side, open trade's move and age, exposure, reward state."""
        c = self.config
        equity = np.maximum(self.value(mark), EPS)
        held = self.contracts != 0
        trade = np.where(held, np.log(equity / np.maximum(self.trade_start, EPS)), 0.0)
        volatility = np.broadcast_to(np.asarray(daily_volatility, dtype=np.float64), trade.shape)
        trade = np.where(volatility > 0, trade / np.where(volatility > 0, volatility, 1.0), 0.0)
        credit_window_bars = c.credit_window * c.decision_bars
        age = np.where(held, np.log1p(self.age / BAR_MINUTES / credit_window_bars), 0.0)
        exposure = np.log1p(np.abs(self.contracts) * mark / equity)
        continuous = np.stack([trade, age, exposure, *observation_features(self.dsr_mean, self.dsr_second, c)],
                              axis=1)
        continuous = np.clip(continuous, -c.observation_clip, c.observation_clip)
        side = np.sign(self.contracts).astype(np.int64) + 1
        return np.concatenate([np.eye(3)[side], continuous], axis=1).astype(np.float32)

    def state_dict(self):
        return {name: value.copy() for name, value in vars(self).items() if isinstance(value, np.ndarray)}

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value.copy())


def advance(ledger, market, cursor, actions, config, bound, sizes=None):
    """One decision - ``decision_bars`` five-minute bars - for any vector of accounts.

    The reward statistics advance once per bar, so the reward of a decision is the
    sum of its bars' increments and still telescopes to the change in the ratio.
    """
    actions = np.asarray(actions, dtype=np.int64)
    cursor = np.asarray(cursor, dtype=np.int64)
    if actions.shape != cursor.shape or not np.isin(actions, [SHORT, FLAT, LONG]).all():
        raise ValueError("invalid actions")
    if (cursor + config.decision_bars * BAR_MINUTES > bound).any():
        raise ValueError("step would cross development boundary")
    start = ledger.value(market.prices[cursor - 1, 3])
    before = start.copy()
    fees_before, funding_before = ledger.fees_total.copy(), ledger.funding_total.copy()
    turnover_before = ledger.turnover_total.copy()
    was_terminal = ledger.terminated.copy()
    filled = np.zeros(len(cursor), bool)
    increments = np.zeros(len(cursor))
    for bar in range(config.decision_bars):
        alive = ~ledger.terminated
        for offset in range(BAR_MINUTES):
            minute = cursor + bar * BAR_MINUTES + offset
            prior_mark = market.prices[minute - 1, 3]
            ledger.funding_event(market.funding[minute], prior_mark)
            # Funding alone can breach margin before a pending entry/reversal.
            ledger.mark(prior_mark, elapsed=False)
            eligible = market.present[minute] & ~filled & ~ledger.terminated
            price = np.where(market.present[minute], market.prices[minute, 0], prior_mark)
            ledger.execute(actions, price, eligible, sizes)
            filled |= eligible
            close = market.prices[minute, 3]
            low = np.where(market.present[minute], market.prices[minute, 2], close)
            high = np.where(market.present[minute], market.prices[minute, 1], close)
            adverse = np.where(ledger.contracts > 0, low, high)
            ledger.mark(close, adverse)
        bar_return = np.where(alive, np.log(ledger.equity / np.maximum(before, EPS)), 0.0)
        increment, ledger.dsr_mean, ledger.dsr_second = differential_ratio(
            bar_return, ledger.dsr_mean, ledger.dsr_second, config, active=alive)
        increments += increment
        before = ledger.equity.copy()
    # The raw log return is kept alongside the reward so other objectives can be
    # recomputed from saved logs without retraining.
    log_return = np.where(was_terminal, 0.0, np.log(ledger.equity / np.maximum(start, EPS)))
    reward = (increments * config.reward_scale).astype(np.float32)
    # The resulting side when an order found no trade to fill against.
    executed = np.where(filled, actions, np.sign(ledger.contracts).astype(int) + 1)
    terminal = ledger.terminated & ~was_terminal
    return reward, terminal, {
        "log_return": log_return, "differential_ratio": increments, "equity": ledger.equity.copy(),
        "contracts": ledger.contracts.copy(), "filled": filled,
        "fees": ledger.fees_total - fees_before, "funding": ledger.funding_total - funding_before,
        "turnover": ledger.turnover_total - turnover_before,
        "active": ~was_terminal, "executed_action": executed}


class Environment:
    def __init__(self, market, bounds, config, rng, training=True):
        self.market, self.bounds, self.config, self.rng = market, bounds, config, rng
        self.training = training
        self.span = config.decision_bars * BAR_MINUTES
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
            upper = last - (self.config.episode_steps + 1) * self.span
            if upper <= first:
                raise ValueError("training interval too short for contiguous episodes")
            epoch = self.market.epoch_start
            starts = (self.rng.integers((first + epoch + BAR_MINUTES - 1) // BAR_MINUTES,
                                        (upper + epoch) // BAR_MINUTES, size=count) * BAR_MINUTES - epoch)
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
        """Decision minutes before the cursor that rebuild the recurrent state."""
        mask = np.ones(self.size, bool) if mask is None else mask
        return self.cursor[mask, None] - np.arange(self.config.burn_in, 0, -1)[None, :] * self.span

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.size,):
            raise ValueError("invalid actions")
        reward, terminal, info = advance(self.ledger, self.market, self.cursor, actions, self.config, self.bounds[1])
        self.cursor += self.span
        self.steps += 1
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
    flat seldom practises an exit. A prefix of ``burn_in`` decisions with real
    fees, funding and entry prices builds the starting position - and doubles as
    the recurrent burn-in - before the episode proper begins. One third of
    accounts stay flat; the rest enter a uniformly chosen side at a uniformly
    chosen step of the prefix.

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
        self.bounds = (self.bounds[0], self.bounds[1] - self.config.burn_in * self.span)
        try:
            super().reset(mask)
        finally:
            self.bounds = original_bounds
        if not hasattr(self, "warm_minutes"):
            self.warm_minutes = np.zeros((self.size, self.config.burn_in), np.int64)
        rows = np.flatnonzero(mask)
        count = len(rows)
        ledger = Ledger(count, self.config)
        cursor = self.cursor[rows].copy()
        untouched_flat = self.rng.random(count) < self.config.flat_start_fraction
        side = np.where(self.rng.random(count) < 0.5, SHORT, LONG)
        entry_step = self.rng.integers(0, self.config.burn_in, count)
        for step in range(self.config.burn_in):
            self.warm_minutes[rows, step] = cursor
            action = np.where(untouched_flat | (step < entry_step), FLAT, side)
            advance(ledger, self.market, cursor, action, self.config, self.bounds[1])
            cursor += self.span
        self.cursor[rows] = cursor
        for name, values in ledger.state_dict().items():
            getattr(self.ledger, name)[rows] = values
        # A rare liquidation inside the prefix starts a fresh flat account instead.
        failed = np.zeros(self.size, bool)
        failed[rows[ledger.terminated]] = True
        if failed.any():
            self.ledger.reset(failed)

    def initial_prefix(self, mask=None):
        mask = np.ones(self.size, bool) if mask is None else mask
        return self.warm_minutes[mask].copy()

    def state_dict(self):
        return {**super().state_dict(), "warm_minutes": self.warm_minutes.copy()}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.warm_minutes = state["warm_minutes"].copy()
