"""Training starts include valid flat and open-position histories."""
from __future__ import annotations

import numpy as np
from perpetual.env import Environment, Ledger


class TrainingEnvironment(Environment):
    def reset(self, mask):
        # Leave room for the exploratory prefix as well as the actual episode.
        original_bounds = self.bounds
        self.bounds = (self.bounds[0], self.bounds[1]-self.config.burn_in*5)
        try:
            super().reset(mask)
        finally:
            self.bounds = original_bounds
        if not hasattr(self, 'warm_minutes'):
            self.warm_minutes = np.zeros((self.size, self.config.burn_in), np.int64)
            self.warm_account = np.zeros((self.size, self.config.burn_in, 16), np.float32)
        shadow = object.__new__(Environment)
        shadow.size = int(mask.sum())
        shadow.market, shadow.bounds, shadow.config = self.market, self.bounds, self.config
        shadow.training, shadow.resets, shadow.rng = False, 0, self.rng
        shadow.cursor, shadow.steps = self.cursor[mask].copy(), np.zeros(shadow.size, np.int64)
        shadow.ledger = Ledger(shadow.size, self.config)
        for step in range(self.config.burn_in):
            minute, account = shadow.observation()
            self.warm_minutes[mask, step], self.warm_account[mask, step] = minute, account
            current = np.sign(shadow.ledger.contracts).astype(int)+1
            alternative = (current+1+self.rng.integers(0, 2, shadow.size)) % 3
            change = self.rng.random(shadow.size) > self.config.initial_persistence
            shadow.step(np.where(change, alternative, current))
        self.cursor[mask] = shadow.cursor
        for name, values in shadow.ledger.state_dict().items():
            getattr(self.ledger, name)[mask] = values
        # These are starting-state experiences, not fake on-policy reward rows.
        # A rare prefix liquidation starts a fresh flat account at that boundary.
        bad = self.ledger.terminated & mask
        if bad.any():
            self.ledger.reset(bad)
            minute = self.cursor[bad, None]-np.arange(self.config.burn_in, 0, -1)[None]*5
            blank = Ledger(int(bad.sum()), self.config).portfolio(self.market.prices[self.cursor[bad]-1, 3])
            self.warm_minutes[bad] = minute
            self.warm_account[bad] = blank[:, None]

    def initial_prefix(self, mask=None):
        mask = np.ones(self.size, bool) if mask is None else mask
        return self.warm_minutes[mask].copy(), self.warm_account[mask].copy()

    def state_dict(self):
        return {**super().state_dict(), 'warm_minutes': self.warm_minutes.copy(), 'warm_account': self.warm_account.copy()}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.warm_minutes, self.warm_account = state['warm_minutes'].copy(), state['warm_account'].copy()
