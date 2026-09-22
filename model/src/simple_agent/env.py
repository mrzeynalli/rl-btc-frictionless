"""The trading environment: a vector of walkers through one continuous stretch of history.

Account (deliberately minimal): the position is -1, 0 or +1 of the account's value, and there are no fees,
slippage, funding or margin. Holding position a_t over the next hour earns R_{t+1} = a_t * (P_{t+1}/P_t - 1).

Reward: the differential Sharpe ratio (Moody & Saffell 2001) - the marginal effect of this hour's return on an
exponentially weighted Sharpe ratio S = A / sqrt(B - A^2), where
    A_t = A_{t-1} + eta * (R_t - A_{t-1})          running mean of returns
    B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})        running mean of squared returns
    D_t = (B_{t-1} * dA - 0.5 * A_{t-1} * dB) / (B_{t-1} - A_{t-1}^2)^(3/2),   dA = R_t - A_{t-1},  dB = R_t^2 - B_{t-1}
D_t is positive when the return raises the Sharpe ratio and negative when it lowers it, so the agent is paid for
return per unit of risk, not for return alone.

There are no episodes: each walker steps hour by hour through the whole period. When a walker reaches the end it
jumps back to the start, flat and with fresh reward statistics, and that step is flagged `done` so the LSTM memory
and the advantage estimate do not cross the jump.
"""
import numpy as np

from .config import ACTIONS

POSITIONS = np.array(ACTIONS, dtype=np.float64)       # action index -> position


def differential_sharpe(R, A, B):
    """One step of the differential Sharpe ratio. Returns (D_t, dA, dB) for vectors R, A, B."""
    dA = R - A                                        # dA = R_t - A_{t-1}
    dB = R * R - B                                    # dB = R_t^2 - B_{t-1}
    variance = np.maximum(B - A * A, 1e-12)           # B - A^2, floored only against division by zero
    D = (B * dA - 0.5 * A * dB) / variance ** 1.5
    return D, dA, dB


class Env:
    def __init__(self, features, returns, n_envs, eta, initial_variance):
        """features: (T, F) standardised states; returns: (T,) next-hour simple returns (last one NaN)."""
        self.features, self.returns = features, returns
        self.n, self.eta, self.b0 = n_envs, eta, initial_variance
        self.last = len(returns) - 1                  # the last row has no next return, so walkers stop before it
        # Walkers start at evenly spaced points, so together they cover the whole period at every moment.
        self.index = (np.arange(n_envs) * self.last // n_envs).astype(np.int64)
        self.position = np.zeros(n_envs)              # every walker starts flat
        self.A = np.zeros(n_envs)                     # A_0 = 0: no measured edge yet
        self.B = np.full(n_envs, initial_variance)    # B_0 = variance of hourly returns: the ratio is defined from step 1

    def observation(self):
        """State = market features of the current row + one-hot of the position currently held."""
        one_hot = np.eye(3, dtype=np.float32)[(self.position + 1).astype(int)]   # -1,0,1 -> [1,0,0],[0,1,0],[0,0,1]
        return np.concatenate([self.features[self.index], one_hot], axis=1)

    def step(self, actions):
        """Take positions, earn the next hour, move one row on. Returns (reward, hourly return, done)."""
        self.position = POSITIONS[actions]
        R = self.position * self.returns[self.index]  # R_{t+1} = a_t * (P_{t+1} / P_t - 1)
        D, dA, dB = differential_sharpe(R, self.A, self.B)
        self.A += self.eta * dA                       # A_t = A_{t-1} + eta * dA
        self.B += self.eta * dB                       # B_t = B_{t-1} + eta * dB
        self.index += 1
        done = self.index >= self.last                # reached the end of the period: wrap around
        self.index[done] = 0
        self.position[done] = 0.0
        self.A[done] = 0.0
        self.B[done] = self.b0
        return D.astype(np.float32), R, done
