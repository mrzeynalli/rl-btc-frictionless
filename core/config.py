"""Every setting of the agent, in one place."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / 'data/processed/xbtusd_hourly_features.parquet'   # built by data_pipeline/hourly_features.py

# Every period is [start, end) in UTC. The feature table starts in June 2017; the first rows whose one-year look-back
# windows are incomplete are dropped by the feature pipeline, so training effectively starts on 12 August 2017.
DATA_START = pd.Timestamp('2017-06-10', tz='UTC')
DATA_END = pd.Timestamp('2026-08-02 01:00', tz='UTC')


def year(y):
    return pd.Timestamp(f'{y}-01-01', tz='UTC')


def fold(validation_year):
    """(training period, validation period) of a selection fold: train on [DATA_START, V), score on year V."""
    return (DATA_START, year(validation_year)), (year(validation_year), year(validation_year + 1))


ACTIONS = (-1, 0, 1)            # action index 0, 1, 2 -> position short, flat, long
HOURS_PER_YEAR = 8760


@dataclass(frozen=True)
class Config:
    # --- network ---
    # Defaults. The walk-forward (scripts/walkforward_nested.py) chooses lr, hidden, entropy_coef and eta for every
    # test year from the years before it and fixes the remaining settings at these values. CleanRL values in [].
    hidden: int = 64            # LSTM and hidden-layer width, for both actor and critic [128]

    # --- rollout ---
    envs: int = 8               # parallel walkers through the training period, started at evenly spaced points
    rollout: int = 64           # hours each walker steps between two updates -> 8 x 64 = 512 samples per update [128]
    minibatches: int = 4        # each minibatch = 2 walkers x 64 hours
    epochs: int = 3             # passes over one rollout per update [4]

    # --- PPO (CleanRL defaults) ---
    gamma: float = 0.98         # discount per hour: rewards ~50 hours ahead still count 1/e
    gae_lambda: float = 0.95    # GAE smoothing between one-step and long-horizon advantage
    clip: float = 0.2           # PPO ratio clip: the new policy may move each probability ratio by at most +-20%
    entropy_coef: float = 0.03  # bonus for keeping the policy random enough to explore [0.01]
    lr: float = 8.38429594542857e-05  # Adam step size of the actor (policy): small, cautious policy steps [2.5e-4]
    critic_lr: float = 9.742244550984722e-04  # Adam step size of the critic (value function): a fast value estimate [2.5e-4]
    max_grad_norm: float = 1.0  # gradient clipping [0.5]

    # --- reward: differential Sharpe ratio (Moody & Saffell 2001) ---
    eta: float = 1 / 336        # adaptation rate of the running mean/variance: a two-week (336-hour) memory [1/120]

    # --- budget ---
    passes: int = 10            # passes over the training period; the model after the last pass is the one used
    obs_clip: float = 5.0       # standardised features are clipped to +-5
    warmup_hours: int = 168     # evaluation feeds the week before a period through the LSTM first, without trading
