"""Load the hourly feature table and turn it into the arrays the agent uses.

The table itself is built outside the agent, by data_pipeline/hourly_features.py.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import FEATURE_FILE

# State v4 (2026-09-21): 120 features = state v3's 108 + 12 trend-persistence features. The agent reads EVERY
# feature the pipeline builds - covering one hour
# to one year in both relative and rank form (data_pipeline/hourly_features.py; ARCHITECTURE.md S3.5). The list is
# taken from the file itself, so the agent and the pipeline can never drift apart. Correlated inputs are kept on
# purpose this time: the redundancy rule is reported (scripts/feature_selection.py) but not applied.
FEATURES = tuple(name for name in pq.read_schema(FEATURE_FILE).names if name not in ('time', 'close'))


def load(end=None):
    """The feature table, cut to rows before `end`. Training and validation always pass their fold's test start,
    so no test row is ever loaded; only the test script loads the whole table."""
    table = pd.read_parquet(FEATURE_FILE)
    if end is not None:
        table = table[table.time < end]
    return table.reset_index(drop=True)


def rows(table, period):
    """Row positions whose time lies in [start, end)."""
    start, end = period
    return np.flatnonzero(((table.time >= start) & (table.time < end)).to_numpy())


class Scaler:
    """Standardise each feature with the TRAINING period's mean and standard deviation: z = (x - mean) / std."""

    def __init__(self, table, train_rows, clip, names=None):
        # `names` picks a feature subset (e.g. the 108 of state v3 to reload its checkpoints); default: all of them.
        self.names = list(FEATURES if names is None else names)
        values = table.loc[train_rows, self.names].to_numpy(np.float64)
        self.mean, self.std, self.clip = values.mean(0), values.std(0), clip

    def transform(self, table):
        z = (table[self.names].to_numpy(np.float64) - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip).astype(np.float32)      # clip rare extremes to +-clip

    def state_dict(self):
        return {'names': self.names, 'mean': self.mean.tolist(), 'std': self.std.tolist(), 'clip': self.clip}


def next_returns(table):
    """R_{t+1} = P_{t+1} / P_t - 1: the simple return of the hour AFTER row t (what a position taken at t earns).

    The last row has no next hour and gets NaN.
    """
    close = table.close.to_numpy(np.float64)
    out = np.full(len(close), np.nan)
    out[:-1] = close[1:] / close[:-1] - 1
    return out
