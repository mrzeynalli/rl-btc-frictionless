"""Load the hourly feature table and turn it into the arrays the agent uses.

The table itself is built outside the agent, by data_pipeline/hourly_features.py.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import FEATURE_FILE

# The 120 market variables: the agent reads every feature the pipeline builds (data_pipeline/hourly_features.py),
# covering one hour to one year in relative and percentile form. The list is taken from the file itself, so the
# agent and the pipeline cannot drift apart.
FEATURES = tuple(name for name in pq.read_schema(FEATURE_FILE).names if name not in ('time', 'close'))


def load(end=None):
    """The feature table, cut to rows before `end`. Training passes the end of the last period it uses, so no
    later row is loaded; only the test evaluation loads the test period."""
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
        # `names` fixes the feature order stored with a trained model; default: all features of the table.
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
