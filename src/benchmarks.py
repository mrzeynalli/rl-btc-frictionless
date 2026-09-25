"""Rule based benchmarks and the trade win rate, evaluated on the same hourly, frictionless account as the agents.

Signals are computed from daily closes known at each 00:00 UTC and held until the next midnight:
  buy and hold; 50 and 200 day moving average (long above, short below; Brock, Lakonishok & LeBaron 1992);
  time series momentum (sign of the past 12-month return, rebalanced monthly; Moskowitz, Ooi & Pedersen 2012);
  MACD(12, 26, 9) crossover (Appel 2005); RSI(14) (long below 30, short above 70, flat otherwise; Wilder 1978).
"""
import numpy as np
import pandas as pd


def daily_signal_positions(table, test_rows, rule):
    """Hourly positions for a rule evaluated on daily closes known at each 00:00 UTC and held until the next one."""
    daily = table.set_index('time').close
    daily = daily[daily.index.hour == 0]                       # the close of each completed day, known at midnight
    signal = rule(daily)                                       # position per midnight, in {-1, 0, 1}
    times = table.time.iloc[test_rows]
    return signal.reindex(times, method='ffill').fillna(0.0).to_numpy()


def moving_average(days):
    return lambda close: np.sign(close - close.rolling(days).mean())


def time_series_momentum(close):
    """Sign of the trailing 12-month return, fixed at each month's first midnight and held for the month."""
    raw = np.sign(close / close.shift(365) - 1)
    first_of_month = close.index.to_series().dt.day == 1
    return raw.where(first_of_month).ffill()


def macd(close):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return np.sign(line - line.ewm(span=9, adjust=False).mean())


def rsi(close):
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()      # Wilder's smoothing
    loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    value = 100 - 100 / (1 + gain / loss)
    return pd.Series(np.where(value < 30, 1.0, np.where(value > 70, -1.0, 0.0)), index=close.index)


RULES = {'Buy and hold': None, '50 day MA': moving_average(50), '200 day MA': moving_average(200),
         'Time series momentum': time_series_momentum, 'MACD': macd, 'RSI': rsi}




def win_rate(R, positions):
    """Share of trades that made money; a trade is a maximal run of hours with the same non-flat position."""
    change = np.flatnonzero(np.diff(positions) != 0) + 1
    wins = [np.prod(1 + R[a:b]) > 1 for a, b in zip(np.r_[0, change], np.r_[change, len(R)]) if positions[a] != 0]
    return float(np.mean(wins) * 100) if wins else float('nan')
