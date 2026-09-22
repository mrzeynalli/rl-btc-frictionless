"""Build the hourly BTC feature table used by the simple agent (src/simple_agent).

Input:  data/bitmex_xbtusd_1m_2016_2026_08_01/  (BitMEX XBTUSD one-minute bars and funding settlements)
Output: data/processed/xbtusd_hourly_features.parquet

One row per hour. Row t describes the hour that CLOSES at `time` (the bar [time - 1h, time)). Every feature in
row t uses only bars up to and including that hour, so the agent can act on row t at `time` and earn the next
hour's return. Nothing looks ahead.

State v3 (2026-09-20, USER): a wide picture of the market - 108 features, every one relative (a ratio, a
volatility-normalised move, an efficiency) or a percentile (a rank inside a trailing window), never a level.
The blocks follow Deng et al. (IEEE TNNLS 2017), who fed "the raw price changes of the last 45 min and the
momentum change to the previous 3 h, 5 h, 1 day, 3 days, and 10 days" (f_t in R^50) - the same structure at an
hourly clock - and extend it with longer horizons and rank versions of the same quantities.

    .venv/bin/python data_pipeline/hourly_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/bitmex_xbtusd_1m_2016_2026_08_01'
OUT = ROOT / 'data/processed/xbtusd_hourly_features.parquet'

# Window lengths in hours.
HOUR, DAY, WEEK, MONTH, YEAR = 1, 24, 168, 672, 8760
QUARTER = 2160                                    # three months
LAGS = 45                                         # Deng et al.'s 45 raw changes, here 45 hours
MOMENTUM = {'3h': 3, '5h': 5, '1d': 24, '3d': 72, '10d': 240,               # Deng's five horizons
            '2d': 48, '1w': 168, '2w': 336, '4w': 672, '8w': 1344, '13w': 2184, '26w': 4368}
MOMENTUM_RANKED = ('1d', '1w', '4w', '13w')       # which momenta also enter as a rank within the past year
RV_RATIOS = {'6h': 6, '12h': 12, '1d': 24, '3d': 72, '1w': 168, '2w': 336}  # realised variance against the month
RV_RANKED = {'1d': (DAY, ('1m', '3m', '1y')), '1w': (WEEK, ('1m', '3m', '1y')), '4w': (MONTH, ('3m', '1y'))}
VOLUME_RATIOS = {'1h': 1, '6h': 6, '1d': 24, '1w': 168}                     # volume against the month's average
VOLUME_RANKED = {'1h': 1, '1d': 24, '1w': 168}
RANK_WINDOWS = {'1m': 720, '3m': QUARTER, '1y': YEAR}
RANGE_POSITION = {'1d': 24, '1w': 168, '4w': 672, '13w': 2184, '1y': 8760}  # stochastic %K horizons
DRAWDOWN = {'4w': 672, '13w': 2184, '1y': 8760}
TREND_EFFICIENCY = {'1d': 24, '3d': 72, '1w': 168, '2w': 336, '4w': 672, '13w': 2184}
# State v4 additions (2026-09-21): trend persistence and regime smoothness, the properties that set 2023 apart
# (ARCHITECTURE.md S8.5): how established and how far-run the trend is, how persistent, how choppy, and whether the
# turbulence comes from rallies or selloffs.
MOVING_AVERAGES = {'50d': 50 * DAY, '200d': 200 * DAY}
UP_DAY_WINDOWS = {'1w': 7, '4w': 28, '13w': 91}                   # in days
CROSSING_WINDOWS = {'4w': MONTH, '13w': 2184}
SEMIVOL_WINDOWS = {'1w': WEEK, '4w': MONTH}


def hourly_bars():
    """Minute bars -> hourly close and volume. The label is the hour's END, so row t is known at time t."""
    bars = pd.read_parquet(RAW / 'XBTUSD-1m-OHLCV.parquet', columns=['open_time', 'close', 'volume'])
    bars = bars.set_index('open_time')
    hourly = pd.DataFrame({
        'close': bars.close.resample('1h', label='right', closed='left').last(),
        'volume': bars.volume.resample('1h', label='right', closed='left').sum(),
    })
    hourly['close'] = hourly.close.ffill()            # an hour with no trade keeps the last price
    return hourly.dropna()


def funding_series(index):
    """Two hourly series: the last rate settled at or before each hour, and the rate settled IN that hour (else 0)."""
    funding = pd.read_parquet(RAW / 'funding/XBTUSD-funding.parquet', columns=['timestamp', 'fundingRate'])
    rate = funding.set_index('timestamp').fundingRate.sort_index()
    settled = rate.resample('1h', label='right', closed='left').sum().reindex(index).fillna(0.0)
    return rate.reindex(index, method='ffill').fillna(0.0), settled


def build():
    h = hourly_bars()
    log_close = np.log(h.close)
    r = log_close.diff()                                               # r_t = ln(P_t / P_{t-1}), hourly log return

    # Realised variance over w hours: RV_w = mean(r^2 over the last w hours); sigma_d = sqrt(24 * RV_4w) is the
    # four-week estimate of a DAY's return volatility, and is the unit every price move below is measured in.
    rv = {name: (r ** 2).rolling(window).mean() for name, window in
          {'1d': DAY, '1w': WEEK, '4w': MONTH, **{k: v for k, v in RV_RATIOS.items()}}.items()}
    sigma_d = np.sqrt(DAY * rv['4w'])
    sigma_h = sigma_d / np.sqrt(DAY)                                   # one hour's typical move

    f = pd.DataFrame(index=h.index)

    # --- A. Deng's raw block: the last 45 hourly returns, each in units of a normal hour -------------------------
    for i in range(LAGS):
        f[f'ret_lag_{i:02d}'] = r.shift(i) / sigma_h

    # --- B, C. Momentum over twelve horizons ---------------------------------------------------------------------
    # m_k = ln(P_t / P_{t-k}) / (sigma_d * sqrt(k / 24)): the move over k hours in units of a normal k-hour move.
    momentum = {}
    for name, k in MOMENTUM.items():
        momentum[name] = (log_close - log_close.shift(k)) / (sigma_d * np.sqrt(k / DAY))
        f[f'mom_{name}'] = momentum[name]

    # --- D. Momentum percentiles: where today's move sits among the past year's moves of the same length ---------
    for name in MOMENTUM_RANKED:
        f[f'mom_pct_{name}'] = momentum[name].rolling(YEAR).rank(pct=True)

    # --- E. Volatility ratios: how turbulent the last k hours were against the four-week baseline -----------------
    for name in RV_RATIOS:
        f[f'rv_ratio_{name}'] = 0.5 * np.log(rv[name] / rv['4w'])

    # --- F. Volatility percentiles: the same turbulence as a rank inside 1 month / 3 months / 1 year -------------
    for name, (_, windows) in RV_RANKED.items():
        for label in windows:
            f[f'rv{name}_pct_{label}'] = rv[name].rolling(RANK_WINDOWS[label]).rank(pct=True)

    # --- G. Volume ratios: recent activity against the month's average hourly volume -----------------------------
    month_volume = h.volume.rolling(MONTH).mean()
    volume = {}
    for name, k in VOLUME_RATIOS.items():
        volume[name] = h.volume.rolling(k).mean()
        f[f'vol_ratio_{name}'] = np.log1p(volume[name]) - np.log1p(month_volume)

    # --- H. Volume percentiles: activity as a rank, which absorbs the six-fold decline in BitMEX volume ----------
    for name, k in VOLUME_RANKED.items():
        series = h.volume.rolling(k).mean()
        for label, window in RANK_WINDOWS.items():
            f[f'v{name}_pct_{label}'] = series.rolling(window).rank(pct=True)

    # --- I. Position in range (stochastic %K): 0 at the low of the window, 1 at the high ------------------------
    for name, k in RANGE_POSITION.items():
        low, high = h.close.rolling(k).min(), h.close.rolling(k).max()
        f[f'range_pos_{name}'] = (h.close - low) / (high - low)

    # --- J. Distance below the running high, in daily volatilities ----------------------------------------------
    for name, k in DRAWDOWN.items():
        f[f'dd_{name}'] = (log_close - log_close.rolling(k).max()) / sigma_d

    # --- K. Trend efficiency (Kaufman 1995): net move / path length, in [-1, 1] ----------------------------------
    path = r.abs()
    for name, k in TREND_EFFICIENCY.items():
        f[f'teff_{name}'] = (log_close - log_close.shift(k)) / path.rolling(k).sum()

    # --- M. Trend persistence and smoothness (state v4) ------------------------------------------------------
    close = h.close
    for name, k in MOVING_AVERAGES.items():
        average = close.rolling(k).mean()
        # How far the price has run from its moving average, in daily volatilities: (ln P - ln MA) / sigma_d.
        f[f'ma_dist_{name}'] = (log_close - np.log(average)) / sigma_d
        # How established the trend is: log(1 + hours since the price last crossed its moving average).
        side = np.sign(close - average)
        run = (side != side.shift()).cumsum()
        f[f'ma_age_{name}'] = np.log1p(side.groupby(run).cumcount())
        if name == '50d':
            crossed = (side != side.shift()).astype(float)
    # Persistence: the share of up-days among the last n days, from daily returns taken at this hour of the day.
    up = ((log_close - log_close.shift(DAY)) > 0).astype(float)
    phase = np.arange(len(up)) % DAY
    for name, days in UP_DAY_WINDOWS.items():
        f[f'up_days_{name}'] = up.groupby(phase).transform(lambda x: x.rolling(days).mean())
    # Choppiness: how often the price crossed its 50-day average recently, as a rank within the past year.
    for name, k in CROSSING_WINDOWS.items():
        f[f'ma_cross_pct_{name}'] = crossed.rolling(k).sum().rolling(YEAR).rank(pct=True)
    # Where the turbulence comes from: 1/2 ln(sum r^2 on up-hours / sum r^2 on down-hours); > 0 when rallies dominate.
    for name, k in SEMIVOL_WINDOWS.items():
        upside = (r.clip(lower=0) ** 2).rolling(k).sum()
        downside = (r.clip(upper=0) ** 2).rolling(k).sum()
        f[f'semivol_ratio_{name}'] = 0.5 * np.log(upside / downside)
    # Regime stability: the standard deviation of log daily realised variance over four weeks (volatility of volatility).
    f['vol_of_vol_4w'] = np.log(rv['1d']).rolling(MONTH).std()

    # --- L. Funding: level as a rank, weekly change, and what was actually paid over a week and a month ----------
    rate, settled = funding_series(h.index)
    for label, window in RANK_WINDOWS.items():
        f[f'fund_pct_{label}'] = rate.rolling(window).rank(pct=True)
    f['fund_chg_1w'] = (rate - rate.rolling(WEEK).mean()) / rate.rolling(YEAR).std()
    for name, k in (('1w', WEEK), ('4w', MONTH)):
        paid = settled.rolling(k).sum()                                # total funding rate paid over the window
        f[f'fund_cum_{name}'] = paid / paid.rolling(YEAR).std()

    f = f.replace([np.inf, -np.inf], np.nan)
    # A year of identical funding rates has zero spread; "no change" is then the right value, not a missing one.
    for name in ('fund_chg_1w', 'fund_cum_1w', 'fund_cum_4w'):
        f[name] = f[name].fillna(0.0)
    table = pd.concat([h[['close']], f[list(FEATURES)]], axis=1).dropna()              # drop the warm-up year
    table.index.name = 'time'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_parquet(OUT, index=False)
    print(f'{len(table.columns) - 1} features, {len(table)} hours, {table.index[0]} -> {table.index[-1]}, '
          f'written to {OUT}')
    return table


FEATURES = tuple(
    [f'ret_lag_{i:02d}' for i in range(LAGS)]
    + [f'mom_{name}' for name in MOMENTUM]
    + [f'mom_pct_{name}' for name in MOMENTUM_RANKED]
    + [f'rv_ratio_{name}' for name in RV_RATIOS]
    + [f'rv{name}_pct_{label}' for name, (_, windows) in RV_RANKED.items() for label in windows]
    + [f'vol_ratio_{name}' for name in VOLUME_RATIOS]
    + [f'v{name}_pct_{label}' for name in VOLUME_RANKED for label in RANK_WINDOWS]
    + [f'range_pos_{name}' for name in RANGE_POSITION]
    + [f'dd_{name}' for name in DRAWDOWN]
    + [f'teff_{name}' for name in TREND_EFFICIENCY]
    + [f'ma_dist_{name}' for name in MOVING_AVERAGES] + [f'ma_age_{name}' for name in MOVING_AVERAGES]
    + [f'up_days_{name}' for name in UP_DAY_WINDOWS]
    + [f'ma_cross_pct_{name}' for name in CROSSING_WINDOWS]
    + [f'semivol_ratio_{name}' for name in SEMIVOL_WINDOWS] + ['vol_of_vol_4w']
    + [f'fund_pct_{label}' for label in RANK_WINDOWS] + ['fund_chg_1w', 'fund_cum_1w', 'fund_cum_4w'])
# State v3 (108 features) is every feature above except the twelve state-v4 additions; kept so the state-v3
# checkpoints can still be loaded and evaluated.
V4_ADDITIONS = tuple(name for name in FEATURES if name.startswith(('ma_dist_', 'ma_age_', 'up_days_', 'ma_cross_pct_',
                                                                   'semivol_ratio_', 'vol_of_vol_')))
FEATURES_V3 = tuple(name for name in FEATURES if name not in V4_ADDITIONS)

if __name__ == '__main__':
    built = build()
    assert list(built.columns) == ['close'] + list(FEATURES), 'FEATURES does not match the built table'
