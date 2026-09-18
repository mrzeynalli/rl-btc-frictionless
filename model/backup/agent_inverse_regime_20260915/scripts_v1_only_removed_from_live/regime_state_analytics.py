"""Evidence for the regime state: relevance, redundancy and scale stability of candidate features.

Every candidate is computed on the five-minute grid from bars strictly before each
boundary. Relevance and redundancy are measured on the training years 2017-2023
only; 2024-2025 are read for scale stability alone, and nothing from 2026 is read.

Outcomes are regime properties of the next day or week - realised volatility,
trend efficiency, the payoff of holding the prevailing trend - not only the sign
of the next return. Daily samples at 00:00 UTC make the one-day outcomes
non-overlapping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent.config import DATA, TEST_START  # noqa: E402

OUT = ROOT / 'results/state_analytics'
DAY, WEEK, MONTH = 288, 2016, 8064


def spearman(a, b):
    """Rank correlation on jointly observed rows (scipy is not installed)."""
    frame = pd.DataFrame({'a': a, 'b': b}).dropna()
    return float(frame.a.rank().corr(frame.b.rank()))


bars = pd.read_parquet(DATA / 'XBTUSD-1m-OHLCV.parquet', columns=['open_time', 'close', 'volume'],
                       filters=[('open_time', '<', TEST_START)])
funding = pd.read_parquet(DATA / 'funding/XBTUSD-funding.parquet', filters=[('timestamp', '<', TEST_START)])
bars = bars.set_index('open_time').sort_index()
# Value known at each five-minute boundary: the bin [t-5, t) labelled by its end.
close = bars.close.resample('5min', label='right', closed='left').last().ffill()
volume = bars.volume.resample('5min', label='right', closed='left').sum()
close = close[close.index >= pd.Timestamp('2016-12-01', tz='UTC')]
volume = volume.reindex(close.index).fillna(0)
log_close = np.log(close)
r5 = log_close.diff().fillna(0)

f = pd.DataFrame(index=close.index)
rv_day = (r5 ** 2).rolling(DAY).sum()
rv_month_daily = (r5 ** 2).rolling(MONTH).sum() / 28
sd_month = np.sqrt(rv_month_daily)
f['volatility_regime'] = 0.5 * np.log(rv_day / rv_month_daily)
f['volatility_level'] = 0.5 * np.log(rv_month_daily)
f['log_daily_vol'] = 0.5 * np.log(rv_day)
move_week = log_close - log_close.shift(WEEK)
f['trend_efficiency_1w'] = move_week / r5.abs().rolling(WEEK).sum()
f['momentum_1w'] = move_week / (sd_month * np.sqrt(7))
f['trend_efficiency_1d'] = (log_close - log_close.shift(DAY)) / r5.abs().rolling(DAY).sum()
f['trend_efficiency_4w'] = (log_close - log_close.shift(MONTH)) / r5.abs().rolling(MONTH).sum()
r_hour = log_close - log_close.shift(12)
# Lo & MacKinlay (1988) variance ratio with overlapping hourly returns over the last week.
f['variance_ratio_1w'] = np.log((r_hour ** 2).rolling(WEEK).sum() / 12 / (r5 ** 2).rolling(WEEK).sum())
f['variance_ratio_1d'] = np.log((r_hour ** 2).rolling(DAY).sum() / 12 / (r5 ** 2).rolling(DAY).sum())
f['drawdown_4w'] = (log_close - log_close.rolling(MONTH).max()) / sd_month
f['rally_4w'] = (log_close - log_close.rolling(MONTH).min()) / sd_month
week_volume = volume.rolling(WEEK).sum() / 7
before_volume = (volume.rolling(5 * WEEK).sum() - volume.rolling(WEEK).sum()) / 28
f['volume_regime'] = np.log1p(week_volume) - np.log1p(before_volume)
rate = funding.set_index('timestamp').fundingRate.sort_index()
f['funding_rate'] = rate.reindex(f.index.union(rate.index)).ffill().reindex(f.index)
f = f.replace([np.inf, -np.inf], np.nan)

# Outcomes over the next day and week, from the boundary forward.
o = pd.DataFrame(index=close.index)
ahead_day = log_close.shift(-DAY) - log_close
o['next_log_daily_vol'] = 0.5 * np.log((r5 ** 2).rolling(DAY).sum().shift(-DAY))
o['next_day_return_z'] = ahead_day / sd_month
o['next_day_efficiency'] = ahead_day / r5.abs().rolling(DAY).sum().shift(-DAY)
o['next_day_abs_efficiency'] = o.next_day_efficiency.abs()
o['next_week_return_z'] = (log_close.shift(-WEEK) - log_close) / (sd_month * np.sqrt(7))
o['next_week_variance_ratio'] = np.log((r_hour ** 2).rolling(WEEK).sum().shift(-WEEK) / 12
                                       / (r5 ** 2).rolling(WEEK).sum().shift(-WEEK))
o['next_day_trend_payoff'] = np.sign(f.trend_efficiency_1w) * ahead_day
o['next_day_return'] = ahead_day

daily = f.index[(f.index.hour == 0) & (f.index.minute == 0)]
train = daily[(daily >= pd.Timestamp('2017-01-01', tz='UTC')) & (daily < pd.Timestamp('2024-01-01', tz='UTC'))]
F, O = f.loc[train], o.loc[train]
names = list(f.columns)
result = {'samples_train_days': int(len(train)), 'features': names}

relevance = {}
for outcome in ('next_log_daily_vol', 'next_day_abs_efficiency', 'next_week_variance_ratio',
                'next_day_return_z', 'next_week_return_z', 'next_day_trend_payoff'):
    relevance[outcome] = {n: round(spearman(F[n], O[outcome]), 3) for n in names}
result['spearman_relevance_2017_2023'] = relevance

yearly = {}
for n in names:
    rows = {}
    for year in range(2017, 2024):
        mask = train.year == year
        rows[year] = {'vol': round(spearman(F[n][mask], O.next_log_daily_vol[mask]), 3),
                      'payoff': round(spearman(F[n][mask], O.next_day_trend_payoff[mask]), 3)}
    yearly[n] = rows
result['yearly_relevance'] = yearly

result['redundancy_spearman'] = F[names].rank().corr().round(2).to_dict()


def quintile_sharpe(feature, payoff):
    frame = pd.DataFrame({'x': F[feature], 'y': O[payoff]}).dropna()
    frame['q'] = pd.qcut(frame.x.rank(method='first'), 5, labels=False)
    out = []
    for q, g in frame.groupby('q'):
        out.append({'quintile': int(q) + 1, 'feature_median': round(float(g.x.median()), 4),
                    'annualised_sharpe': round(float(g.y.mean() / g.y.std() * np.sqrt(365)), 2),
                    'days': int(len(g))})
    return out


conditional = {}
for n in ('trend_efficiency_1w', 'variance_ratio_1w', 'volatility_regime', 'volatility_level',
          'drawdown_4w', 'rally_4w', 'volume_regime', 'funding_rate'):
    conditional[n] = {'trend_payoff': quintile_sharpe(n, 'next_day_trend_payoff'),
                      'long_payoff': quintile_sharpe(n, 'next_day_return')}
absolute = F.trend_efficiency_1w.abs()
frame = pd.DataFrame({'x': absolute, 'y': O.next_day_trend_payoff}).dropna()
frame['q'] = pd.qcut(frame.x.rank(method='first'), 5, labels=False)
conditional['abs_trend_efficiency_1w'] = {'trend_payoff': [
    {'quintile': int(q) + 1, 'feature_median': round(float(g.x.median()), 4),
     'annualised_sharpe': round(float(g.y.mean() / g.y.std() * np.sqrt(365)), 2)} for q, g in frame.groupby('q')]}
# Trend payoff among strong-trend days, split by the hourly variance ratio.
strong = frame.x >= frame.x.quantile(0.6)
vr = F.variance_ratio_1w.reindex(frame.index)
split = {}
for label, mask in (('variance_ratio_above_median', strong & (vr >= vr.median())),
                    ('variance_ratio_below_median', strong & (vr < vr.median()))):
    y = frame.y[mask]
    split[label] = {'days': int(mask.sum()), 'annualised_sharpe': round(float(y.mean() / y.std() * np.sqrt(365)), 2)}
conditional['strong_trend_by_variance_ratio'] = split
result['conditional_sharpe_2017_2023'] = conditional

stability = {}
for n in names:
    rows = {}
    for year in range(2017, 2026):
        values = f[n][f.index.year == year].dropna()
        rows[year] = {'median': round(float(values.median()), 4),
                      'iqr': round(float(values.quantile(0.75) - values.quantile(0.25)), 4)}
    stability[n] = rows
result['scale_stability_by_year'] = stability
(OUT / 'regime_state_analytics.json').write_text(json.dumps(result, indent=1))

print('train days', len(train))
print(pd.DataFrame(relevance).to_string())
print(F[names].rank().corr().round(2).to_string())
for n, v in conditional.items():
    if n == 'strong_trend_by_variance_ratio':
        print(n, v)
        continue
    print(n, 'trend', [q['annualised_sharpe'] for q in v['trend_payoff']],
          'long', [q['annualised_sharpe'] for q in v.get('long_payoff', [])])
print('yearly vol relevance')
print(pd.DataFrame({n: {y: yearly[n][y]['vol'] for y in yearly[n]} for n in names}).T.to_string())
print('yearly trend-payoff relevance')
print(pd.DataFrame({n: {y: yearly[n][y]['payoff'] for y in yearly[n]} for n in names}).T.to_string())
print('scale medians')
print(pd.DataFrame({n: {y: stability[n][y]['median'] for y in stability[n]} for n in names}).T.to_string())
print('scale IQR')
print(pd.DataFrame({n: {y: stability[n][y]['iqr'] for y in stability[n]} for n in names}).T.to_string())

# --- incremental information: does each feature add to the others? -----------------
# Linear R-squared on standardised ranks. A feature that only repeats another adds
# nothing here even when it is individually relevant.
CANDIDATES = ['trend_efficiency_1w', 'trend_efficiency_1d', 'trend_efficiency_4w', 'momentum_1w',
              'variance_ratio_1w', 'variance_ratio_1d', 'drawdown_4w', 'rally_4w',
              'volatility_level', 'volatility_regime', 'volume_regime', 'funding_rate']
CHOSEN = ['trend_efficiency_1w', 'drawdown_4w', 'volatility_level',
          'volatility_regime', 'volume_regime', 'funding_rate']
OUTCOMES = ['next_log_daily_vol', 'next_day_abs_efficiency', 'next_week_variance_ratio',
            'next_day_return_z', 'next_week_return_z', 'next_day_trend_payoff']
joint = pd.concat([F[CANDIDATES], O[OUTCOMES]], axis=1).dropna()
ranked = joint.rank()
Z = (ranked - ranked.mean()) / ranked.std()


def r_squared(columns, outcome):
    X = np.column_stack([np.ones(len(Z))] + [Z[c].to_numpy() for c in columns])
    y = Z[outcome].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - np.var(y - X @ beta) / np.var(y))


incremental = {'samples': int(len(Z)), 'forward_selection': {}, 'drop_one_from_chosen': {}, 'add_to_chosen': {}}
for outcome in OUTCOMES:
    chosen, total, path = [], 0.0, []
    for _ in range(5):
        gains = {c: r_squared(chosen + [c], outcome) - total for c in CANDIDATES if c not in chosen}
        best = max(gains, key=gains.get)
        chosen.append(best)
        total += gains[best]
        path.append([best, round(gains[best], 4)])
    incremental['forward_selection'][outcome] = path
    full = r_squared(CHOSEN, outcome)
    incremental['drop_one_from_chosen'][outcome] = {'r2': round(full, 4), **{
        c: round(full - r_squared([x for x in CHOSEN if x != c], outcome), 4) for c in CHOSEN}}
    incremental['add_to_chosen'][outcome] = {
        c: round(r_squared(CHOSEN + [c], outcome) - full, 4) for c in CANDIDATES if c not in CHOSEN}
result['incremental_rank_r2_2017_2023'] = incremental
result['variance_ratio_persistence_by_year'] = {
    year: round(spearman(F.variance_ratio_1w[train.year == year], O.next_week_variance_ratio[train.year == year]), 3)
    for year in range(2017, 2024)}
(OUT / 'regime_state_analytics.json').write_text(json.dumps(result, indent=1))
print('INCREMENTAL')
print(json.dumps(incremental, indent=1))
print('variance ratio persistence by year', result['variance_ratio_persistence_by_year'])
