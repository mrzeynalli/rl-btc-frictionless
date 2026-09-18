"""Pre-registered test of BitMEX XBTUSD as the pre-2020 proxy for Binance USD-M BTCUSDT.

Rules (results/agent_v2/PREREGISTRATION.md §1), on development data only:
- hourly log returns of the two venues correlate at rho >= 0.99, else the proxy is rejected;
- each candidate market feature computed from each venue alone correlates at Spearman rho >= 0.90,
  else that feature leaves the state.
Features are compared only once both venues have the longest look-back of history (Binance
launched 2019-09-08; 26 weeks later is 2020-03-08), from 2020-04-01; returns from 2020-01-01.

    PYTHONPATH=src .venv/bin/python scripts/proxy_audit.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent.config import BINANCE, BITMEX, OUTPUT, write_json
from agent.data import CANDIDATE_FEATURES, causal_arrays, load_sources

RETURNS_FROM = pd.Timestamp('2020-01-01', tz='UTC')
FEATURES_FROM = pd.Timestamp('2020-04-01', tz='UTC')
UNTIL = pd.Timestamp('2026-01-01', tz='UTC')
HISTORY_FROM = pd.Timestamp('2019-09-08', tz='UTC')


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())[0, 1])


def hourly_close(bars):
    frame = bars.dropna(subset=['close']).set_index('open_time').close
    return frame.resample('1h', label='right', closed='left').last()


def main():
    bitmex = pd.read_parquet(BITMEX / 'XBTUSD-1m-OHLCV.parquet',
                             filters=[('open_time', '>=', HISTORY_FROM), ('open_time', '<', UNTIL)])
    binance = pd.read_parquet(BINANCE / 'BTCUSDT-1m-OHLCV.parquet',
                              filters=[('open_time', '>=', HISTORY_FROM), ('open_time', '<', UNTIL)])
    hours = pd.concat({'bitmex': hourly_close(bitmex), 'binance': hourly_close(binance)}, axis=1)
    hours = hours[hours.index > RETURNS_FROM]
    returns = np.log(hours).diff().dropna()
    rho_returns = float(returns.corr().iloc[0, 1])
    gap = (returns.bitmex - returns.binance).abs()
    level = (hours.bitmex / hours.binance - 1).dropna()

    funding_b = pd.read_parquet(BITMEX / 'funding/XBTUSD-funding.parquet',
                                filters=[('timestamp', '>=', RETURNS_FROM), ('timestamp', '<', UNTIL)])
    funding_n = pd.read_parquet(BINANCE / 'BTCUSDT-funding.parquet',
                                filters=[('timestamp', '>=', RETURNS_FROM), ('timestamp', '<', UNTIL)])
    daily_funding = pd.concat({'bitmex': funding_b.set_index('timestamp').fundingRate.resample('1D').sum(),
                               'binance': funding_n.set_index('timestamp').fundingRate.resample('1D').sum()},
                              axis=1).dropna()

    venues = {
        'bitmex': causal_arrays(bitmex[['open_time', 'open', 'high', 'low', 'close', 'volume']],
                                funding_b.rename(columns={}), HISTORY_FROM, UNTIL),
        'binance': causal_arrays(binance[['open_time', 'open', 'high', 'low', 'close']].assign(volume=binance.quote_volume),
                                 funding_n, HISTORY_FROM, UNTIL),
    }
    # BitMEX funding before 2020 is needed for the BitMEX funding feature's first rows only.
    grid = pd.date_range(FEATURES_FROM, UNTIL, freq='1h', inclusive='left')
    rows = {}
    for name, market in venues.items():
        index = (grid.as_unit('ns').asi8 // 60_000_000_000 - market.epoch_start).astype(np.int64)
        rows[name] = np.asarray(market.features[index], dtype=np.float64)
    features = {}
    for column, feature in enumerate(CANDIDATE_FEATURES):
        rho = spearman(rows['bitmex'][:, column], rows['binance'][:, column])
        features[feature] = {'spearman': round(rho, 4), 'passes': bool(rho >= 0.90)}
    _, _, splice = load_sources(pd.Timestamp('2020-02-01', tz='UTC'))
    report = {
        'hourly_returns': {'from': str(RETURNS_FROM), 'until': str(UNTIL), 'hours': int(len(returns)),
                           'pearson': round(rho_returns, 5), 'passes': bool(rho_returns >= 0.99),
                           'median_abs_difference_bp': round(float(gap.median() * 1e4), 3),
                           'p99_abs_difference_bp': round(float(gap.quantile(0.99) * 1e4), 2),
                           'median_price_level_gap_bp': round(float(level.median() * 1e4), 2)},
        'daily_funding': {'days': int(len(daily_funding)),
                          'pearson': round(float(daily_funding.corr().iloc[0, 1]), 4),
                          'mean_bitmex_bp': round(float(daily_funding.bitmex.mean() * 1e4), 3),
                          'mean_binance_bp': round(float(daily_funding.binance.mean() * 1e4), 3)},
        'features': features,
        'splice': splice,
        'proxy_accepted': bool(rho_returns >= 0.99),
        'features_removed': [name for name, item in features.items() if not item['passes']],
    }
    write_json(OUTPUT / 'audit/proxy_audit.json', report)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
