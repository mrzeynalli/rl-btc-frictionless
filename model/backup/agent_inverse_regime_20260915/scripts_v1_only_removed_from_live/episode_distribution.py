"""What regimes the episode sampler actually drew, and in what proportion.

The training environment draws a uniformly random start in the fold's training span and
runs one contiguous quarter-long episode from it. Uniform is unbiased for the *base rate*
of the span: it reproduces whatever mix of rising and falling quarters the history holds,
it does not balance them. This measures that mix, the number of genuinely independent
episodes behind it, and what a constant long and a constant short earned in the training
years through the real ledger.

    PYTHONPATH=src .venv/bin/python scripts/episode_distribution.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent.config import BAR_MINUTES, OUTPUT, Config, TRAIN_START, fold_config
from agent.data import Scaler, prepare, training_decisions
from agent.evaluate import evaluate

FOLD_END = {2024: pd.Timestamp('2024-01-01', tz='UTC'), 2025: pd.Timestamp('2025-01-01', tz='UTC')}
TEST_DAYS, TEST_RETURN = 212, -0.284


def forward_returns(market, first, last, minutes):
    """Log return over `minutes` from every start the sampler could draw, 5-minute aligned."""
    close = np.asarray(market.prices[:, 3])
    starts = np.arange(first, last - minutes, BAR_MINUTES)
    return np.log(close[starts + minutes] / close[starts]), starts


def main():
    config = Config()
    market = prepare()
    episode_minutes = config.episode_steps * config.decision_bars * config.decision_minutes
    prefix_minutes = config.burn_in * config.decision_bars * config.decision_minutes
    print(f'episode: {config.episode_steps} hourly decisions = {episode_minutes / 1440:.1f} days, '
          f'plus a {prefix_minutes / 1440:.1f}-day exploring-start prefix')
    print(f'{config.environments} parallel environments, {config.rollout} decisions per rollout\n')

    for year, end in FOLD_END.items():
        first, last = market.bounds(TRAIN_START, end)
        span_days = (last - first) / 1440
        draws = round(({2024: 5991, 2025: 6849}[year] * config.samples_per_update) / config.episode_steps)
        independent = span_days / (episode_minutes / 1440)
        returns, starts = forward_returns(market, first, last, episode_minutes)
        rising = float((returns > 0).mean())
        print(f'== fold {year}: training span {TRAIN_START.date()} .. {end.date()} '
              f'({span_days / 365.25:.1f} years, {len(starts):,} distinct legal starts)')
        print(f'   episodes drawn during training      {draws:,}')
        print(f'   non-overlapping episodes available  {independent:.0f}   '
              f'(every draw overlaps ~{draws / independent:.0f} others)')
        print(f'   quarters that rose                  {rising:6.1%}   fell {1 - rising:6.1%}')
        print(f'   mean quarter log return             {returns.mean():+.3f}   '
              f'median {np.median(returns):+.3f}')
        for q in (5, 25, 50, 75, 95):
            print(f'     p{q:<3} {np.percentile(returns, q):+.3f}', end='')
        print()
        # how often the span held a decline as deep and as long as the test window
        long_minutes = TEST_DAYS * 1440
        deep, _ = forward_returns(market, first, last, long_minutes)
        share = float((deep <= np.log1p(TEST_RETURN)).mean())
        print(f'   {TEST_DAYS}-day stretches as bad as the test window ({TEST_RETURN:+.1%}): {share:6.1%}\n')

    # the baselines need a fold's measured reward prior to construct a ledger; it does
    # not enter a constant position's return, but the configuration must be a real one
    scaler = Scaler.from_dict(json.loads((OUTPUT / 'sharpe_bars12/scalers/2025.json').read_text()))
    config = fold_config(config, scaler.bar_return_sd,
                         training_decisions(market, FOLD_END[2025], config))
    print('== what the training years paid a constant position, through the real ledger '
          '(BTC log return, 12 bp round trip, funding included)')
    print('(a constant 1x inverse long is LIQUIDATED in 2018 and 2020, so those rows are wipeouts,')
    print(' not a measure of what being long paid; the quarter-return distribution above is that measure)')
    print(f"{'year':<6}{'constant long':>16}{'liq':>5}{'constant short':>16}{'liq':>5}{'flat':>8}{'BTC price':>14}")
    totals = {'long': 0.0, 'short': 0.0}
    for year in range(2017, 2025):
        begin, finish = (pd.Timestamp(f'{year}-01-01', tz='UTC'), pd.Timestamp(f'{year + 1}-01-01', tz='UTC'))
        row, liq = {}, {}
        for name in ('long', 'short'):
            aggregate, frame = evaluate([], market, None, config, begin, finish, baseline=name)
            row[name] = aggregate['per_seed'][0]['total_log_return_btc']
            liq[name] = aggregate['per_seed'][0]['liquidations']
            totals[name] += row[name]
            price = frame.mark.iloc[-1] / frame.mark.iloc[0] - 1
        print(f'{year:<6}{row["long"]:>+16.3f}{liq["long"]:>5}{row["short"]:>+16.3f}{liq["short"]:>5}'
              f'{0.0:>8.1f}{price:>+14.1%}')
    print(f'{"total":<6}{totals["long"]:>+16.3f}{"":>5}{totals["short"]:>+16.3f}')


if __name__ == '__main__':
    main()
