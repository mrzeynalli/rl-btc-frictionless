"""THE test entrypoint. It did not exist until now, and running it spends the window.

USER (2026-09-12): "test all the seeds individually (not the ensemble) on test data
(make sure not to save any training logs so we dont learn anything; just report the
returns and sharpe ratios of each one and its comparison with always long and always
short strategies)".

One pass over 2026-01-01 .. 2026-08-01 (the extent of the collected data), every run at
the checkpoint already selected on validation, four sampling seeds each, from a flat
account. Nothing is written that a later decision could be fitted to: no per-decision
frames, no per-checkpoint scores, no model state. The single file it writes is
TEST_LOCK.json, and only to record that the seal is broken.

Features are causal, so the test market is rebuilt from the raw bars with the full
2017-2025 history flowing in; each run keeps the scaler saved inside its own checkpoint,
fitted on its own training span alone. Nothing is refitted on the test window.

USER (2026-09-13): "test the fully trained agent on 2026 test data without saving any
training logs again just to see the performance" - a second, descriptive look at the
already-spent window, with --checkpoint final (pass 100). It cannot inform a selection.

    PYTHONPATH=src .venv/bin/python scripts/test_window.py --spend-the-test-window [--checkpoint final]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import agent.config as config
import agent.data as data
from agent.config import OUTPUT, SEEDS, YEARS, Config, write_json

BEGIN = pd.Timestamp('2026-01-01', tz='UTC')
END = pd.Timestamp('2026-08-01', tz='UTC')   # the last whole month of collected data
FOLD_TRAINING = {2024: '2017-2023', 2025: '2017-2024'}


def usd_sharpe(frame):
    """Annualised Sharpe of the account measured in dollars rather than in BTC."""
    value = np.r_[frame.equity_usd.iloc[0] / (1 + frame.log_return.iloc[0]), frame.equity_usd.to_numpy()]
    daily = pd.Series(np.diff(np.log(value)), index=frame.time).groupby(frame.time.dt.floor('D').values).sum()
    daily = np.expm1(daily.to_numpy())
    spread = daily.std(ddof=1)
    return float(daily.mean() / spread * np.sqrt(365)) if spread else 0.0


def describe(aggregate, frame):
    """One row per evaluated policy: the spread over sampling seeds, never one number."""
    per_seed = aggregate['per_seed']
    usd = [usd_sharpe(frame[frame.seed == r['seed']]) for r in per_seed]
    take = lambda key: np.array([r[key] for r in per_seed], dtype=float)
    return {
        'return_btc': take('total_return_btc'), 'sharpe_btc': take('daily_sharpe_btc'),
        'return_usd': take('total_return_usd'), 'sharpe_usd': np.array(usd),
        'drawdown': take('max_drawdown_btc'), 'orders': take('orders'),
        'holding_hours': take('median_holding_spell_hours'),
        'long': np.array([r['side_share']['long'] for r in per_seed]),
        'short': np.array([r['side_share']['short'] for r in per_seed]),
        'flat': np.array([r['side_share']['flat'] for r in per_seed]),
        'liquidations': int(sum(r['liquidations'] for r in per_seed)),
    }


def row(name, d):
    span = lambda v: f'{v.mean():+7.2%} [{v.min():+6.2%},{v.max():+6.2%}]' if len(v) > 1 else f'{v.mean():+7.2%}'
    sharpe = lambda v: f'{v.mean():+5.2f}' + (f' [{v.min():+5.2f},{v.max():+5.2f}]' if len(v) > 1 else '')
    return (f"{name:<16} {span(d['return_btc']):>28} {sharpe(d['sharpe_btc']):>20} "
            f"{span(d['return_usd']):>28} {sharpe(d['sharpe_usd']):>20} "
            f"{d['drawdown'].mean():+7.1%} {d['orders'].mean():>7.0f} {d['holding_hours'].mean():>7.0f} "
            f"{d['long'].mean():>5.0%}/{d['short'].mean():<4.0%} {d['liquidations']:>3}")


HEAD = (f"{'policy':<16} {'return BTC (mean [min,max])':>28} {'Sharpe BTC':>20} "
        f"{'return USD (mean [min,max])':>28} {'Sharpe USD':>20} {'maxDD':>7} {'orders':>7} {'hold h':>7} "
        f"{'long/short':>10} {'liq':>3}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spend-the-test-window', action='store_true', required=True,
                        help='acknowledge that this evaluates the sealed window')
    parser.add_argument('--checkpoint', choices=('selected', 'final'), default='selected',
                        help='the checkpoint selected on validation, or the last one (pass 100)')
    parser.add_argument('--experiment', type=Path, default=OUTPUT / 'sharpe_bars12')
    parser.add_argument('--cache', type=Path, default=Path('/root/.cache/agent_test_window'),
                        help='market arrays including the test window, outside the project')
    args = parser.parse_args()

    # The one deliberate unlock, authorised by the user on 2026-09-12. Everything
    # downstream reads the guard at call time, so this single assignment is the whole
    # entrypoint; no guard is removed from the code.
    config.TEST_START = data.TEST_START = END
    from agent.data import Scaler, prepare
    from agent.evaluate import evaluate
    from agent.experiment import fold_configs
    from agent.network import ActorCritic
    from agent.train import run_path

    print(f'TEST WINDOW OPENED {BEGIN.date()} .. {END.date()}  ({(END - BEGIN).days} days)', flush=True)
    market = prepare(args.cache)
    print(f'market rebuilt through {market.audit["end_exclusive"]}', flush=True)
    selection = json.loads((args.experiment / 'comparison/selection.json').read_text())
    if args.checkpoint == 'selected':
        updates = selection['selected']['updates']
        label = f"selected checkpoint {selection['selected']['checkpoint']} of 20 (pass {round(selection['selected']['passes'])})"
    else:
        updates = {}
        for year in YEARS:
            for seed in SEEDS:
                fold = json.loads((run_path(args.experiment, year, seed) / 'training.json').read_text())['fold']
                updates[f'{year}_{seed}'] = fold['checkpoints'][-1]
        label = 'final checkpoint 20 of 20 (pass 100)'
    base = Config()
    configs = fold_configs(args.experiment, base, market)

    actions = ('argmax actions, one deterministic path per policy' if base.evaluation_policy == 'argmax'
               else f'sampled actions over {base.evaluation_seeds} seeds per policy')
    print(f"\n{label}; {actions}, flat start, "
          f'{base.fee * 1e4:.0f} bp fee + {base.slippage * 1e4:.0f} bp slippage per leg\n')

    config_2025 = configs[2025][0]
    benchmarks = {}
    for name in ('long', 'short', 'flat'):
        aggregate, frame = evaluate([], market, None, config_2025, BEGIN, END, baseline=name)
        benchmarks[name] = describe(aggregate, frame)
    price = None
    for year in YEARS:
        fold_config_, _ = configs[year]
        print(f'== models trained {FOLD_TRAINING[year]}, validated on {year}, {label} '
              f'(update {updates[f"{year}_{SEEDS[0]}"]})')
        print(HEAD)
        for seed in SEEDS:
            update = updates[f'{year}_{seed}']
            saved = torch.load(run_path(args.experiment, year, seed) / f'policy_{update:05d}.pt',
                               map_location='cpu', weights_only=False)
            model = ActorCritic(fold_config_)
            model.load_state_dict(saved['model'])
            aggregate, frame = evaluate([model], market, Scaler.from_dict(saved['scaler']),
                                        fold_config_, BEGIN, END)
            if price is None:
                price = (float(frame.mark.iloc[0]), float(frame[frame.seed == 0].mark.iloc[-1]))
            print(row(f'seed {seed}', describe(aggregate, frame)), flush=True)
        print()
    print('== benchmarks over the same window')
    print(HEAD)
    for name in ('long', 'short', 'flat'):
        print(row({'long': 'always long', 'short': 'always short', 'flat': 'flat (hold BTC)'}[name], benchmarks[name]))
    print(f'\nBTC {price[0]:,.0f} -> {price[1]:,.0f} ({price[1] / price[0] - 1:+.2%}) over the test window.')

    lock_path = args.experiment / 'TEST_LOCK.json'
    lock = json.loads(lock_path.read_text())
    accesses = lock.pop('accesses', None) or [{
        'accessed_at_utc': lock.get('accessed_at_utc'), 'checkpoint': 'selected',
        'authorised_by': lock.get('authorised_by'), 'policies_evaluated': lock.get('policies_evaluated')}]
    accesses.append({'accessed_at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': args.checkpoint,
                     'updates': updates, 'policies_evaluated': len(SEEDS) * len(YEARS),
                     'authorised_by': {'selected': 'USER (2026-09-12): test every seed individually on the test data',
                                       'final': 'USER (2026-09-13): test the fully trained agent on 2026 test data'
                                       }[args.checkpoint]})
    write_json(lock_path, {
        'locked': False, 'start_inclusive': str(BEGIN), 'end_exclusive': str(END),
        'performance_accessed': True, 'unlock_entrypoint': 'scripts/test_window.py',
        'accesses': accesses,
        'note': 'each access is one pass at a checkpoint fixed before the look; no per-decision frames, '
                'per-checkpoint scores or model state were written. The window is spent: it cannot '
                'select or tune anything'})
    print(f'\nTEST_LOCK.json records {len(accesses)} accesses. No other file was written.')


if __name__ == '__main__':
    main()
