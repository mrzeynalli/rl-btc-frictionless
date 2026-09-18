"""Compare an experiment's selected checkpoint with the permanent benchmark set, on its validation folds.

The agent's side of the comparison comes from the decision-level validation frames already saved at the
selected checkpoint (every seed); every benchmark is simulated through the same ledger, quarter by quarter
from a flat account (src/agent/benchmarks.py). The turnover-matched random policy is fitted to the agent's
own hour-to-hour position changes in each fold. Development data only; the test window stays locked.

    PYTHONPATH=src .venv/bin/python scripts/benchmark_report.py --experiment results/agent_v2/main [--paths 1000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.benchmarks import (BENCHMARKS, RANDOM, annualised_sharpe, fit_transitions, path_metrics, simulate,
                              volatility_scale)
from agent.config import ROOT, TRAIN_START, Config, write_json
from agent.data import prepare
from agent.evaluate import quarter_windows
from agent.experiment import fold_configs
from agent.scoring import interquartile_mean
from agent.train import fold_end, run_path


def agent_paths(experiment, base, year, updates):
    """The selected checkpoint's validation quarters, one path per seed."""
    quarters = []
    for seed in base.seeds:
        update = updates[f'{year}_{seed}']
        run = run_path(experiment, year, seed)
        frame = pd.read_csv(run/f'validation_{update:05d}.csv.gz', parse_dates=['time'],
                            usecols=['time', 'quarter', 'log_return', 'equity', 'contracts'])
        record = {q['quarter']: q for q in json.loads((run/f'validation_{update:05d}.json').read_text())['quarters']}
        quarters.append((frame, record))
    out = []
    for q in range(1, base.fold_quarters[year] + 1):
        frames = [f[f.quarter == q].sort_values('time') for f, _ in quarters]
        records = [r[q] for _, r in quarters]
        out.append({'log_return': np.stack([f.log_return.to_numpy() for f in frames]),
                    'equity': np.stack([f.equity.to_numpy() for f in frames]),
                    'side': np.stack([np.sign(f.contracts.to_numpy()) for f in frames]),
                    'times': pd.DatetimeIndex(frames[0].time),
                    'orders': np.array([r['per_seed'][0]['orders'] for r in records], dtype=float),
                    'fees': np.array([r['per_seed'][0]['fees'] for r in records]),
                    'funding': np.array([r['per_seed'][0]['funding'] for r in records])})
    return out


def summarise(quarters):
    """Across a fold's quarters, per path: year log return, Sharpe, worst drawdown, trading and exposure."""
    metrics = [path_metrics(q['log_return'], q['equity'], q['side'], q['times']) for q in quarters]
    quarterly = np.stack([m['total_log_return'] for m in metrics], axis=1)          # paths x quarters
    daily = np.concatenate([m['daily_returns'] for m in metrics], axis=1)
    return {'quarterly': quarterly, 'year': quarterly.sum(1), 'sharpe': annualised_sharpe(daily),
            'worst_drawdown': np.min(np.stack([m['max_drawdown'] for m in metrics], axis=1), axis=1),
            'orders_per_quarter': np.mean(np.stack([q['orders'] for q in quarters], axis=1), axis=1),
            'fees_per_year': np.sum(np.stack([q['fees'] for q in quarters], axis=1), axis=1),
            'funding_per_year': np.sum(np.stack([q['funding'] for q in quarters], axis=1), axis=1),
            'long_share': np.mean(np.stack([m['long_share'] for m in metrics], axis=1), axis=1),
            'short_share': np.mean(np.stack([m['short_share'] for m in metrics], axis=1), axis=1)}


def describe(summary, kind):
    def stat(values):
        values = np.asarray(values, dtype=float)
        if kind == 'random':
            return {'median': float(np.median(values)), 'p05': float(np.percentile(values, 5)),
                    'p95': float(np.percentile(values, 95))}
        if kind == 'agent':
            return {'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max()),
                    'per_seed': values.round(4).tolist()}
        return {'value': float(values[0])}
    return {key: stat(value) for key, value in summary.items() if key != 'quarterly'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT/'results/agent_v2/main')
    parser.add_argument('--paths', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    output = args.output or ROOT/'results/benchmarks'/experiment.relative_to(ROOT/'results').as_posix().replace('/', '_')
    base = Config.from_dict(json.loads((experiment/'protocol.json').read_text())['config'])
    selection = json.loads((experiment/'comparison/selection.json').read_text())
    updates = selection['selected']['updates']
    market = prepare()
    configs = fold_configs(experiment, base, market)
    report = {'experiment': str(experiment), 'selected_checkpoint': selection['selected']['checkpoint'],
              'selected_passes': selection['selected']['passes'], 'paths': args.paths, 'folds': {},
              'test_locked': True, 'test_performance_accessed': False}
    quarterly_rows, pooled = [], {}
    for year in base.fold_quarters:
        config, _ = configs[year]
        agent = agent_paths(experiment, base, year, updates)
        transitions = fit_transitions([side for q in agent for side in q['side']])
        first, last = market.bounds(TRAIN_START, fold_end(year))
        scale = volatility_scale(market, first, last)
        fold = {'agent': summarise(agent)}
        for name in BENCHMARKS:
            quarters = []
            for number, (begin, finish) in enumerate(quarter_windows(year, base.fold_quarters[year]), 1):
                quarters.append(simulate(name, market, config, begin, finish, paths=args.paths,
                                         seed=year * 10 + number, transitions=transitions, scale=scale))
                print(f'{year} Q{number} {name} done', flush=True)
            fold[name] = summarise(quarters)
        report['folds'][str(year)] = {
            'matched_transitions': transitions.round(4).tolist(), 'volatility_scale_c': scale,
            'strategies': {name: describe(s, 'agent' if name == 'agent' else 'random' if name in RANDOM else 'rule')
                           for name, s in fold.items()},
            'agent_percentile_in_random': {
                name: {'seed_mean_year': float((fold[name]['year'] < fold['agent']['year'].mean()).mean()),
                       'per_seed_year': [float((fold[name]['year'] < v).mean()) for v in fold['agent']['year']]}
                for name in RANDOM}}
        for name, s in fold.items():
            pooled.setdefault(name, []).append(s)
            for path_index in range(min(len(s['quarterly']), 5 if name in RANDOM else len(s['quarterly']))):
                for q, value in enumerate(s['quarterly'][path_index], 1):
                    quarterly_rows.append({'fold': year, 'quarter': q, 'strategy': name, 'path': path_index,
                                           'log_return': value})
    both = {}
    for name, parts in pooled.items():
        years = np.sum([p['year'] for p in parts], axis=0)
        quarters = np.concatenate([p['quarterly'] for p in parts], axis=1)
        both[name] = {'two_year_log_return': years, 'quarter_iqm': interquartile_mean(quarters) if name == 'agent'
                      else np.array([interquartile_mean(row) for row in quarters])}
    report['both_folds'] = {name: {k: (describe({k: v}, 'agent' if name == 'agent' else 'random' if name in RANDOM
                                                else 'rule')[k] if np.ndim(v) else float(v)) for k, v in item.items()}
                            for name, item in both.items()}
    report['both_folds_agent_percentile_in_random'] = {
        name: float((both[name]['two_year_log_return'] < both['agent']['two_year_log_return'].mean()).mean())
        for name in RANDOM}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'benchmarks.json', report)
    pd.DataFrame(quarterly_rows).to_csv(output/'quarterly_sample.csv', index=False)
    print(json.dumps({'output': str(output), 'percentile': report['both_folds_agent_percentile_in_random']}, indent=1))


if __name__ == '__main__':
    main()
