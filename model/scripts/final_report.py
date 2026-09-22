"""Report-ready results of the final agent (cfg_016): trained 2017-08 -> 2024, tested on 2025-01-01 -> 2026-08-01.

The five models are the ones the multi-year search trained for its hold-out check
(results/simple_agent_hpo_multiyear/final/cfg_016/test_2026/seed_*/checkpoints/pass_10.pt): the final model of each
seed, no checkpoint chosen. Each trades one continuous, cost-free account from a flat $10,000 on 1 January 2025 to
1 August 2026 (USER 2026-09-22: the whole of 2025 and 2026 to August is the test window, covering the rally to the
all-time high, the top and the decline). Hourly equity curves are saved for plotting.

    PYTHONPATH=src .venv/bin/python scripts/final_report.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from simple_agent.config import DATA_END, SEEDS, Config, periods, year
from simple_agent.data import Scaler, load, next_returns, rows
from simple_agent.evaluate import POSITIONS, benchmarks, run_policy, score
from simple_agent.model import ActorCritic

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results/simple_agent_hpo_multiyear/final/cfg_016/test_2026'
OUT = ROOT / 'results/simple_agent_final'
KEYS = ('total_return_pct', 'equity_from_10k', 'sharpe', 'max_drawdown', 'position_changes', 'long_share',
        'short_share', 'flat_share')


def main():
    config = Config()
    train, _, _ = periods(2026)                       # trained 2017-08 -> 2024-12-31
    test = (year(2025), DATA_END)
    table = load(end=test[1])
    run = json.loads((RUNS / f'seed_{SEEDS[0]}' / 'config.json').read_text())
    scaler = Scaler(table, rows(table, train), config.obs_clip, names=run['features'])
    features, returns = scaler.transform(table), next_returns(table)
    report = {'settings': run['config'], 'features': len(run['features']), 'periods': run['periods'], 'seeds': {}}
    curves = {}
    for label, period in (('test', test),):
        period_rows = rows(table, period)
        times = table.time.iloc[period_rows[:-1]].to_numpy()
        for seed in SEEDS:
            model = ActorCritic(len(run['features']) + 3, run['config']['hidden'])
            model.load_state_dict(torch.load(RUNS / f'seed_{seed}' / 'checkpoints' / 'pass_10.pt')['model'])
            result, positions, R, _ = run_policy(model, features, returns, period_rows, config.warmup_hours)
            report['seeds'].setdefault(seed, {})[label] = {k: result[k] for k in KEYS}
            curves[(label, f'seed_{seed}')] = 10_000 * np.cumprod(1 + R)
            curves[(label, f'position_{seed}')] = positions        # -1 short, 0 flat, +1 long, held for the next hour
        r = returns[period_rows[:-1]]
        bench = benchmarks(table, returns, period_rows)
        report[f'benchmarks_{label}'] = {'buy_and_hold': {k: bench['buy_and_hold'][k] for k in KEYS},
                                         'sma_50d': {k: bench['sma_50d'][k] for k in KEYS},
                                         'random_uniform_total_log_return': bench['random_uniform_total_log_return']}
        curves[(label, 'buy_and_hold')] = 10_000 * np.cumprod(1 + r)
        frame = pd.DataFrame({name: values for (lab, name), values in curves.items() if lab == label})
        frame.insert(0, 'time', times)
        OUT.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUT / f'equity_{label}.csv', index=False)
        agent = [report['seeds'][s][label] for s in SEEDS]
        report[f'average_{label}'] = {'equity_from_10k': float(np.mean([a['equity_from_10k'] for a in agent])),
                                      'total_return_pct': float(np.mean([a['total_return_pct'] for a in agent])),
                                      'sharpe': float(np.mean([a['sharpe'] for a in agent])),
                                      'profitable_seeds': int(sum(a['total_return_pct'] > 0 for a in agent))}
        print(f'\n{label}:')
        for s in SEEDS:
            a = report['seeds'][s][label]
            print(f"  seed {s}: {a['total_return_pct']:+6.1f}%  ${a['equity_from_10k']:>9,.0f}  Sharpe {a['sharpe']:+.2f}  "
                  f"max DD {a['max_drawdown'] * 100:5.1f}%  changes {a['position_changes']:5d}  "
                  f"long {a['long_share'] * 100:.0f}% short {a['short_share'] * 100:.0f}% flat {a['flat_share'] * 100:.0f}%")
        av = report[f'average_{label}']
        print(f"  average: {av['total_return_pct']:+.1f}%  ${av['equity_from_10k']:,.0f}  Sharpe {av['sharpe']:+.2f}  "
              f"profitable {av['profitable_seeds']}/5")
        for name in ('buy_and_hold', 'sma_50d'):
            b = report[f'benchmarks_{label}'][name]
            print(f"  {name}: {b['total_return_pct']:+.1f}%  ${b['equity_from_10k']:,.0f}  Sharpe {b['sharpe']:+.2f}  "
                  f"max DD {b['max_drawdown'] * 100:.1f}%")
    (OUT / 'final_report.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
