"""Choose one checkpoint on 2025, then test it once on 2026.

Selection: the (seed, pass) with the highest 2025 Sharpe ratio. Only that checkpoint is run on 2026, next to the
benchmarks on the same hours. The test is run once; a second run is refused unless --again is given.

    PYTHONPATH=src .venv/bin/python -m simple_agent.select_and_test [--output results/simple_agent]
"""
import argparse
import json
from pathlib import Path

import torch

from .config import OUTPUT, SEEDS, Config, periods
from .data import FEATURES, Scaler, load, next_returns, rows
from .evaluate import benchmarks, run_policy
from .model import ActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--again', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    parser.add_argument('--test-year', type=int, default=2026)
    args = parser.parse_args()
    TRAIN, VALIDATION, TEST = periods(args.test_year)
    if (args.output / 'test.json').exists() and not args.again:
        raise SystemExit('the test has already been run: see test.json')
    config = Config()

    # 1. Every seed's validation curve; pick the best 2025 Sharpe.
    candidates = []
    for seed in args.seeds:
        for line in open(args.output / f'seed_{seed}' / 'validation.jsonl'):
            item = json.loads(line)
            candidates.append({'seed': seed, **item})
    best = max(candidates, key=lambda c: c['sharpe'])

    # 2. The chosen checkpoint on 2026 (the only place 2026 rows are loaded).
    table = load(end=TEST[1])
    # The chosen run's own settings: its feature list and network width, whatever the current defaults are.
    run = json.loads((args.output / f'seed_{best["seed"]}' / 'config.json').read_text())
    scaler = Scaler(table, rows(table, TRAIN), config.obs_clip, names=run['features'])
    features, returns = scaler.transform(table), next_returns(table)
    model = ActorCritic(len(run['features']) + 3, run['config']['hidden'])
    checkpoint = torch.load(args.output / f'seed_{best["seed"]}' / 'checkpoints' / f'pass_{best["pass"]:02d}.pt')
    model.load_state_dict(checkpoint['model'])
    test_rows = rows(table, TEST)
    result, positions, R, probabilities = run_policy(model, features, returns, test_rows, config.warmup_hours)
    report = {'test_year': args.test_year, 'selected': {'seed': best['seed'], 'pass': best['pass'], 'validation_sharpe': best['sharpe'],
                           'validation_log_return': best['total_log_return']},
              'test_period': [str(TEST[0]), str(table.time.iloc[test_rows[-1]])],
              'agent': result, 'mean_probabilities': probabilities.mean(0).round(4).tolist(),
              'benchmarks': benchmarks(table, returns, test_rows),
              'validation_benchmarks': benchmarks(table, returns, rows(table, VALIDATION)),
              'all_validation': [{k: c[k] for k in ('seed', 'pass', 'sharpe', 'total_log_return', 'position_changes')}
                                 for c in candidates]}
    (args.output / 'test.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('selected', 'agent', 'benchmarks')}, indent=1))


if __name__ == '__main__':
    main()
