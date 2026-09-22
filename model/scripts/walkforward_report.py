"""Test results of the nested walk-forward (scripts/walkforward_nested.py), 1 January 2021 -> 1 August 2026.

Every year Y is traded by the models trained on the data before Y with the configuration chosen for Y. Each random
initialization trades one continuous account: it starts flat with 10,000 USD on 1 January 2021, every year starts flat
after a warm-up week without trading, and the equity carries over from year to year. The reported agent is the mean
over the ten initializations. The rule benchmarks trade the same hours under the same frictionless accounting.

Overfitting diagnostics:
  generalization gap   annualized Sharpe ratio of each final model on its own training period against its test year;
  PBO                  probability of backtest overfitting by combinatorially symmetric cross-validation over the
                       candidate configurations and the validation years 2019-2024 (Bailey et al. 2017);
  deflated Sharpe      probability that the test Sharpe ratio exceeds the best of N trials expected by chance
                       (Bailey & Lopez de Prado 2014), with N the number of candidate configurations.

    PYTHONPATH=src .venv/bin/python scripts/walkforward_report.py --algo ppo
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch

from simple_agent.config import DATA_END, DATA_START, HOURS_PER_YEAR, year
from simple_agent.data import Scaler, load, next_returns, rows
from simple_agent.evaluate import POSITIONS, run_policy, score
from simple_agent.model import ActorCritic

import walkforward_nested as nested
from paper_figures import RULES, daily_signal_positions, win_rate

ROOT = Path(__file__).resolve().parents[1]
START = year(2021)
KEYS = ('total_return_pct', 'equity_from_10k', 'sharpe', 'max_drawdown', 'position_changes', 'long_share',
        'short_share', 'flat_share', 'win_rate_pct')


def period(test_year):
    return year(test_year), min(year(test_year + 1), DATA_END)


def evaluate_model(job):
    """One (test year, seed): positions and returns on the test year, and the Sharpe ratio on its training period."""
    out, test_year, seed, passes, hidden = job
    table = load(end=DATA_END)
    train_rows = rows(table, (DATA_START, year(test_year)))
    run = json.loads((nested.final_dir(out, test_year, seed) / 'config.json').read_text())
    scaler = Scaler(table, train_rows, run['config']['obs_clip'], names=run['features'])
    features, returns = scaler.transform(table), next_returns(table)
    model = ActorCritic(len(run['features']) + 3, hidden)
    checkpoint = nested.final_dir(out, test_year, seed) / 'checkpoints' / f'pass_{passes:02d}.pt'
    model.load_state_dict(torch.load(checkpoint)['model'])
    _, positions, R, _ = run_policy(model, features, returns, rows(table, period(test_year)), run['config']['warmup_hours'])
    in_sample, _, _, _ = run_policy(model, features, returns, train_rows, run['config']['warmup_hours'])
    return test_year, seed, positions, R, in_sample['sharpe']


def pbo(matrix):
    """CSCV over a (configurations x time blocks) performance matrix; returns the share of negative logits."""
    blocks = matrix.shape[1]
    logits = []
    for inside in combinations(range(blocks), blocks // 2):
        outside = [j for j in range(blocks) if j not in inside]
        best = int(np.argmax(matrix[:, inside].mean(1)))
        oos = matrix[:, outside].mean(1)
        omega = ((oos < oos[best]).sum() + 0.5) / len(oos)
        logits.append(math.log(omega / (1 - omega)))
    return float(np.mean(np.array(logits) <= 0)), logits


def deflated_sharpe(R, trial_sharpes):
    """Bailey & Lopez de Prado (2014), with per-period (hourly) Sharpe ratios."""
    sr = R.mean() / R.std()
    z = (R - R.mean()) / R.std()
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    n = len(trial_sharpes)
    gamma = 0.5772156649
    inv = NormalDist().inv_cdf
    sr0 = math.sqrt(np.var(trial_sharpes, ddof=1)) * ((1 - gamma) * inv(1 - 1 / n) + gamma * inv(1 - 1 / (n * math.e)))
    stat = (sr - sr0) * math.sqrt(len(R) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(NormalDist().cdf(stat)), float(sr0 * math.sqrt(HOURS_PER_YEAR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=('ppo', 'reinforce'), required=True)
    args = parser.parse_args()
    out = ROOT / f'results/walkforward_nested_{args.algo}'
    choices = {int(k): v for k, v in json.loads((out / 'choices.json').read_text()).items()}
    configs = json.loads((out / 'candidates.json').read_text())
    jobs = [(out, y, s, choices[y]['selected']['passes'], configs[choices[y]['selected']['config']]['hidden'])
            for y in nested.TEST_YEARS for s in nested.FINAL_SEEDS]
    with ProcessPoolExecutor(6) as pool:
        results = list(pool.map(evaluate_model, jobs))

    table = load(end=DATA_END)
    returns = next_returns(table)
    test_rows = np.concatenate([rows(table, period(y))[:-1] for y in nested.TEST_YEARS])
    r = returns[test_rows]
    times = table.time.iloc[test_rows].reset_index(drop=True)
    year_of = times.dt.year.to_numpy()

    agent, curves, gap = {}, {}, []
    for seed in nested.FINAL_SEEDS:
        parts = sorted((y, p, R) for y, s, p, R, _ in results if s == seed)
        positions = np.concatenate([p for _, p, _ in parts])
        R = np.concatenate([R for _, _, R in parts])
        agent[seed] = {**score(R, positions), 'win_rate_pct': win_rate(R, positions),
                       'by_year': {str(y): float(np.expm1(np.log1p(R[year_of == y]).sum()) * 100)
                                   for y in nested.TEST_YEARS}}
        curves[f'seed_{seed}'] = 10_000 * np.cumprod(1 + R)
        curves[f'position_{seed}'] = positions
    for y, s, _, R, in_sample in results:
        gap.append({'test_year': y, 'seed': s, 'train_sharpe': in_sample,
                    'test_sharpe': float(R.mean() / R.std() * np.sqrt(HOURS_PER_YEAR)) if R.std() > 0 else 0.0})
    mean_agent = {k: float(np.mean([a[k] for a in agent.values()])) for k in KEYS}
    mean_agent['by_year'] = {str(y): float(np.mean([a['by_year'][str(y)] for a in agent.values()]))
                             for y in nested.TEST_YEARS}

    bench = {}
    for name, rule in RULES.items():
        positions = np.ones(len(test_rows)) if rule is None else daily_signal_positions(table, test_rows, rule)
        R = positions * r
        bench[name] = {**score(R, positions), 'win_rate_pct': win_rate(R, positions),
                       'by_year': {str(y): float(np.expm1(np.log1p(R[year_of == y]).sum()) * 100)
                                   for y in nested.TEST_YEARS}}
        curves[name] = 10_000 * np.cumprod(1 + R)
    rng = np.random.default_rng(0)
    random_scores = [score(p * r, p) for p in (POSITIONS[rng.integers(0, 3, len(r))] for _ in range(1000))]
    bench['Random (median)'] = {k: float(np.median([s[k] for s in random_scores]))
                                for k in ('total_return_pct', 'equity_from_10k', 'sharpe', 'max_drawdown')}

    # Overfitting diagnostics.
    blocks = (2019, 2020, 2021, 2022, 2023, 2024)
    matrix = np.array([[np.mean([nested.curve(out, name, y, s)[9] for s in nested.SELECTION_SEEDS]) for y in blocks]
                       for name in configs])
    pbo_value, logits = pbo(matrix)
    trial_sharpes = matrix.mean(1) / np.sqrt(HOURS_PER_YEAR)                 # per-period Sharpe of each trial
    mean_R = np.mean([np.diff(np.r_[10_000, curves[f'seed_{s}']]) / np.r_[10_000, curves[f'seed_{s}'][:-1]]
                      for s in nested.FINAL_SEEDS], axis=0)
    dsr, sr_expected = deflated_sharpe(mean_R, trial_sharpes)
    gap_frame = pd.DataFrame(gap)
    diagnostics = {'pbo': pbo_value, 'pbo_logits': logits, 'deflated_sharpe_probability': dsr,
                   'expected_max_sharpe_of_trials_annualized': sr_expected,
                   'generalization_gap': gap_frame.groupby('test_year')[['train_sharpe', 'test_sharpe']].mean()
                   .round(3).to_dict(orient='index'),
                   'mean_train_sharpe': float(gap_frame.train_sharpe.mean()),
                   'mean_test_sharpe': float(gap_frame.test_sharpe.mean())}

    report = {'choices': {y: c['selected'] for y, c in choices.items()}, 'agent_mean': mean_agent,
              'agent_seeds': agent, 'benchmarks': bench, 'diagnostics': diagnostics}
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    frame = pd.DataFrame(curves)
    frame.insert(0, 'time', times)
    frame.to_csv(out / 'equity_test.csv', index=False)
    print(json.dumps({'agent_mean': mean_agent, 'diagnostics': {k: v for k, v in diagnostics.items()
                                                                 if k != 'pbo_logits'}}, indent=1, default=float))
    for name, b in bench.items():
        print(f"{name:22} {b['total_return_pct']:+8.1f}%  Sharpe {b['sharpe']:+.2f}  maxDD {b['max_drawdown'] * 100:.1f}%")


if __name__ == '__main__':
    main()
