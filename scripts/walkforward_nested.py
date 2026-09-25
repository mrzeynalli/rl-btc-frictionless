"""Nested expanding walk-forward evaluation, 2021 -> 2026-08.

For every test year Y the configuration and the number of passes are chosen using only years before Y, the agent is
retrained on everything before Y with ten random initializations, and it then trades year Y with fixed parameters:

    test year   selection (validation years, each trained on the data before it)   final training
    2021        2019, 2020                                                          2017-08 -> 2020
    2022        2019, 2020, 2021                                                    2017-08 -> 2021
    2023        2020, 2021, 2022                                                    2017-08 -> 2022
    2024        2021, 2022, 2023                                                    2017-08 -> 2023
    2025        2022, 2023, 2024                                                    2017-08 -> 2024
    2026        2023, 2024, 2025                                                    2017-08 -> 2025 (test to 1 Aug)

Algorithm: recurrent PPO, at most 15 passes.

Candidates: a 2^4 factorial design over the actor learning rate, the network width, the entropy coefficient and
the adaptation rate of the reward, all other settings fixed. Every candidate is trained with three random
initializations on each validation year 2019-2025 and scored after every pass; the runs are shared between test years.

Selection for test year Y (Li et al. 2018; Hastie, Tibshirani & Friedman 2009): the seed-averaged validation Sharpe
ratio is averaged over the validation years of Y after each pass and smoothed over three passes; the pass count is the
smallest one within one standard error (across years) of the best, and the candidate with the highest mean minus one
standard error at its pass count is selected.

    PYTHONPATH=. .venv/bin/python scripts/walkforward_nested.py [--workers 6]

Results are written to agents/ppo/results/walkforward/.
"""
import argparse
from itertools import product
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'agents' / 'ppo' / 'results' / 'walkforward'
TEST_YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
VALIDATION_YEARS = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
SELECTION_SEEDS = (11, 29, 47)
FINAL_SEEDS = (11, 29, 47, 61, 83, 101, 131, 163, 197, 229)
MIN_PASSES, SMOOTH = 3, 3
MAX_PASSES = 15

FIXED = {'gamma': 0.98, 'gae_lambda': 0.95, 'clip': 0.2, 'critic_lr': 9.742244550984722e-04,
         'rollout': 64, 'epochs': 3, 'minibatches': 4, 'max_grad_norm': 1.0, 'obs_clip': 5.0}
FACTORS = {'lr': (8.38429594542857e-05, 2.5e-4), 'hidden': (64, 128), 'entropy_coef': (0.01, 0.03),
           'eta': (1 / 120, 1 / 336)}


def candidates():
    names = list(FACTORS)
    return {f'c{i:02d}': {**FIXED, **dict(zip(names, values))}
            for i, values in enumerate(product(*FACTORS.values()))}


def validation_dir(out, name, year, seed):
    return out / 'selection' / name / f'validation_{year}' / f'seed_{seed}'


def curve(out, name, year, seed):
    path = validation_dir(out, name, year, seed) / 'validation.jsonl'
    return [json.loads(line)['sharpe'] for line in path.read_text().splitlines()] if path.exists() else []


def final_dir(out, test_year, seed):
    return out / 'final' / f'test_{test_year}' / f'seed_{seed}'


def run_all(jobs, workers, log):
    """jobs: (target_dir, seed, args, done). Runs `workers` trainings at a time, skipping finished ones."""
    pending = [job for job in jobs if not job[3]()]
    running, total, started = [], len(pending), time.time()
    while pending or running:
        while pending and len(running) < workers:
            target, seed, args, _ = job = pending.pop(0)
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = open(target.parent / f'seed_{seed}.log', 'w')
            process = subprocess.Popen([sys.executable, '-m', 'core.train', '--seed', str(seed),
                                        '--output', str(target.parent), *args],
                                       stdout=handle, stderr=subprocess.STDOUT)
            running.append((job, process, handle))
        time.sleep(15)
        for item in list(running):
            job, process, handle = item
            if process.poll() is None:
                continue
            handle.close()
            running.remove(item)
            done = total - len(pending) - len(running)
            status = 'ok' if process.returncode == 0 else f'FAILED ({process.returncode})'
            print(f'{time.strftime("%Y-%m-%d %H:%M")} {job[0].relative_to(ROOT)} {status} '
                  f'[{done}/{total}, {(time.time() - started) / 3600:.1f} h]', file=log, flush=True)


def smooth(x):
    return np.array([x[max(0, i - SMOOTH // 2):i + SMOOTH // 2 + 1].mean() for i in range(len(x))])


def assess(out, name, years):
    """Seed-averaged curves per validation year -> (score, passes, detail)."""
    folds = np.array([np.mean([curve(out, name, y, s)[:MAX_PASSES] for s in SELECTION_SEEDS], axis=0) for y in years])
    mean = smooth(folds.mean(0))
    se = smooth(folds.std(0, ddof=1) / np.sqrt(len(years)))
    eligible = np.arange(MIN_PASSES - 1, MAX_PASSES)
    best = eligible[np.argmax(mean[eligible])]
    chosen = int(next(i for i in eligible if mean[i] >= mean[best] - se[best]))      # one-standard-error rule
    return {'config': name, 'passes': chosen + 1, 'score': float(mean[chosen] - se[chosen]),
            'mean_sharpe': float(mean[chosen]), 'standard_error': float(se[chosen]),
            'per_year': {str(y): float(v) for y, v in zip(years, folds[:, chosen])}}


def selection_years(test_year):
    return tuple(y for y in VALIDATION_YEARS if test_year - 3 <= y < test_year)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    out = RESULTS
    out.mkdir(parents=True, exist_ok=True)
    configs = candidates()
    (out / 'candidates.json').write_text(json.dumps(configs, indent=1))
    log = open(out / 'progress.log', 'a')

    def overrides(settings):
        return ['--set', *[f'{k}={v}' for k, v in settings.items()]]

    # 1. Selection runs, shared by all test years.
    jobs = []
    for name, settings in configs.items():
        for year in VALIDATION_YEARS:
            for seed in SELECTION_SEEDS:
                target = validation_dir(out, name, year, seed)
                jobs.append((target, seed, ['--passes', str(MAX_PASSES), '--validation-year', str(year),
                                            *overrides(settings)],
                             lambda n=name, y=year, s=seed: len(curve(out, n, y, s)) >= MAX_PASSES))
    run_all(jobs, args.workers, log)

    # 2. Choose a configuration and a pass count for every test year from the years before it.
    choices = {}
    for test_year in TEST_YEARS:
        years = selection_years(test_year)
        ranked = sorted((assess(out, name, years) for name in configs), key=lambda r: -r['score'])
        choices[test_year] = {'validation_years': list(years), 'selected': ranked[0], 'ranking': ranked}
    (out / 'choices.json').write_text(json.dumps(choices, indent=1))

    # 3. Final models: train on everything before the test year, ten random initializations.
    jobs = []
    for test_year, choice in choices.items():
        chosen = choice['selected']
        for seed in FINAL_SEEDS:
            target = final_dir(out, test_year, seed)
            checkpoint = target / 'checkpoints' / f'pass_{chosen["passes"]:02d}.pt'
            jobs.append((target, seed, ['--passes', str(chosen['passes']), '--train-end', str(test_year),
                                        *overrides(configs[chosen['config']])],
                         lambda c=checkpoint: c.exists()))
    run_all(jobs, args.workers, log)
    print(f'{time.strftime("%Y-%m-%d %H:%M")} walk-forward training finished', file=log, flush=True)


if __name__ == '__main__':
    main()
