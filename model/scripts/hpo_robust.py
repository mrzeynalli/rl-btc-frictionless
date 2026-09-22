"""Robust walk-forward hyperparameter search with ten seeds.

Every choice the final agent needs, including how long it trains, is made on four walk-forward folds; no separate
validation year is set aside and the test period (2025-01-01 onward) is never read:

    training              validation (scored)
    2017-08 -> 2020       2021
    2017-08 -> 2021       2022
    2017-08 -> 2022       2023
    2017-08 -> 2023       2024

Measures against selection bias:
  1. Cautious score. For a configuration, the validation Sharpe ratio is averaged over seeds within each fold, then
     over folds; its score is that mean minus one standard error across folds, so a configuration that is good in
     one or two years and poor in the others ranks below a consistent one.
  2. Fewer free settings. Clipping (0.2, Schulman et al. 2017), gradient norm limit (0.5, CleanRL), minibatches (4,
     CleanRL) and observation clipping (5) are fixed; nine settings are searched. The adaptation rate eta of the
     differential Sharpe ratio stays in the search, with longer memories added because the previous search chose
     its longest value (1/336).
  3. Training length from the folds. Every run is scored after every pass. The number of passes is the smallest one
     whose smoothed mean score lies within one standard error of the best (the one-standard-error rule, Hastie,
     Tibshirani & Friedman 2009), and the configuration is scored at that pass.
  4. Ten seeds in the last rung and in the final training.
  5. The probability of backtest overfitting (Bailey et al. 2017) is estimated by combinatorially symmetric cross
     validation over the folds of rungs 2 and 3 and written to pbo.json.

Successive halving (Li et al. 2018):
  Rung 1  48 random configurations + cfg_016 (the previous winner, carried as a control), 1 seed, validation years
          2022 and 2024, 10 passes.
  Rung 2  the best 12 (+ control), 2 seeds, four validation years, 20 passes.
  Rung 3  the best 4 (+ control), 10 seeds, four validation years, 20 passes (rung 2's two seeds are reused).
  Final   the winner trained on everything before 2025 with the ten seeds, for its chosen number of passes.
The test evaluation is a separate step (scripts/final_report.py) run once on the final models.

Resumable: finished runs are skipped. Everything is written to results/simple_agent_hpo_robust/.

    PYTHONPATH=src .venv/bin/python scripts/hpo_robust.py [--workers 6]
"""
import argparse
from itertools import combinations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/simple_agent_hpo_robust'
YEARS = (2021, 2022, 2023, 2024)
SEEDS = (11, 29, 47, 61, 83, 101, 131, 163, 197, 229)
RUNG1 = {'years': (2022, 2024), 'seeds': SEEDS[:1], 'passes': 10, 'keep': 12}
RUNG2 = {'years': YEARS, 'seeds': SEEDS[:2], 'passes': 20, 'keep': 4}
RUNG3 = {'years': YEARS, 'seeds': SEEDS, 'passes': 20}
N_RANDOM = 48
MIN_PASSES = 3                  # the training length is chosen from pass 3 onward
SMOOTH = 3                      # passes in the centred moving average of the score curve

FIXED = {'clip': 0.2, 'max_grad_norm': 0.5, 'minibatches': 4, 'obs_clip': 5.0}
SPACE = {
    'gamma': [0.95, 0.97, 0.98, 0.99, 0.995],
    'gae_lambda': [0.8, 0.9, 0.95, 0.98],
    'lr': ('log', 5e-5, 1e-3),
    'critic_lr': ('log', 5e-5, 1e-3),
    'hidden': [64, 128, 256],
    'entropy_coef': [0.0, 0.003, 0.01, 0.03],
    'eta': [1 / 72, 1 / 120, 1 / 336, 1 / 720, 1 / 1440],     # memory of the reward: 3 days to 2 months
    'rollout': [64, 128, 256],
    'epochs': [3, 4, 6],
}
CONTROL = {'gamma': 0.98, 'gae_lambda': 0.95, 'lr': 8.38429594542857e-05, 'critic_lr': 9.742244550984722e-04,
           'hidden': 64, 'entropy_coef': 0.03, 'clip': 0.2, 'eta': 1 / 336, 'rollout': 64, 'epochs': 3,
           'minibatches': 4, 'max_grad_norm': 1.0, 'obs_clip': 5.0}          # cfg_016, exactly as reported


def sample(rng):
    config = dict(FIXED)
    for name, choices in SPACE.items():
        if isinstance(choices, tuple):
            config[name] = float(np.exp(rng.uniform(np.log(choices[1]), np.log(choices[2]))))
        else:
            config[name] = choices[int(rng.integers(len(choices)))]
    return config


def configurations():
    path = OUT / 'configs.json'
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(20260922)
    configs = {'cfg_016': CONTROL, **{f'r{i:03d}': sample(rng) for i in range(1, N_RANDOM + 1)}}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configs, indent=1))
    return configs


def run_dir(rung, name, year, seed):
    stage = 'rung1' if rung == 1 else 'runs'          # rungs 2 and 3 share runs: same passes, more seeds
    return OUT / stage / name / f'validation_{year}' / f'seed_{seed}'


def curve(rung, name, year, seed):
    path = run_dir(rung, name, year, seed) / 'validation.jsonl'
    return [json.loads(line)['sharpe'] for line in path.read_text().splitlines()] if path.exists() else []


def launch(jobs, configs, workers, log):
    """jobs: (rung, name, year, seed, passes); runs `workers` trainings at a time, skipping finished ones."""
    pending = [job for job in jobs if len(curve(*job[:4])) < job[4]]
    running, total, started = [], len(pending), time.time()
    while pending or running:
        while pending and len(running) < workers:
            rung, name, year, seed, passes = job = pending.pop(0)
            target = run_dir(rung, name, year, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = open(target.parent / f'seed_{seed}.log', 'w')
            process = subprocess.Popen(
                [sys.executable, '-m', 'simple_agent.train', '--seed', str(seed), '--passes', str(passes),
                 '--validation-year', str(year), '--output', str(target.parent),
                 '--set', *[f'{k}={v}' for k, v in configs[name].items()]],
                stdout=handle, stderr=subprocess.STDOUT)
            running.append((job, process, handle))
        time.sleep(15)
        for item in list(running):
            job, process, handle = item
            if process.poll() is None:
                continue
            handle.close()
            running.remove(item)
            status = 'ok' if process.returncode == 0 else f'FAILED ({process.returncode})'
            done = total - len(pending) - len(running)
            print(f'{time.strftime("%Y-%m-%d %H:%M")} rung {job[0]} {job[1]} {job[2]} seed {job[3]}: {status} '
                  f'[{done}/{total}, {(time.time() - started) / 3600:.1f} h]', file=log, flush=True)


def fold_curves(rung, name, spec):
    """Per validation year, the seed-averaged Sharpe ratio after each pass: array (years, passes), or None."""
    rows = []
    for year in spec['years']:
        curves = [curve(rung, name, year, seed)[:spec['passes']] for seed in spec['seeds']]
        if any(len(c) < spec['passes'] for c in curves):
            return None
        rows.append(np.mean(curves, axis=0))
    return np.array(rows)


def smooth(x):
    return np.array([x[max(0, i - SMOOTH // 2):i + SMOOTH // 2 + 1].mean() for i in range(len(x))])


def assess(rung, name, spec):
    folds = fold_curves(rung, name, spec)
    if folds is None:
        return None
    mean = smooth(folds.mean(0))
    se = smooth(folds.std(0, ddof=1) / np.sqrt(len(folds)))
    eligible = np.arange(MIN_PASSES - 1, folds.shape[1])
    best = eligible[np.argmax(mean[eligible])]
    chosen = int(next(i for i in eligible if mean[i] >= mean[best] - se[best]))     # one-standard-error rule
    return {'config': name, 'passes': chosen + 1, 'score': float(mean[chosen] - se[chosen]),
            'mean_sharpe': float(mean[chosen]), 'standard_error': float(se[chosen]),
            'worst_year_sharpe': float(folds[:, chosen].min()),
            'per_year': {str(y): float(v) for y, v in zip(spec['years'], folds[:, chosen])},
            'mean_curve': [round(float(v), 4) for v in mean]}


def ranking(rung, names, spec):
    rows = [assess(rung, name, spec) for name in names]
    return sorted([r for r in rows if r], key=lambda r: -r['score'])


def pbo(rows):
    """Probability of backtest overfitting by CSCV over the validation years (Bailey et al. 2017)."""
    names = [r['config'] for r in rows]
    matrix = np.array([[r['per_year'][str(y)] for y in YEARS] for r in rows])       # configs x years
    logits = []
    for inside in combinations(range(len(YEARS)), len(YEARS) // 2):
        outside = [j for j in range(len(YEARS)) if j not in inside]
        chosen = int(np.argmax(matrix[:, inside].mean(1)))
        oos = matrix[:, outside].mean(1)
        omega = (oos < oos[chosen]).sum() / len(names) + 0.5 / len(names)          # relative rank in (0, 1)
        logits.append(float(np.log(omega / (1 - omega))))
    return {'configs': names, 'logits': logits, 'pbo': float(np.mean(np.array(logits) <= 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    configs = configurations()
    log = open(OUT / 'progress.log', 'a')

    launch([(1, n, y, s, RUNG1['passes']) for n in configs for y in RUNG1['years'] for s in RUNG1['seeds']],
           configs, args.workers, log)
    rung1 = ranking(1, list(configs), RUNG1)
    (OUT / 'rung1.json').write_text(json.dumps(rung1, indent=1))

    names2 = [r['config'] for r in rung1[:RUNG1['keep']]]
    names2 += [] if 'cfg_016' in names2 else ['cfg_016']
    launch([(2, n, y, s, RUNG2['passes']) for n in names2 for y in RUNG2['years'] for s in RUNG2['seeds']],
           configs, args.workers, log)
    rung2 = ranking(2, names2, RUNG2)
    (OUT / 'rung2.json').write_text(json.dumps(rung2, indent=1))

    names3 = [r['config'] for r in rung2[:RUNG2['keep']]]
    names3 += [] if 'cfg_016' in names3 else ['cfg_016']
    launch([(3, n, y, s, RUNG3['passes']) for n in names3 for y in RUNG3['years'] for s in RUNG3['seeds']],
           configs, args.workers, log)
    rung3 = ranking(3, names3, RUNG3)
    (OUT / 'rung3.json').write_text(json.dumps(rung3, indent=1))
    (OUT / 'pbo.json').write_text(json.dumps({'rung2': pbo(rung2), 'rung3': pbo(rung3)}, indent=1))

    winner = rung3[0]
    (OUT / 'winner.json').write_text(json.dumps({**winner, 'settings': configs[winner['config']]}, indent=1))
    target = OUT / 'final' / winner['config']
    pending = [s for s in SEEDS if not (target / f'seed_{s}' / 'checkpoints' / f'pass_{winner["passes"]:02d}.pt').exists()]
    running = []
    while pending or running:
        while pending and len(running) < args.workers:
            seed = pending.pop(0)
            target.mkdir(parents=True, exist_ok=True)
            handle = open(target / f'seed_{seed}.log', 'w')
            running.append((subprocess.Popen(
                [sys.executable, '-m', 'simple_agent.train', '--seed', str(seed), '--passes', str(winner['passes']),
                 '--final', '--output', str(target), '--set',
                 *[f'{k}={v}' for k, v in configs[winner['config']].items()]],
                stdout=handle, stderr=subprocess.STDOUT), handle))
        time.sleep(15)
        for item in list(running):
            if item[0].poll() is not None:
                item[1].close()
                running.remove(item)
    print(f'{time.strftime("%Y-%m-%d %H:%M")} search finished; winner {winner["config"]} '
          f'({winner["passes"]} passes)', file=log, flush=True)


if __name__ == '__main__':
    main()
