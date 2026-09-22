"""The multi-year hyperparameter search that selected the reported configuration (cfg_016).

Every configuration is scored on four validation years, each seen by a model trained only on the years before it:

    training              validation (scored)
    2017-08 -> 2020       2021
    2017-08 -> 2021       2022
    2017-08 -> 2022       2023
    2017-08 -> 2023       2024

A configuration's score is the mean over folds and seeds of the final model's validation Sharpe ratio; no checkpoint
is chosen. The test period (2025-01-01 onward) never enters the search.

Successive halving (Li et al., Hyperband, JMLR 2018): screen many configurations cheaply, spend more on the survivors.
  Rung 1  48 random configurations + the initial settings + 2 configurations from an earlier single-year search,
          1 seed, validation years 2022 and 2024, 5 passes.
  Rung 2  the best 12, 2 seeds, all four validation years, 10 passes.
  Rung 3  the best 5, 3 more seeds (5 in all).
  Final   the winner trained on everything before 2025 with the five seeds (simple_agent.train --final); these are
          the models evaluated on the test period by scripts/final_report.py.
Result directories are named by the year after the validation year (test_<V+1>). The initial settings' four-fold runs
in rungs 2 and 3 are reused from an earlier walk-forward run with identical code and settings (CONTROL_RUNS).

Resumable: finished runs are skipped. Everything is written to results/simple_agent_hpo_multiyear/.

    PYTHONPATH=src .venv/bin/python scripts/hpo_multiyear.py [--workers 6]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/simple_agent_hpo_multiyear'
CONTROL_RUNS = ROOT / 'results/simple_agent_walkforward_state4'
SEEDS = (11, 29, 47, 61, 83)
RUNG1 = {'folds': (2023, 2025), 'seeds': (11,), 'passes': 5}
RUNG2 = {'folds': (2022, 2023, 2024, 2025), 'seeds': (11, 29), 'passes': 10, 'keep': 12}
RUNG3 = {'folds': (2022, 2023, 2024, 2025), 'seeds': (11, 29, 47, 61, 83), 'passes': 10, 'keep': 5}
N_RANDOM = 48

SPACE = {
    'gamma': [0.95, 0.97, 0.98, 0.99, 0.995],
    'gae_lambda': [0.8, 0.9, 0.95, 0.98],
    'lr': ('log', 5e-5, 1e-3),
    'critic_lr': ('log', 5e-5, 1e-3),
    'hidden': [64, 128, 256],
    'entropy_coef': [0.0, 0.003, 0.01, 0.03],
    'clip': [0.1, 0.2, 0.3],
    'eta': [1 / 24, 1 / 72, 1 / 120, 1 / 336],
    'rollout': [64, 128, 256],
    'epochs': [3, 4, 6],
    'minibatches': [2, 4, 8],                 # minibatches per epoch: 4, 2 or 1 walker sequences each
    'max_grad_norm': [0.5, 1.0],
    'obs_clip': [3.0, 5.0, 10.0],
}
CURRENT = {'gamma': 0.98, 'gae_lambda': 0.95, 'lr': 2.5e-4, 'critic_lr': 2.5e-4, 'hidden': 128, 'entropy_coef': 0.01,
           'clip': 0.2, 'eta': 1 / 120, 'rollout': 128, 'epochs': 4, 'minibatches': 4, 'max_grad_norm': 0.5,
           'obs_clip': 5.0}
SINGLE_YEAR_WINNERS = {
    'known_cfg001': {**CURRENT, 'gamma': 0.995, 'gae_lambda': 0.8, 'lr': 3.4e-4, 'critic_lr': 2.03e-4, 'hidden': 64,
                     'entropy_coef': 0.003, 'eta': 1 / 72, 'rollout': 256, 'epochs': 6},
    'known_cfg029': {**CURRENT, 'gamma': 0.98, 'gae_lambda': 0.8, 'lr': 2.02e-4, 'critic_lr': 9.3e-5, 'hidden': 64,
                     'entropy_coef': 0.003, 'rollout': 256, 'epochs': 6},
}


def sample(rng):
    config = {}
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
    rng = np.random.default_rng(20260921)
    configs = {'current': CURRENT, **SINGLE_YEAR_WINNERS,
               **{f'cfg_{i:03d}': sample(rng) for i in range(1, N_RANDOM + 1)}}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configs, indent=1))
    return configs


def run_path(rung, name, fold, seed):
    if name == 'current' and rung > 1:
        return CONTROL_RUNS / f'test_{fold}' / f'seed_{seed}'          # reused: same code, same settings, 10 passes
    return OUT / f'rung{1 if rung == 1 else 2}' / name / f'test_{fold}' / f'seed_{seed}'


def curve(rung, name, fold, seed):
    path = run_path(rung, name, fold, seed) / 'validation.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def finished(rung, name, fold, seed, passes):
    return len(curve(rung, name, fold, seed)) >= passes


def launch(jobs, configs, workers, log_file):
    """jobs: (rung, name, fold, seed, passes). Runs `workers` trainings at a time."""
    pending = [job for job in jobs if not finished(*job)]
    running = []
    started = time.time()
    total = len(pending)
    while pending or running:
        while pending and len(running) < workers:
            rung, name, fold, seed, passes = pending.pop(0)
            target = run_path(rung, name, fold, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            log = open(target.parent / f'seed_{seed}.log', 'w')
            overrides = [f'{k}={v}' for k, v in configs[name].items()]
            process = subprocess.Popen(
                [sys.executable, '-m', 'simple_agent.train', '--seed', str(seed), '--passes', str(passes),
                 '--validation-year', str(fold - 1), '--output', str(target.parent), '--set', *overrides],
                stdout=log, stderr=subprocess.STDOUT)
            running.append(((rung, name, fold, seed), process, log))
        time.sleep(15)
        for item in list(running):
            job, process, log = item
            if process.poll() is not None:
                log.close()
                running.remove(item)
                done = total - len(pending) - len(running)
                status = 'ok' if process.returncode == 0 else f'FAILED ({process.returncode})'
                elapsed = (time.time() - started) / 3600
                print(f'{time.strftime("%Y-%m-%d %H:%M")} rung {job[0]} {job[1]} fold {job[2]} seed {job[3]}: {status} '
                      f'[{done}/{total}, {elapsed:.1f} h]', flush=True, file=log_file)


def score(rung, name, folds, seeds, passes):
    """Mean over folds of the mean over seeds of the final model's validation Sharpe, and per-fold detail."""
    per_fold = {}
    for fold in folds:
        values = []
        for seed in seeds:
            items = curve(rung, name, fold, seed)
            final = [i for i in items if i['pass'] == passes]
            if final:
                values.append(final[0])
        if values:
            per_fold[fold] = {'sharpe': float(np.mean([v['sharpe'] for v in values])),
                              'return_pct': float(np.mean([np.expm1(v['total_log_return']) * 100 for v in values])),
                              'seeds': len(values)}
    if len(per_fold) < len(folds):
        return None
    sharpes = [v['sharpe'] for v in per_fold.values()]
    return {'config': name, 'mean_sharpe': float(np.mean(sharpes)), 'worst_fold_sharpe': float(np.min(sharpes)),
            'fold_standard_error': float(np.std(sharpes, ddof=1) / np.sqrt(len(sharpes))) if len(sharpes) > 1 else None,
            'per_fold': per_fold}


def ranking(rung, names, spec):
    rows = [score(rung, name, spec['folds'], spec['seeds'], spec['passes']) for name in names]
    return sorted([r for r in rows if r], key=lambda r: -r['mean_sharpe'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    configs = configurations()
    log = open(OUT / 'progress.log', 'a')

    # Rung 1: everything, cheaply.
    launch([(1, name, fold, seed, RUNG1['passes']) for name in configs for fold in RUNG1['folds']
            for seed in RUNG1['seeds']], configs, args.workers, log)
    rung1 = ranking(1, list(configs), RUNG1)
    (OUT / 'rung1.json').write_text(json.dumps(rung1, indent=1))

    # Rung 2: the best 12 (the current settings always carried along as the control), four folds, two seeds.
    finalists2 = [r['config'] for r in rung1[:RUNG2['keep']]]
    if 'current' not in finalists2:
        finalists2.append('current')
    launch([(2, name, fold, seed, RUNG2['passes']) for name in finalists2 for fold in RUNG2['folds']
            for seed in RUNG2['seeds']], configs, args.workers, log)
    rung2 = ranking(2, finalists2, RUNG2)
    (OUT / 'rung2.json').write_text(json.dumps(rung2, indent=1))

    # Rung 3: the best 5, five seeds.
    finalists3 = [r['config'] for r in rung2[:RUNG3['keep']]]
    if 'current' not in finalists3:
        finalists3.append('current')
    launch([(3, name, fold, seed, RUNG3['passes']) for name in finalists3 for fold in RUNG3['folds']
            for seed in RUNG3['seeds']], configs, args.workers, log)
    rung3 = ranking(3, finalists3, RUNG3)
    (OUT / 'rung3.json').write_text(json.dumps(rung3, indent=1))
    winner = rung3[0]['config']
    (OUT / 'winner.json').write_text(json.dumps({'config': winner, 'settings': configs[winner], **rung3[0]}, indent=1))

    # Final: the winner and the initial settings trained on everything before 2025, the models of the test period.
    contenders = [winner] if winner == 'current' else [winner, 'current']
    final_jobs = []
    for name in contenders:
        for seed in SEEDS:
            target = OUT / 'final' / name / 'test_2026' / f'seed_{seed}'
            if not (target / 'checkpoints' / f'pass_{RUNG3["passes"]:02d}.pt').exists():
                final_jobs.append((name, seed, target))
    running = []
    while final_jobs or running:
        while final_jobs and len(running) < args.workers:
            name, seed, target = final_jobs.pop(0)
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = open(target.parent / f'seed_{seed}.log', 'w')
            overrides = [f'{k}={v}' for k, v in configs[name].items()]
            running.append((subprocess.Popen(
                [sys.executable, '-m', 'simple_agent.train', '--seed', str(seed), '--passes', str(RUNG3['passes']),
                 '--final', '--output', str(target.parent), '--set', *overrides],
                stdout=handle, stderr=subprocess.STDOUT), handle))
        time.sleep(15)
        for item in list(running):
            if item[0].poll() is not None:
                item[1].close()
                running.remove(item)
    print(f'{time.strftime("%Y-%m-%d %H:%M")} search finished; winner {winner}', file=log, flush=True)


if __name__ == '__main__':
    main()
