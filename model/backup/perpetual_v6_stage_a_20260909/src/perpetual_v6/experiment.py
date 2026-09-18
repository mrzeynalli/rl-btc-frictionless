"""Stage A: horizon fix on one architecture, many seeds; development data only."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from perpetual.config import ROOT, TEST_START, write_json
from perpetual.data import Scaler, load_market
from perpetual.train import now
from .config import CACHE, OUTPUT, SEEDS, YEARS, Config, protocol, source_hash, variants
from .evaluate import evaluate
from .network import ActorCritic


def run_path(output, job):
    variant, year, seed = job
    return Path(output)/'runs'/variant/str(year)/f'seed_{seed}'


def status(output, jobs, phase, active=()):
    active_pids = {job: process.pid for job, process, _ in active}
    entries, speeds = [], []
    for job in jobs:
        path = run_path(output, job)/'training.json'
        item = {'variant': job[0], 'year': job[1], 'seed': job[2], 'status': 'pending', 'updates_completed': 0}
        if path.exists():
            record = json.loads(path.read_text())
            item.update({k: record.get(k) for k in ('status', 'updates_completed', 'elapsed_seconds', 'last_update_at')})
            item['last_training_summary'] = record.get('last_training_summary')
            item['last_validation'] = record.get('validation_curve', [None])[-1] if record.get('validation_curve') else None
            if item['last_training_summary']:
                speeds.append(item['last_training_summary']['update_seconds'])
        if job in active_pids:
            item['pid'] = active_pids[job]
        entries.append(item)
    write_json(Path(output)/'status.json', {'phase': phase, 'updated_at': now(), 'jobs': entries,
               'completed_jobs': sum(e['status'] == 'completed' for e in entries), 'total_jobs': len(jobs),
               'active_jobs': len(active), 'median_latest_update_seconds': float(np.median(speeds)) if speeds else None,
               'test_locked': True, 'test_performance_accessed': False})


def dispatch(output, jobs, base, workers):
    pending = [j for j in jobs if not (run_path(output, j)/'training.json').exists()
               or json.loads((run_path(output, j)/'training.json').read_text()).get('status') != 'completed']
    active, attempts = [], {}
    expected_hash = json.loads((output/'protocol.json').read_text())['source_sha256']
    def stop(signum, frame):
        raise InterruptedError(f'coordinator received signal {signum}')
    previous_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or active:
            while pending and len(active) < workers:
                if source_hash() != expected_hash:
                    raise RuntimeError('source changed during the frozen experiment')
                job = pending.pop(0)
                attempts[job] = attempts.get(job, 0)+1
                variant, year, seed = job
                handle = (output/'logs'/f'{variant}_{year}_{seed}.log').open('a', buffering=1)
                command = [sys.executable, '-u', '-m', 'perpetual_v6.train', '--variant', variant,
                           '--year', str(year), '--seed', str(seed), '--output', str(output),
                           '--updates', str(base.updates), '--threads', str(base.threads)]
                env = dict(os.environ)
                env.update({'PYTHONPATH': str(ROOT/'src'), 'OMP_NUM_THREADS': str(base.threads),
                            'MKL_NUM_THREADS': str(base.threads), 'OPENBLAS_NUM_THREADS': str(base.threads)})
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active.append((job, process, handle))
                print(json.dumps({'event': 'started', 'job': job, 'pid': process.pid, 'at': now()}), flush=True)
            remaining = []
            for job, process, handle in active:
                code = process.poll()
                if code is None:
                    remaining.append((job, process, handle)); continue
                handle.close()
                outcome = {'job': job, 'exit_code': code, 'at': now(), 'attempt': attempts[job]}
                with (output/'logs/jobs.jsonl').open('a') as log:
                    log.write(json.dumps(outcome)+'\n')
                if code:
                    if attempts[job] >= 3:
                        raise RuntimeError(f'{job} failed three times; inspect its log')
                    pending.insert(0, job)
                print(json.dumps({'event': 'finished', **outcome}), flush=True)
            active = remaining
            status(output, jobs, 'training_and_validation', active)
            if pending or active:
                time.sleep(10)
    finally:
        for _, process, handle in active:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            handle.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def summarize(output, base, evaluate_mixtures=True):
    rankings = []
    torch.set_num_threads(base.threads)
    market = load_market(CACHE) if evaluate_mixtures else None
    for variant, config in variants(base).items():
        curves = []
        for year in YEARS:
            for seed in SEEDS:
                record = json.loads((run_path(output, (variant, year, seed))/'training.json').read_text())
                if record['status'] != 'completed':
                    raise RuntimeError('selection requires all completed development runs')
                curves.append({v['update']: v for v in record['validation_curve']})
        budgets = sorted(set.intersection(*(set(c) for c in curves)))
        candidates = []
        for budget in budgets:
            values = [c[budget] for c in curves]
            logs = [x['total_log_return_btc'] for x in values]
            candidates.append({'update': budget, 'median_log_return': float(np.median(logs)),
                               'mean_log_return': float(np.mean(logs)),
                               'positive_run_share': float(np.mean(np.array(logs)>0)),
                               'liquidations': sum(v['liquidations'] for v in values),
                               'per_year_median_log_return': {str(y): float(np.median(logs[i*len(SEEDS):(i+1)*len(SEEDS)])) for i, y in enumerate(YEARS)}})
        candidates.sort(key=lambda x: (-x['median_log_return'], -x['mean_log_return'], x['update']))
        selected = candidates[0]
        item = {'variant': variant, 'selected': selected, 'budgets': candidates, 'ensembles': {},
                'test_locked': True, 'test_performance_accessed': False}
        report_path = output/'comparison'/f'{variant}.json'
        if evaluate_mixtures:
            for year in YEARS:
                cached_path = output/'comparison'/f'{variant}_{year}_6bp.json'
                if cached_path.exists():
                    item['ensembles'][str(year)] = json.loads(cached_path.read_text())
                    continue
                models, scaler = [], None
                for seed in SEEDS:
                    path = run_path(output, (variant, year, seed))/f"policy_{selected['update']:04d}.pt"
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    model = ActorCritic(config); model.load_state_dict(saved['model']); models.append(model)
                    current = Scaler.from_dict(saved['scaler'])
                    if scaler is not None and scaler.as_dict() != current.as_dict():
                        raise ValueError('ensemble scaler mismatch')
                    scaler = current
                start, end = f'{year}-01-01T00:00:00Z', f'{year+1}-01-01T00:00:00Z'
                result, frame = evaluate(models, market, scaler, config, start, end)
                item['ensembles'][str(year)] = result
                write_json(cached_path, result)
                frame.to_csv(output/'comparison'/f'{variant}_{year}_6bp.csv.gz', index=False)
            item['development_candidate'] = (selected['median_log_return'] > 0 and selected['positive_run_share'] >= 2/3
                and selected['liquidations'] == 0 and all(v['total_log_return_btc'] > 0 and v['liquidations'] == 0 for v in item['ensembles'].values()))
        write_json(report_path, item)
        rankings.append(item)
        write_json(output/'comparison/ranking.json', {'architectures': rankings, 'test_locked': True,
                   'test_performance_accessed': False, 'selection_is_development_only': True})
    return rankings


def run(output=OUTPUT, base=Config(), workers=2):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('logs', 'scalers', 'comparison'):
        (output/folder).mkdir(exist_ok=True)
    with (output/'.experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Stage A exists to give the advantage estimator a window longer than
        # the ~22 h a 12 bp round trip needs to break even. Launching without
        # that would repeat v5 at greater expense, so refuse rather than run.
        for name, config in variants(base).items():
            if config.credit_window_hours < 20:
                raise ValueError(f'{name}: credit window {config.credit_window_hours:.1f} h is below the '
                                 'cost break-even holding period; stage A would not test its own hypothesis')
        spec = protocol(base)
        if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text()) != spec:
            raise ValueError('existing experiment source/config mismatch')
        write_json(output/'protocol.json', spec)
        write_json(output/'TEST_LOCK.json', {'locked': True, 'start_inclusive': str(TEST_START),
                   'performance_accessed': False, 'unlock_entrypoint': None,
                   'enforcement': 'read-only pre-2026 v4 cache; date guards; development fold whitelist; training-only branch bounds'})
        market = load_market(CACHE)
        for year in YEARS:
            path = output/'scalers'/f'{year}.json'
            if not path.exists():
                write_json(path, Scaler.fit(market, pd.Timestamp(f'{year}-01-01', tz='UTC')).as_dict())
        del market
        jobs = [(variant, year, seed) for seed in SEEDS for year in YEARS for variant in variants(base)]
        write_json(output/'queue.json', {'jobs': jobs, 'workers': workers, 'threads_per_worker': base.threads,
                   'updates_per_job': base.updates, 'total_jobs': len(jobs), 'test_locked': True})
        dispatch(output, jobs, base, workers)
        status(output, jobs, 'comparing_actual_ensembles')
        if source_hash() != spec['source_sha256']:
            raise RuntimeError('source changed before comparison')
        summarize(output, base)
        status(output, jobs, 'completed_test_locked')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--updates', type=int, default=400)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error('this host uses at most three concurrent jobs')
    try:
        run(args.output, replace(Config(), updates=args.updates, threads=args.threads), args.workers)
    except Exception as error:
        write_json(args.output/'failure.json', {'at': now(), 'error': str(error), 'test_locked': True})
        status_path = args.output/'status.json'
        if status_path.exists():
            saved = json.loads(status_path.read_text())
            saved.update({'phase': 'failed_test_locked', 'error': str(error), 'updated_at': now()})
            write_json(status_path, saved)
        raise


if __name__ == '__main__':
    main()
