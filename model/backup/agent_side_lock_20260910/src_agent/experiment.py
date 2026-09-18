"""Resumable seed sweep over three development folds; no test stage exists."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
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

from .config import CACHE, OUTPUT, ROOT, SEEDS, TEST_START, YEARS, Config, protocol, source_hash, write_json
from .data import Scaler, prepare
from .evaluate import evaluate, evaluate_blocks
from .network import ActorCritic
from .train import now, run_path


def status(output, jobs, phase, active=()):
    active_pids = {job: process.pid for job, process, _ in active}
    entries, speeds = [], []
    for job in jobs:
        path = run_path(output, *job)/'training.json'
        item = {'year': job[0], 'seed': job[1], 'status': 'pending', 'updates_completed': 0}
        if path.exists():
            record = json.loads(path.read_text())
            item.update({k: record.get(k) for k in ('status', 'updates_completed', 'elapsed_seconds', 'last_update_at')})
            item['last_training_summary'] = record.get('last_training_summary')
            curve = record.get('validation_curve') or []
            item['last_validation'] = {k: v for k, v in curve[-1].items() if k != 'per_seed'} if curve else None
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
    pending = [j for j in jobs if not (run_path(output, *j)/'training.json').exists()
               or json.loads((run_path(output, *j)/'training.json').read_text()).get('status') != 'completed']
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
                year, seed = job
                handle = (output/'logs'/f'{year}_{seed}.log').open('a', buffering=1)
                command = [sys.executable, '-u', '-m', 'agent.train', '--year', str(year),
                           '--seed', str(seed), '--output', str(output),
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


def summarize(output, base):
    """Pick one common budget across every seed and fold, then characterise it.

    Selection uses the median across runs rather than the mean: with outcomes
    this dispersed a single lucky seed should not choose the checkpoint.
    """
    output = Path(output)
    torch.set_num_threads(base.threads)
    curves = []
    for year in YEARS:
        for seed in SEEDS:
            record = json.loads((run_path(output, year, seed)/'training.json').read_text())
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
                           'positive_run_share': float(np.mean(np.array(logs) > 0)),
                           'median_holding_hours': float(np.median([x['median_holding_spell_hours'] for x in values])),
                           'median_orders': float(np.median([x['orders'] for x in values])),
                           'liquidations': int(sum(v['liquidations'] for v in values))})
    candidates.sort(key=lambda x: (-x['median_log_return'], -x['mean_log_return'], x['update']))
    selected = candidates[0]
    report = {'selected': selected, 'budgets': candidates, 'blocks': {}, 'stress': {},
              'test_locked': True, 'test_performance_accessed': False}
    market = prepare(CACHE)
    # Each seed's own policy is scored separately rather than blended into one
    # ensemble. After a predecessor whose entire outcome was decided by which
    # direction a seed happened to pick, the distribution across seeds IS the
    # result; an ensemble would average exactly the disagreement worth seeing.
    # It is also seven times cheaper per step, which is what makes a block-level
    # distribution affordable at all.
    for year in YEARS:
        blocks, stress = [], {f'{s*1e4:.0f}bp_slippage': [] for s in base.stress_slippage}
        for seed in SEEDS:
            saved = torch.load(run_path(output, year, seed)/f"policy_{selected['update']:04d}.pt",
                               map_location='cpu', weights_only=False)
            model = ActorCritic(base); model.load_state_dict(saved['model'])
            scaler = Scaler.from_dict(saved['scaler'])
            blocks.append({'seed': seed, **evaluate_blocks([model], market, scaler, base, year,
                                                           seeds=base.block_evaluation_seeds)})
            for slippage in base.stress_slippage:
                aggregate, _ = evaluate([model], market, scaler, base, f'{year}-01-01T00:00:00Z',
                                        f'{year+1}-01-01T00:00:00Z', slippage=slippage,
                                        seeds=base.block_evaluation_seeds)
                stress[f'{slippage*1e4:.0f}bp_slippage'].append(
                    {'seed': seed, **{k: v for k, v in aggregate.items() if k != 'per_seed'}})
        medians = [b['median_block_log_return'] for b in blocks]
        shares = [b['positive_block_share'] for b in blocks]
        report['blocks'][str(year)] = {
            'per_seed': blocks,
            'median_across_seeds': float(np.median(medians)),
            'seed_share_with_positive_median': float(np.mean(np.array(medians) > 0)),
            'median_positive_block_share': float(np.median(shares))}
        report['stress'][str(year)] = stress
    medians = [b['median_across_seeds'] for b in report['blocks'].values()]
    shares = [b['median_positive_block_share'] for b in report['blocks'].values()]
    report['development_candidate'] = bool(min(medians) > 0 and min(shares) >= 0.55
                                           and selected['liquidations'] == 0)
    report['criteria'] = (f'median {base.block_days}-day block log return above zero in every fold, at '
                          'least 55 per cent of blocks positive, and no liquidations; a development '
                          'screen only, not a claim of deployability')
    write_json(output/'comparison/selection.json', report)
    return report


def run(output=OUTPUT, base=Config(), workers=2):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('logs', 'scalers', 'comparison'):
        (output/folder).mkdir(exist_ok=True)
    with (output/'.experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Refuse to launch a configuration that cannot test its own hypothesis:
        # an advantage window below the cost break-even reproduces the collapse
        # this design exists to remove, at full expense.
        if base.credit_window_hours < 20:
            raise ValueError(f'credit window {base.credit_window_hours:.1f} h is below the cost '
                             'break-even holding period')
        spec = protocol(base)
        if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text()) != spec:
            raise ValueError('existing experiment source/config mismatch')
        write_json(output/'protocol.json', spec)
        write_json(output/'TEST_LOCK.json', {'locked': True, 'start_inclusive': str(TEST_START),
                   'performance_accessed': False, 'unlock_entrypoint': None,
                   'enforcement': 'cache built only from pre-2026 rows; date guards; development fold '
                                  'whitelist; training-only counterfactual bounds'})
        market = prepare(CACHE)
        for year in YEARS:
            path = output/'scalers'/f'{year}.json'
            if not path.exists():
                write_json(path, Scaler.fit(market, pd.Timestamp(f'{year}-01-01', tz='UTC')).as_dict())
        del market
        jobs = [(year, seed) for seed in SEEDS for year in YEARS]
        write_json(output/'queue.json', {'jobs': jobs, 'workers': workers, 'threads_per_worker': base.threads,
                   'updates_per_job': base.updates, 'total_jobs': len(jobs), 'test_locked': True})
        dispatch(output, jobs, base, workers)
        status(output, jobs, 'selecting')
        if source_hash() != spec['source_sha256']:
            raise RuntimeError('source changed before selection')
        summarize(output, base)
        status(output, jobs, 'completed_test_locked')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--updates', type=int, default=Config.updates)
    args = parser.parse_args()
    # Measured at roughly 0.9 GB per worker on this host, so five fit inside the
    # available memory. Nine jobs over five workers is two waves rather than
    # three, which is the difference between four and six days.
    if not 1 <= args.workers <= 5:
        parser.error('this host uses at most five concurrent jobs')
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
