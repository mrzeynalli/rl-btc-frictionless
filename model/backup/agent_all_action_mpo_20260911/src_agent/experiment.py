"""Resumable seed sweep over the two walk-forward folds; no test stage exists."""
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

from .config import (CACHE, COST_BREAK_EVEN_HOURS, OUTPUT, ROOT, SEEDS, TEST_START, YEARS, Config, protocol,
                     source_hash, write_json)
from .data import Scaler, prepare
from .evaluate import QUARTER_MONTHS, evaluate_quarters
from .network import ActorCritic
from .scoring import interquartile_mean, probability_of_backtest_overfitting, stratified_bootstrap_interval
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
            item['last_validation'] = curve[-1] if curve else None
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


def quarter_scores(records, budget, key='total_log_return_btc_mean'):
    """(seeds, slices) matrix: each slice is one validation quarter of one fold."""
    matrix = np.zeros((len(SEEDS), len(YEARS) * len(QUARTER_MONTHS)))
    for s, seed in enumerate(SEEDS):
        for y, year in enumerate(YEARS):
            for q, quarter in enumerate(records[(year, seed)][budget]['quarters']):
                matrix[s, y * len(QUARTER_MONTHS) + q] = quarter[key]
    return matrix


def summarize(output, base):
    """Characterise every common checkpoint, choose one, and measure what choosing cost.

    Selection maximises the interquartile mean of quarterly log return across
    seeds and quarters (Agarwal et al. 2021). The interval is a stratified
    bootstrap over seeds within each quarter. Because the best of ten checkpoints
    is chosen on the same data it is scored on, the probability of backtest
    overfitting is estimated by CSCV over the eight quarters (Bailey et al.
    2017). No pass or fail threshold is imposed on any of these.
    """
    output = Path(output)
    torch.set_num_threads(base.threads)
    records = {}
    for year in YEARS:
        for seed in SEEDS:
            record = json.loads((run_path(output, year, seed)/'training.json').read_text())
            if record['status'] != 'completed':
                raise RuntimeError('selection requires all completed development runs')
            records[(year, seed)] = {v['update']: v for v in record['validation_curve']}
    budgets = sorted(set.intersection(*(set(c) for c in records.values())))
    candidates = []
    for budget in budgets:
        scores = quarter_scores(records, budget)
        shares = [q['long_entry_share'] for (year, seed) in records for q in records[(year, seed)][budget]['quarters']
                  if q['long_entry_share'] is not None]
        candidates.append({'update': budget, 'iqm_quarter_log_return': interquartile_mean(scores),
                           'median_quarter_log_return': float(np.median(scores)),
                           'mean_quarter_log_return': float(np.mean(scores)),
                           'positive_quarter_share': float(np.mean(scores > 0)),
                           'liquidations': int(sum(records[k][budget]['liquidations'] for k in records)),
                           'median_long_entry_share': float(np.median(shares)) if shares else None,
                           'mean_long_share': float(np.mean([records[k][budget]['long_share'] for k in records])),
                           'mean_short_share': float(np.mean([records[k][budget]['short_share'] for k in records]))})
    selected = max(candidates, key=lambda x: (x['iqm_quarter_log_return'], -x['update']))
    interval = stratified_bootstrap_interval(quarter_scores(records, selected['update']), interquartile_mean,
                                             base.bootstrap_replications, base.confidence_level)
    performance = np.stack([quarter_scores(records, b) for b in budgets], -1).transpose(1, 0, 2)
    report = {'selected': selected, 'budgets': candidates, 'selected_interval': interval,
              'probability_of_backtest_overfitting': probability_of_backtest_overfitting(performance),
              'passive': {}, 'stress': {}, 'test_locked': True, 'test_performance_accessed': False}
    market = prepare(CACHE)
    for year in YEARS:
        report['passive'][str(year)] = {
            name: [q['total_log_return_btc'] for q in evaluate_quarters([], market, None, base, year, baseline=name)[0]]
            for name in ('long', 'short', 'flat')}
    for slippage in base.stress_slippage:
        matrix = np.zeros((len(SEEDS), len(YEARS) * len(QUARTER_MONTHS)))
        for s, seed in enumerate(SEEDS):
            for y, year in enumerate(YEARS):
                saved = torch.load(run_path(output, year, seed)/f"policy_{selected['update']:04d}.pt",
                                   map_location='cpu', weights_only=False)
                model = ActorCritic(base); model.load_state_dict(saved['model'])
                quarters, _ = evaluate_quarters([model], market, Scaler.from_dict(saved['scaler']), base, year,
                                                slippage=slippage)
                matrix[s, y * len(QUARTER_MONTHS):(y + 1) * len(QUARTER_MONTHS)] = [
                    q['total_log_return_btc_mean'] for q in quarters]
        report['stress'][f'{slippage*1e4:.0f}bp_slippage'] = {
            'iqm_quarter_log_return': interquartile_mean(matrix), 'scores': matrix.tolist()}
    write_json(output/'comparison/selection.json', report)
    return report


def run(output=OUTPUT, base=Config(), workers=2):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('logs', 'scalers', 'comparison'):
        (output/folder).mkdir(exist_ok=True)
    with (output/'.experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A credit window below the cost break-even reproduces the collapse the
        # horizon exists to remove, at full expense.
        if base.credit_window_hours < COST_BREAK_EVEN_HOURS:
            raise ValueError(f'credit window {base.credit_window_hours:.1f} h is below the cost '
                             'break-even holding period')
        spec = protocol(base)
        if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text()) != spec:
            raise ValueError('existing experiment source/config mismatch')
        write_json(output/'protocol.json', spec)
        write_json(output/'TEST_LOCK.json', {'locked': True, 'start_inclusive': str(TEST_START),
                   'performance_accessed': False, 'unlock_entrypoint': None,
                   'enforcement': 'cache built only from pre-2026 rows; date guards; development fold '
                                  'whitelist'})
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
    if not 1 <= args.workers <= len(SEEDS) * len(YEARS):
        parser.error('at most one worker per job')
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
