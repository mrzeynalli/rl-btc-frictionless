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

from .config import (CACHE, OUTPUT, REWARDS, ROOT, SEEDS, TEST_START, YEARS, Config, check_horizon, fold_config,
                     protocol, quarter_episode, source_hash, write_json)
from .data import Scaler, prepare, training_decisions
from .evaluate import QUARTER_MONTHS, evaluate_quarters
from .network import ActorCritic
from .scoring import interquartile_mean, probability_of_backtest_overfitting, stratified_bootstrap_interval
from .train import fold_end, now, run_path


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
                           '--passes', str(base.training_passes), '--validation-passes', str(base.validation_passes),
                           '--threads', str(base.threads), '--reward', base.reward,
                           '--decision-bars', str(base.decision_bars)]
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


def quarter_scores(records, checkpoint, key='total_log_return_btc_mean'):
    """(seeds, slices) matrix at one checkpoint index: each slice is one validation quarter of one fold."""
    matrix = np.zeros((len(SEEDS), len(YEARS) * len(QUARTER_MONTHS)))
    for s, seed in enumerate(SEEDS):
        for y, year in enumerate(YEARS):
            for q, quarter in enumerate(records[(year, seed)][checkpoint]['quarters']):
                matrix[s, y * len(QUARTER_MONTHS) + q] = quarter[key]
    return matrix


def break_even_slippage(results):
    """Extra slippage per leg, in bp, at which each evaluated path's log return would reach zero.

    First order: extra slippage s on every leg costs s x turnover in log return, where
    turnover sums each leg's notional over the equity its trade started with. Derived from
    the evaluations themselves, so no grid of stress levels has to be chosen.
    """
    values = np.asarray([r['total_log_return_btc'] / r['turnover'] * 1e4 for r in results if r['turnover'] > 0])
    if not len(values):
        return {'paths': 0}
    return {'paths': int(len(values)), 'iqm_bp': interquartile_mean(values), 'median_bp': float(np.median(values)),
            'share_above_zero': float(np.mean(values > 0))}


def fold_configs(output, base, market):
    """Each fold's configuration, from the scaler measured on its training span."""
    configs = {}
    for year in YEARS:
        scaler = Scaler.from_dict(json.loads((Path(output)/'scalers'/f'{year}.json').read_text()))
        configs[year] = (fold_config(base, scaler.bar_return_sd, training_decisions(market, fold_end(year), base)),
                         scaler)
    return configs


def summarize(output, base):
    """Characterise every checkpoint, choose one, and measure what choosing cost.

    Checkpoints are aligned by index: the k-th checkpoint of every run lies after the same
    number of passes over its fold's training span. Selection maximises the interquartile
    mean of quarterly log return across seeds and quarters (Agarwal et al. 2021) - the
    validation-based stopping of Moody & Saffell (2001) and Deng et al. (2017), made across
    checkpoints. The interval is a stratified bootstrap over seeds within each quarter.
    Because the best checkpoint is chosen on the same data it is scored on, the probability
    of backtest overfitting is estimated by CSCV over the eight quarters (Bailey et al.
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
            records[(year, seed)] = sorted(record['validation_curve'], key=lambda v: v['update'])
    count = min(len(curve) for curve in records.values())
    candidates = []
    for checkpoint in range(count):
        scores = quarter_scores(records, checkpoint)
        entries = [records[key][checkpoint] for key in records]
        shares = [q['long_entry_share'] for entry in entries for q in entry['quarters'] if q['long_entry_share'] is not None]
        candidates.append({'checkpoint': checkpoint + 1, 'passes': float(np.mean([e.get('passes', np.nan) for e in entries])),
                           'updates': {f'{year}_{seed}': records[(year, seed)][checkpoint]['update'] for year, seed in records},
                           'iqm_quarter_log_return': interquartile_mean(scores),
                           'median_quarter_log_return': float(np.median(scores)),
                           'mean_quarter_log_return': float(np.mean(scores)),
                           'positive_quarter_share': float(np.mean(scores > 0)),
                           'liquidations': int(sum(e['liquidations'] for e in entries)),
                           'median_long_entry_share': float(np.median(shares)) if shares else None,
                           'mean_long_share': float(np.mean([e['long_share'] for e in entries])),
                           'mean_short_share': float(np.mean([e['short_share'] for e in entries]))})
    selected = max(candidates, key=lambda x: (x['iqm_quarter_log_return'], -x['checkpoint']))
    index = selected['checkpoint'] - 1
    interval = stratified_bootstrap_interval(quarter_scores(records, index), interquartile_mean,
                                             base.bootstrap_replications, base.confidence_level)
    performance = np.stack([quarter_scores(records, k) for k in range(count)], -1).transpose(1, 0, 2)
    report = {'selected': selected, 'checkpoints': candidates, 'selected_interval': interval,
              'probability_of_backtest_overfitting': probability_of_backtest_overfitting(performance),
              'passive': {}, 'test_locked': True, 'test_performance_accessed': False}
    market = prepare(CACHE)
    configs = fold_configs(output, base, market)
    for year in YEARS:
        config, _ = configs[year]
        report['passive'][str(year)] = {
            name: [q['total_log_return_btc'] for q in evaluate_quarters([], market, None, config, year, baseline=name)[0]]
            for name in ('long', 'short', 'flat')}
    paths = []
    for seed in SEEDS:
        for year in YEARS:
            config, _ = configs[year]
            update = records[(year, seed)][index]['update']
            saved = torch.load(run_path(output, year, seed)/f'policy_{update:05d}.pt', map_location='cpu', weights_only=False)
            model = ActorCritic(config)
            model.load_state_dict(saved['model'])
            quarters, _ = evaluate_quarters([model], market, Scaler.from_dict(saved['scaler']), config, year)
            paths.extend(result for quarter in quarters for result in quarter['per_seed'])
    report['break_even_slippage'] = break_even_slippage(paths)
    write_json(output/'comparison/selection.json', report)
    return report


def run(output=OUTPUT, base=Config(), workers=2):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('logs', 'scalers', 'comparison'):
        (output/folder).mkdir(exist_ok=True)
    with (output/'.experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = protocol(base)
        if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text()) != spec:
            raise ValueError('existing experiment source/config mismatch')
        write_json(output/'protocol.json', spec)
        write_json(output/'TEST_LOCK.json', {'locked': True, 'start_inclusive': str(TEST_START),
                   'performance_accessed': False, 'unlock_entrypoint': None,
                   'enforcement': 'cache built only from pre-2026 rows; date guards; development fold '
                                  'whitelist'})
        market = prepare(CACHE)
        folds = {}
        for year in YEARS:
            path = output/'scalers'/f'{year}.json'
            if not path.exists():
                write_json(path, Scaler.fit(market, fold_end(year)).as_dict())
        for year, (config, scaler) in fold_configs(output, base, market).items():
            # A credit window below the cost break-even reproduces the collapse the
            # horizon exists to remove, at full expense.
            folds[str(year)] = {'training_decisions': int(training_decisions(market, fold_end(year), base)),
                                'updates': config.updates, 'bar_return_sd': scaler.bar_return_sd,
                                'cost_break_even_hours': check_horizon(config, scaler.bar_return_sd),
                                'credit_window_hours': config.credit_window_hours}
        del market
        jobs = [(year, seed) for seed in SEEDS for year in YEARS]
        write_json(output/'queue.json', {'jobs': jobs, 'workers': workers, 'threads_per_worker': base.threads,
                   'folds': folds, 'total_jobs': len(jobs), 'test_locked': True})
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
    parser.add_argument('--passes', type=int, default=Config.training_passes)
    parser.add_argument('--validation-passes', type=int, default=Config.validation_passes)
    parser.add_argument('--reward', choices=REWARDS, default=Config.reward)
    parser.add_argument('--decision-bars', type=int, default=Config.decision_bars)
    args = parser.parse_args()
    if not 1 <= args.workers <= len(SEEDS) * len(YEARS):
        parser.error('at most one worker per job')
    try:
        run(args.output, replace(Config(), threads=args.threads, training_passes=args.passes,
                                 validation_passes=args.validation_passes, reward=args.reward,
                                 decision_bars=args.decision_bars,
                                 episode_steps=quarter_episode(args.decision_bars)), args.workers)
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
