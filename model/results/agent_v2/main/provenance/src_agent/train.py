"""One training run: one validation fold, one seed. Development data only."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .config import (CACHE, DEVELOPMENT_FOLDS, OUTPUT, TRAIN_START, Config, check_horizon, checkpoint_updates,
                     fold_config, source_hash, write_json)
from .data import Scaler, prepare, training_decisions
from .evaluate import evaluate_quarters
from .ledger import TrainingEnvironment
from .network import ActorCritic
from .ppo import Trainer


def now():
    return datetime.now(timezone.utc).isoformat()


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def run_path(output, year, seed):
    return Path(output)/'runs'/str(year)/f'seed_{seed}'


def fold_end(year):
    return pd.Timestamp(f'{year}-01-01', tz='UTC')


def validation_summary(quarters, update):
    """Compact per-quarter record for the curve; per-seed detail is saved separately."""
    compact = [{k: v for k, v in q.items() if k != 'per_seed'} for q in quarters]
    return {'update': update, 'quarters': compact,
            'log_return_mean': float(sum(q['total_log_return_mean'] for q in quarters)),
            'liquidations': int(sum(q['liquidations'] for q in quarters)),
            'orders_median': float(np.median([q['orders'] for q in quarters])),
            'long_share': float(np.mean([q['long_share'] for q in quarters])),
            'short_share': float(np.mean([q['short_share'] for q in quarters])),
            'test_locked': True, 'test_performance_accessed': False}


def train(year, seed, output=OUTPUT, base=Config()):
    if year not in base.fold_quarters or year not in DEVELOPMENT_FOLDS:
        raise ValueError('only the development folds of this experiment are allowed')
    output = Path(output)
    expected = json.loads((output/'protocol.json').read_text())
    if expected['source_sha256'] != source_hash() or expected['config'] != base.as_dict():
        raise ValueError('source/config differs from frozen protocol')
    torch.set_num_threads(base.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already fixed for this process; only settable before parallel work.
        pass
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    market = prepare(CACHE)
    end = fold_end(year)
    scaler_path = output/'scalers'/f'{year}.json'
    scaler = Scaler.from_dict(json.loads(scaler_path.read_text())) if scaler_path.exists() else Scaler.fit(market, end)
    if pd.Timestamp(scaler.fitted_end) != end:
        raise ValueError('scaler does not match training cutoff')
    # The update budget, the reward prior and the cost break-even come from this fold's
    # training span alone.
    decisions = training_decisions(market, end, base)
    config = fold_config(base, scaler.bar_return_sd, decisions)
    break_even = check_horizon(config, scaler.bar_return_sd, config.enforce_horizon)
    schedule = checkpoint_updates(config)
    # Training pays the inflated cost; validation below pays the real one.
    training_costs = replace(config, fee=config.fee * config.training_cost_multiplier,
                             slippage=config.slippage * config.training_cost_multiplier)
    env = TrainingEnvironment(market, market.bounds(TRAIN_START, end), training_costs, np.random.default_rng(seed))
    model = ActorCritic(config)
    trainer = Trainer(model, env, scaler, config, seed)
    run = run_path(output, year, seed)
    run.mkdir(parents=True, exist_ok=True)
    fingerprint = {'source_sha256': source_hash(), 'year': year, 'seed': seed, 'config': base.as_dict()}
    fold = {'training_decisions': int(decisions), 'updates': config.updates, 'checkpoints': list(schedule),
            'passes_per_update': config.samples_per_update / decisions, 'bar_return_sd': scaler.bar_return_sd,
            'dsr_initial_second_moment': config.dsr_initial_second_moment, 'cost_break_even_hours': break_even,
            'credit_window_hours': config.credit_window_hours}
    record = {'status': 'running', 'started_at': now(), 'specification': fingerprint, 'fold': fold,
              'train_end_exclusive': str(end), 'parameters': sum(p.numel() for p in model.parameters()),
              'updates_completed': 0, 'validation_curve': [], 'test_locked': True,
              'test_performance_accessed': False}
    if (run/'training.json').exists():
        previous = json.loads((run/'training.json').read_text())
        if previous['specification'] != fingerprint:
            raise ValueError('checkpoint experiment identity mismatch')
        if previous['status'] == 'completed':
            return previous
    if (run/'resume.pt').exists():
        saved = torch.load(run/'resume.pt', map_location='cpu', weights_only=False)
        if saved['record']['specification'] != fingerprint:
            raise ValueError('resume provenance mismatch')
        trainer.load_state_dict(saved['trainer'])
        record = saved['record']
        if (run/'updates.jsonl').exists():
            kept = []
            for line in (run/'updates.jsonl').read_text().splitlines():
                try:
                    if json.loads(line)['update'] <= trainer.updates:
                        kept.append(line)
                except ValueError:
                    pass
            (run/'updates.jsonl').write_text('\n'.join(kept)+'\n' if kept else '')
    else:
        (run/'updates.jsonl').write_text('')
    elapsed_before, started = record.get('elapsed_seconds', 0.), time.monotonic()
    record['status'] = 'running'
    write_json(run/'training.json', record)
    with (run/'updates.jsonl').open('a', buffering=1) as log:
        while trainer.updates < config.updates:
            update_started = time.monotonic()
            summary = trainer.run_update()
            summary['update_seconds'] = time.monotonic()-update_started
            summary['elapsed_seconds'] = elapsed_before+time.monotonic()-started
            summary['passes'] = trainer.updates * fold['passes_per_update']
            log.write(json.dumps(summary, allow_nan=False)+'\n')
            record.update({'status': 'running', 'updates_completed': trainer.updates, 'last_update_at': now(),
                           'last_training_summary': summary, 'elapsed_seconds': summary['elapsed_seconds']})
            due = trainer.updates in schedule
            if due:
                policy = {'model': model.state_dict(), 'scaler': scaler.as_dict(), 'specification': fingerprint,
                          'fold': fold, 'update': trainer.updates, 'test_locked': True}
                save_checkpoint(run/f'policy_{trainer.updates:05d}.pt', policy)
                record['status'] = 'validating'
                write_json(run/'training.json', record)
                quarters, frame = evaluate_quarters([model], market, scaler, config, year)
                write_json(run/f'validation_{trainer.updates:05d}.json',
                           {'update': trainer.updates, 'quarters': quarters})
                frame.to_csv(run/f'validation_{trainer.updates:05d}.csv.gz', index=False)
                result = validation_summary(quarters, trainer.updates)
                result['passes'] = summary['passes']
                # The same exam on the last training year: separates failing to learn from failing
                # to generalise when validation is poor.
                inside, inside_frame = evaluate_quarters([model], market, scaler, config, year - 1, count=4)
                write_json(run/f'in_sample_{trainer.updates:05d}.json',
                           {'update': trainer.updates, 'year': year - 1, 'quarters': inside})
                inside_frame.to_csv(run/f'in_sample_{trainer.updates:05d}.csv.gz', index=False)
                result['in_sample'] = {k: v for k, v in validation_summary(inside, trainer.updates).items()
                                       if k in ('log_return_mean', 'orders_median', 'long_share', 'short_share')}
                result['in_sample']['quarters'] = [q['total_log_return_mean'] for q in inside]
                np.savez_compressed(run/f'rollout_{trainer.updates:05d}.npz', **trainer.last_rollout)
                record['validation_curve'] = [v for v in record['validation_curve']
                                              if v['update'] != trainer.updates]+[result]
                print(json.dumps({'event': 'validation', 'year': year, 'seed': seed,
                                  **{k: v for k, v in result.items() if k != 'quarters'}}), flush=True)
                record['status'] = 'running'
            record['elapsed_seconds'] = elapsed_before+time.monotonic()-started
            if trainer.updates % 50 == 0 or due:
                save_checkpoint(run/'resume.pt', {'trainer': trainer.state_dict(), 'record': record})
            write_json(run/'training.json', record)
            if trainer.updates % 25 == 0 or trainer.updates == 1:
                print(json.dumps({'year': year, 'seed': seed, **summary}), flush=True)
    record.update({'status': 'completed', 'finished_at': now()})
    write_json(run/'training.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description='Training/development validation only; test locked')
    parser.add_argument('--year', type=int, required=True, choices=sorted(DEVELOPMENT_FOLDS))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    # The configuration is the experiment's frozen specification, never re-typed.
    base = Config.from_dict(json.loads((args.output/'protocol.json').read_text())['config'])
    train(args.year, args.seed, args.output, base)


if __name__ == '__main__':
    main()
