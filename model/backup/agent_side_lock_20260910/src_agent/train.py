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

from .config import CACHE, OUTPUT, TRAIN_START, YEARS, Config, source_hash, write_json
from .data import Scaler, prepare
from .evaluate import evaluate
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


def train(year, seed, output=OUTPUT, base=Config()):
    if year not in YEARS:
        raise ValueError('only development validation folds are allowed')
    config = base
    output = Path(output)
    expected = json.loads((output/'protocol.json').read_text())
    if expected['source_sha256'] != source_hash() or expected['config'] != config.as_dict():
        raise ValueError('source/config differs from frozen protocol')
    torch.set_num_threads(config.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already fixed for this process; only settable before parallel work.
        pass
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    market = prepare(CACHE)
    end = pd.Timestamp(f'{year}-01-01', tz='UTC')
    scaler_path = output/'scalers'/f'{year}.json'
    scaler = Scaler.from_dict(json.loads(scaler_path.read_text())) if scaler_path.exists() else Scaler.fit(market, end)
    if pd.Timestamp(scaler.fitted_end) != end:
        raise ValueError('scaler does not match training cutoff')
    env = TrainingEnvironment(market, market.bounds(TRAIN_START, end), config, np.random.default_rng(seed))
    model = ActorCritic(config)
    trainer = Trainer(model, env, scaler, config, seed)
    run = run_path(output, year, seed)
    run.mkdir(parents=True, exist_ok=True)
    fingerprint = {'source_sha256': source_hash(), 'year': year, 'seed': seed, 'config': config.as_dict()}
    record = {'status': 'running', 'started_at': now(), 'specification': fingerprint,
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
            (run/'updates.jsonl').write_text('\n'.join(kept)+'\n')
    else:
        trainer.calibrate()
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
            log.write(json.dumps(summary, allow_nan=False)+'\n')
            record.update({'status': 'running', 'updates_completed': trainer.updates, 'last_update_at': now(),
                           'last_training_summary': summary, 'elapsed_seconds': summary['elapsed_seconds']})
            due = trainer.updates % config.validation_every == 0 or trainer.updates == config.updates
            if due:
                policy = {'model': model.state_dict(), 'scaler': scaler.as_dict(), 'specification': fingerprint,
                          'update': trainer.updates, 'test_locked': True}
                save_checkpoint(run/f'policy_{trainer.updates:04d}.pt', policy)
                record['status'] = 'validating'
                write_json(run/'training.json', record)
                result, frame = evaluate([model], market, scaler, config, end,
                                         pd.Timestamp(f'{year+1}-01-01', tz='UTC'))
                result['update'] = trainer.updates
                write_json(run/f'validation_{trainer.updates:04d}.json', result)
                frame.to_csv(run/f'validation_{trainer.updates:04d}.csv.gz', index=False)
                record['validation_curve'] = [v for v in record['validation_curve']
                                              if v['update'] != trainer.updates]+[result]
                print(json.dumps({'event': 'validation', 'year': year, 'seed': seed,
                                  **{k: v for k, v in result.items() if k != 'per_seed'}}), flush=True)
                record['status'] = 'running'
            record['elapsed_seconds'] = elapsed_before+time.monotonic()-started
            if trainer.updates % 10 == 0 or due:
                save_checkpoint(run/'resume.pt', {'trainer': trainer.state_dict(), 'record': record})
            write_json(run/'training.json', record)
            if trainer.updates % 5 == 0 or trainer.updates == 1:
                print(json.dumps({'year': year, 'seed': seed, **summary}), flush=True)
    record.update({'status': 'completed', 'finished_at': now()})
    write_json(run/'training.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description='Training/development validation only; test locked')
    parser.add_argument('--year', type=int, required=True, choices=YEARS)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--updates', type=int, default=Config.updates)
    parser.add_argument('--threads', type=int, default=Config.threads)
    args = parser.parse_args()
    train(args.year, args.seed, args.output, replace(Config(), updates=args.updates, threads=args.threads))


if __name__ == '__main__':
    main()
