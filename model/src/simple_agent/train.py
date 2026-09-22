"""Train one seed of the agent.

Search fold (walk forward): train on everything before year V and score the model on year V after every pass.
    PYTHONPATH=src .venv/bin/python -m simple_agent.train --seed 11 --validation-year 2023 --output <dir> [--set ...]
Final agent: train on everything before 2025 (config.TRAIN); nothing after the training period is loaded.
    PYTHONPATH=src .venv/bin/python -m simple_agent.train --seed 11 --final --output <dir> [--passes 10] [--set ...]

Writes, per seed:  updates.jsonl (every PPO update), checkpoints/pass_XX.pt (weights after each pass), config.json,
                   and for a search fold validation.jsonl (the validation-year score after each pass).
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .config import OUTPUT, TRAIN, Config, fold
from .data import FEATURES, Scaler, load, next_returns, rows
from .env import Env
from .evaluate import run_policy
from .model import ActorCritic
from .ppo import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--passes', type=int, default=Config.passes)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--validation-year', type=int, help='search fold: train before this year, score on it')
    target.add_argument('--final', action='store_true', help='train on config.TRAIN for the reported agent')
    parser.add_argument('--set', nargs='*', default=[], metavar='FIELD=VALUE',
                        help='override Config fields, e.g. --set gamma=0.99 hidden=256 (hyperparameter search)')
    args = parser.parse_args()
    if args.final:
        train_period, validation_period = TRAIN, None
    else:
        train_period, validation_period = fold(args.validation_year)
    overrides = {}
    for item in args.set:
        name, value = item.split('=', 1)
        overrides[name] = type(getattr(Config, name))(float(value) if '.' in value or 'e' in value else int(value))
    config = replace(Config(), passes=args.passes, **overrides)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    last = validation_period[1] if validation_period else train_period[1]
    table = load(end=last)                                      # no row after the period in use is loaded
    train_rows = rows(table, train_period)
    validation_rows = rows(table, validation_period) if validation_period else None
    scaler = Scaler(table, train_rows, config.obs_clip)
    features = scaler.transform(table)                          # z-scores with training mean and std
    returns = next_returns(table)
    train_returns = returns[train_rows]
    initial_variance = float(np.nanvar(train_returns[:-1]))     # B_0 for the reward: variance of hourly returns

    env = Env(features[train_rows], train_returns, config.envs, config.eta, initial_variance)
    model = ActorCritic(len(FEATURES) + 3, config.hidden)
    ppo = PPO(model, env, config)
    # One pass = as many samples as the training period has hours.
    updates_per_pass = math.ceil(len(train_rows) / (config.envs * config.rollout))

    out = args.output / f'seed_{args.seed}'
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({'config': asdict(config), 'seed': args.seed,
                                                 'features': list(FEATURES), 'scaler': scaler.state_dict(),
                                                 'train_hours': int(len(train_rows)),
                                                 'validation_year': args.validation_year,
                                                 'periods': {k: [str(v[0]), str(v[1])] for k, v in
                                                             (('train', train_period), ('validation', validation_period))
                                                             if v is not None},
                                                 'updates_per_pass': updates_per_pass,
                                                 'initial_variance': initial_variance}, indent=1))
    started = time.time()
    validation_log = open(out / 'validation.jsonl', 'w') if validation_rows is not None else None
    with open(out / 'updates.jsonl', 'w') as update_log:
        for p in range(1, config.passes + 1):
            for u in range(updates_per_pass):
                summary = ppo.update()
                summary.update({'pass': p, 'update': (p - 1) * updates_per_pass + u + 1,
                                'seconds': round(time.time() - started, 1)})
                update_log.write(json.dumps(summary) + '\n')
            update_log.flush()
            torch.save({'model': model.state_dict(), 'pass': p, 'seed': args.seed}, out / 'checkpoints' / f'pass_{p:02d}.pt')
            if validation_log is None:
                print(json.dumps({'seed': args.seed, 'pass': p, 'entropy': round(summary['entropy'], 3),
                                  'minutes': round((time.time() - started) / 60, 1)}), flush=True)
                continue
            result, positions, _, probabilities = run_policy(model, features, returns, validation_rows, config.warmup_hours)
            result.update({'pass': p, 'mean_probabilities': probabilities.mean(0).round(4).tolist(),
                           'seconds': round(time.time() - started, 1)})
            validation_log.write(json.dumps(result) + '\n')
            validation_log.flush()
            print(json.dumps({'seed': args.seed, 'pass': p, 'val_sharpe': round(result['sharpe'], 3),
                              'val_log_return': round(result['total_log_return'], 3),
                              'changes': result['position_changes'], 'entropy': round(summary['entropy'], 3),
                              'minutes': round((time.time() - started) / 60, 1)}), flush=True)
    if validation_log is not None:
        validation_log.close()


if __name__ == '__main__':
    main()
