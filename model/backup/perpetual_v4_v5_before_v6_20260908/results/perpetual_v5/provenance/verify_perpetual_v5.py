"""Production-configuration development smoke and clearly labelled toy learning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import torch

from perpetual.config import TRAIN_START, write_json
from perpetual.data import Scaler, causal_arrays, load_market
from perpetual_v5.config import CACHE, Config, variants, source_hash
from perpetual_v5.environment import TrainingEnvironment
from perpetual_v5.evaluate import evaluate
from perpetual_v5.network import ActorCritic
from perpetual_v5.ppo import Trainer


def toy_market():
    # Deterministic, readily learnable regime changes plus small observation noise.
    # This diagnoses learning capability, not realistic market profitability.
    size = 115200
    time_grid = pd.date_range('2016-12-01', periods=size, freq='min', tz='UTC')
    rng = np.random.default_rng(2026)
    drift = np.where((np.arange(size)//720) % 2 == 0, .0003, -.0003)
    close = 10000*np.exp(np.cumsum(drift+rng.normal(0, .00004, size)))
    opening = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame({'open_time': time_grid, 'open': opening, 'high': np.maximum(opening, close),
                         'low': np.minimum(opening, close), 'close': close, 'volume': np.full(size, 100000.)})
    funding = pd.DataFrame({'timestamp': pd.DatetimeIndex([], tz='UTC'), 'fundingRate': [], 'funding_interval_hours': []})
    return causal_arrays(bars, funding, time_grid[0], time_grid[-1]+pd.Timedelta(minutes=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['real', 'toy'], required=True)
    parser.add_argument('--variant', choices=list(variants()), required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    c = variants()[args.variant]
    torch.set_num_threads(c.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(11); torch.use_deterministic_algorithms(True)
    market = load_market(CACHE) if args.mode == 'real' else toy_market()
    train_end = pd.Timestamp('2023-01-01' if args.mode == 'real' else '2017-02-01', tz='UTC')
    val_end = train_end+pd.Timedelta(days=2 if args.mode == 'real' else 7)
    scaler = Scaler.fit(market, train_end)
    env = TrainingEnvironment(market, market.bounds(TRAIN_START, train_end), c, np.random.default_rng(11))
    trainer = Trainer(ActorCritic(c), env, scaler, c, 11)
    trainer.calibrate()
    record = {'mode': args.mode, 'variant': args.variant, 'config': c.as_dict(), 'source_sha256': source_hash(),
              'production_hyperparameters': True, 'test_locked': True, 'test_performance_accessed': False,
              'toy_is_not_market_evidence': args.mode == 'toy', 'updates': [], 'validations': []}
    start = time.monotonic()
    for update in range(args.updates):
        begin = time.monotonic()
        summary = trainer.run_update()
        summary['seconds'] = time.monotonic()-begin
        record['updates'].append(summary)
        if update+1 in (20, 40, 60, 80) or update+1 == args.updates:
            result, frame = evaluate([trainer.model], market, scaler, c, train_end, val_end)
            result['update'] = update+1
            record['validations'].append(result)
            frame.to_csv(args.output/f"{args.mode}_{args.variant}_{update+1}.csv.gz", index=False)
        record['elapsed_seconds'] = time.monotonic()-start
        record['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        write_json(args.output/f'{args.mode}_{args.variant}.json', record)
        print(json.dumps({'mode': args.mode, 'variant': args.variant, **summary}), flush=True)
    print(json.dumps({'completed': True, 'mode': args.mode, 'last_validation': record['validations'][-1]}), flush=True)


if __name__ == '__main__':
    main()
