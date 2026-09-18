"""Can this learner find a signal we planted ourselves?

Trains on a synthetic market whose drift is predictable from one observation
column, and whose oracle policy is known to beat buy-and-hold after real costs.
A failure here is a failure of the machinery, not of the market.

Also sweeps network capacity, which is the one architectural choice no citation
can settle: Andrychowicz et al. are explicit that optimal policy width is
environment-dependent and that both too-small and too-large hurt.

    python scripts/planted_edge_test.py --capacity small --updates 500
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from agent.config import Config, write_json
from agent.evaluate import evaluate
from agent.ledger import Environment, LONG, TrainingEnvironment
from agent.network import ActorCritic
from agent.ppo import Trainer
from agent.synthetic import edge_economics, oracle_actions, planted_market, scaler_for

CAPACITY = {
    # actor width dominates cost: it runs at every rollout step and in every
    # minibatch, so 64 -> 128 roughly doubles wall clock while the critic barely
    # moves it.
    'small': dict(hidden=64, local_width=32, head_width=64, position_hidden=16),
    'large': dict(),
}


def reference_returns(market, regime, bounds, config):
    """Oracle and buy-and-hold on the same window, under identical costs."""
    steps = (bounds[1] - bounds[0]) // 5
    out = {}
    for name, actions in (('oracle', oracle_actions(market, regime, bounds)),
                          ('constant_long', np.full(steps, LONG))):
        env = Environment(market, bounds, config, np.random.default_rng(0), training=False)
        out[name] = sum(float(env.step(np.array([a]))[2]['log_return'][0]) for a in actions)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity', choices=sorted(CAPACITY), default='large')
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--evaluate-every', type=int, default=100)
    parser.add_argument('--oracle-sharpe', type=float, default=3.0)
    parser.add_argument('--output', type=Path, default=Path('results/planted_edge'))
    args = parser.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    config = replace(Config(), updates=args.updates, **CAPACITY[args.capacity])
    market, regime = planted_market(minutes=600_000, seed=7, oracle_sharpe=args.oracle_sharpe)
    scaler = scaler_for(market, config)
    train_bounds = market.bounds(market.timestamp(20_000), market.timestamp(450_000))
    start, end = market.timestamp(460_000), market.timestamp(590_000)
    references = reference_returns(market, regime, market.bounds(start, end), config)
    economics = edge_economics(market, config)

    run = args.output/args.capacity
    run.mkdir(parents=True, exist_ok=True)
    header = {'capacity': args.capacity, 'widths_actor': config.branch_widths(True),
              'widths_critic': config.branch_widths(False), 'updates': args.updates,
              'seed': args.seed, 'edge': economics, 'references': references,
              'validation_window': [str(start), str(end)]}
    write_json(run/'setup.json', header)
    print(json.dumps(header, indent=2), flush=True)

    env = TrainingEnvironment(market, train_bounds, config, np.random.default_rng(args.seed))
    model = ActorCritic(config)
    trainer = Trainer(model, env, scaler, config, args.seed)
    trainer.calibrate()
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}", flush=True)

    started = time.monotonic()
    with (run/'updates.jsonl').open('w', buffering=1) as log:
        while trainer.updates < args.updates:
            tick = time.monotonic()
            summary = trainer.run_update()
            summary['update_seconds'] = time.monotonic()-tick
            log.write(json.dumps(summary, allow_nan=False)+'\n')
            if trainer.updates % 10 == 0 or trainer.updates == 1:
                print(f"upd {summary['update']:>4} {summary['update_seconds']:5.1f}s "
                      f"| gateH {summary['gate_entropy']:.4f} a {summary['gate_entropy_alpha']:.5f} "
                      f"| dirH {summary['direction_entropy']:.3f} "
                      f"| hold {summary['policy_implied_holding_hours']:7.1f}h "
                      f"| MIflat {summary['state_action_mutual_information_flat']:.4f} "
                      f"| flat {summary['account_side_share'][1]:.3f} "
                      f"| turn {summary['turnover_per_decision']:.5f} "
                      f"| EV {summary['explained_variance_post'] or 0:.2f} "
                      f"| rew {summary['mean_reward']:+.5f}", flush=True)
            if trainer.updates % args.evaluate_every == 0 or trainer.updates == args.updates:
                result, _ = evaluate([model], market, scaler, config, start, end, seeds=2)
                result['update'] = trainer.updates
                write_json(run/f'validation_{trainer.updates:04d}.json',
                           {k: v for k, v in result.items() if k != 'per_seed'})
                print(f"  VALIDATION @{trainer.updates}: log return median {result['total_log_return_btc']:+.4f} "
                      f"[{result['total_log_return_btc_min']:+.4f}, {result['total_log_return_btc_max']:+.4f}] "
                      f"| orders {result['orders']:.0f} | hold {result['median_holding_spell_hours']:.1f}h "
                      f"| oracle {references['oracle']:+.3f} | long {references['constant_long']:+.3f}",
                      flush=True)
    print(f"finished in {(time.monotonic()-started)/3600:.2f} h", flush=True)


if __name__ == '__main__':
    main()
