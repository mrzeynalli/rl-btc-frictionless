"""Can this learner find a signal we planted, and does it learn the market rather than the drift?

Two pre-registered tests on synthetic markets whose optimal policy is known
(src/agent/synthetic.py), both scored on a held-out 90-day window after real costs:

- symmetric: the regime drift only. Can the machinery find and exploit it at all?
  Identical to the previous generation's planted test, so results compare directly.
- drift: BTC's measured in-sample Sharpe ratio (0.63) added as unconditional drift,
  over a training span as long as BTC's 2017-2023 (seven years), so the sample
  realises the drift the way BTC's did. Does the agent learn the regime, or the
  sample's direction?

PASS in both modes if the mean of the last three evaluations beats the BETTER of
constant long and constant short on the held-out window. A policy locked onto
either side cannot pass; one lucky checkpoint cannot pass either.

    python scripts/planted_edge_test.py --mode drift --seed 11 --updates 1000
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agent.config import Config, write_json
from agent.evaluate import evaluate
from agent.ledger import Environment, LONG, SHORT, TrainingEnvironment
from agent.network import ActorCritic
from agent.ppo import Trainer
from agent.synthetic import BTC_IN_SAMPLE_SHARPE, edge_economics, oracle_actions, planted_market, scaler_for

MARKET_SEED = 7
YEAR_MINUTES = 525_600
SEVEN_YEARS = 20_000 + 7 * YEAR_MINUTES
SPECS = {
    'symmetric': dict(unconditional_sharpe=0.0, minutes=600_000, start='2020-01-01',
                      train=(20_000, 450_000), validation=(460_000, 590_000)),
    # A 299-day span did not realise the drift: on this seed constant long lost
    # 1.61 log over it while constant short gained 0.59. Over seven years 59.8% of
    # 92-day training windows rise, against 57.3% for BTC in 2017-2023 (measured).
    # Dated from 2010 so seven synthetic years end before the 2026 test lock; the
    # generator does not depend on the calendar.
    'drift': dict(unconditional_sharpe=BTC_IN_SAMPLE_SHARPE, minutes=SEVEN_YEARS + 150_000, start='2010-01-01',
                  train=(20_000, SEVEN_YEARS), validation=(SEVEN_YEARS + 10_000, SEVEN_YEARS + 140_000)),
}


def reference_returns(market, regime, bounds, config):
    """Oracle and constant positions on the same window, under identical costs."""
    steps = (bounds[1] - bounds[0]) // 5
    out = {}
    for name, actions in (('oracle', oracle_actions(market, regime, bounds)),
                          ('constant_long', np.full(steps, LONG)), ('constant_short', np.full(steps, SHORT))):
        env = Environment(market, bounds, config, np.random.default_rng(0), training=False)
        out[name] = sum(float(env.step(np.array([a]))[2]['log_return'][0]) for a in actions)
    return out


def training_drift(market, bounds, config):
    """How often a training episode's price rises: the pull toward always-long."""
    window = config.episode_steps * config.decision_minutes
    price = np.asarray(market.prices[:, 3])
    starts = np.arange(bounds[0], bounds[1] - window, 1440)
    change = np.log(price[starts + window] / price[starts])
    return {'episode_windows': int(len(starts)), 'rising_share': float(np.mean(change > 0)),
            'mean_log_change': float(change.mean()), 'btc_2017_2023_rising_share': 0.573}


def regime_conditioning(frame, market, regime):
    """Share of time long and short while the planted regime is up and down."""
    # Unit-safe: evaluation frames carry second-resolution timestamps.
    minutes = ((frame.time - market.timestamp(0)) // pd.Timedelta(minutes=1)).to_numpy()
    up = regime[minutes] > 0
    side = np.sign(frame.contracts.to_numpy())
    return {'long_when_up': float(np.mean(side[up] > 0)), 'long_when_down': float(np.mean(side[~up] > 0)),
            'short_when_up': float(np.mean(side[up] < 0)), 'short_when_down': float(np.mean(side[~up] < 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=sorted(SPECS), default='symmetric')
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--evaluate-every', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('results/planted_edge'))
    args = parser.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    spec = SPECS[args.mode]
    config = replace(Config(), updates=args.updates)
    market, regime = planted_market(minutes=spec['minutes'], seed=MARKET_SEED, start=spec['start'],
                                    unconditional_sharpe=spec['unconditional_sharpe'])
    scaler = scaler_for(market)
    train_bounds = market.bounds(market.timestamp(spec['train'][0]), market.timestamp(spec['train'][1]))
    start, end = market.timestamp(spec['validation'][0]), market.timestamp(spec['validation'][1])
    references = reference_returns(market, regime, market.bounds(start, end), config)
    criterion = max(references['constant_long'], references['constant_short'])

    run = args.output/f'{args.mode}_seed_{args.seed}'
    run.mkdir(parents=True, exist_ok=True)
    header = {'mode': args.mode, 'seed': args.seed, 'market_seed': MARKET_SEED, 'updates': args.updates,
              'edge': edge_economics(market, config), 'references': references,
              'training_drift': training_drift(market, train_bounds, config),
              'pass_if_last_three_mean_above': criterion, 'training_window': [str(market.timestamp(b)) for b in train_bounds],
              'validation_window': [str(start), str(end)],
              'widths_actor': config.branch_widths(True), 'widths_critic': config.branch_widths(False)}
    write_json(run/'setup.json', header)
    print(json.dumps(header, indent=2), flush=True)

    env = TrainingEnvironment(market, train_bounds, config, np.random.default_rng(args.seed))
    model = ActorCritic(config)
    trainer = Trainer(model, env, scaler, config, args.seed)
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}", flush=True)

    started, curve = time.monotonic(), []
    with (run/'updates.jsonl').open('w', buffering=1) as log:
        while trainer.updates < args.updates:
            tick = time.monotonic()
            summary = trainer.run_update()
            summary['update_seconds'] = time.monotonic()-tick
            log.write(json.dumps(summary, allow_nan=False)+'\n')
            if trainer.updates % 10 == 0 or trainer.updates == 1:
                share, hold = summary['long_entry_share'], summary['policy_implied_holding_hours']
                print(f"upd {summary['update']:>4} {summary['update_seconds']:5.1f}s "
                      f"| H {summary['policy_entropy'] or 0:.3f} "
                      f"| hold {'-' if hold is None else f'{hold:7.1f}'}h "
                      f"| MI {summary['state_action_mutual_information']:.4f} "
                      f"| side S/F/L {'/'.join(f'{x:.2f}' for x in summary['account_side_share'])} "
                      f"| long entries {'-' if share is None else f'{share:.2f}'} "
                      f"| EV {summary['explained_variance_post'] or 0:.2f} "
                      f"| liq {summary['liquidations_in_rollout']}", flush=True)
            if trainer.updates % args.evaluate_every == 0 or trainer.updates == args.updates:
                result, frame = evaluate([model], market, scaler, config, start, end, seeds=2)
                entry = {'update': trainer.updates, **{k: v for k, v in result.items() if k != 'per_seed'},
                         **regime_conditioning(frame, market, regime)}
                curve.append(entry)
                write_json(run/f'validation_{trainer.updates:04d}.json', entry)
                print(f"  VALIDATION @{trainer.updates}: log return {result['total_log_return_btc_mean']:+.4f} "
                      f"[{result['total_log_return_btc_min']:+.4f}, {result['total_log_return_btc_max']:+.4f}] "
                      f"| orders {result['orders']:.0f} | hold {result['median_holding_spell_hours']:.1f}h "
                      f"| long up/down {entry['long_when_up']:.2f}/{entry['long_when_down']:.2f} "
                      f"| short up/down {entry['short_when_up']:.2f}/{entry['short_when_down']:.2f} "
                      f"| oracle {references['oracle']:+.3f} long {references['constant_long']:+.3f} "
                      f"short {references['constant_short']:+.3f}", flush=True)
    last = float(np.mean([v['total_log_return_btc_mean'] for v in curve[-3:]]))
    verdict = {'mode': args.mode, 'seed': args.seed, 'last_three_mean_log_return': last,
               'criterion': criterion, 'passed': bool(last > criterion),
               'oracle': references['oracle'], 'share_of_oracle': last / references['oracle'],
               'final_conditioning': {k: curve[-1][k] for k in ('long_when_up', 'long_when_down',
                                                                'short_when_up', 'short_when_down')},
               'hours': (time.monotonic()-started)/3600}
    write_json(run/'verdict.json', verdict)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == '__main__':
    main()
