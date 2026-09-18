"""Pre-registered verification on synthetic markets, run the way the real experiment runs.

The market's optimal policy is known (src/agent/synthetic.py). A run trains for a number
of passes over a synthetic training span with the production configuration, validates a
checkpoint every few passes on a validation window and also scores it on a later test
window, selects the checkpoint with the best validation return - as the real experiment
selects among its checkpoints - and judges that checkpoint on the test window, which
plays no part in the choice. The whole validation and test curve is kept, so where
held-out performance peaks, in passes, is measured rather than assumed.

- latent: bear, range and bull regimes generated as minute bars and passed through the
  real feature pipeline; the regime is never shown. Training on three years, validation
  on the next 26 weeks, test on the last 25.
- observable: the regime is written into the trend feature; only PPO, the ledger and the
  network are under test.

PASS if the selected checkpoint's test-window log return beats the best of flat,
constant long and constant short on the test window, AND on the test window it is long
more often in up-regimes than in down-regimes and short more often in down-regimes than
in up-regimes.

    python scripts/planted_edge_test.py --mode latent --reward sortino --seed 11 --passes 100 --evaluate-passes 2
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

from agent.config import Config, checkpoint_updates, fold_config, quarter_episode, write_json
from agent.evaluate import evaluate
from agent.ledger import LONG, SHORT, Environment, TrainingEnvironment
from agent.network import ActorCritic
from agent.ppo import Trainer
from agent.synthetic import (BEAR, BULL, decision_minutes, edge_economics, latent_regime_market, oracle_actions,
                             planted_market, scaler_for, span_decisions, trend_rule_actions)

MARKET_SEED = 7
YEAR = 525_600
WEEK = 10_080
CONDITIONING = ('long_when_up', 'long_when_down', 'short_when_up', 'short_when_down')
KEPT = ('total_log_return_btc_mean', 'total_log_return_btc_min', 'total_log_return_btc_max', 'orders',
        'median_holding_spell_hours', 'long_share', 'short_share', 'liquidations')


def build(args):
    """Market, regime, and the training, validation and test windows in minutes."""
    if args.mode == 'observable':
        market, regime = planted_market(minutes=600_000, seed=MARKET_SEED, oracle_sharpe=args.oracle_sharpe,
                                        start='2020-01-01')
        return market, regime, (20_000, 450_000), (460_000, 525_000), (525_000, 590_000)
    market, regime = latent_regime_market(years=4.0, seed=MARKET_SEED, trend_sharpe=args.trend_sharpe,
                                          regime_days=args.regime_days)
    validation_start = 3 * YEAR + WEEK
    test_start = validation_start + 26 * WEEK
    return market, regime, (0, 3 * YEAR), (validation_start, test_start), (test_start, len(market))


def reference_returns(market, regime, bounds, config):
    """Oracle, the causal trend rule and constant positions on one window, under identical costs."""
    steps = len(decision_minutes(bounds, config))
    plans = {'oracle': oracle_actions(market, regime, bounds, config),
             'trend_rule': trend_rule_actions(market, bounds, config),
             'constant_long': np.full(steps, LONG), 'constant_short': np.full(steps, SHORT)}
    out = {'flat': 0.0}
    for name, actions in plans.items():
        env = Environment(market, bounds, config, np.random.default_rng(0), training=False)
        out[name] = sum(float(env.step(np.array([a]))[2]['log_return'][0]) for a in actions)
    return out


def regime_conditioning(frame, market, regime, latent):
    """Share of time long and short while the planted regime is up and down."""
    # Unit-safe: evaluation frames carry second-resolution timestamps.
    minutes = ((frame.time - market.timestamp(0)) // pd.Timedelta(minutes=1)).to_numpy()
    state = regime[minutes]
    up, down = (state == BULL, state == BEAR) if latent else (state > 0, state < 0)
    side = np.sign(frame.contracts.to_numpy())

    def share(mask, sign):
        return float(np.mean(side[mask] == sign)) if mask.any() else 0.0
    return {'long_when_up': share(up, 1), 'long_when_down': share(down, 1),
            'short_when_up': share(up, -1), 'short_when_down': share(down, -1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('observable', 'latent'), default='latent')
    parser.add_argument('--reward', choices=('sharpe', 'sortino'), default=Config.reward)
    parser.add_argument('--decision-bars', type=int, default=Config.decision_bars)
    parser.add_argument('--passes', type=int, default=Config.training_passes)
    parser.add_argument('--evaluate-passes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--oracle-sharpe', type=float, default=30.0)
    parser.add_argument('--trend-sharpe', type=float, default=0.5)
    parser.add_argument('--regime-days', type=float, default=10.0)
    parser.add_argument('--output', type=Path, default=Path('results/verification'))
    args = parser.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    latent = args.mode == 'latent'
    base = replace(Config(), reward=args.reward, decision_bars=args.decision_bars,
                   episode_steps=quarter_episode(args.decision_bars), training_passes=args.passes,
                   validation_passes=args.evaluate_passes)
    market, regime, train, validation, test = build(args)
    train_bounds = market.bounds(market.timestamp(train[0]), market.timestamp(train[1]))
    scaler = scaler_for(market, end=train_bounds[1])
    decisions = span_decisions(train_bounds, base)
    config = fold_config(base, scaler.bar_return_sd, decisions)
    schedule = checkpoint_updates(config)
    windows = {'validation': (market.timestamp(validation[0]), market.timestamp(validation[1])),
               'test': (market.timestamp(test[0]), market.timestamp(test[1]))}
    references = {name: reference_returns(market, regime, market.bounds(*window), config)
                  for name, window in windows.items()}
    criterion = max(references['test']['constant_long'], references['test']['constant_short'], references['test']['flat'])

    run = args.output/f'{args.mode}_{args.reward}_bars{args.decision_bars}_seed{args.seed}'
    run.mkdir(parents=True, exist_ok=True)
    header = {'mode': args.mode, 'reward': args.reward, 'decision_bars': args.decision_bars, 'seed': args.seed,
              'market_seed': MARKET_SEED, 'passes': args.passes, 'evaluate_passes': args.evaluate_passes,
              'training_decisions': int(decisions), 'updates': config.updates, 'checkpoints': list(schedule),
              'bar_return_sd': scaler.bar_return_sd, 'config': config.as_dict(),
              'edge': edge_economics(market, config), 'references': references,
              'pass_if_selected_test_return_above': criterion,
              'training_window': [str(market.timestamp(b)) for b in train_bounds],
              'windows': {name: [str(a), str(b)] for name, (a, b) in windows.items()},
              'credit_window_hours': config.credit_window_hours, 'reward_scale': config.reward_scale}
    write_json(run/'setup.json', header)
    print(json.dumps({k: v for k, v in header.items() if k not in ('config', 'checkpoints')}, indent=2), flush=True)

    env = TrainingEnvironment(market, train_bounds, config, np.random.default_rng(args.seed))
    model = ActorCritic(config)
    trainer = Trainer(model, env, scaler, config, args.seed)
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}", flush=True)

    started, curve = time.monotonic(), []
    with (run/'updates.jsonl').open('w', buffering=1) as log, (run/'checkpoints.jsonl').open('w', buffering=1) as book:
        while trainer.updates < config.updates:
            tick = time.monotonic()
            summary = trainer.run_update()
            summary['update_seconds'] = time.monotonic()-tick
            summary['passes'] = trainer.updates * config.samples_per_update / decisions
            log.write(json.dumps(summary, allow_nan=False)+'\n')
            if trainer.updates % 50 == 0 or trainer.updates == 1:
                share, hold = summary['long_entry_share'], summary['policy_implied_holding_hours']
                print(f"upd {summary['update']:>5} pass {summary['passes']:5.1f} {summary['update_seconds']:4.1f}s "
                      f"| H {summary['policy_entropy'] or 0:.3f} "
                      f"| hold {'-' if hold is None else f'{hold:7.1f}'}h "
                      f"| MI {summary['state_action_mutual_information']:.4f} "
                      f"| side S/F/L {'/'.join(f'{x:.2f}' for x in summary['account_side_share'])} "
                      f"| long entries {'-' if share is None else f'{share:.2f}'} "
                      f"| EV {summary['explained_variance_post'] or 0:.2f} "
                      f"| KL {summary['approx_kl']:.4f} clip {summary['clip_fraction']:.3f} "
                      f"| liq {summary['liquidations_in_rollout']}", flush=True)
            if trainer.updates in schedule:
                entry = {'update': trainer.updates, 'passes': summary['passes'],
                         'train_net_log_return_per_decision': summary['mean_net_log_return']}
                for name, (start, end) in windows.items():
                    result, frame = evaluate([model], market, scaler, config, start, end, seeds=2)
                    entry[name] = {**{k: result[k] for k in KEPT}, **regime_conditioning(frame, market, regime, latent)}
                curve.append(entry)
                book.write(json.dumps(entry, allow_nan=False)+'\n')
                v, t = entry['validation'], entry['test']
                print(f"  CHECKPOINT @{trainer.updates} pass {entry['passes']:.1f}: validation {v['total_log_return_btc_mean']:+.3f} "
                      f"| test {t['total_log_return_btc_mean']:+.3f} (orders {t['orders']:.0f}, hold {t['median_holding_spell_hours']:.1f}h, "
                      f"long up/down {t['long_when_up']:.2f}/{t['long_when_down']:.2f}, short up/down "
                      f"{t['short_when_up']:.2f}/{t['short_when_down']:.2f})", flush=True)
    selected = max(curve, key=lambda e: (e['validation']['total_log_return_btc_mean'], -e['update']))
    test_result = selected['test']
    conditioned = (test_result['long_when_up'] > test_result['long_when_down']
                   and test_result['short_when_down'] > test_result['short_when_up'])
    rule = references['test']['trend_rule']
    verdict = {'mode': args.mode, 'reward': args.reward, 'decision_bars': args.decision_bars, 'seed': args.seed,
               'selected_update': selected['update'], 'selected_passes': selected['passes'],
               'selected_validation_return': selected['validation']['total_log_return_btc_mean'],
               'selected_test_return': test_result['total_log_return_btc_mean'], 'criterion': criterion,
               'beats_best_constant': bool(test_result['total_log_return_btc_mean'] > criterion),
               'regime_conditioned': bool(conditioned),
               'passed': bool(test_result['total_log_return_btc_mean'] > criterion and conditioned),
               'selected_test_conditioning': {k: test_result[k] for k in CONDITIONING},
               'test_oracle': references['test']['oracle'], 'test_trend_rule': rule,
               'share_of_trend_rule': test_result['total_log_return_btc_mean'] / rule if rule > 0 else None,
               'final_checkpoint_test_return': curve[-1]['test']['total_log_return_btc_mean'],
               'peak_validation_passes': selected['passes'],
               'hours': (time.monotonic()-started)/3600}
    write_json(run/'verdict.json', verdict)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == '__main__':
    main()
