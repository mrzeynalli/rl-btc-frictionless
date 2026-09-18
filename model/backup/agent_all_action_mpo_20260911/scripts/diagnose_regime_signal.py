"""Is the planted regime reaching the learner, and does the learning signal favour it?

Measurements on the symmetric planted market (seed 7), per 10 updates:
1. signal presence: the scaled regime column agrees with regime[t-1] at decision minutes;
2. signal transmission, two ways:
   - sampled: mean GAE advantage of sampled actions that agree with the regime minus those
     that oppose it, with a naive t-statistic (ignores autocorrelation, so optimistic);
   - all-action: Q(visited account, LONG) - Q(visited account, SHORT) on the same bar, when
     the regime is up minus when it is down (positive = the action values carry the edge);
3. policy conditioning: P(long | regime up) - P(long | regime down) and the mirror for short.
--oracle-sharpe raises the planted edge; with zero costs and a strong edge a working pipeline
must learn the conditioning quickly, so failure there would indicate a bug rather than slow learning.

    PYTHONPATH=src python scripts/diagnose_regime_signal.py --cost real --oracle-sharpe 3 --updates 100
"""
import argparse
import json
from dataclasses import replace

import numpy as np
import torch

from agent.config import Config
from agent.data import FEATURE_NAMES
from agent.ledger import LONG, SHORT, TrainingEnvironment
from agent.network import ActorCritic
from agent.ppo import Trainer, advantages
from agent.synthetic import REGIME_FEATURE, planted_market, scaler_for

parser = argparse.ArgumentParser()
parser.add_argument('--cost', choices=('real', 'zero'), required=True)
parser.add_argument('--updates', type=int, default=150)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--oracle-sharpe', type=float, default=3.0)
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(args.seed)
torch.use_deterministic_algorithms(True)

config = replace(Config(), updates=args.updates)
if args.cost == 'zero':
    config = replace(config, fee=0.0, slippage=0.0)
market, regime = planted_market(minutes=600_000, seed=7, oracle_sharpe=args.oracle_sharpe)
scaler = scaler_for(market)
bounds = market.bounds(market.timestamp(20_000), market.timestamp(450_000))
column = FEATURE_NAMES.index(REGIME_FEATURE)

probe = np.arange(bounds[0] + 5, bounds[1] - 5, 5)[::50]
observed = scaler.observations(market, probe, config.observation_clip)[:, column]
print(json.dumps({'cost': args.cost, 'oracle_sharpe': args.oracle_sharpe,
                  'regime_column_agrees_with_previous_minute_regime': float(np.mean(np.sign(observed) == regime[probe - 1])),
                  'drift_per_bar_in_bar_sd': market.audit['drift_per_minute'] * 5 / (market.audit['minute_sd'] * 5 ** 0.5)}),
      flush=True)

env = TrainingEnvironment(market, bounds, config, np.random.default_rng(args.seed))
trainer = Trainer(ActorCritic(config), env, scaler, config, args.seed)
window = []
while trainer.updates < args.updates:
    rows, bootstrap = trainer.collect()
    raw, _ = advantages(rows['rewards'], rows['values'], bootstrap, rows['terminals'], rows['valid'],
                        config.gamma, config.gae_lambda)
    up = regime[rows['minutes'] - 1] > 0
    valid, actions, probabilities = rows['valid'], rows['actions'], rows['probabilities']
    agree = valid & (((actions == LONG) & up) | ((actions == SHORT) & ~up))
    oppose = valid & (((actions == SHORT) & up) | ((actions == LONG) & ~up))
    difference = float(raw[agree].mean() - raw[oppose].mean()) if agree.any() and oppose.any() else 0.
    scale = float(raw[valid].std()) * np.sqrt(1 / max(agree.sum(), 1) + 1 / max(oppose.sum(), 1))
    summary = trainer.update(rows, bootstrap)
    critic = trainer.evaluate_critic(rows, trainer.rollout_market(rows))
    visited = critic['action_values'].gather(
        2, torch.from_numpy(rows['actual_index'])[..., None, None].expand(-1, -1, 1, 3)).squeeze(2).numpy()
    gap = visited[..., LONG] - visited[..., SHORT]
    window.append({
        'difference': difference, 't': difference / scale if scale else 0., 'sd': float(raw[valid].std()),
        'q_gap': float(gap[valid & up].mean() - gap[valid & ~up].mean()) if (valid & up).any() and (valid & ~up).any() else 0.,
        'p_long': float(probabilities[..., LONG][valid & up].mean() - probabilities[..., LONG][valid & ~up].mean()),
        'p_short': float(probabilities[..., SHORT][valid & ~up].mean() - probabilities[..., SHORT][valid & up].mean()),
        'held_long': float(np.mean(rows['sides'][valid & up] == LONG) - np.mean(rows['sides'][valid & ~up] == LONG)),
        'entropy': summary['policy_entropy'], 'hold_h': summary['policy_implied_holding_hours'],
        'side_share': summary['account_side_share'], 'net_bp': summary['mean_net_log_return'] * 1e4})
    if trainer.updates % 10 == 0:
        part = window[-10:]
        mean = lambda k: float(np.mean([w[k] for w in part]))
        print(json.dumps({'update': trainer.updates, 'cost': args.cost, 'oracle_sharpe': args.oracle_sharpe,
                          'adv_agree_minus_oppose': round(mean('difference'), 4), 'naive_t': round(mean('t'), 2),
                          'adv_sd': round(mean('sd'), 3),
                          'q_long_minus_short_up_minus_down': round(mean('q_gap'), 4),
                          'p_long_up_minus_down': round(mean('p_long'), 3),
                          'p_short_down_minus_up': round(mean('p_short'), 3),
                          'held_long_up_minus_down': round(mean('held_long'), 3),
                          'entropy': round(mean('entropy'), 3), 'hold_h': round(mean('hold_h'), 2),
                          'side_share': [round(x, 2) for x in np.mean([w['side_share'] for w in part], 0)],
                          'net_bp_per_step': round(mean('net_bp'), 2)}), flush=True)
