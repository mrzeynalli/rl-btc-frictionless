"""Training-only, action-enumerated account branches on an unchanged market."""
from __future__ import annotations

import numpy as np
import torch

from perpetual.env import Environment, Ledger
from .network import map_hidden


def cat_hidden(items):
    if items[0] is None:
        return None
    if isinstance(items[0], tuple):
        return tuple(cat_hidden([x[i] for x in items]) for i in range(len(items[0])))
    return torch.cat(items, dim=1)


def branch_environment(environment, anchors):
    """Independent ledgers; no mutation of the policy's actual environment."""
    env = object.__new__(Environment)
    env.market, env.bounds, env.config = environment.market, environment.bounds, environment.config
    env.size = sum(len(a['state']['cursor']) for a in anchors) * 3
    env.training, env.resets = False, 0
    env.rng = np.random.default_rng(0)
    env.cursor = np.concatenate([a['state']['cursor'] for a in anchors]).repeat(3)
    env.steps = np.zeros(env.size, np.int64)
    env.ledger = Ledger(env.size, env.config)
    env.ledger.load_state_dict({key: np.concatenate([a['state']['ledger'][key] for a in anchors]).repeat(3)
                               for key in anchors[0]['state']['ledger']})
    return env


@torch.no_grad()
def simulate(model, environment, scaler, anchors, masks, generator):
    c = environment.config
    if not anchors:
        return None
    env = branch_environment(environment, anchors)
    unique_minutes = env.cursor[::3, None] + np.arange(c.counterfactual_horizon+1)[None, :] * 5
    # Environment bounds are training bounds, stricter than the global test lock.
    if unique_minutes.max() >= env.bounds[1]:
        raise ValueError('counterfactual continuation crosses training boundary')
    fast, slow_np = scaler.observations(env.market, unique_minutes, c.fast_minutes)
    fast, slow = torch.from_numpy(fast), torch.from_numpy(slow_np)
    initial_hidden = cat_hidden([a['hidden'] for a in anchors])
    cached, position_hidden = [], []
    for b, branch in enumerate((model.actor, model.critic)):
        local = branch.local(fast)
        market, _ = branch.encode_market(local, slow, initial_hidden[b][0])
        cached.append((local.repeat_interleave(3, 0), market.repeat_interleave(3, 0)))
        position_hidden.append(map_hidden(initial_hidden[b][1], lambda x: x.repeat_interleave(3, 1)))
    slow = slow.repeat_interleave(3, 0)
    branch_masks = tuple(torch.cat([m for _ in anchors], 0).repeat_interleave(3, 0) for m in masks)
    total = np.zeros(env.size, np.float64)
    first_probabilities = None
    held_count = np.zeros(3, np.int64)
    for step in range(c.counterfactual_horizon+1):
        _, account = env.observation()
        account = torch.from_numpy(account[:, None])
        outputs = []
        for b, branch in enumerate((model.actor, model.critic)):
            local, market = cached[b]
            output, position_hidden[b] = branch.decision(local[:, step:step+1], slow[:, step:step+1], account,
                                                         market[:, step:step+1], position_hidden[b], branch_masks[b])
            outputs.append(output[:, 0])
        if step == c.counterfactual_horizon:
            total += c.gamma**step * outputs[1][:, 0].numpy() * ~env.ledger.terminated
            break
        probabilities = outputs[0].softmax(-1)
        if step == 0:
            first_probabilities = probabilities[::3].numpy()
            action = np.tile(np.arange(3), env.size//3)
        else:
            # Common random numbers reduce variance between alternative actions.
            u = torch.rand(env.size//3, 1, generator=generator).repeat_interleave(3, 0)
            action = (u > probabilities.cumsum(-1)).sum(-1).clamp(max=2).numpy()
        reward, _, _ = env.step(action)
        total += c.gamma**step * reward
        held_count += np.bincount(np.sign(env.ledger.contracts).astype(int)+1, minlength=3)
    target = total.reshape(-1, 3).astype(np.float32)
    if not np.isfinite(target).all():
        raise RuntimeError('non-finite counterfactual return')
    return {'targets': target, 'old_probabilities': first_probabilities,
            'steps': [a['step'] for a in anchors], 'path_side_counts': held_count.tolist(),
            'branch_decisions': env.size*c.counterfactual_horizon,
            'liquidations': int(env.ledger.liquidations.sum())}
