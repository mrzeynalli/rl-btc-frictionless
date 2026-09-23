"""Critic-free policy gradient: One-Batch REINFORCE with a batch-mean baseline.

The critic-free alternative to PPO, following the One-Batch REINFORCE of FlashREINFORCE (NVIDIA, 2026), which in
turn is REINFORCE (Williams 1992) with the reward centred on the batch mean. One update:
 1. collect: every walker takes `rollout` steps with the current policy (actions sampled from pi);
 2. returns, computed backwards and cut at a path restart (no bootstrap, as there is no critic):
        G_t = D_{t+1} + gamma * (1 - done_{t+1}) * G_{t+1}
 3. advantages centred on the batch mean and scaled by the batch standard deviation:
        A_t = (G_t - mean(G)) / std(G)
 4. one gradient step on the whole batch, which is then discarded:
        L = -mean(log pi(a_t | h_t) * A_t) - c_ent * entropy
    The sequences are replayed from the LSTM memory stored at the start of the rollout.
"""
import numpy as np
import torch
from torch import nn


class Reinforce:
    def __init__(self, model, env, config):
        self.model, self.env, self.c = model, env, config
        self.actor_opt = torch.optim.Adam(model.actor.parameters(), lr=config.lr, eps=1e-5)
        self.actor_state = model.actor.initial_state(env.n)
        self.obs = torch.from_numpy(env.observation())
        self.done = torch.zeros(env.n)                 # 1.0 where the current observation starts a new path

    @torch.no_grad()
    def collect(self):
        n, T = self.env.n, self.c.rollout
        start = self.actor_state
        buf = {'obs': torch.zeros(T, n, self.obs.shape[1]), 'actions': torch.zeros(T, n, dtype=torch.long),
               'rewards': torch.zeros(T, n), 'dones': torch.zeros(T, n), 'returns_hourly': np.zeros((T, n))}
        for t in range(T):
            buf['obs'][t], buf['dones'][t] = self.obs, self.done
            logits, self.actor_state = self.model.actor(self.obs[None], self.actor_state, self.done[None])
            action = torch.distributions.Categorical(logits=logits[0]).sample()
            reward, R, done = self.env.step(action.numpy())
            buf['actions'][t], buf['rewards'][t], buf['returns_hourly'][t] = action, torch.from_numpy(reward), R
            self.obs, self.done = torch.from_numpy(self.env.observation()), torch.from_numpy(done.astype(np.float32))
        return buf, start, self.done.clone()

    def returns(self, buf, next_done):
        T = self.c.rollout
        G = torch.zeros_like(buf['rewards'])
        running = torch.zeros(self.env.n)
        for t in reversed(range(T)):
            not_done = 1.0 - (next_done if t == T - 1 else buf['dones'][t + 1])      # 0 across a wrap-around
            running = buf['rewards'][t] + self.c.gamma * not_done * running
            G[t] = running
        return G

    def update(self):
        c = self.c
        buf, start, next_done = self.collect()
        G = self.returns(buf, next_done)
        advantage = (G - G.mean()) / (G.std() + 1e-8)                       # batch-mean baseline, unit scale
        logits, _ = self.model.actor(buf['obs'], start, buf['dones'])
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(buf['actions'])
        entropy = dist.entropy().mean()
        loss = -(logp * advantage).mean() - c.entropy_coef * entropy
        self.actor_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.actor.parameters(), c.max_grad_norm)
        self.actor_opt.step()
        return {'policy_loss': float(loss.detach()), 'entropy': float(entropy.detach()), 'mean_return_to_go': float(G.mean()),
                'mean_reward': float(buf['rewards'].mean()),
                'mean_hourly_return': float(buf['returns_hourly'].mean()),
                'action_share': (torch.bincount(buf['actions'].flatten(), minlength=3).float()
                                 / buf['actions'].numel()).tolist()}
