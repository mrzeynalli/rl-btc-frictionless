"""Proximal Policy Optimisation (Schulman et al. 2017) as CleanRL's ppo_atari_lstm implements it.

One update:
 1. collect: every walker takes `rollout` steps with the current policy (actions sampled from pi);
 2. advantages by GAE (Schulman et al. 2016):
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)          one-step surprise
        A_t     = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}           smoothed over the future
        R_t     = A_t + V(s_t)                                                     the critic's target
 3. `epochs` passes over the rollout in minibatches of whole walker sequences:
        ratio   = pi_new(a|s) / pi_old(a|s)
        L_actor = -mean(min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)) - c_ent * entropy
        L_critic = 0.5 * mean((V(s) - R)^2)
    Each sequence restarts from the LSTM memory stored at the start of the rollout.
"""
import numpy as np
import torch
from torch import nn


class PPO:
    def __init__(self, model, env, config):
        self.model, self.env, self.c = model, env, config
        self.actor_opt = torch.optim.Adam(model.actor.parameters(), lr=config.lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(model.critic.parameters(), lr=config.critic_lr, eps=1e-5)
        n = env.n
        self.actor_state = model.actor.initial_state(n)
        self.critic_state = model.critic.initial_state(n)
        self.obs = torch.from_numpy(env.observation())
        self.done = torch.zeros(n)                     # 1.0 where the current observation starts a new path

    @torch.no_grad()
    def collect(self):
        c, n, T = self.c, self.env.n, self.c.rollout
        # The memory at the rollout's start: every training sequence below is replayed from it.
        start = {'actor': self.actor_state, 'critic': self.critic_state}
        buf = {'obs': torch.zeros(T, n, self.obs.shape[1]), 'actions': torch.zeros(T, n, dtype=torch.long),
               'logp': torch.zeros(T, n), 'values': torch.zeros(T, n), 'rewards': torch.zeros(T, n),
               'dones': torch.zeros(T, n), 'returns_hourly': np.zeros((T, n))}
        for t in range(T):
            buf['obs'][t], buf['dones'][t] = self.obs, self.done
            logits, self.actor_state = self.model.actor(self.obs[None], self.actor_state, self.done[None])
            value, self.critic_state = self.model.critic(self.obs[None], self.critic_state, self.done[None])
            dist = torch.distributions.Categorical(logits=logits[0])
            action = dist.sample()                     # training samples from pi, so it keeps exploring
            reward, R, done = self.env.step(action.numpy())
            buf['actions'][t], buf['logp'][t] = action, dist.log_prob(action)
            buf['values'][t], buf['rewards'][t] = value[0, :, 0], torch.from_numpy(reward)
            buf['returns_hourly'][t] = R
            self.obs, self.done = torch.from_numpy(self.env.observation()), torch.from_numpy(done.astype(np.float32))
        # V(s_T) of the state after the rollout, to bootstrap the last advantages.
        next_value, _ = self.model.critic(self.obs[None], self.critic_state, self.done[None])
        return buf, start, next_value[0, :, 0], self.done.clone()

    def advantages(self, buf, next_value, next_done):
        c, T = self.c, self.c.rollout
        adv = torch.zeros_like(buf['rewards'])
        running = torch.zeros(self.env.n)
        for t in reversed(range(T)):
            not_done = 1.0 - (next_done if t == T - 1 else buf['dones'][t + 1])      # 0 across a wrap-around
            next_v = next_value if t == T - 1 else buf['values'][t + 1]
            delta = buf['rewards'][t] + c.gamma * next_v * not_done - buf['values'][t]
            running = delta + c.gamma * c.gae_lambda * not_done * running
            adv[t] = running
        return adv, adv + buf['values']

    def update(self):
        c = self.c
        buf, start, next_value, next_done = self.collect()
        adv, targets = self.advantages(buf, next_value, next_done)
        stats = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': [], 'clip_fraction': []}
        walkers_per_batch = self.env.n // c.minibatches
        for _ in range(c.epochs):
            for batch in torch.randperm(self.env.n).split(walkers_per_batch):
                obs, dones = buf['obs'][:, batch], buf['dones'][:, batch]
                # Actor: replay the sequences from the stored start memory and apply the clipped objective.
                state = (start['actor'][0][:, batch], start['actor'][1][:, batch])
                logits, _ = self.model.actor(obs, state, dones)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(buf['actions'][:, batch])
                log_ratio = logp - buf['logp'][:, batch]
                ratio = log_ratio.exp()                                          # pi_new / pi_old
                a = adv[:, batch]
                a = (a - a.mean()) / (a.std() + 1e-8)                            # normalise advantages per minibatch
                surrogate = torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a)
                entropy = dist.entropy().mean()
                actor_loss = -surrogate.mean() - c.entropy_coef * entropy
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.model.actor.parameters(), c.max_grad_norm)
                self.actor_opt.step()
                # Critic: regress V(s) on the GAE targets R_t.
                state = (start['critic'][0][:, batch], start['critic'][1][:, batch])
                values, _ = self.model.critic(obs, state, dones)
                value_loss = 0.5 * ((values[..., 0] - targets[:, batch]) ** 2).mean()
                self.critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), c.max_grad_norm)
                self.critic_opt.step()
                with torch.no_grad():
                    stats['policy_loss'].append(float(-surrogate.mean()))
                    stats['value_loss'].append(float(value_loss))
                    stats['entropy'].append(float(entropy))
                    stats['approx_kl'].append(float(((ratio - 1) - log_ratio).mean()))   # KL(old || new) estimate
                    stats['clip_fraction'].append(float(((ratio - 1).abs() > c.clip).float().mean()))
        summary = {k: float(np.mean(v)) for k, v in stats.items()}
        variance = float(targets.var())
        summary.update({
            'mean_reward': float(buf['rewards'].mean()),
            'mean_hourly_return': float(buf['returns_hourly'].mean()),
            'action_share': (torch.bincount(buf['actions'].flatten(), minlength=3).float() / buf['actions'].numel()).tolist(),
            'explained_variance': 1 - float((targets - buf['values']).var()) / variance if variance > 0 else 0.0,
        })
        return summary
