"""Recurrent PPO for a single trading account.

PPO's clipped surrogate (Schulman et al. 2017) on generalised advantage estimates
(Schulman et al., ICLR 2016), configured as Andrychowicz et al. (ICLR 2021)
recommend from their study of 250,000 agents and as CleanRL's ppo_atari_lstm
implements it: separate policy and value networks, advantages normalised per
minibatch and recomputed before every pass, an unclipped squared value loss, Adam
with a linearly decaying step, and global gradient clipping.

Recurrent state. Each training sequence starts from a state rebuilt under the
current parameters over the stored burn-in prefix (Kapturowski et al., R2D2, ICLR
2019), so no update reads a stale state. Before the first gradient step the whole
rollout is replayed and must reproduce the collected probabilities, values and
bootstrap; a misaligned observation, account or recurrent state stops training
instead of silently biasing it.

A known failure mode. With a softmax policy the clipped objective can settle on a
suboptimal discrete action that keeps being sampled while a better one is not
(Hsu, Mendler-Dunner & Hardt 2020). Here that looks like a policy locked on one
side whatever the market does. Every update reports the mutual information
between visited state and action - zero for a policy that ignores the market -
and the realised direction of its entries.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .ledger import ACTION_NAMES, FLAT, LONG, PORTFOLIO_WIDTH, SHORT
from .network import market_tensor, policy_entropy

ACTIONS = len(ACTION_NAMES)
# Float32 recurrent kernels stepped one decision at a time and run over a whole
# sequence agree to about 1e-6; a misalignment moves outputs by orders more.
REPLAY_TOLERANCE = 5e-5


def explained_variance(target, prediction):
    variance = float(np.var(target))
    return None if variance < 1e-12 else 1 - float(np.var(target - prediction)) / variance


def advantages(rewards, values, bootstrap, terminals, valid, gamma, lam):
    """Economic termination masks bootstrap; artificial time limits do not (Pardo et al. 2018)."""
    result = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[0], dtype=np.float32)
    after = bootstrap.copy()
    for step in reversed(range(rewards.shape[1])):
        continuation = ~terminals[:, step]
        delta = rewards[:, step] + gamma * after * continuation - values[:, step]
        running = (delta + gamma * lam * running * continuation) * valid[:, step]
        result[:, step] = running
        after = values[:, step]
    return result, result + values


def normalised(advantage, mask):
    """Per-minibatch advantage normalisation over the steps that count."""
    selected = advantage[mask]
    scale = selected.std() if selected.numel() > 1 else torch.ones(())
    return (advantage - selected.mean()) / (scale + 1e-8)


def clipped_surrogate(log_ratio, advantage, clip):
    """PPO's per-step objective: the smaller of the unclipped and the clipped ratio times the advantage."""
    ratio = log_ratio.exp()
    return torch.min(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)


def state_action_mutual_information(probabilities, mask):
    """Mutual information between visited state and action, in nats.

    Zero exactly when the policy is the same distribution in every state, which
    is what a side-locked policy looks like. Defined on absolute actions, so it
    names no side.
    """
    if not bool(mask.any()):
        return 0.
    p = probabilities[mask]
    average = p.mean(0, keepdim=True)
    divergence = (p * (p.clamp_min(1e-12).log() - average.clamp_min(1e-12).log())).sum(-1).mean()
    return float(divergence.clamp_min(0))


def masked_mean(values, mask):
    return float(values[mask].mean()) if bool(mask.any()) else None


class Trainer:
    def __init__(self, model, environment, scaler, config, seed):
        self.model, self.env, self.scaler, self.config = model, environment, scaler, config
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=config.critic_lr)
        self.generator = torch.Generator().manual_seed(seed + 1000)
        self.prefix_minutes = self.env.initial_prefix()
        self.updates = 0

    def learning_rates(self):
        """Linear decay to zero over the run (Andrychowicz et al.; CleanRL)."""
        c = self.config
        fraction = max(1.0 - self.updates / max(c.updates, 1), 0.0) if c.anneal_lr else 1.0
        return c.actor_lr * fraction, c.critic_lr * fraction

    def market(self, minutes):
        return market_tensor(self.scaler, self.env.market, minutes, self.config)

    def rollout_market(self, rows):
        """Prefix, rollout and final observation: the whole sequence the memory needs."""
        return self.market(np.concatenate([rows['prefix_minutes'], rows['minutes'], rows['final_minute'][:, None]], 1))

    @torch.no_grad()
    def collect(self):
        c, env = self.config, self.env
        E, T = env.size, c.rollout
        reset = (env.steps >= c.episode_steps) | env.ledger.terminated
        if reset.any():
            env.reset(reset)
            self.prefix_minutes[reset] = env.initial_prefix(reset)
        hidden = self.model.burn(self.market(self.prefix_minutes))
        rows = {k: np.zeros((E, T), d) for k, d in (
            ('minutes', np.int64), ('actions', np.int64), ('sides', np.int64), ('logp', np.float32),
            ('values', np.float32), ('rewards', np.float32), ('terminals', bool), ('valid', bool),
            ('turnover', np.float32), ('net_log_return', np.float32), ('position_age_minutes', np.float32))}
        rows['account'] = np.zeros((E, T, PORTFOLIO_WIDTH), np.float32)
        rows['probabilities'] = np.zeros((E, T, ACTIONS), np.float32)
        for step in range(T):
            minute, account = env.observation()
            logp, value, hidden = self.model(self.market(minute[:, None]), torch.from_numpy(account[:, None]), hidden)
            logp, value = logp[:, 0], value[:, 0]
            action = torch.multinomial(logp.exp(), 1, generator=self.generator).squeeze(1)
            rows['minutes'][:, step], rows['account'][:, step] = minute, account
            rows['sides'][:, step] = account[:, :3].argmax(1)
            rows['position_age_minutes'][:, step] = env.ledger.age
            rows['actions'][:, step] = action.numpy()
            rows['probabilities'][:, step] = logp.exp().numpy()
            rows['logp'][:, step] = logp.gather(-1, action[:, None]).squeeze(-1).numpy()
            rows['values'][:, step] = value.numpy()
            reward, terminal, info = env.step(action.numpy())
            rows['rewards'][:, step], rows['terminals'][:, step] = reward, terminal
            rows['valid'][:, step], rows['turnover'][:, step] = info['active'], info['turnover']
            rows['net_log_return'][:, step] = info['log_return']
        minute, account = env.observation()
        _, value, _ = self.model(self.market(minute[:, None]), torch.from_numpy(account[:, None]), hidden)
        bootstrap = np.where(env.ledger.terminated, 0., value[:, 0].numpy()).astype(np.float32)
        rows['final_minute'], rows['final_account'] = minute, account
        rows['terminated_at_end'] = env.ledger.terminated.copy()
        rows['prefix_minutes'] = self.prefix_minutes.copy()
        self.prefix_minutes = np.concatenate([self.prefix_minutes, rows['minutes']], 1)[:, -c.burn_in:].copy()
        return rows, bootstrap

    def sliced(self, branch, market, account, index, begin, end):
        """One minibatch of sequences from a recurrent state rebuilt under current parameters."""
        c = self.config
        with torch.no_grad():
            _, hidden = branch.encode(market[index, :c.burn_in + begin])
        memory, _ = branch.encode(market[index, c.burn_in + begin:c.burn_in + end], hidden)
        return branch.read(memory, account[index, begin:end])

    @torch.no_grad()
    def replay(self, rows, market):
        """Log-probabilities, values and bootstrap under the current parameters, over whole sequences."""
        c, T = self.config, rows['rewards'].shape[1]
        account = torch.from_numpy(rows['account'])
        actor_memory, _ = self.model.actor.encode(market)
        logp = self.model.actor.read(actor_memory[:, c.burn_in:c.burn_in + T], account)
        critic_memory, _ = self.model.critic.encode(market)
        values = self.model.critic.read(critic_memory[:, c.burn_in:c.burn_in + T], account)[..., 0]
        final = self.model.critic.read(critic_memory[:, c.burn_in + T], torch.from_numpy(rows['final_account']))[..., 0]
        bootstrap = torch.where(torch.from_numpy(rows['terminated_at_end']), 0., final)
        return logp, values.numpy(), bootstrap.numpy()

    def update(self, rows, bootstrap):
        c = self.config
        E, T = rows['rewards'].shape
        gamma, lam = c.decision_gamma, c.decision_lambda
        market = self.rollout_market(rows)
        account = torch.from_numpy(rows['account'])
        actions = torch.from_numpy(rows['actions'])
        old_logp = torch.from_numpy(rows['logp'])
        old_probabilities = torch.from_numpy(rows['probabilities'])
        valid = torch.from_numpy(rows['valid'])
        selected = rows['valid']
        actor_lr, critic_lr = self.learning_rates()
        for group in self.actor_optimizer.param_groups:
            group['lr'] = actor_lr
        for group in self.critic_optimizer.param_groups:
            group['lr'] = critic_lr

        replayed_logp, replayed_values, replayed_bootstrap = self.replay(rows, market)
        replay_probability = float((replayed_logp.exp() - old_probabilities).abs()[valid].max()) if selected.any() else 0.
        replay_value = float(np.abs(replayed_values - rows['values'])[selected].max()) if selected.any() else 0.
        replay_bootstrap = float(np.abs(replayed_bootstrap - bootstrap).max())
        if max(replay_probability, replay_value, replay_bootstrap) > REPLAY_TOLERANCE:
            raise RuntimeError(f'PPO replay mismatch: {replay_probability}, {replay_value}, {replay_bootstrap}')
        _, collected_returns = advantages(rows['rewards'], rows['values'], bootstrap, rows['terminals'],
                                          rows['valid'], gamma, lam)

        stats = {k: [] for k in ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction')}
        values, final = rows['values'], bootstrap
        for epoch in range(c.epochs):
            if epoch and c.recompute_advantages_per_pass:
                # Andrychowicz et al.: recomputing advantages each pass performs best.
                _, values, final = self.replay(rows, market)
            advantage, returns = advantages(rows['rewards'], values, final, rows['terminals'], rows['valid'],
                                            gamma, lam)
            advantage, returns = torch.from_numpy(advantage), torch.from_numpy(returns)
            for chunk in torch.randperm(T // c.sequence, generator=self.generator).tolist():
                begin, end = chunk * c.sequence, (chunk + 1) * c.sequence
                for index in torch.randperm(E, generator=self.generator).split(c.minibatch_sequences):
                    mask = valid[index, begin:end]
                    if not mask.any():
                        continue
                    logp = self.sliced(self.model.actor, market, account, index, begin, end)
                    chosen = logp.gather(-1, actions[index, begin:end, None]).squeeze(-1)
                    log_ratio = chosen - old_logp[index, begin:end]
                    ratio = log_ratio.exp()
                    scaled = normalised(advantage[index, begin:end], mask)
                    surrogate = clipped_surrogate(log_ratio, scaled, c.clip)
                    policy_loss = -surrogate[mask].mean()
                    entropy = policy_entropy(logp)[mask].mean()
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    (policy_loss - c.entropy_coefficient * entropy).backward()
                    nn.utils.clip_grad_norm_(self.model.actor.parameters(), c.gradient_clip, error_if_nonfinite=True)
                    self.actor_optimizer.step()
                    value = self.sliced(self.model.critic, market, account, index, begin, end)[..., 0]
                    value_loss = .5 * ((value - returns[index, begin:end]) ** 2)[mask].mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.critic.parameters(), c.gradient_clip, error_if_nonfinite=True)
                    self.critic_optimizer.step()
                    with torch.no_grad():
                        # Schulman's low-variance estimator of KL(old || new), as CleanRL logs it.
                        stats['approx_kl'].append(float(((ratio - 1) - log_ratio)[mask].mean()))
                        stats['clip_fraction'].append(float(((ratio - 1).abs() > c.clip)[mask].float().mean()))
                    stats['policy_loss'].append(float(policy_loss.detach()))
                    stats['entropy'].append(float(entropy.detach()))
                    stats['value_loss'].append(float(value_loss.detach()))

        logp, values, final = self.replay(rows, market)
        _, returns = advantages(rows['rewards'], values, final, rows['terminals'], rows['valid'], gamma, lam)
        probabilities = logp.exp()
        side, sampled = rows['sides'], rows['actions']
        flat, held = selected & (side == FLAT), selected & (side != FLAT)
        entries = flat & (sampled != FLAT)
        exits = held & (sampled != side)
        long_entries = int((entries & (sampled == LONG)).sum())
        keep = probabilities.gather(-1, torch.from_numpy(side)[..., None]).squeeze(-1)
        switch = masked_mean(1 - keep, valid)
        flat_mask = torch.from_numpy(flat)
        decision_hours = c.decision_bars * c.decision_minutes / 60
        summary = {'update': self.updates + 1, **{k: float(np.mean(v)) if v else 0. for k, v in stats.items()}}
        summary.update({
            'policy_entropy': masked_mean(policy_entropy(logp), valid),
            'mean_reward': float(rows['rewards'][selected].mean()) if selected.any() else 0.,
            'reward_sd': float(rows['rewards'][selected].std()) if selected.any() else 0.,
            'active_steps': int(selected.sum()),
            'mean_net_log_return': float(rows['net_log_return'][selected].mean()) if selected.any() else 0.,
            'median_net_log_return': float(np.median(rows['net_log_return'][selected])) if selected.any() else 0.,
            'turnover_per_decision': float(rows['turnover'][selected].mean()) if selected.any() else 0.,
            'sampled_action_share': (np.bincount(sampled[selected], minlength=3) / max(selected.sum(), 1)).tolist(),
            'account_side_share': (np.bincount(side[selected], minlength=3) / max(selected.sum(), 1)).tolist(),
            'entry_decisions': int(entries.sum()),
            # Realised direction of entries from flat. A probability average over flat
            # rows once read 0.97 long in a run whose every actual entry went short.
            'long_entry_share': long_entries / int(entries.sum()) if entries.any() else None,
            'exit_decisions': int((held & (sampled == FLAT)).sum()),
            'reversal_decisions': int((held & (sampled != FLAT) & (sampled != side)).sum()),
            'mean_age_at_exit_hours': float(rows['position_age_minutes'][exits].mean() / 60) if exits.any() else None,
            'flat_long_probability': masked_mean(probabilities[..., LONG], flat_mask),
            'flat_short_probability': masked_mean(probabilities[..., SHORT], flat_mask),
            'switch_probability_mean': switch,
            'policy_implied_holding_hours': decision_hours / max(switch, 1e-9) if switch is not None else None,
            'state_action_mutual_information': state_action_mutual_information(probabilities, valid),
            'state_action_mutual_information_flat': state_action_mutual_information(probabilities, flat_mask),
            'explained_variance_pre': explained_variance(collected_returns[selected], rows['values'][selected]),
            'explained_variance_post': explained_variance(returns[selected], values[selected]),
            'target_std': float(returns[selected].std()) if selected.any() else 0.,
            'replay_probability_error': replay_probability, 'replay_value_error': replay_value,
            'replay_bootstrap_error': replay_bootstrap,
            'post_update_kl': masked_mean((old_probabilities * (old_probabilities.clamp_min(1e-12).log() - logp)).sum(-1),
                                          valid),
            'actor_lr': actor_lr, 'critic_lr': critic_lr,
            'liquidations_in_rollout': int(rows['terminals'].sum()),
            'mean_open_position_age_minutes': float(rows['position_age_minutes'][held].mean()) if held.any() else 0.,
            'max_open_position_age_minutes': float(rows['position_age_minutes'][held].max()) if held.any() else 0.})
        for number, name in ((SHORT, 'short'), (FLAT, 'flat'), (LONG, 'long')):
            mask = selected & (side == number)
            summary[f'value_rmse_{name}'] = float(np.sqrt(np.mean((returns[mask] - values[mask]) ** 2))) if mask.any() else None
        self.updates += 1
        return summary

    def run_update(self):
        return self.update(*self.collect())

    def state_dict(self):
        return {'model': self.model.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(), 'environment': self.env.state_dict(),
                'prefix_minutes': self.prefix_minutes, 'generator': self.generator.get_state(),
                'torch_rng': torch.get_rng_state(), 'updates': self.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        for name in ('actor_optimizer', 'critic_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        self.env.load_state_dict(state['environment'])
        self.prefix_minutes = state['prefix_minutes']
        self.generator.set_state(state['generator'])
        torch.set_rng_state(state['torch_rng'])
        self.updates = state['updates']
