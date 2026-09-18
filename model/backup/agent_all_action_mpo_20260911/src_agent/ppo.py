"""Recurrent actor-critic that compares every action on the same bar.

Why not sampled advantages. On a planted market (ARCHITECTURE.md §11.4) a per-bar
edge of 0.009 sd was invisible to GAE advantages that sum a 286-bar credit window
of market noise, and with costs the policy committed to one side within 100
updates. Mei et al. (NeurIPS 2021) prove that fast policy optimisation must commit
with positive probability "in the absence of external oracle information, which
allows an algorithm to determine the difference between optimal and sub-optimal
actions given only on-policy samples" - their stochastic setting observes only the
sampled action's reward.

In this market that information exists. A one-contract trader does not move the
price, so the reward of every action on a bar is computable exactly (Huang 2018,
"action augmentation"). Three consequences are used:

1. Action values Q(s, a) = r(s, a) + gamma V(s'_a) for all three actions, from exact
   branch rewards on the same bar and the critic at each branch's next account.
2. The policy improvement integrates over the actions instead of using the sampled
   one (expected policy gradients, Ciosek & Whiteson 2018; Mean Actor-Critic, Allen
   et al. 2017). The step itself is MPO's (Abdolmaleki et al., ICLR 2018): build the
   improved distribution q proportional to pi_old * exp(advantage / temperature) under
   a KL bound, then fit the network to it. PPO's ratio clip was tried first and failed
   (§11.6): its gradient on a logit carries that action's own probability as a factor,
   so once the policy went deterministic it stopped following its critic - the
   "gravitational pull of softmax" (Mei et al. 2020).
3. Updates are made at all three accounts reachable over the previous bar, not only
   at the visited one: exploring starts (Sutton & Barto 2018, §5.3) at every step.
   A policy that holds short still learns what to do when long - which policy
   improvement needs before it can leave the lock - and the critic learns those
   accounts' values from the same exact branches instead of extrapolating to states
   it never saw (Fujimoto, Meger & Precup, ICML 2019).

The visited account keeps its GAE lambda-return value target, which carries value
across the horizon; the other two use the one-step expected target over the
policy's own action probabilities (expected SARSA).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .ledger import ACTION_NAMES, FLAT, LONG, PORTFOLIO_WIDTH, SHORT, branch_every_action, replicate
from .network import market_tensor, policy_entropy

ACTIONS = len(ACTION_NAMES)
TRACKED = ('wallet', 'contracts', 'entry', 'dsr_mean', 'dsr_second', 'terminated')


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
    divergence = (p*(p.clamp_min(1e-12).log()-average.clamp_min(1e-12).log())).sum(-1).mean()
    return float(divergence.clamp_min(0))


def masked_mean(values, mask):
    return float(values[mask].mean()) if bool(mask.any()) else None


def relative_gap(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if not first.size:
        return 0.
    scale = np.maximum(np.maximum(np.abs(first), np.abs(second)), 1e-12)
    return float(np.max(np.abs(first - second) / scale))


class Trainer:
    def __init__(self, model, environment, scaler, config, seed):
        self.model, self.env, self.scaler, self.config = model, environment, scaler, config
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=config.critic_lr)
        self.generator = torch.Generator().manual_seed(seed + 1000)
        self.prefix_minutes = self.env.initial_prefix()
        # Three accounts per environment: those reachable over the previous bar
        # under SHORT, FLAT and LONG. ``actual_index`` marks the visited one.
        self.siblings, self.actual_index = self.env.starting_siblings(np.ones(self.env.size, bool))
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

    def load_siblings(self, mask, branches, actual):
        rows = np.flatnonzero(mask)
        targets = (rows[:, None] * ACTIONS + np.arange(ACTIONS)).ravel()
        for name, values in branches.state_dict().items():
            getattr(self.siblings, name)[targets] = values
        self.actual_index[rows] = actual

    def sibling_gap(self):
        """Largest relative difference between the tracked visited sibling and the environment."""
        chosen = np.arange(self.env.size) * ACTIONS + self.actual_index
        tracked, actual = self.siblings.state_dict(), self.env.ledger.state_dict()
        return max(relative_gap(tracked[name][chosen], actual[name]) for name in TRACKED)

    def account_features(self, ledger, minutes, copies):
        market = self.env.market
        return ledger.portfolio(np.repeat(market.prices[minutes - 1, 3], copies),
                                np.repeat(market.daily_volatility[minutes], copies))

    @torch.no_grad()
    def collect(self):
        c, env = self.config, self.env
        E, T, A = env.size, c.rollout, ACTIONS
        reset = (env.steps >= c.episode_steps) | env.ledger.terminated
        if reset.any():
            env.reset(reset)
            self.prefix_minutes[reset] = env.initial_prefix(reset)
            self.load_siblings(reset, *env.starting_siblings(reset))
        worst_gap = self.sibling_gap()
        actor_hidden, critic_hidden = self.model.burn(self.market(self.prefix_minutes))
        rows = {k: np.zeros((E, T), d) for k, d in (
            ('minutes', np.int64), ('actions', np.int64), ('sides', np.int64), ('actual_index', np.int64),
            ('logp', np.float32), ('values', np.float32), ('rewards', np.float32), ('terminals', bool),
            ('valid', bool), ('turnover', np.float32), ('net_log_return', np.float32),
            ('position_age_minutes', np.float32))}
        rows['account'] = np.zeros((E, T, PORTFOLIO_WIDTH), np.float32)
        rows['probabilities'] = np.zeros((E, T, A), np.float32)
        rows['sibling_account'] = np.zeros((E, T, A, PORTFOLIO_WIDTH), np.float32)
        rows['sibling_probabilities'] = np.zeros((E, T, A, A), np.float32)
        rows['sibling_active'] = np.zeros((E, T, A), bool)
        rows['branch_reward'] = np.zeros((E, T, A, A), np.float32)
        rows['branch_terminal'] = np.zeros((E, T, A, A), bool)
        rows['branch_account'] = np.zeros((E, T, A, A, PORTFOLIO_WIDTH), np.float32)
        everyone = np.arange(E)
        for step in range(T):
            minute = env.cursor.copy()
            market = self.market(minute[:, None])
            actor_memory, actor_hidden = self.model.actor.encode(market, actor_hidden)
            critic_memory, critic_hidden = self.model.critic.encode(market, critic_hidden)
            sibling = self.account_features(self.siblings, minute, A).reshape(E, A, -1)
            sibling_logp = self.model.actor.read(actor_memory.expand(E, A, -1), torch.from_numpy(sibling))
            account = sibling[everyone, self.actual_index]
            logp = sibling_logp[everyone, self.actual_index]
            probabilities = logp.exp()
            action = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(1)
            value = self.model.critic.read(critic_memory, torch.from_numpy(account[:, None]))[..., 0]
            branches, branch_reward, branch_terminal = branch_every_action(
                self.siblings, env.market, np.repeat(minute, A), c, env.bounds[1])
            rows['minutes'][:, step], rows['account'][:, step] = minute, account
            rows['sides'][:, step] = account[:, :3].argmax(1)
            rows['actual_index'][:, step] = self.actual_index
            rows['position_age_minutes'][:, step] = env.ledger.age
            rows['actions'][:, step] = action.numpy()
            rows['probabilities'][:, step] = probabilities.numpy()
            rows['logp'][:, step] = logp.gather(-1, action[:, None]).squeeze(-1).numpy()
            rows['values'][:, step] = value[:, 0].numpy()
            rows['sibling_account'][:, step] = sibling
            rows['sibling_probabilities'][:, step] = sibling_logp.exp().numpy()
            rows['sibling_active'][:, step] = ~self.siblings.terminated.reshape(E, A)
            rows['branch_reward'][:, step] = branch_reward.reshape(E, A, A)
            rows['branch_terminal'][:, step] = branch_terminal.reshape(E, A, A)
            reward, terminal, info = env.step(action.numpy())
            rows['rewards'][:, step], rows['terminals'][:, step] = reward, terminal
            rows['valid'][:, step], rows['turnover'][:, step] = info['active'], info['turnover']
            rows['net_log_return'][:, step] = info['log_return']
            rows['branch_account'][:, step] = self.account_features(branches, env.cursor, A * A).reshape(E, A, A, -1)
            worst_gap = max(worst_gap, relative_gap(reward, rows['branch_reward'][everyone, step, self.actual_index,
                                                                                  action.numpy()]))
            chosen = (everyone * A + self.actual_index)[:, None] * A + np.arange(A)
            self.siblings = replicate(branches, chosen.ravel(), 1)
            self.actual_index = action.numpy().copy()
            worst_gap = max(worst_gap, self.sibling_gap())
        if worst_gap > 1e-9:
            raise RuntimeError(f'counterfactual branches diverged from the environment: {worst_gap}')
        minute = env.cursor.copy()
        final_account = self.account_features(self.siblings, minute, A).reshape(E, A, -1)[everyone, self.actual_index]
        critic_memory, _ = self.model.critic.encode(self.market(minute[:, None]), critic_hidden)
        final_value = self.model.critic.read(critic_memory, torch.from_numpy(final_account[:, None]))[..., 0][:, 0].numpy()
        bootstrap = np.where(env.ledger.terminated, 0., final_value)
        rows['final_minute'], rows['final_account'] = minute, final_account
        rows['terminated_at_end'] = env.ledger.terminated.copy()
        rows['sibling_gap'] = worst_gap
        rows['prefix_minutes'] = self.prefix_minutes.copy()
        self.prefix_minutes = np.concatenate([self.prefix_minutes, rows['minutes']], 1)[:, -c.burn_in:].copy()
        return rows, bootstrap

    def sliced(self, branch, market, accounts, index, begin, end):
        """One minibatch of sequences, every sibling account read from the same market memory.

        The recurrent state is rebuilt under current parameters from the stored
        prefix, never taken stale from collection.
        """
        c = self.config
        with torch.no_grad():
            _, hidden = branch.encode(market[index, :c.burn_in + begin])
        memory, _ = branch.encode(market[index, c.burn_in + begin:c.burn_in + end], hidden)
        return branch.read(memory[:, :, None].expand(-1, -1, accounts.shape[2], -1), accounts[index, begin:end])

    @torch.no_grad()
    def evaluate_critic(self, rows, market):
        """State values, learned action values, and the targets for both critic heads.

        The policy step needs Q^pi, not one bar's outcome. MPO's E-step reads a learned
        critic, and Sinclair et al. distil counterfactual values into a network for the
        same reason. The exact one-step quantity r(s, a) + gamma V(s'_a) is therefore the
        action-value head's regression target, and the head - which averages a bar's noise
        across states with the same features - is what the policy step reads.
        """
        c = self.config
        E, T, A = rows['branch_reward'].shape[:3]
        memory, _ = self.model.critic.encode(market)
        memory = memory[:, c.burn_in:]
        sibling_account = torch.from_numpy(rows['sibling_account'])
        branch_account = torch.from_numpy(rows['branch_account'])
        sibling_values = torch.empty(E, T, A)
        sibling_action_values = torch.empty(E, T, A, A)
        next_values = torch.empty(E, T, A, A)
        for begin in range(0, T, c.sequence):
            end = min(T, begin + c.sequence)
            read = self.model.critic.read(memory[:, begin:end, None].expand(-1, -1, A, -1),
                                          sibling_account[:, begin:end])
            sibling_values[:, begin:end], sibling_action_values[:, begin:end] = read[..., 0], read[..., 1:]
            next_values[:, begin:end] = self.model.critic.read(
                memory[:, begin + 1:end + 1, None, None].expand(-1, -1, A, A, -1),
                branch_account[:, begin:end])[..., 0]
        final = self.model.critic.read(memory[:, T], torch.from_numpy(rows['final_account']))[..., 0]
        bootstrap = torch.where(torch.from_numpy(rows['terminated_at_end']), 0., final).numpy()
        actual = torch.from_numpy(rows['actual_index'])
        values = sibling_values.gather(-1, actual[..., None]).squeeze(-1)
        continuation = (~torch.from_numpy(rows['branch_terminal'])).float()
        one_step = torch.from_numpy(rows['branch_reward']) + c.gamma * continuation * next_values
        _, returns = advantages(rows['rewards'], values.numpy(), bootstrap, rows['terminals'], rows['valid'],
                                c.gamma, c.gae_lambda)
        expected = (torch.from_numpy(rows['sibling_probabilities']) * one_step).sum(-1)
        targets = torch.where(F.one_hot(actual, A).bool(), torch.from_numpy(returns)[..., None], expected)
        return {'values': values, 'sibling_values': sibling_values, 'action_values': sibling_action_values,
                'one_step_values': one_step, 'targets': targets, 'returns': returns}

    @torch.no_grad()
    def policy_target(self, action_values, old, mask):
        """MPO's E-step: the improved distribution, and the temperature setting its KL step.

        Abdolmaleki et al. (ICLR 2018) take q proportional to pi_old * exp(Q / eta), with
        eta from a convex dual under the constraint that the mean KL(q || pi_old) stays
        below epsilon. With three actions the integral over actions is exact, so eta is
        found by bisection on the realised mean KL, which decreases in it. Adding a
        per-state constant to Q leaves q unchanged, so the centred advantage is used.
        """
        c = self.config
        log_old = old.clamp_min(1e-12).log()
        advantage = action_values - (old * action_values).sum(-1, keepdim=True)
        spread = float(advantage[mask].abs().max()) if bool(mask.any()) else 0.
        if spread == 0.:
            return old.clone(), 0., 0.

        def improved(temperature):
            target = torch.softmax(log_old + advantage / temperature, -1)
            divergence = (target * (target.clamp_min(1e-12).log() - log_old)).sum(-1)
            return target, float(divergence[mask].mean())

        low, high = spread * 1e-6, spread * 1e6
        for _ in range(60):
            middle = math.sqrt(low * high)
            if improved(middle)[1] > c.policy_kl_step:
                low = middle
            else:
                high = middle
        temperature = math.sqrt(low * high)
        target, divergence = improved(temperature)
        return target, temperature, divergence

    def update(self, rows, bootstrap):
        c = self.config
        E, T, A = rows['branch_reward'].shape[:3]
        market = self.rollout_market(rows)
        sibling_account = torch.from_numpy(rows['sibling_account'])
        old = torch.from_numpy(rows['sibling_probabilities'])
        valid = torch.from_numpy(rows['valid'])
        sibling_mask = valid[..., None] & torch.from_numpy(rows['sibling_active'])
        actual = torch.from_numpy(rows['actual_index'])
        actor_lr, critic_lr = self.learning_rates()
        for group in self.actor_optimizer.param_groups:
            group['lr'] = actor_lr
        for group in self.critic_optimizer.param_groups:
            group['lr'] = critic_lr
        _, collected_returns = advantages(rows['rewards'], rows['values'], bootstrap, rows['terminals'],
                                          rows['valid'], c.gamma, c.gae_lambda)

        @torch.no_grad()
        def replay_policy():
            memory, _ = self.model.actor.encode(market)
            return self.model.actor.read(memory[:, c.burn_in:c.burn_in + T, None].expand(-1, -1, A, -1), sibling_account)

        replayed = replay_policy().exp()
        replay_ratio = float((replayed - old).abs()[sibling_mask].max()) if sibling_mask.any() else 0.
        critic = self.evaluate_critic(rows, market)

        replay_value = float((critic['values'] - torch.from_numpy(rows['values'])).abs()[valid].max()) if valid.any() else 0.
        if max(replay_ratio, replay_value) > 5e-5:
            raise RuntimeError(f'PPO replay mismatch: {replay_ratio}, {replay_value}')

        stats = {k: [] for k in ('policy_loss', 'value_loss', 'entropy', 'approx_kl')}
        improved, temperature, step = self.policy_target(critic['action_values'], old, sibling_mask)
        for epoch in range(c.epochs):
            if c.recompute_advantages_per_pass and epoch:
                # Andrychowicz et al.: stale advantages hurt; recompute each pass.
                critic = self.evaluate_critic(rows, market)
                improved, temperature, step = self.policy_target(critic['action_values'], old, sibling_mask)
            for chunk in torch.randperm(T // c.sequence, generator=self.generator).tolist():
                begin, end = chunk * c.sequence, (chunk + 1) * c.sequence
                for index in torch.randperm(E, generator=self.generator).split(c.minibatch_sequences):
                    mask = sibling_mask[index, begin:end]
                    if not mask.any():
                        continue
                    logits = self.sliced(self.model.actor, market, sibling_account, index, begin, end)
                    # M-step: fit the policy to the improved distribution. MPO use the
                    # forward KL here, which "reduces the tendency of the entropy of the
                    # parametric policy to collapse"; its gradient is pi_theta - target,
                    # which stays large for an action the critic likes even when that
                    # action's own probability has decayed to nothing.
                    policy_loss = -(improved[index, begin:end] * logits).sum(-1)[mask].mean()
                    entropy = policy_entropy(logits)[mask].mean()
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    (policy_loss - c.entropy_coefficient * entropy).backward()
                    nn.utils.clip_grad_norm_(self.model.actor.parameters(), c.gradient_clip, error_if_nonfinite=True)
                    self.actor_optimizer.step()
                    prediction = self.sliced(self.model.critic, market, sibling_account, index, begin, end)
                    value_error = (prediction[..., 0] - critic['targets'][index, begin:end]) ** 2
                    action_error = (prediction[..., 1:] - critic['one_step_values'][index, begin:end]) ** 2
                    value_loss = .5 * (value_error[mask].mean() + action_error[mask].mean())
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.critic.parameters(), c.gradient_clip, error_if_nonfinite=True)
                    self.critic_optimizer.step()
                    with torch.no_grad():
                        previous = old[index, begin:end]
                        kl = (previous * (previous.clamp_min(1e-12).log() - logits)).sum(-1)
                        stats['approx_kl'].append(float(kl[mask].mean()))
                    stats['policy_loss'].append(float(policy_loss.detach()))
                    stats['entropy'].append(float(entropy.detach()))
                    stats['value_loss'].append(float(value_loss.detach()))

        sibling_logits = replay_policy()
        logits = sibling_logits.gather(2, actual[..., None, None].expand(-1, -1, 1, A)).squeeze(2)
        critic = self.evaluate_critic(rows, market)
        probabilities = logits.exp()
        selected = rows['valid']
        side, sampled = rows['sides'], rows['actions']
        flat, held = selected & (side == FLAT), selected & (side != FLAT)
        entries = flat & (sampled != FLAT)
        long_entries = int((entries & (sampled == LONG)).sum())
        keep = probabilities.gather(-1, torch.from_numpy(side)[..., None]).squeeze(-1)
        switch = masked_mean(1 - keep, valid)
        flat_mask = torch.from_numpy(flat)
        sibling_sides = np.sort(rows['sibling_account'][..., :3].argmax(-1), axis=-1)
        returns, values = critic['returns'], critic['values'].numpy()
        summary = {'update': self.updates + 1, **{k: float(np.mean(v)) if v else 0. for k, v in stats.items()}}
        summary.update({
            'policy_entropy': masked_mean(policy_entropy(logits), valid),
            'sibling_policy_entropy': masked_mean(policy_entropy(sibling_logits), sibling_mask),
            'mean_reward': float(rows['rewards'][selected].mean()) if selected.any() else 0.,
            'reward_sd': float(rows['rewards'][selected].std()) if selected.any() else 0.,
            'active_steps': int(selected.sum()),
            'mean_net_log_return': float(rows['net_log_return'][selected].mean()) if selected.any() else 0.,
            'median_net_log_return': float(np.median(rows['net_log_return'][selected])) if selected.any() else 0.,
            'turnover_per_decision': float(rows['turnover'][selected].mean()) if selected.any() else 0.,
            'sampled_action_share': (np.bincount(sampled[selected], minlength=3) / max(selected.sum(), 1)).tolist(),
            'account_side_share': (np.bincount(side[selected], minlength=3) / max(selected.sum(), 1)).tolist(),
            'entry_decisions': int(entries.sum()),
            # Realised direction of entries from flat. The previous agent's
            # probability average over flat rows read 0.97 long in a run whose
            # every actual entry went short.
            'long_entry_share': long_entries / int(entries.sum()) if entries.any() else None,
            'exit_decisions': int((held & (sampled == FLAT)).sum()),
            'reversal_decisions': int((held & (sampled != FLAT) & (sampled != side)).sum()),
            'flat_long_probability': masked_mean(probabilities[..., LONG], flat_mask),
            'flat_short_probability': masked_mean(probabilities[..., SHORT], flat_mask),
            'switch_probability_mean': switch,
            'policy_implied_holding_hours': 5 / (60 * max(switch, 1e-9)) if switch is not None else None,
            'state_action_mutual_information': state_action_mutual_information(probabilities, valid),
            'state_action_mutual_information_flat': state_action_mutual_information(probabilities, flat_mask),
            'explained_variance_pre': explained_variance(collected_returns[selected], rows['values'][selected]),
            'explained_variance_post': explained_variance(returns[selected], values[selected]),
            'target_std': float(returns[selected].std()) if selected.any() else 0.,
            'action_value_spread': masked_mean(critic['action_values'].std(-1), sibling_mask),
            'action_value_head_error': masked_mean(
                (critic['action_values'] - critic['one_step_values']).abs().mean(-1), sibling_mask),
            'policy_temperature': temperature, 'policy_target_kl': step,
            'policy_target_entropy': masked_mean(
                -(improved.clamp_min(1e-12) * improved.clamp_min(1e-12).log()).sum(-1), sibling_mask),
            'sibling_value_error': masked_mean((critic['sibling_values'] - critic['targets']).abs(), sibling_mask),
            'sibling_side_coverage': float((sibling_sides[selected] == np.arange(A)).all(-1).mean()) if selected.any() else 0.,
            'counterfactual_consistency_gap': float(rows['sibling_gap']),
            'replay_ratio_error': replay_ratio, 'replay_value_error': replay_value,
            'post_update_kl': masked_mean((torch.from_numpy(rows['probabilities'])
                                           * (torch.from_numpy(rows['probabilities']).clamp_min(1e-12).log() - logits)).sum(-1),
                                          valid),
            'actor_lr': actor_lr, 'critic_lr': critic_lr,
            'liquidations_in_rollout': int(rows['terminals'].sum()),
            'mean_open_position_age_minutes': float(rows['position_age_minutes'][held].mean()) if held.any() else 0.,
            'max_open_position_age_minutes': float(rows['position_age_minutes'][held].max()) if held.any() else 0.})
        for index, name in ((SHORT, 'short'), (FLAT, 'flat'), (LONG, 'long')):
            mask = selected & (side == index)
            summary[f'value_rmse_{name}'] = float(np.sqrt(np.mean((returns[mask] - values[mask]) ** 2))) if mask.any() else None
        self.updates += 1
        return summary

    def run_update(self):
        return self.update(*self.collect())

    def state_dict(self):
        return {'model': self.model.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(), 'environment': self.env.state_dict(),
                'prefix_minutes': self.prefix_minutes, 'siblings': self.siblings.state_dict(),
                'actual_index': self.actual_index.copy(), 'generator': self.generator.get_state(),
                'torch_rng': torch.get_rng_state(), 'updates': self.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        for name in ('actor_optimizer', 'critic_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        self.env.load_state_dict(state['environment'])
        self.prefix_minutes = state['prefix_minutes']
        self.siblings.load_state_dict(state['siblings'])
        self.actual_index = state['actual_index'].copy()
        self.generator.set_state(state['generator'])
        torch.set_rng_state(state['torch_rng'])
        self.updates = state['updates']
