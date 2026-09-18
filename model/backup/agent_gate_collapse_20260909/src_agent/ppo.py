from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .counterfactual import simulate
from .ledger import PORTFOLIO_WIDTH
from .network import direction_entropy, gate_entropy, map_hidden, tensors


def explained_variance(target, prediction):
    variance = float(np.var(target))
    return None if variance < 1e-12 else 1 - float(np.var(target - prediction)) / variance


def advantages(rewards, values, bootstrap, terminals, valid, gamma, lam):
    """Economic termination masks bootstrap; artificial time limits do not."""
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
    is the failure that made each v5 run a bet on the year's direction rather
    than a trading rule. Defined on absolute actions, so it names no side and is
    invariant to relabelling them.
    """
    if not bool(mask.any()):
        return 0.
    p = probabilities[mask]
    average = p.mean(0, keepdim=True)
    divergence = (p*(p.clamp_min(1e-12).log()-average.clamp_min(1e-12).log())).sum(-1).mean()
    return float(divergence.clamp_min(0))


class Trainer:
    def __init__(self, model, environment, scaler, config, seed):
        self.model, self.env, self.scaler, self.config = model, environment, scaler, config
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=config.critic_lr)
        # Dual variable for the direction-entropy target, tuned by gradient the
        # way SAC tunes its temperature (Haarnoja et al. 2018, arXiv:1812.05905).
        # A fixed coefficient cannot work here: advantages are rescaled to order
        # one, so any constant is either negligible or overwhelming, and the
        # predecessor's 0.005 was roughly two hundred times too weak to matter.
        self.log_alpha = torch.tensor(float(np.log(config.entropy_alpha_init)), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.entropy_alpha_lr)
        # A SECOND dual, on the gate. Every collapse observed - to permanent flat
        # and to a side held for months - happened here while direction entropy
        # stayed near its maximum and its own dual therefore never engaged.
        self.log_gate_alpha = torch.tensor(float(np.log(config.entropy_alpha_init)), requires_grad=True)
        self.gate_alpha_optimizer = torch.optim.Adam([self.log_gate_alpha], lr=config.entropy_alpha_lr)
        # Slow global advantage scale. Per-minibatch standardisation guarantees
        # unit-scale updates even from a batch containing no signal at all.
        self.advantage_scale = None
        self.generator = torch.Generator().manual_seed(seed+1000)
        self.branch_generator = torch.Generator().manual_seed(seed+2000)
        self.prefix_minutes, self.prefix_account = self.env.initial_prefix()
        self.updates = 0

    def alpha(self):
        c = self.config
        return self.log_alpha.exp().clamp(c.entropy_alpha_min, c.entropy_alpha_max)

    def gate_alpha(self):
        c = self.config
        return self.log_gate_alpha.exp().clamp(c.entropy_alpha_min, c.entropy_alpha_max)

    def learning_rates(self):
        """Linear decay to zero over the run, as CleanRL anneals by default."""
        c = self.config
        fraction = 1.0 - self.updates/max(c.updates, 1) if c.anneal_lr else 1.0
        fraction = max(fraction, 0.0)
        return c.actor_lr*fraction, c.critic_lr*fraction

    @torch.no_grad()
    def calibrate(self):
        if self.config.normalization != 'batch':
            return
        # Calibration uses only the training environment. Account paths include
        # each side and the actual fees/funding; no validation normalization.
        state = self.env.state_dict()
        minutes, accounts = [], []
        for _ in range(self.config.burn_in):
            minute, account = self.env.observation()
            minutes.append(minute); accounts.append(account)
            self.env.step(np.arange(self.env.size) % 3)
        self.env.load_state_dict(state)
        self.model.calibrate(*tensors(self.scaler, self.env.market, np.stack(minutes, 1), np.stack(accounts, 1), self.config))

    @torch.no_grad()
    def collect(self):
        c, env = self.config, self.env
        reset = (env.steps >= c.episode_steps) | env.ledger.terminated
        if reset.any():
            env.reset(reset)
            minute, account = env.initial_prefix(reset)
            self.prefix_minutes[reset], self.prefix_account[reset] = minute, account
        hidden = self.model.burn(*tensors(self.scaler, env.market, self.prefix_minutes, self.prefix_account, c))
        masks = self.model.masks(env.size, self.generator)
        shape = (env.size, c.rollout)
        rows = {k: np.zeros(shape, d) for k, d in (('minutes', np.int64), ('actions', np.int64),
                 ('logp', np.float32), ('values', np.float32), ('rewards', np.float32),
                 ('terminals', bool), ('valid', bool), ('turnover', np.float32),
                 ('net_log_return', np.float32))}
        rows['account'] = np.zeros((*shape, PORTFOLIO_WIDTH), np.float32)
        rows['holding_age_minutes'] = np.zeros(shape, np.float32)
        rows['probabilities'] = np.zeros((*shape, 3), np.float32)
        rows['masks'] = masks
        anchors = []
        for step in range(c.rollout):
            if c.counterfactual_enabled and step % c.counterfactual_stride == c.counterfactual_stride//2:
                anchors.append({'step': step, 'state': env.state_dict(),
                                'hidden': map_hidden(hidden, lambda x: x.clone())})
            minute, account = env.observation()
            inputs = tensors(self.scaler, env.market, minute[:, None], account[:, None], c)
            logits, value, _, hidden = self.model(*inputs, hidden, masks)
            distribution = torch.distributions.Categorical(logits=logits[:, 0])
            action = torch.multinomial(distribution.probs, 1, generator=self.generator).squeeze(1)
            rows['minutes'][:, step], rows['account'][:, step] = minute, account
            rows['holding_age_minutes'][:, step] = env.ledger.age
            rows['actions'][:, step] = action.numpy()
            rows['probabilities'][:, step] = distribution.probs.numpy()
            rows['logp'][:, step] = distribution.log_prob(action).numpy()
            rows['values'][:, step] = value[:, 0].numpy()
            reward, terminal, info = env.step(action.numpy())
            rows['rewards'][:, step], rows['terminals'][:, step] = reward, terminal
            rows['valid'][:, step], rows['turnover'][:, step] = info['active'], info['turnover']
            # The plain log-equity objective is kept next to the reward so it can
            # be recomputed from saved logs without retraining.
            rows['net_log_return'][:, step] = info['log_return']
        minute, account = env.observation()
        _, value, _, _ = self.model(*tensors(self.scaler, env.market, minute[:, None], account[:, None], c), hidden, masks)
        bootstrap = np.where(env.ledger.terminated, 0., value[:, 0].numpy())
        rows['prefix_minutes'], rows['prefix_account'] = self.prefix_minutes.copy(), self.prefix_account.copy()
        self.prefix_minutes = np.concatenate([self.prefix_minutes, rows['minutes']], 1)[:, -c.burn_in:].copy()
        self.prefix_account = np.concatenate([self.prefix_account, rows['account']], 1)[:, -c.burn_in:].copy()
        before = env.state_dict()
        cf = simulate(self.model, env, self.scaler, anchors, masks, self.branch_generator)
        for key, value in before['ledger'].items():
            if not np.array_equal(value, env.ledger.state_dict()[key]):
                raise RuntimeError('counterfactual branch mutated the actual ledger')
        if cf is not None:
            expected = np.concatenate([rows['probabilities'][:, a['step']] for a in anchors])
            error = float(np.abs(cf['old_probabilities']-expected).max())
            if error > 5e-5:
                raise RuntimeError(f'counterfactual initial policy mismatch: {error}')
            cf['replay_error'] = error
        return rows, bootstrap, cf

    def update(self, rows, bootstrap, cf):
        c = self.config
        advantage, target = advantages(rows['rewards'], rows['values'], bootstrap, rows['terminals'], rows['valid'], c.gamma, c.gae_lambda)
        full_minute = np.concatenate([rows['prefix_minutes'], rows['minutes']], 1)
        full_account = np.concatenate([rows['prefix_account'], rows['account']], 1)
        data = tensors(self.scaler, self.env.market, full_minute, full_account, c)
        valid, actions = torch.from_numpy(rows['valid']), torch.from_numpy(rows['actions'])
        old_logp, old_probs = torch.from_numpy(rows['logp']), torch.from_numpy(rows['probabilities'])
        adv, targets = torch.from_numpy(advantage), torch.from_numpy(target)
        # Centre per batch, but scale by a slowly moving global estimate so that
        # a batch with genuinely nothing to learn produces a genuinely small
        # update instead of being stretched to unit variance.
        centred = adv-adv[valid].mean()
        batch_scale = float(centred[valid].square().mean().sqrt())
        if self.advantage_scale is None:
            self.advantage_scale = batch_scale
        else:
            self.advantage_scale += c.advantage_scale_decay*(batch_scale-self.advantage_scale)
        adv = centred/(self.advantage_scale+1e-8)
        masks = rows['masks']

        def forward_slice(index, begin, end, branch=None):
            inputs = [a[index] for a in data]
            # Actual prefix recomputed under current parameters, no stale states.
            if branch is None:
                hidden = self.model.burn(*(a[:, :c.burn_in+begin] for a in inputs))
                return self.model(*(a[:, c.burn_in+begin:c.burn_in+end] for a in inputs), hidden,
                                  tuple(m[index] for m in masks))
            network = getattr(self.model, branch)
            hidden = None
            with torch.no_grad():
                for start in range(0, c.burn_in+begin, 64):
                    stop = min(start+64, c.burn_in+begin)
                    hidden = network(*(a[:, start:stop] for a in inputs), hidden)[1]
            b = 0 if branch == 'actor' else 1
            return network(*(a[:, c.burn_in+begin:c.burn_in+end] for a in inputs), hidden, masks[b][index])[0]

        @torch.no_grad()
        def replay():
            outputs, values = [], []
            hidden = self.model.burn(*(a[:, :c.burn_in] for a in data))
            for start in range(0, c.rollout, c.sequence):
                logp, value, _, hidden = self.model(*(a[:, c.burn_in+start:c.burn_in+start+c.sequence] for a in data), hidden, masks)
                outputs.append(logp); values.append(value)
            return torch.cat(outputs, 1), torch.cat(values, 1)

        logp, value = replay()
        replay_ratio = float((torch.exp(logp.gather(-1, actions[..., None]).squeeze(-1)-old_logp)-1).abs().max())
        replay_value = float((value-torch.from_numpy(rows['values'])).abs().max())
        if max(replay_ratio, replay_value) > 5e-5:
            raise RuntimeError(f'PPO replay mismatch: {replay_ratio}, {replay_value}')
        stats = {k: [] for k in ('policy_loss', 'value_loss', 'entropy', 'direction_entropy',
                                 'gate_entropy', 'kl', 'clip_fraction', 'cf_actor_loss',
                                 'cf_q_loss', 'cf_kl')}
        stopped = False
        current_action = torch.from_numpy((rows['account'][..., 0]+1).astype(np.int64))
        actor_lr, critic_lr = self.learning_rates()
        for group in self.actor_optimizer.param_groups:
            group['lr'] = actor_lr
        for group in self.critic_optimizer.param_groups:
            group['lr'] = critic_lr

        def rescale(raw):
            centred = raw-raw[valid].mean()
            return centred/(self.advantage_scale+1e-8)

        @torch.no_grad()
        def recomputed_advantages():
            """Fresh advantages and value targets from the current critic.

            Andrychowicz et al. measure that stale advantages hurt and that
            recomputing them at the start of each pass "performs best among all
            variants". The bootstrap past the end of the rollout is left as
            collected, since the state one step beyond it is not in this buffer;
            it affects only the final step's advantage.
            """
            values, hidden = [], self.model.critic(*(a[:, :c.burn_in] for a in data))[1]
            for start in range(0, c.rollout, c.sequence):
                window = tuple(a[:, c.burn_in+start:c.burn_in+start+c.sequence] for a in data)
                output, hidden = self.model.critic(*window, hidden, masks[1])
                values.append(output[..., 0])
            fresh = torch.cat(values, 1).numpy()
            raw, fresh_target = advantages(rows['rewards'], fresh, bootstrap, rows['terminals'],
                                           rows['valid'], c.gamma, c.gae_lambda)
            return rescale(torch.from_numpy(raw)), torch.from_numpy(fresh_target)
        for epoch in range(c.epochs+c.critic_epochs):
            critic_only = epoch >= c.epochs
            # A fresh pass over the same data: recompute advantages from the
            # updated critic, and release a KL stop scoped to the previous pass.
            if c.kl_stop_scope == 'pass':
                stopped = False
            if c.recompute_advantages_per_pass and epoch and not critic_only:
                adv, targets = recomputed_advantages()
            for chunk in torch.randperm(c.rollout//c.sequence, generator=self.generator).tolist():
                begin, end = chunk*c.sequence, (chunk+1)*c.sequence
                for index in torch.randperm(c.environments, generator=self.generator).split(c.minibatch_sequences):
                    mask = valid[index, begin:end]
                    if not mask.any():
                        continue
                    if critic_only or stopped:
                        prediction = forward_slice(index, begin, end, 'critic')[..., 0]
                    else:
                        logits, prediction, _, _ = forward_slice(index, begin, end)
                        logp = logits.gather(-1, actions[index, begin:end, None]).squeeze(-1)
                        difference = logp-old_logp[index, begin:end]
                        ratio = difference.exp()
                        kl = (old_probs[index, begin:end]*(old_probs[index, begin:end].clamp_min(1e-12).log()-logits)).sum(-1)[mask].mean()
                        if c.kl_stop_scope != 'off' and float(kl.detach()) > 1.5*c.target_kl:
                            stopped = True
                        else:
                            selected_adv = adv[index, begin:end]
                            loss = -torch.minimum(ratio*selected_adv, ratio.clamp(1-c.clip, 1+c.clip)*selected_adv)[mask].mean()
                            entropy = -(logits.exp()*logits).sum(-1)[mask].mean()
                            # Regularise the DIRECTION choice only, uniformly
                            # across states. Weighting this by P(switch) would
                            # hand the agent an escape: drive switching to zero
                            # and the penalty disappears with it.
                            direction = direction_entropy(logits, current_action[index, begin:end])[mask].mean()
                            gate = gate_entropy(logits, current_action[index, begin:end])[mask].mean()
                            self.actor_optimizer.zero_grad(set_to_none=True)
                            (loss-self.alpha().detach()*direction
                                 -self.gate_alpha().detach()*gate).backward()
                            nn.utils.clip_grad_norm_(self.model.actor.parameters(), c.gradient_clip, error_if_nonfinite=True)
                            self.actor_optimizer.step()
                            # alpha rises while direction entropy sits below the
                            # target and falls once it is above it.
                            for parameter, optimizer, value, goal in (
                                    (self.log_alpha, self.alpha_optimizer, direction, c.entropy_target),
                                    (self.log_gate_alpha, self.gate_alpha_optimizer, gate, c.gate_entropy_target)):
                                dual_loss = parameter.exp()*(value.detach()-goal)
                                optimizer.zero_grad(set_to_none=True)
                                dual_loss.backward()
                                optimizer.step()
                                # Clamp the parameter itself, not only its reading:
                                # the dual loss is exp(log_alpha) * gap, so a
                                # drifting parameter makes its own gradient blow up.
                                with torch.no_grad():
                                    parameter.clamp_(float(np.log(c.entropy_alpha_min)),
                                                     float(np.log(c.entropy_alpha_max)))
                            for key, item in (('policy_loss', loss), ('entropy', entropy),
                                              ('direction_entropy', direction), ('gate_entropy', gate),
                                              ('kl', kl),
                                              ('clip_fraction', ((ratio-1).abs() > c.clip)[mask].float().mean())):
                                stats[key].append(float(item.detach()))
                    loss = .5*((prediction-targets[index, begin:end])**2)[mask].mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.critic.parameters(), c.gradient_clip, error_if_nonfinite=True)
                    self.critic_optimizer.step()
                    stats['value_loss'].append(float(loss.detach()))

            if cf is not None:
                # Enumerated action-return estimator; these are not on-policy
                # rollout rows and do not enter GAE or its normalization.
                for anchor_id in torch.randperm(len(cf['steps']), generator=self.generator).tolist():
                    step = cf['steps'][anchor_id]
                    index = torch.arange(c.environments)
                    chosen = slice(anchor_id*c.environments, (anchor_id+1)*c.environments)
                    returns = torch.from_numpy(cf['targets'][chosen])
                    prior = torch.from_numpy(cf['old_probabilities'][chosen])
                    active = valid[:, step]
                    if not active.any():
                        continue
                    if not critic_only and not stopped and self.updates >= c.counterfactual_warmup:
                        logits = forward_slice(index, step, step+1, 'actor')[:, 0]
                        ratio = logits.exp()/prior.clamp_min(1e-12)
                        kl = (prior*(prior.clamp_min(1e-12).log()-logits)).sum(-1)[active].mean()
                        stats['cf_kl'].append(float(kl.detach()))
                        if c.kl_stop_scope != 'off' and float(kl.detach()) > 1.5*c.target_kl:
                            stopped = True
                        else:
                            advantage_cf = returns-(prior*returns).sum(-1, keepdim=True)
                            # One scale per complete anchor batch preserves action comparisons.
                            advantage_cf = advantage_cf/(advantage_cf[active].square().mean().sqrt()+1e-8)
                            surrogate = torch.minimum(ratio*advantage_cf, ratio.clamp(1-c.clip, 1+c.clip)*advantage_cf)
                            loss = -(prior*surrogate).sum(-1)[active].mean()
                            self.actor_optimizer.zero_grad(set_to_none=True)
                            (c.counterfactual_actor_weight*loss).backward()
                            nn.utils.clip_grad_norm_(self.model.actor.parameters(), c.gradient_clip, error_if_nonfinite=True)
                            self.actor_optimizer.step()
                            stats['cf_actor_loss'].append(float(loss.detach()))
                    prediction = forward_slice(index, step, step+1, 'critic')[:, 0, 1:]
                    loss = .5*((prediction-returns)**2)[active].mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    (c.counterfactual_critic_weight*loss).backward()
                    nn.utils.clip_grad_norm_(self.model.critic.parameters(), c.gradient_clip, error_if_nonfinite=True)
                    self.critic_optimizer.step()
                    stats['cf_q_loss'].append(float(loss.detach()))

        logits, prediction = replay()
        summary = {k: float(np.mean(v)) if v else 0. for k, v in stats.items()}
        selected = rows['valid']
        summary.update({'update': self.updates+1, 'mean_reward': float(rows['rewards'][selected].mean()),
                        'active_steps': int(selected.sum()), 'turnover_per_decision': float(rows['turnover'][selected].mean()),
                        'sampled_action_share': (np.bincount(rows['actions'][selected], minlength=3)/selected.sum()).tolist(),
                        'account_side_share': (np.bincount((rows['account'][..., 0][selected]+1).astype(int), minlength=3)/selected.sum()).tolist(),
                        'explained_variance_pre': explained_variance(target[selected], rows['values'][selected]),
                        'explained_variance_post': explained_variance(target[selected], prediction.numpy()[selected]),
                        'target_std': float(target[selected].std()), 'replay_ratio_error': replay_ratio,
                        'replay_value_error': replay_value,
                        'entropy_alpha': float(self.alpha().detach()),
                        'gate_entropy_alpha': float(self.gate_alpha().detach()),
                        'actor_lr': actor_lr, 'critic_lr': critic_lr,
                        'advantage_scale': float(self.advantage_scale),
                        'mean_net_log_return': float(rows['net_log_return'][selected].mean()),
                        'net_log_return_sd': float(rows['net_log_return'][selected].std()),
                        'reward_sd': float(rows['rewards'][selected].std()),
                        'reward_clip_fraction': float(np.mean(np.abs(rows['rewards'][selected]) >= self.config.dsr_clip*self.config.reward_scale-1e-9)),
                        'post_update_kl': float((old_probs*(old_probs.clamp_min(1e-12).log()-logits)).sum(-1)[valid].mean()),
                        'kl_stopped': stopped, 'liquidations_in_rollout': int(rows['terminals'].sum()),
                        'counterfactual_branches': 0 if cf is None else int(cf['targets'].size),
                        'counterfactual_replay_error': 0. if cf is None else cf['replay_error'],
                        'counterfactual_side_counts': [0, 0, 0] if cf is None else cf['path_side_counts'],
                        'counterfactual_target_std': 0. if cf is None else float(cf['targets'].std())})
        current = (rows['account'][..., 0]+1).astype(int)
        held = selected & (current != 1)
        flat = selected & (current == 1)
        summary.update({'entry_decisions': int((flat & (rows['actions'] != 1)).sum()),
                        'exit_decisions': int((held & (rows['actions'] == 1)).sum()),
                        'reversal_decisions': int((held & (rows['actions'] != 1) & (rows['actions'] != current)).sum()),
                        'mean_open_position_age_minutes': float(rows['holding_age_minutes'][held].mean()) if held.any() else 0.,
                        'max_open_position_age_minutes': float(rows['holding_age_minutes'][held].max()) if held.any() else 0.,
                        'counterfactual_best_action_counts': [0, 0, 0] if cf is None else np.bincount(cf['targets'].argmax(-1), minlength=3).tolist()})
        # Stage A diagnostics on the post-update policy. Mutual information
        # between state and action is zero exactly when the policy is a constant
        # function of the market, which is the failure that turned every v5 run
        # into a bet on the year's direction rather than a trading rule. These
        # are measurements only: neither names a side, neither enters a gradient.
        probabilities = logits.exp()
        keep = probabilities.gather(-1, torch.from_numpy(current)[..., None]).squeeze(-1)
        switch = (1-keep)[valid]
        mean_switch = float(switch.mean()) if switch.numel() else 0.
        summary.update({'state_action_mutual_information': state_action_mutual_information(probabilities, valid),
                        'state_action_mutual_information_flat': state_action_mutual_information(probabilities, torch.from_numpy(flat)),
                        'switch_probability_mean': mean_switch,
                        'switch_probability_sd': float(switch.std(unbiased=False)) if switch.numel() else 0.,
                        'policy_implied_holding_hours': 5/(60*max(mean_switch, 1e-9))})
        for side, name in ((-1, 'short'), (0, 'flat'), (1, 'long')):
            mask = selected & (rows['account'][..., 0] == side)
            summary[f'value_rmse_{name}'] = float(np.sqrt(np.mean((target[mask]-prediction.numpy()[mask])**2))) if mask.any() else None
        self.updates += 1
        return summary

    def run_update(self):
        return self.update(*self.collect())

    def state_dict(self):
        return {'model': self.model.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(), 'environment': self.env.state_dict(),
                'prefix_minutes': self.prefix_minutes, 'prefix_account': self.prefix_account,
                'generator': self.generator.get_state(), 'branch_generator': self.branch_generator.get_state(),
                'log_alpha': self.log_alpha.detach().clone(),
                'alpha_optimizer': self.alpha_optimizer.state_dict(),
                'log_gate_alpha': self.log_gate_alpha.detach().clone(),
                'gate_alpha_optimizer': self.gate_alpha_optimizer.state_dict(),
                'advantage_scale': self.advantage_scale,
                'torch_rng': torch.get_rng_state(), 'updates': self.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        for name in ('actor_optimizer', 'critic_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        self.env.load_state_dict(state['environment'])
        self.prefix_minutes, self.prefix_account = state['prefix_minutes'], state['prefix_account']
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
            self.log_gate_alpha.copy_(state['log_gate_alpha'])
        self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
        self.gate_alpha_optimizer.load_state_dict(state['gate_alpha_optimizer'])
        self.advantage_scale = state['advantage_scale']
        self.generator.set_state(state['generator']); self.branch_generator.set_state(state['branch_generator'])
        torch.set_rng_state(state['torch_rng'])
        self.updates = state['updates']
