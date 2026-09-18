from __future__ import annotations

import numpy as np
import torch
from torch import nn

from perpetual.network import tensors
from perpetual.ppo import advantages, explained_variance
from .counterfactual import simulate
from .network import map_hidden


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
        self.generator = torch.Generator().manual_seed(seed+1000)
        self.branch_generator = torch.Generator().manual_seed(seed+2000)
        self.prefix_minutes, self.prefix_account = self.env.initial_prefix()
        self.updates = 0

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
                 ('terminals', bool), ('valid', bool), ('turnover', np.float32))}
        rows['account'] = np.zeros((*shape, 16), np.float32)
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
        adv = (adv-adv[valid].mean())/(adv[valid].std(unbiased=False)+1e-8)
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
        stats = {k: [] for k in ('policy_loss', 'value_loss', 'entropy', 'kl', 'clip_fraction', 'cf_actor_loss', 'cf_q_loss', 'cf_kl')}
        stopped = False
        entropy_coef = c.entropy_start + min(1., self.updates/max(c.entropy_decay_updates, 1))*(c.entropy_end-c.entropy_start)
        for epoch in range(c.epochs+c.critic_epochs):
            critic_only = epoch >= c.epochs
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
                        if float(kl.detach()) > 1.5*c.target_kl:
                            stopped = True
                        else:
                            selected_adv = adv[index, begin:end]
                            loss = -torch.minimum(ratio*selected_adv, ratio.clamp(1-c.clip, 1+c.clip)*selected_adv)[mask].mean()
                            entropy = -(logits.exp()*logits).sum(-1)[mask].mean()
                            self.actor_optimizer.zero_grad(set_to_none=True)
                            (loss-entropy_coef*entropy).backward()
                            nn.utils.clip_grad_norm_(self.model.actor.parameters(), c.gradient_clip, error_if_nonfinite=True)
                            self.actor_optimizer.step()
                            for key, item in (('policy_loss', loss), ('entropy', entropy), ('kl', kl), ('clip_fraction', ((ratio-1).abs() > c.clip)[mask].float().mean())):
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
                        if float(kl.detach()) > 1.5*c.target_kl:
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
                        'replay_value_error': replay_value, 'entropy_coefficient': entropy_coef,
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
                'torch_rng': torch.get_rng_state(), 'updates': self.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        for name in ('actor_optimizer', 'critic_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        self.env.load_state_dict(state['environment'])
        self.prefix_minutes, self.prefix_account = state['prefix_minutes'], state['prefix_account']
        self.generator.set_state(state['generator']); self.branch_generator.set_state(state['branch_generator'])
        torch.set_rng_state(state['torch_rng'])
        self.updates = state['updates']
