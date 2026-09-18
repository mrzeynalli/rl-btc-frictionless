from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .network import tensors


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


class Trainer:
    def __init__(self, model, environment, scaler, config, seed):
        self.model, self.env, self.scaler, self.config = model, environment, scaler, config
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=config.critic_lr)
        self.generator = torch.Generator().manual_seed(seed + 1000)
        self.prefix_minutes, self.prefix_portfolio = self.env.initial_prefix()
        self.updates = 0

    @torch.no_grad()
    def collect(self):
        config, env = self.config, self.env
        reset = (env.steps >= config.episode_steps) | env.ledger.terminated
        if reset.any():
            env.reset(reset)
            minute, portfolio = env.initial_prefix(reset)
            self.prefix_minutes[reset], self.prefix_portfolio[reset] = minute, portfolio
        prefix = tensors(self.scaler, env.market, self.prefix_minutes, self.prefix_portfolio, config)
        hidden = self.model.burn(*prefix)
        size, steps = config.environments, config.rollout
        rows = {name: np.zeros((size, steps), dtype=dtype) for name, dtype in (
            ("minutes", np.int64), ("actions", np.int64), ("logp", np.float32),
            ("values", np.float32), ("rewards", np.float32),
            ("terminals", bool), ("valid", bool), ("turnover", np.float32))}
        rows["portfolio"] = np.zeros((size, steps, self.prefix_portfolio.shape[-1]), np.float32)
        for step in range(steps):
            minute, portfolio = env.observation()
            inputs = tensors(self.scaler, env.market, minute[:, None], portfolio[:, None], config)
            logits, value, hidden = self.model(*inputs, hidden)
            distribution = torch.distributions.Categorical(logits=logits[:, 0])
            action = torch.multinomial(distribution.probs, 1, generator=self.generator).squeeze(1)
            rows["minutes"][:, step], rows["portfolio"][:, step] = minute, portfolio
            rows["actions"][:, step] = action.numpy()
            rows["logp"][:, step] = distribution.log_prob(action).numpy()
            rows["values"][:, step] = value[:, 0].numpy()
            reward, terminal, info = env.step(action.numpy())
            rows["rewards"][:, step], rows["terminals"][:, step] = reward, terminal
            rows["valid"][:, step], rows["turnover"][:, step] = info["active"], info["turnover"]
        minute, portfolio = env.observation()
        _, value, _ = self.model(*tensors(self.scaler, env.market, minute[:, None], portfolio[:, None], config), hidden)
        bootstrap = np.where(env.ledger.terminated, 0.0, value[:, 0].numpy())
        rows["prefix_minutes"], rows["prefix_portfolio"] = self.prefix_minutes.copy(), self.prefix_portfolio.copy()
        self.prefix_minutes = np.concatenate([self.prefix_minutes, rows["minutes"]], axis=1)[:, -config.burn_in:].copy()
        self.prefix_portfolio = np.concatenate([self.prefix_portfolio, rows["portfolio"]], axis=1)[:, -config.burn_in:].copy()
        return rows, bootstrap

    def update(self, rows, bootstrap):
        config = self.config
        advantage, target = advantages(rows["rewards"], rows["values"], bootstrap,
                                       rows["terminals"], rows["valid"], config.gamma, config.gae_lambda)
        full_minute = np.concatenate([rows["prefix_minutes"], rows["minutes"]], axis=1)
        full_portfolio = np.concatenate([rows["prefix_portfolio"], rows["portfolio"]], axis=1)
        data = tensors(self.scaler, self.env.market, full_minute, full_portfolio, config)
        valid = torch.from_numpy(rows["valid"])
        actions = torch.from_numpy(rows["actions"])
        old_logp = torch.from_numpy(rows["logp"])
        advantage_t = torch.from_numpy(advantage)
        target_t = torch.from_numpy(target)
        if not valid.any():
            raise RuntimeError("no active observations in PPO rollout")
        advantage_t = (advantage_t - advantage_t[valid].mean()) / (advantage_t[valid].std(unbiased=False) + 1e-8)
        prefix_length = config.burn_in

        def replay():
            hidden = self.model.burn(*(a[:, :prefix_length] for a in data))
            return self.model(*(a[:, prefix_length:] for a in data), hidden)

        with torch.no_grad():
            logits, replay_values, _ = replay()
            logp = torch.distributions.Categorical(logits=logits).log_prob(actions)
            ratio_error = float((torch.exp(logp - old_logp) - 1).abs().max())
            value_error = float((replay_values - torch.from_numpy(rows["values"])).abs().max())
        if ratio_error > 5e-5 or value_error > 5e-5:
            raise RuntimeError(f"recurrent replay mismatch: probability={ratio_error}, value={value_error}")
        progress = min(1.0, self.updates / max(1, config.entropy_decay_updates))
        entropy_coef = config.entropy_start + progress * (config.entropy_end - config.entropy_start)
        statistics = {name: [] for name in ("policy_loss", "value_loss", "entropy", "kl", "clip_fraction", "actor_gradient_norm", "critic_gradient_norm")}
        stop = False
        for epoch in range(config.epochs + config.critic_epochs):
            critic_only = epoch >= config.epochs
            if stop and not critic_only:
                continue
            starts = torch.randperm(config.rollout // config.sequence, generator=self.generator).tolist()
            for chunk in starts:
                begin, end = chunk * config.sequence, (chunk + 1) * config.sequence
                order = torch.randperm(config.environments, generator=self.generator)
                for index in order.split(config.minibatch_sequences):
                    mask = valid[index, begin:end]
                    if not mask.any():
                        continue
                    # Recompute the actual prefix with CURRENT parameters on every
                    # epoch. Portfolio/action/reward histories remain as recorded.
                    inputs = [a[index] for a in data]
                    with torch.no_grad():
                        hidden = self.model.burn(*(a[:, :prefix_length + begin] for a in inputs))
                    batch = [a[:, prefix_length + begin:prefix_length + end] for a in inputs]
                    if critic_only:
                        values = self.model.critic(*batch, hidden[1])[0].squeeze(-1)
                    else:
                        logits, values, _ = self.model(*batch, hidden)
                        distribution = torch.distributions.Categorical(logits=logits)
                        logp = distribution.log_prob(actions[index, begin:end])
                        difference = logp - old_logp[index, begin:end]
                        ratio = torch.exp(difference)
                        kl = (ratio - 1 - difference)[mask].mean()
                        if float(kl.detach()) > 1.5 * config.target_kl:
                            stop = True
                            break
                        adv = advantage_t[index, begin:end]
                        policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - config.clip, 1 + config.clip) * adv)[mask].mean()
                        entropy = distribution.entropy()[mask].mean()
                        actor_loss = policy_loss - entropy_coef * entropy
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        norm = nn.utils.clip_grad_norm_(self.model.actor.parameters(), config.gradient_clip, error_if_nonfinite=True)
                        self.actor_optimizer.step()
                        for name, value in (("policy_loss", policy_loss), ("entropy", entropy), ("kl", kl),
                                            ("clip_fraction", ((ratio - 1).abs() > config.clip)[mask].float().mean()),
                                            ("actor_gradient_norm", norm)):
                            statistics[name].append(float(value.detach()))
                    # Independent optimizer AND clipping budget: critic cannot
                    # suppress the actor via a common gradient clipping scale.
                    value_loss = 0.5 * ((values - target_t[index, begin:end]) ** 2)[mask].mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    norm = nn.utils.clip_grad_norm_(self.model.critic.parameters(), config.gradient_clip, error_if_nonfinite=True)
                    self.critic_optimizer.step()
                    statistics["value_loss"].append(float(value_loss.detach()))
                    statistics["critic_gradient_norm"].append(float(norm))
                if stop and not critic_only:
                    break
        with torch.no_grad():
            logits, post, _ = replay()
            distribution = torch.distributions.Categorical(logits=logits)
            difference = distribution.log_prob(actions) - old_logp
            post_kl = float((difference.exp() - 1 - difference)[valid].mean())
        summary = {name: float(np.mean(values)) if values else 0.0 for name, values in statistics.items()}
        mask = rows["valid"]
        summary.update({"explained_variance_pre": explained_variance(target[mask], rows["values"][mask]),
                        "explained_variance_post": explained_variance(target[mask], post.numpy()[mask]),
                        "target_std": float(target[mask].std()), "prediction_std": float(post.numpy()[mask].std()),
                        "critic_rmse_post": float(np.sqrt(np.mean((target[mask] - post.numpy()[mask]) ** 2))),
                        "replay_ratio_error": ratio_error, "replay_value_error": value_error,
                        "post_update_kl": post_kl, "kl_stopped": stop, "entropy_coefficient": entropy_coef,
                        "mean_reward": float(rows["rewards"][mask].mean()),
                        "turnover_per_decision": float(rows["turnover"][mask].mean()),
                        "sampled_action_share": (np.bincount(rows["actions"][mask], minlength=3) / mask.sum()).tolist(),
                        "liquidations_in_rollout": int(rows["terminals"].sum()),
                        "active_steps": int(mask.sum()), "reward_scale": config.reward_scale})
        self.updates += 1
        summary["update"] = self.updates
        return summary

    def run_update(self):
        return self.update(*self.collect())

    def state_dict(self):
        return {"model": self.model.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(), "environment": self.env.state_dict(),
                "prefix_minutes": self.prefix_minutes, "prefix_portfolio": self.prefix_portfolio,
                "generator_state": self.generator.get_state(), "torch_rng": torch.get_rng_state(),
                "updates": self.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.env.load_state_dict(state["environment"])
        self.prefix_minutes, self.prefix_portfolio = state["prefix_minutes"], state["prefix_portfolio"]
        self.generator.set_state(state["generator_state"])
        torch.set_rng_state(state["torch_rng"])
        self.updates = state["updates"]
