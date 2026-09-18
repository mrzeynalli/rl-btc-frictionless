"""Recurrent PPO for the long-only Spot BUY/SELL/HOLD environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn

from env import ACTION_COUNT, PORTFOLIO_WIDTH, VectorHourlyEnvironment
from execution import executable_action_logits
from model import MarketActorCritic, RecurrentState


@dataclass(frozen=True)
class PPOConfig:
    environments: int = 32
    rollout_steps: int = 128
    minibatches: int = 2
    actor_epochs: int = 4
    extra_critic_epochs: int = 4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    value_coefficient: float = 1.0
    entropy_coefficient: float = 0.02
    entropy_final_coefficient: float = 0.005
    entropy_anneal_updates: int = 225
    adaptive_entropy: bool = False
    entropy_target_fraction: float = 0.60
    entropy_adaptation_rate: float = 0.05
    entropy_min_coefficient: float = 0.005
    entropy_max_coefficient: float = 0.05
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-4
    max_gradient_norm: float = 0.5
    normalise_advantages: bool = True
    normalise_returns: bool = True
    burn_in_hours: int = 240
    execution_kappa: float = 0.0
    measure_gradient_conflict: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class ReturnScale:
    def __init__(self, gamma: float, environments: int, decay: float = 0.99):
        self.gamma = float(gamma)
        self.decay = float(decay)
        self.accumulator = np.zeros(environments, dtype=np.float64)
        self.variance = 1.0
        self.initialised = False

    @property
    def scale(self) -> float:
        return float(np.sqrt(self.variance) + 1e-8)

    def observe(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        trace = np.empty_like(rewards, dtype=np.float64)
        for step in range(rewards.shape[0]):
            self.accumulator = self.gamma * self.accumulator + rewards[step]
            trace[step] = self.accumulator
            self.accumulator = np.where(dones[step], 0.0, self.accumulator)
        variance = float(np.var(trace))
        if np.isfinite(variance) and variance > 0:
            if not self.initialised:
                self.variance = variance
                self.initialised = True
            else:
                self.variance = self.decay * self.variance + (1 - self.decay) * variance


class RolloutBuffer:
    def __init__(
        self,
        steps: int,
        environments: int,
        slow_width: int,
        fast_shape: tuple[int, int],
        recurrent: RecurrentState,
    ) -> None:
        self.fast = np.zeros((steps, environments, *fast_shape), dtype=np.float32)
        self.slow = np.zeros((steps, environments, slow_width), dtype=np.float32)
        self.portfolio = np.zeros(
            (steps, environments, PORTFOLIO_WIDTH), dtype=np.float32
        )
        self.current_position = np.zeros((steps, environments), dtype=np.float32)
        self.actions = np.zeros((steps, environments), dtype=np.int64)
        self.log_probabilities = np.zeros((steps, environments), dtype=np.float32)
        self.values = np.zeros((steps, environments), dtype=np.float32)
        self.rewards = np.zeros((steps, environments), dtype=np.float32)
        self.dones = np.zeros((steps, environments), dtype=bool)
        self.truncated_values = np.zeros((steps, environments), dtype=np.float32)
        actor_layers, _, actor_hidden = recurrent.actor.shape
        self.reset_actor = np.zeros(
            (steps, actor_layers, environments, actor_hidden), dtype=np.float32
        )
        if recurrent.critic is None:
            self.reset_critic = None
        else:
            critic_layers, _, critic_hidden = recurrent.critic.shape
            self.reset_critic = np.zeros(
                (steps, critic_layers, environments, critic_hidden), dtype=np.float32
            )


def _explained_variance(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.var(target))
    if denominator <= 1e-12:
        return 0.0
    return 1.0 - float(np.var(target - prediction)) / denominator


class PPOTrainer:
    def __init__(
        self,
        model: MarketActorCritic,
        environment: VectorHourlyEnvironment,
        config: PPOConfig,
        generator: torch.Generator,
    ) -> None:
        if config.environments % config.minibatches:
            raise ValueError("environment count must divide into minibatches")
        self.model = model
        self.environment = environment
        self.config = config
        self.generator = generator
        self.actor_parameters = model.actor_parameters()
        self.critic_parameters = model.critic_parameters()
        self.optimiser = torch.optim.Adam([
            {"params": self.actor_parameters, "lr": config.actor_learning_rate},
            {"params": self.critic_parameters, "lr": config.critic_learning_rate},
        ])
        self.return_scale = ReturnScale(config.gamma, config.environments)
        self.hidden = self._burned_hidden()
        self._adaptive_entropy_coefficient = float(config.entropy_coefficient)
        self.updates_done = 0
        self.steps_done = 0

    @torch.no_grad()
    def _burned_hidden(self, mask: np.ndarray | None = None) -> RecurrentState:
        return self.model.burn_in(
            torch.from_numpy(self.environment.burn_in_fast(mask))
        )

    def entropy_coefficient(self) -> float:
        if self.config.adaptive_entropy:
            return self._adaptive_entropy_coefficient
        progress = min(
            self.updates_done / max(self.config.entropy_anneal_updates - 1, 1), 1.0
        )
        return float(
            self.config.entropy_coefficient
            + progress
            * (self.config.entropy_final_coefficient - self.config.entropy_coefficient)
        )

    def _adapt_entropy(self, observed_entropy: float) -> float:
        if not self.config.adaptive_entropy:
            return self.entropy_coefficient()
        target = self.config.entropy_target_fraction * math.log(2.0)
        multiplier = math.exp(
            self.config.entropy_adaptation_rate * (target - observed_entropy)
        )
        self._adaptive_entropy_coefficient = float(np.clip(
            self._adaptive_entropy_coefficient * multiplier,
            self.config.entropy_min_coefficient,
            self.config.entropy_max_coefficient,
        ))
        return self._adaptive_entropy_coefficient

    @torch.no_grad()
    def collect(self) -> tuple[RolloutBuffer, torch.Tensor, dict]:
        env, config = self.environment, self.config
        buffer = RolloutBuffer(
            config.rollout_steps,
            config.environments,
            env.slow_width,
            tuple(env.fast.shape[1:]),
            self.hidden,
        )
        action_counts = np.zeros(ACTION_COUNT, dtype=np.int64)
        turnover_total = 0.0
        episode_growth: list[float] = []
        for step in range(config.rollout_steps):
            fast, slow, portfolio, _ = env.observation()
            current_position = env.position.copy()
            market_logits, value, self.hidden = self.model(
                torch.from_numpy(fast).unsqueeze(1),
                torch.from_numpy(slow).unsqueeze(1),
                torch.from_numpy(portfolio).unsqueeze(1),
                self.hidden,
            )
            execution_logits = executable_action_logits(
                market_logits[:, 0], torch.from_numpy(current_position).float(),
                env.config.cost_rate, config.execution_kappa,
            )
            distribution = torch.distributions.Categorical(logits=execution_logits)
            action = distribution.sample()
            buffer.fast[step] = fast
            buffer.slow[step] = slow
            buffer.portfolio[step] = portfolio
            buffer.current_position[step] = current_position
            buffer.actions[step] = action.numpy()
            buffer.log_probabilities[step] = distribution.log_prob(action).numpy()
            buffer.values[step] = value[:, 0].numpy()
            reward, done, info = env.step(action.numpy())
            buffer.rewards[step] = reward
            buffer.dones[step] = done
            action_counts += np.bincount(action.numpy(), minlength=ACTION_COUNT)
            turnover_total += float(info["turnover"].sum())
            if done.any():
                _, continuation, _ = self.model(
                    torch.from_numpy(info["continuation_fast"]).unsqueeze(1),
                    torch.from_numpy(info["continuation_slow"]).unsqueeze(1),
                    torch.from_numpy(info["continuation_portfolio"]).unsqueeze(1),
                    self.hidden,
                )
                buffer.truncated_values[step] = continuation[:, 0].numpy()
                episode_growth.extend(np.log(info["equity"][done]).tolist())
                burned = self._burned_hidden(done)
                done_tensor = torch.from_numpy(done)
                if step + 1 < config.rollout_steps:
                    indices = np.flatnonzero(done)
                    buffer.reset_actor[step + 1][:, indices] = burned.actor.cpu().numpy()
                    if buffer.reset_critic is not None:
                        assert burned.critic is not None
                        buffer.reset_critic[step + 1][:, indices] = burned.critic.cpu().numpy()
                self.hidden = self.hidden.replace_where(burned, done_tensor)

        fast, slow, portfolio, _ = env.observation()
        _, bootstrap, _ = self.model(
            torch.from_numpy(fast).unsqueeze(1),
            torch.from_numpy(slow).unsqueeze(1),
            torch.from_numpy(portfolio).unsqueeze(1),
            self.hidden,
        )
        self.steps_done += config.rollout_steps * config.environments
        observations = config.rollout_steps * config.environments
        return buffer, bootstrap[:, 0], {
            "sampled_action_share": (action_counts / max(action_counts.sum(), 1)).tolist(),
            "episode_log_growth": episode_growth,
            "mean_reward": float(buffer.rewards.mean()),
            "turnover_per_step": turnover_total / observations,
            "starting_position_counts": env.start_position_counts.tolist(),
            "episode_length_counts": dict(env.episode_length_counts),
            "regime_sample_counts": env.regime_sample_counts.tolist(),
        }

    def advantages(
        self, buffer: RolloutBuffer, bootstrap: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        rewards = buffer.rewards
        if self.config.normalise_returns:
            rewards = rewards / self.return_scale.scale
        advantages = np.zeros_like(buffer.rewards)
        running = np.zeros(buffer.rewards.shape[1], dtype=np.float32)
        next_value = bootstrap.numpy().astype(np.float32)
        for step in reversed(range(buffer.rewards.shape[0])):
            done = buffer.dones[step]
            value_after = np.where(done, buffer.truncated_values[step], next_value)
            running = np.where(done, 0.0, running)
            delta = rewards[step] + self.config.gamma * value_after - buffer.values[step]
            running = delta + self.config.gamma * self.config.gae_lambda * running
            advantages[step] = running
            next_value = buffer.values[step]
        return advantages, advantages + buffer.values

    def _tensors(self, buffer: RolloutBuffer) -> dict[str, torch.Tensor | None]:
        tensors: dict[str, torch.Tensor | None] = {
            "fast": torch.from_numpy(buffer.fast).permute(1, 0, 2, 3).contiguous(),
            "slow": torch.from_numpy(buffer.slow).permute(1, 0, 2).contiguous(),
            "portfolio": torch.from_numpy(buffer.portfolio).permute(1, 0, 2).contiguous(),
            "position": torch.from_numpy(buffer.current_position).T.contiguous(),
            "actions": torch.from_numpy(buffer.actions).T.contiguous(),
            "old_logp": torch.from_numpy(buffer.log_probabilities).T.contiguous(),
            "dones": torch.from_numpy(buffer.dones).T.contiguous(),
            "reset_actor": torch.from_numpy(buffer.reset_actor).permute(1, 2, 0, 3).contiguous(),
            "reset_critic": None,
        }
        if buffer.reset_critic is not None:
            tensors["reset_critic"] = torch.from_numpy(buffer.reset_critic).permute(
                1, 2, 0, 3
            ).contiguous()
        return tensors

    def update(
        self, buffer: RolloutBuffer, bootstrap: torch.Tensor, initial_state: RecurrentState
    ) -> dict:
        config = self.config
        if config.normalise_returns:
            self.return_scale.observe(buffer.rewards, buffer.dones)
        advantages, returns = self.advantages(buffer, bootstrap)
        data = self._tensors(buffer)
        assert all(data[key] is not None for key in (
            "fast", "slow", "portfolio", "position", "actions", "old_logp", "dones",
            "reset_actor",
        ))
        resets = torch.zeros_like(data["dones"])
        resets[:, 1:] = data["dones"][:, :-1]
        reset_state = RecurrentState(data["reset_actor"], data["reset_critic"])
        advantage_tensor = torch.from_numpy(advantages).T.contiguous()
        return_tensor = torch.from_numpy(returns).T.contiguous()
        per_minibatch = config.environments // config.minibatches
        diagnostics = {key: [] for key in (
            "policy_loss", "value_loss", "entropy", "clip_fraction",
            "approximate_kl", "gradient_norm",
        )}
        entropy_coefficient = self.entropy_coefficient()
        recurrent_gradient_cosine = None

        with torch.no_grad():
            replay_logits, replay_values, _ = self.model(
                data["fast"], data["slow"], data["portfolio"], initial_state,
                resets, reset_state,
            )
            replay_executable = executable_action_logits(
                replay_logits, data["position"], self.environment.config.cost_rate,
                config.execution_kappa,
            )
            replay_logp = torch.distributions.Categorical(
                logits=replay_executable
            ).log_prob(data["actions"])
            replay_ratio_error = float(
                (torch.exp(replay_logp - data["old_logp"]) - 1.0).abs().max()
            )
            replay_value_error = float(
                (replay_values.T - torch.from_numpy(buffer.values)).abs().max()
            )
        if replay_ratio_error > 2e-5 or replay_value_error > 2e-5:
            raise RuntimeError(
                "recurrent replay invariant failed: "
                f"ratio_error={replay_ratio_error:.3e}, "
                f"value_error={replay_value_error:.3e}"
            )

        for epoch in range(config.actor_epochs):
            order = torch.randperm(config.environments, generator=self.generator)
            for start in range(0, config.environments, per_minibatch):
                index = order[start : start + per_minibatch]
                logits, values, _ = self.model(
                    data["fast"][index], data["slow"][index], data["portfolio"][index],
                    initial_state.select(index), resets[index], reset_state.select(index),
                )
                executable = executable_action_logits(
                    logits, data["position"][index], self.environment.config.cost_rate,
                    config.execution_kappa,
                )
                distribution = torch.distributions.Categorical(logits=executable)
                logp = distribution.log_prob(data["actions"][index])
                entropy = distribution.entropy().mean()
                batch_advantage = advantage_tensor[index]
                if config.normalise_advantages:
                    batch_advantage = (batch_advantage - batch_advantage.mean()) / (
                        batch_advantage.std() + 1e-8
                    )
                ratio = torch.exp(logp - data["old_logp"][index])
                policy_loss = -torch.minimum(
                    ratio * batch_advantage,
                    torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range)
                    * batch_advantage,
                ).mean()
                value_loss = 0.5 * ((values - return_tensor[index]) ** 2).mean()
                actor_loss = policy_loss - entropy_coefficient * entropy
                critic_loss = config.value_coefficient * value_loss
                if (
                    config.measure_gradient_conflict
                    and self.model.critic_mode == "shared"
                    and recurrent_gradient_cosine is None
                    and epoch == 0
                ):
                    recurrent = self.model.shared_recurrent_parameters()
                    actor_gradients = torch.autograd.grad(
                        actor_loss, recurrent, retain_graph=True, allow_unused=True
                    )
                    critic_gradients = torch.autograd.grad(
                        critic_loss, recurrent, retain_graph=True, allow_unused=True
                    )
                    actor_vector = torch.cat([item.reshape(-1) for item in actor_gradients if item is not None])
                    critic_vector = torch.cat([item.reshape(-1) for item in critic_gradients if item is not None])
                    denominator = actor_vector.norm() * critic_vector.norm()
                    if float(denominator) > 0:
                        recurrent_gradient_cosine = float(
                            torch.dot(actor_vector, critic_vector) / denominator
                        )
                self.optimiser.zero_grad(set_to_none=True)
                (actor_loss + critic_loss).backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.max_gradient_norm
                )
                self.optimiser.step()
                with torch.no_grad():
                    difference = logp - data["old_logp"][index]
                    diagnostics["policy_loss"].append(float(policy_loss))
                    diagnostics["value_loss"].append(float(value_loss))
                    diagnostics["entropy"].append(float(entropy))
                    diagnostics["clip_fraction"].append(float(
                        ((ratio - 1).abs() > config.clip_range).float().mean()
                    ))
                    diagnostics["approximate_kl"].append(float(
                        (torch.exp(difference) - 1 - difference).mean()
                    ))
                    diagnostics["gradient_norm"].append(float(gradient_norm))

        for _ in range(config.extra_critic_epochs):
            order = torch.randperm(config.environments, generator=self.generator)
            for start in range(0, config.environments, per_minibatch):
                index = order[start : start + per_minibatch]
                _, values, _ = self.model(
                    data["fast"][index], data["slow"][index], data["portfolio"][index],
                    initial_state.select(index), resets[index], reset_state.select(index),
                )
                value_loss = 0.5 * ((values - return_tensor[index]) ** 2).mean()
                self.optimiser.zero_grad(set_to_none=True)
                value_loss.backward()
                for parameter in self.actor_parameters:
                    parameter.grad = None
                nn.utils.clip_grad_norm_(self.critic_parameters, config.max_gradient_norm)
                self.optimiser.step()
                diagnostics["value_loss"].append(float(value_loss.detach()))

        with torch.no_grad():
            _, post_values, _ = self.model(
                data["fast"], data["slow"], data["portfolio"], initial_state, resets
                , reset_state
            )
        target = returns
        pre = buffer.values
        post = post_values.T.cpu().numpy()
        summary = {key: float(np.mean(items)) for key, items in diagnostics.items() if items}
        next_entropy_coefficient = self._adapt_entropy(summary.get("entropy", 0.0))
        summary.update({
            "explained_variance_pre_update": _explained_variance(target, pre),
            "explained_variance_post_update": _explained_variance(target, post),
            "critic_prediction_std_pre": float(pre.std()),
            "critic_prediction_std_post": float(post.std()),
            "advantage_std": float(advantages.std()),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "entropy_coefficient": entropy_coefficient,
            "entropy_coefficient_next": next_entropy_coefficient,
            "entropy_target": self.config.entropy_target_fraction * math.log(2.0),
            "replay_ratio_max_abs_error": replay_ratio_error,
            "replay_value_max_abs_error": replay_value_error,
            "recurrent_gradient_cosine": recurrent_gradient_cosine,
            "actor_learning_rate": config.actor_learning_rate,
            "critic_learning_rate": config.critic_learning_rate,
            "return_scale": self.return_scale.scale,
        })
        return summary

    def run_update(self) -> dict:
        initial_state = self.hidden.clone()
        buffer, bootstrap, statistics = self.collect()
        summary = self.update(buffer, bootstrap, initial_state)
        self.hidden = self.hidden.detach()
        self.updates_done += 1
        growth = statistics.pop("episode_log_growth")
        summary.update(statistics)
        summary["completed_episodes"] = len(growth)
        summary["episode_log_growth_mean"] = float(np.mean(growth)) if growth else float("nan")
        summary["update"] = self.updates_done
        summary["environment_steps"] = self.steps_done
        return summary
