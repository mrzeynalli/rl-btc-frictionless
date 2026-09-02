"""Recurrent PPO over contiguous hourly sequences.

Every minibatch is a whole ``rollout_steps``-hour stretch of history for a
subset of the parallel environments, replayed from the recurrent state that
started it.  Sequences are never shuffled at the step level and are never
resampled to balance market regimes -- the agent sees history in the order it
happened.

Episode length is constrained to a whole number of rollouts, so an episode
boundary can only fall on a rollout boundary and never inside a backpropagated
sequence.  When it does fall there the boundary is treated as a *truncation*:
the value of the state the episode would have continued into is bootstrapped
through it, rather than pretending the world ended.

Four things differ from the ``hourly_eq`` trainer, each a response to a
measured pathology of that study.

*Discount.*  An action's causal reach in this environment is exactly two
steps: it sets the P&L of the interval it opens, and it sets the turnover and
the funding-on-previous of the interval after that.  Nothing it does survives
into a third step, because the position is re-chosen from scratch every hour.
``hourly_eq`` ran ``gamma=0.99, lambda=0.95``, an effective horizon of
``1/(1 - gamma*lambda) = 17`` steps, so every advantage carried fifteen steps
of returns the action provably could not influence -- a measured 2.76x
inflation of advantage noise over a myopic estimator.  The defaults here are
sized to the real credit horizon instead.

*Return scale.*  The value target's scale moves with ``gamma``, so a fixed
``value_coefficient`` means something different at 0.99 than at 0.5.  Rewards
are divided by a running estimate of the discounted-return standard deviation,
which holds the target near unit scale at any discount and makes the value
coefficient comparable across the grid.

*Learning rate.*  ``hourly_eq`` recorded a clip fraction of exactly 0.0000 at
every update of all 320 runs: the policy never moved far enough for the trust
region to engage even once, so the clip was inert and the run was a very slow
random walk.  The rate here is adapted to hit a target KL, which forces the
policy to move at a measurable pace and puts the clip fraction back in a range
where it reports something.

*Critic.*  Value clipping is off by default -- it constrains a critic that was
never the limiting factor -- and the value head carries its own learning-rate
multiplier so it can chase the target faster than the policy moves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from env import VectorHourlyEnvironment
from model import GruActorCritic


@dataclass(frozen=True)
class PPOConfig:
    # A CPU GRU costs almost the same per timestep at batch 4 as at batch 32,
    # so wide-but-few minibatches buy roughly three times the throughput for
    # the same optimisation work.  Thirty-two environments at 128 hours is
    # 4,096 decisions per update.
    environments: int = 32
    rollout_steps: int = 128
    minibatches: int = 2
    epochs: int = 4
    # Sized to the two-step causal reach of an action, not to a long horizon.
    gamma: float = 0.5
    gae_lambda: float = 0.5
    clip_range: float = 0.20
    # ``None`` disables value clipping entirely.
    value_clip_range: float | None = None
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    max_gradient_norm: float = 0.5
    normalise_advantages: bool = True
    # Hold the discounted return near unit scale so ``value_coefficient`` and
    # ``max_gradient_norm`` mean the same thing at every discount factor.
    normalise_returns: bool = True
    # Adapt the rate to a target KL rather than leaving it fixed.
    adaptive_learning_rate: bool = True
    target_kl: float = 0.02
    learning_rate_minimum: float = 1e-5
    learning_rate_maximum: float = 3e-2
    learning_rate_factor: float = 1.5
    # The value head runs at ``multiplier`` times the policy's rate.
    critic_learning_rate_multiplier: float = 3.0

    def as_dict(self) -> dict:
        return asdict(self)


class ReturnScale:
    """Running standard deviation of the *raw* discounted return.

    Rewards are divided by it before advantage estimation, so the value target
    sits near unit scale whatever the discount factor is.  The statistic is
    accumulated from raw rewards -- ``R_t = gamma * R_{t-1} + r_t`` carried
    across updates and reset at episode boundaries -- never from the rewards
    that have already been divided, which would close a feedback loop on
    itself.  It is a pure rescaling: it multiplies every reward in the buffer
    by one identical positive number and so cannot change the sign or the
    ranking of any advantage.
    """

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
        """Accumulate one rollout of raw ``rewards`` shaped ``[steps, envs]``."""
        trace = np.empty_like(rewards, dtype=np.float64)
        for step in range(rewards.shape[0]):
            self.accumulator = self.gamma * self.accumulator + rewards[step]
            trace[step] = self.accumulator
            self.accumulator = np.where(dones[step], 0.0, self.accumulator)
        variance = float(np.var(trace))
        if not np.isfinite(variance) or variance <= 0.0:
            return
        if not self.initialised:
            self.variance = variance
            self.initialised = True
        else:
            self.variance = self.decay * self.variance + (1.0 - self.decay) * variance


class RolloutBuffer:
    def __init__(self, steps: int, environments: int, slow_width: int, bars: int, features: int):
        self.fast = np.zeros((steps, environments, bars, features), dtype=np.float32)
        self.slow = np.zeros((steps, environments, slow_width), dtype=np.float32)
        self.actions = np.zeros((steps, environments), dtype=np.int64)
        self.log_probabilities = np.zeros((steps, environments), dtype=np.float32)
        self.values = np.zeros((steps, environments), dtype=np.float32)
        self.rewards = np.zeros((steps, environments), dtype=np.float32)
        self.dones = np.zeros((steps, environments), dtype=bool)
        self.truncated_values = np.zeros((steps, environments), dtype=np.float32)


class PPOTrainer:
    """Collect-and-update loop with an exact replay invariant on the rollout."""

    def __init__(
        self,
        model: GruActorCritic,
        environment: VectorHourlyEnvironment,
        config: PPOConfig,
        generator: torch.Generator,
    ):
        if environment.episode_hours % config.rollout_steps != 0:
            raise ValueError("episode_hours must be a whole number of rollouts")
        if config.environments % config.minibatches != 0:
            raise ValueError("environments must divide into whole minibatches")
        self.model = model
        self.environment = environment
        self.config = config
        self.generator = generator

        # The value head chases its target faster than the policy moves; the
        # shared trunk follows the policy's rate so the representation is not
        # dragged around by the critic's larger steps.
        critic_parameters = list(model.critic_hidden.parameters()) + list(
            model.critic_output.parameters()
        )
        critic_ids = {id(parameter) for parameter in critic_parameters}
        other_parameters = [p for p in model.parameters() if id(p) not in critic_ids]
        self.optimiser = torch.optim.Adam(
            [
                {"params": other_parameters, "lr": config.learning_rate},
                {
                    "params": critic_parameters,
                    "lr": config.learning_rate * config.critic_learning_rate_multiplier,
                },
            ]
        )
        self.learning_rate = float(config.learning_rate)
        self.return_scale = ReturnScale(config.gamma, config.environments)
        self.hidden = model.initial_hidden(config.environments)
        self.updates_done = 0
        self.steps_done = 0

    def _apply_learning_rate(self) -> None:
        self.optimiser.param_groups[0]["lr"] = self.learning_rate
        self.optimiser.param_groups[1]["lr"] = (
            self.learning_rate * self.config.critic_learning_rate_multiplier
        )

    # -- collection ---------------------------------------------------------

    @torch.no_grad()
    def collect(self) -> tuple[RolloutBuffer, torch.Tensor, dict]:
        config = self.config
        environment = self.environment
        buffer = RolloutBuffer(
            config.rollout_steps, config.environments, environment.slow_width,
            environment.fast.shape[1], environment.fast.shape[2],
        )
        equity_growth: list[float] = []
        turnover_total = 0.0
        action_counts = np.zeros(environment.action_count, dtype=np.int64)

        for step in range(config.rollout_steps):
            fast, slow = environment.observation()
            fast_tensor = torch.from_numpy(fast).unsqueeze(1)
            slow_tensor = torch.from_numpy(slow).unsqueeze(1)
            logits, value, self.hidden = self.model(fast_tensor, slow_tensor, self.hidden)
            distribution = torch.distributions.Categorical(logits=logits[:, 0])
            action = distribution.sample()
            buffer.fast[step] = fast
            buffer.slow[step] = slow
            buffer.actions[step] = action.numpy()
            buffer.log_probabilities[step] = distribution.log_prob(action).numpy()
            buffer.values[step] = value[:, 0].numpy()

            reward, done, info = environment.step(action.numpy())
            buffer.rewards[step] = reward
            buffer.dones[step] = done
            action_counts += np.bincount(action.numpy(), minlength=environment.action_count)
            turnover_total += float(info["turnover"].sum())

            if done.any():
                # The episode is truncated, not terminated: value the state it
                # would have continued into, captured before the reset.
                _, continuation, _ = self.model(
                    torch.from_numpy(info["continuation_fast"]).unsqueeze(1),
                    torch.from_numpy(info["continuation_slow"]).unsqueeze(1),
                    self.hidden,
                )
                buffer.truncated_values[step] = continuation[:, 0].numpy()
                equity_growth.extend(np.log(info["equity"][done]).tolist())
                self.hidden = self.hidden * torch.from_numpy(~done).reshape(1, -1, 1).float()

        fast, slow = environment.observation()
        _, bootstrap, _ = self.model(
            torch.from_numpy(fast).unsqueeze(1), torch.from_numpy(slow).unsqueeze(1), self.hidden
        )
        self.steps_done += config.rollout_steps * config.environments
        statistics = {
            "sampled_action_share": (action_counts / max(action_counts.sum(), 1)).tolist(),
            "episode_log_growth": equity_growth,
            "mean_reward": float(buffer.rewards.mean()),
            "turnover_per_step": turnover_total / (config.rollout_steps * config.environments),
        }
        return buffer, bootstrap[:, 0], statistics

    # -- advantage estimation ----------------------------------------------

    def advantages(self, buffer: RolloutBuffer, bootstrap: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        steps, environments = buffer.rewards.shape
        rewards = buffer.rewards
        if config.normalise_returns:
            rewards = rewards / self.return_scale.scale
        advantages = np.zeros_like(buffer.rewards)
        running = np.zeros(environments, dtype=np.float32)
        next_value = bootstrap.numpy().astype(np.float32)
        for step in reversed(range(steps)):
            done = buffer.dones[step]
            value_after = np.where(done, buffer.truncated_values[step], next_value)
            running = np.where(done, 0.0, running)
            delta = rewards[step] + config.gamma * value_after - buffer.values[step]
            running = delta + config.gamma * config.gae_lambda * running
            advantages[step] = running
            next_value = buffer.values[step]
        returns = advantages + buffer.values
        return advantages, returns

    # -- optimisation -------------------------------------------------------

    def update(self, buffer: RolloutBuffer, bootstrap: torch.Tensor, initial_hidden: torch.Tensor) -> dict:
        config = self.config
        # The scale is refreshed from this rollout's raw rewards before it is
        # used, so even the first update divides by a calibrated number.
        if config.normalise_returns:
            self.return_scale.observe(buffer.rewards, buffer.dones)
        advantages, returns = self.advantages(buffer, bootstrap)

        fast = torch.from_numpy(buffer.fast).permute(1, 0, 2, 3).contiguous()
        slow = torch.from_numpy(buffer.slow).permute(1, 0, 2).contiguous()
        actions = torch.from_numpy(buffer.actions).permute(1, 0).contiguous()
        old_log_probabilities = torch.from_numpy(buffer.log_probabilities).permute(1, 0).contiguous()
        old_values = torch.from_numpy(buffer.values).permute(1, 0).contiguous()
        advantage_tensor = torch.from_numpy(advantages).permute(1, 0).contiguous()
        return_tensor = torch.from_numpy(returns).permute(1, 0).contiguous()
        resets = torch.from_numpy(buffer.dones).permute(1, 0).contiguous()
        # A done recorded at step t resets the state *before* step t+1, and the
        # last one falls outside this rollout, so no sequence carries a reset.
        sequence_resets = torch.zeros_like(resets)
        sequence_resets[:, 1:] = resets[:, :-1]

        per_minibatch = config.environments // config.minibatches
        order = torch.randperm(config.environments, generator=self.generator)
        diagnostics = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "clip_fraction": [], "approximate_kl": [], "gradient_norm": [],
        }
        for _ in range(config.epochs):
            for start in range(0, config.environments, per_minibatch):
                index = order[start : start + per_minibatch]
                logits, values, _ = self.model(
                    fast[index], slow[index], initial_hidden[:, index].contiguous(),
                    sequence_resets[index],
                )
                distribution = torch.distributions.Categorical(logits=logits)
                log_probabilities = distribution.log_prob(actions[index])
                entropy = distribution.entropy().mean()

                batch_advantages = advantage_tensor[index]
                if config.normalise_advantages:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std() + 1e-8
                    )
                ratio = torch.exp(log_probabilities - old_log_probabilities[index])
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                if config.value_clip_range is None:
                    value_loss = 0.5 * ((values - return_tensor[index]) ** 2).mean()
                else:
                    value_clipped = old_values[index] + torch.clamp(
                        values - old_values[index], -config.value_clip_range, config.value_clip_range
                    )
                    value_loss = 0.5 * torch.max(
                        (values - return_tensor[index]) ** 2,
                        (value_clipped - return_tensor[index]) ** 2,
                    ).mean()

                loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
                self.optimiser.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), config.max_gradient_norm)
                self.optimiser.step()

                with torch.no_grad():
                    difference = log_probabilities - old_log_probabilities[index]
                    diagnostics["policy_loss"].append(float(policy_loss))
                    diagnostics["value_loss"].append(float(value_loss))
                    diagnostics["entropy"].append(float(entropy))
                    diagnostics["clip_fraction"].append(
                        float(((ratio - 1.0).abs() > config.clip_range).float().mean())
                    )
                    diagnostics["approximate_kl"].append(float((torch.exp(difference) - 1 - difference).mean()))
                    diagnostics["gradient_norm"].append(float(gradient_norm))

        explained = 1.0 - float(np.var(returns - buffer.values)) / max(float(np.var(returns)), 1e-12)
        summary = {key: float(np.mean(value)) for key, value in diagnostics.items()}
        summary["explained_variance"] = explained
        summary["advantage_std"] = float(advantages.std())
        summary["return_mean"] = float(returns.mean())
        summary["return_std"] = float(returns.std())

        # The rate is adapted after the update, on the KL this update actually
        # produced, so every minibatch above ran at one single known rate.
        summary["learning_rate"] = self.learning_rate
        summary["return_scale"] = self.return_scale.scale
        if config.adaptive_learning_rate:
            observed = summary["approximate_kl"]
            factor = config.learning_rate_factor
            if observed < config.target_kl / factor:
                self.learning_rate = min(self.learning_rate * factor, config.learning_rate_maximum)
            elif observed > config.target_kl * factor:
                self.learning_rate = max(self.learning_rate / factor, config.learning_rate_minimum)
            self._apply_learning_rate()
        return summary

    def run_update(self) -> dict:
        initial_hidden = self.hidden.clone()
        buffer, bootstrap, statistics = self.collect()
        summary = self.update(buffer, bootstrap, initial_hidden)
        self.hidden = self.hidden.detach()
        self.updates_done += 1
        growth = statistics.pop("episode_log_growth")
        summary.update(statistics)
        summary["completed_episodes"] = len(growth)
        summary["episode_log_growth_mean"] = float(np.mean(growth)) if growth else float("nan")
        summary["update"] = self.updates_done
        summary["environment_steps"] = self.steps_done
        return summary
