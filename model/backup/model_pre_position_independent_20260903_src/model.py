"""A recurrent actor-critic with controlled critic-encoder ablations.

::

    12 five-minute observations
              |
           GRU(32)
              |
      final hidden state  ->  concatenated with the slow hourly variables
           /        \\
      Actor head   Critic head
      3 logits     state value

The GRU reads the five-minute stream continuously.  Hour ``h`` supplies twelve
consecutive inputs and the hidden state after the twelfth is what the heads
read, so the recurrent state carries across hourly decisions instead of being
rebuilt from nothing every hour.  That is what makes "only the hidden state
evolves causally" meaningful at validation and test time.

The slow variables and the three-way position one-hot enter at the head rather
than the recurrent stream: the position is the agent's own state, so it must be
readable at the decision instant without rewriting twelve bars of history.

Initialisation is standard and nothing else: Xavier into the GRU, orthogonal
recurrent weights, zero biases, small-gain orthogonal on the actor output.

The default is the original shared GRU.  Experiments may instead detach the
critic at the shared representation or give the critic a separately recurrent
encoder without changing the actor architecture.
"""

from __future__ import annotations

import torch
from torch import nn

ACTION_COUNT = 3
BAR_FEATURES = 4
BARS_PER_HOUR = 12


def _initialise_gru(gru: nn.GRU) -> list[dict]:
    hidden = gru.hidden_size
    manifest: list[dict] = []
    with torch.no_grad():
        for name, parameter in gru.named_parameters():
            if name.startswith("weight_ih"):
                for gate, gate_name in enumerate(("reset", "update", "candidate")):
                    nn.init.xavier_uniform_(parameter[gate * hidden : (gate + 1) * hidden])
                    manifest.append({"tensor": name, "gate": gate_name, "method": "xavier_uniform"})
            elif name.startswith("weight_hh"):
                for gate, gate_name in enumerate(("reset", "update", "candidate")):
                    nn.init.orthogonal_(parameter[gate * hidden : (gate + 1) * hidden])
                    manifest.append({"tensor": name, "gate": gate_name, "method": "orthogonal"})
            else:
                nn.init.zeros_(parameter)
                manifest.append({"tensor": name, "method": "zeros"})
    return manifest


def _initialise_linear(layer: nn.Linear, gain: float, manifest: list[dict], label: str) -> None:
    with torch.no_grad():
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
    manifest.append({"tensor": label, "method": "orthogonal", "gain": gain})


class GruActorCritic(nn.Module):
    """Shared recurrent trunk with separate policy and value heads."""

    def __init__(
        self,
        slow_features: int,
        hidden_size: int = 32,
        head_size: int = 32,
        action_count: int = ACTION_COUNT,
        bar_features: int = BAR_FEATURES,
        bars_per_hour: int = BARS_PER_HOUR,
        critic_encoder: str = "shared",
    ):
        super().__init__()
        if critic_encoder not in {"shared", "detached", "separate"}:
            raise ValueError(
                "critic_encoder must be 'shared', 'detached', or 'separate'"
            )
        self.hidden_size = int(hidden_size)
        self.slow_features = int(slow_features)
        self.action_count = int(action_count)
        self.bar_features = int(bar_features)
        self.bars_per_hour = int(bars_per_hour)
        self.critic_encoder = critic_encoder

        self.gru = nn.GRU(self.bar_features, self.hidden_size, batch_first=True)
        head_input = self.hidden_size + self.slow_features
        self.actor_hidden = nn.Linear(head_input, head_size)
        self.actor_output = nn.Linear(head_size, self.action_count)
        self.critic_hidden = nn.Linear(head_input, head_size)
        self.critic_output = nn.Linear(head_size, 1)
        self.activation = nn.Tanh()
        # Constructing a module consumes the global RNG even though all weights
        # below are explicitly reinitialised.  Restore it so common parameters
        # are paired exactly across shared/separate ablations.
        rng_state = torch.random.get_rng_state()
        self.critic_gru = (
            nn.GRU(self.bar_features, self.hidden_size, batch_first=True)
            if critic_encoder == "separate" else None
        )
        torch.random.set_rng_state(rng_state)

        self.initialisation_manifest = _initialise_gru(self.gru)
        _initialise_linear(self.actor_hidden, 2.0**0.5, self.initialisation_manifest, "actor_hidden")
        _initialise_linear(self.actor_output, 0.01, self.initialisation_manifest, "actor_output")
        _initialise_linear(self.critic_hidden, 2.0**0.5, self.initialisation_manifest, "critic_hidden")
        _initialise_linear(self.critic_output, 1.0, self.initialisation_manifest, "critic_output")
        # Initialise the optional extra encoder last so every parameter shared
        # with the smaller variants starts identically under the same seed.
        if self.critic_gru is not None:
            for item in _initialise_gru(self.critic_gru):
                self.initialisation_manifest.append(
                    {**item, "tensor": f"critic_{item['tensor']}"}
                )

    def initial_hidden(self, batch: int, device: torch.device | None = None) -> torch.Tensor:
        encoders = 2 if self.critic_gru is not None else 1
        return torch.zeros(encoders, batch, self.hidden_size, device=device)

    def _run_gru(
        self,
        gru: nn.GRU,
        fast: torch.Tensor,
        hidden: torch.Tensor,
        resets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one recurrent encoder and return its end-of-hour states."""
        batch, hours = fast.shape[0], fast.shape[1]
        stream = fast.reshape(batch, hours * self.bars_per_hour, self.bar_features)
        if resets is None or not bool(resets.any()):
            output, hidden = gru(stream, hidden)
            hourly = output.reshape(
                batch, hours, self.bars_per_hour, self.hidden_size
            )[:, :, -1]
            return hourly, hidden
        collected = []
        for step in range(hours):
            keep = (~resets[:, step]).to(stream.dtype).reshape(1, -1, 1)
            hidden = hidden * keep
            chunk = stream[
                :, step * self.bars_per_hour : (step + 1) * self.bars_per_hour
            ]
            output, hidden = gru(chunk, hidden)
            collected.append(output[:, -1])
        return torch.stack(collected, dim=1), hidden

    def trunk(
        self, fast: torch.Tensor, hidden: torch.Tensor, resets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the GRU over ``fast`` of shape ``[batch, hours, 12, 4]``.

        Returns the hidden state read at the end of every hour, shaped
        ``[batch, hours, hidden]``, and the state carried past the last hour.
        ``resets[b, t]`` zeroes the recurrent state before hour ``t`` of
        sequence ``b``; sequences with no reset take the single fused call.
        """
        return self._run_gru(self.gru, fast, hidden[:1], resets)

    def burn_in(self, fast: torch.Tensor) -> torch.Tensor:
        """Build recurrent state causally from preceding fast observations."""
        batch = fast.shape[0]
        hidden = self.initial_hidden(batch, fast.device)
        if fast.shape[1] == 0:
            return hidden
        _, actor_hidden = self._run_gru(self.gru, fast, hidden[:1])
        if self.critic_gru is None:
            return actor_hidden
        _, critic_hidden = self._run_gru(self.critic_gru, fast, hidden[1:2])
        return torch.cat([actor_hidden, critic_hidden], dim=0)

    def actor_recurrent_parameters(self) -> list[nn.Parameter]:
        """Parameters on which actor/critic gradient conflict is measured."""
        return list(self.gru.parameters())

    def forward(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        hidden: torch.Tensor,
        resets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``fast`` is ``[batch, hours, 12, 4]``, ``slow`` is ``[batch, hours, features]``."""
        actor_hourly, actor_state = self._run_gru(
            self.gru, fast, hidden[:1], resets
        )
        actor_joined = torch.cat([actor_hourly, slow], dim=-1)
        logits = self.actor_output(self.activation(self.actor_hidden(actor_joined)))

        if self.critic_gru is not None:
            critic_hourly, critic_state = self._run_gru(
                self.critic_gru, fast, hidden[1:2], resets
            )
            critic_joined = torch.cat([critic_hourly, slow], dim=-1)
            next_hidden = torch.cat([actor_state, critic_state], dim=0)
        elif self.critic_encoder == "detached":
            critic_joined = torch.cat([actor_hourly.detach(), slow], dim=-1)
            next_hidden = actor_state
        else:
            critic_joined = actor_joined
            next_hidden = actor_state
        value = self.critic_output(
            self.activation(self.critic_hidden(critic_joined))
        ).squeeze(-1)
        return logits, value, next_hidden

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))
