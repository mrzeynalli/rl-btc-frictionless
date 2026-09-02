"""One small GRU, an actor head and a critic head.

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
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.slow_features = int(slow_features)
        self.action_count = int(action_count)
        self.bar_features = int(bar_features)
        self.bars_per_hour = int(bars_per_hour)

        self.gru = nn.GRU(self.bar_features, self.hidden_size, batch_first=True)
        head_input = self.hidden_size + self.slow_features
        self.actor_hidden = nn.Linear(head_input, head_size)
        self.actor_output = nn.Linear(head_size, self.action_count)
        self.critic_hidden = nn.Linear(head_input, head_size)
        self.critic_output = nn.Linear(head_size, 1)
        self.activation = nn.Tanh()

        self.initialisation_manifest = _initialise_gru(self.gru)
        _initialise_linear(self.actor_hidden, 2.0**0.5, self.initialisation_manifest, "actor_hidden")
        _initialise_linear(self.actor_output, 0.01, self.initialisation_manifest, "actor_output")
        _initialise_linear(self.critic_hidden, 2.0**0.5, self.initialisation_manifest, "critic_hidden")
        _initialise_linear(self.critic_output, 1.0, self.initialisation_manifest, "critic_output")

    def initial_hidden(self, batch: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden_size, device=device)

    def trunk(
        self, fast: torch.Tensor, hidden: torch.Tensor, resets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the GRU over ``fast`` of shape ``[batch, hours, 12, 4]``.

        Returns the hidden state read at the end of every hour, shaped
        ``[batch, hours, hidden]``, and the state carried past the last hour.
        ``resets[b, t]`` zeroes the recurrent state before hour ``t`` of
        sequence ``b``; sequences with no reset take the single fused call.
        """
        batch, hours = fast.shape[0], fast.shape[1]
        stream = fast.reshape(batch, hours * self.bars_per_hour, self.bar_features)
        if resets is None or not bool(resets.any()):
            output, hidden = self.gru(stream, hidden)
            hourly = output.reshape(batch, hours, self.bars_per_hour, self.hidden_size)[:, :, -1]
            return hourly, hidden
        collected = []
        for step in range(hours):
            keep = (~resets[:, step]).to(stream.dtype).reshape(1, -1, 1)
            hidden = hidden * keep
            chunk = stream[:, step * self.bars_per_hour : (step + 1) * self.bars_per_hour]
            output, hidden = self.gru(chunk, hidden)
            collected.append(output[:, -1])
        return torch.stack(collected, dim=1), hidden

    def forward(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        hidden: torch.Tensor,
        resets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``fast`` is ``[batch, hours, 12, 4]``, ``slow`` is ``[batch, hours, features]``."""
        hourly, hidden = self.trunk(fast, hidden, resets)
        joined = torch.cat([hourly, slow], dim=-1)
        logits = self.actor_output(self.activation(self.actor_hidden(joined)))
        value = self.critic_output(self.activation(self.critic_hidden(joined))).squeeze(-1)
        return logits, value, hidden

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))
