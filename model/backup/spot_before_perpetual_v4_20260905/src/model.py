"""Position-independent recurrent market actor and portfolio-aware critic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from env import PORTFOLIO_WIDTH


BAR_FEATURES = 4
BARS_PER_HOUR = 60
FAST_HIDDEN = 32
SLOW_FEATURES = 10
MARKET_WIDTH = FAST_HIDDEN + SLOW_FEATURES


@dataclass
class RecurrentState:
    actor: torch.Tensor
    critic: torch.Tensor | None = None

    def clone(self) -> "RecurrentState":
        return RecurrentState(
            self.actor.clone(), None if self.critic is None else self.critic.clone()
        )

    def detach(self) -> "RecurrentState":
        return RecurrentState(
            self.actor.detach(), None if self.critic is None else self.critic.detach()
        )

    def select(self, index: torch.Tensor) -> "RecurrentState":
        return RecurrentState(
            self.actor[:, index].contiguous(),
            None if self.critic is None else self.critic[:, index].contiguous(),
        )

    def replace_where(
        self, replacement: "RecurrentState", replace: torch.Tensor
    ) -> "RecurrentState":
        replace = replace.to(dtype=torch.bool, device=self.actor.device)
        actor = self.actor.clone()
        if replacement.actor.shape[1] == self.actor.shape[1]:
            actor[:, replace] = replacement.actor[:, replace]
        else:
            actor[:, replace] = replacement.actor
        if self.critic is None:
            critic = None
        else:
            assert replacement.critic is not None
            critic = self.critic.clone()
            if replacement.critic.shape[1] == self.critic.shape[1]:
                critic[:, replace] = replacement.critic[:, replace]
            else:
                critic[:, replace] = replacement.critic
        return RecurrentState(actor, critic)


def _initialise_recurrent(module: nn.Module, hidden: int, gates: int) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.startswith("weight_ih"):
                for gate in range(gates):
                    nn.init.xavier_uniform_(parameter[gate * hidden : (gate + 1) * hidden])
            elif name.startswith("weight_hh"):
                for gate in range(gates):
                    nn.init.orthogonal_(parameter[gate * hidden : (gate + 1) * hidden])
            else:
                nn.init.zeros_(parameter)


class MarketEncoder(nn.Module):
    def __init__(self, encoder_type: str, input_size: int = BAR_FEATURES, hidden: int = FAST_HIDDEN):
        super().__init__()
        if encoder_type == "gru":
            self.network: nn.GRU | nn.LSTM = nn.GRU(input_size, hidden, batch_first=True)
            gates = 3
        elif encoder_type == "lstm":
            self.network = nn.LSTM(input_size, hidden, batch_first=True)
            gates = 4
        else:
            raise ValueError("encoder_type must be 'gru' or 'lstm'")
        self.encoder_type = encoder_type
        self.hidden = hidden
        _initialise_recurrent(self.network, hidden, gates)

    @property
    def state_layers(self) -> int:
        return 1 if self.encoder_type == "gru" else 2

    def initial_state(self, batch: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(self.state_layers, batch, self.hidden, device=device)

    def _run(self, stream: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_type == "gru":
            output, next_state = self.network(stream, state)
            return output, next_state
        output, (hidden, cell) = self.network(stream, (state[:1], state[1:]))
        return output, torch.cat([hidden, cell], dim=0)

    def forward(
        self,
        fast: torch.Tensor,
        state: torch.Tensor,
        resets: torch.Tensor | None = None,
        reset_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, hours = fast.shape[:2]
        if resets is None:
            stream = fast.reshape(batch, hours * BARS_PER_HOUR, BAR_FEATURES)
            output, next_state = self._run(stream, state)
            hourly = output.reshape(batch, hours, BARS_PER_HOUR, self.hidden)[:, :, -1]
            return hourly, next_state
        if reset_states is not None:
            expected = (self.state_layers, batch, hours, self.hidden)
            if tuple(reset_states.shape) != expected:
                raise ValueError(
                    f"reset state has shape {tuple(reset_states.shape)}, expected {expected}"
                )
        collected = []
        current = state
        for step in range(hours):
            reset = resets[:, step].reshape(1, batch, 1)
            replacement = (
                torch.zeros_like(current)
                if reset_states is None
                else reset_states[:, :, step]
            )
            current = torch.where(reset, replacement, current)
            output, current = self._run(fast[:, step], current)
            collected.append(output[:, -1])
        return torch.stack(collected, dim=1), current


def _initialise_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class MarketActorCritic(nn.Module):
    """Raw BUY/SELL/HOLD market logits never receive portfolio information."""

    def __init__(
        self,
        action_count: int = 3,
        slow_features: int = SLOW_FEATURES,
        encoder_type: str = "gru",
        critic_mode: str = "shared",
        head_size: int = 32,
    ) -> None:
        super().__init__()
        if slow_features != SLOW_FEATURES:
            raise ValueError(f"architecture requires {SLOW_FEATURES} slow features")
        if critic_mode not in {"shared", "detached", "separate"}:
            raise ValueError("critic_mode must be shared, detached, or separate")
        self.action_count = int(action_count)
        self.slow_features = int(slow_features)
        self.encoder_type = encoder_type
        self.critic_mode = critic_mode
        self.actor_encoder = MarketEncoder(encoder_type)
        self.critic_encoder = MarketEncoder(encoder_type) if critic_mode == "separate" else None
        self.actor_hidden = nn.Linear(MARKET_WIDTH, head_size)
        self.actor_output = nn.Linear(head_size, action_count)
        self.critic_hidden = nn.Linear(MARKET_WIDTH + PORTFOLIO_WIDTH, head_size)
        self.critic_output = nn.Linear(head_size, 1)
        self.activation = nn.Tanh()
        _initialise_linear(self.actor_hidden, 2.0**0.5)
        _initialise_linear(self.actor_output, 0.01)
        _initialise_linear(self.critic_hidden, 2.0**0.5)
        _initialise_linear(self.critic_output, 1.0)

    def initial_state(self, batch: int, device: torch.device | None = None) -> RecurrentState:
        return RecurrentState(
            self.actor_encoder.initial_state(batch, device),
            None if self.critic_encoder is None else self.critic_encoder.initial_state(batch, device),
        )

    def _representation(
        self,
        encoder: MarketEncoder,
        fast: torch.Tensor,
        slow: torch.Tensor,
        state: torch.Tensor,
        resets: torch.Tensor | None,
        reset_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, next_state = encoder(fast, state, resets, reset_states)
        return torch.cat([encoded, slow], dim=-1), next_state

    def market_logits(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        state: RecurrentState,
        resets: torch.Tensor | None = None,
        reset_state: RecurrentState | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        market, actor_state = self._representation(
            self.actor_encoder, fast, slow, state.actor, resets,
            None if reset_state is None else reset_state.actor,
        )
        logits = self.actor_output(self.activation(self.actor_hidden(market)))
        return logits, RecurrentState(actor_state, state.critic)

    def forward(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        portfolio: torch.Tensor,
        state: RecurrentState,
        resets: torch.Tensor | None = None,
        reset_state: RecurrentState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        actor_market, actor_state = self._representation(
            self.actor_encoder, fast, slow, state.actor, resets,
            None if reset_state is None else reset_state.actor,
        )
        logits = self.actor_output(self.activation(self.actor_hidden(actor_market)))
        if self.critic_encoder is None:
            critic_market = actor_market.detach() if self.critic_mode == "detached" else actor_market
            critic_state = None
        else:
            assert state.critic is not None
            critic_market, critic_state = self._representation(
                self.critic_encoder, fast, slow, state.critic, resets,
                None if reset_state is None else reset_state.critic,
            )
        critic_input = torch.cat([critic_market, portfolio], dim=-1)
        value = self.critic_output(self.activation(self.critic_hidden(critic_input))).squeeze(-1)
        return logits, value, RecurrentState(actor_state, critic_state)

    def burn_in(self, fast: torch.Tensor) -> RecurrentState:
        state = self.initial_state(fast.shape[0], fast.device)
        if not fast.shape[1]:
            return state
        dummy_slow = torch.zeros(
            fast.shape[0], fast.shape[1], SLOW_FEATURES, device=fast.device
        )
        _, actor_state = self._representation(
            self.actor_encoder, fast, dummy_slow, state.actor, None, None
        )
        critic_state = None
        if self.critic_encoder is not None:
            assert state.critic is not None
            _, critic_state = self._representation(
                self.critic_encoder, fast, dummy_slow, state.critic, None, None
            )
        return RecurrentState(actor_state, critic_state)

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.actor_encoder.parameters()) + list(self.actor_hidden.parameters()) + list(
            self.actor_output.parameters()
        )

    def critic_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.critic_hidden.parameters()) + list(self.critic_output.parameters())
        if self.critic_encoder is not None:
            parameters = list(self.critic_encoder.parameters()) + parameters
        return parameters

    def shared_recurrent_parameters(self) -> list[nn.Parameter]:
        return list(self.actor_encoder.parameters()) if self.critic_mode == "shared" else []

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))
