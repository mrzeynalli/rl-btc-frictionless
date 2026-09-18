from __future__ import annotations

import torch
from torch import nn

from .data import FAST_NAMES, SLOW_NAMES
from .env import PORTFOLIO_WIDTH


class Branch(nn.Module):
    def __init__(self, config, output):
        super().__init__()
        width = config.local_width
        self.local = nn.Sequential(nn.Conv1d(len(FAST_NAMES), 16, 5, stride=2), nn.Tanh(),
                                   nn.Conv1d(16, width, 3, stride=2), nn.Tanh(),
                                   nn.AdaptiveAvgPool1d(1))
        self.input = nn.Sequential(nn.Linear(width + len(SLOW_NAMES) + PORTFOLIO_WIDTH, config.hidden), nn.Tanh())
        self.gru = nn.GRU(config.hidden, config.hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(config.hidden, 32), nn.Tanh(), nn.Linear(32, output))
        for name, parameter in self.gru.named_parameters():
            if "weight_hh" in name:
                for gate in parameter.chunk(3):
                    nn.init.orthogonal_(gate)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01 if output == 3 else 1.0)
        nn.init.zeros_(self.head[-1].bias)

    def features(self, fast, slow, portfolio):
        batch, steps, minutes, channels = fast.shape
        local = self.local(fast.reshape(batch * steps, minutes, channels).transpose(1, 2))
        joined = torch.cat([local.reshape(batch, steps, -1), slow, portfolio], dim=-1)
        return self.input(joined)

    def forward(self, fast, slow, portfolio, hidden=None):
        encoded = self.features(fast, slow, portfolio)
        sequence, hidden = self.gru(encoded, hidden)
        return self.head(sequence), hidden


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.actor = Branch(config, 3)
        self.critic = Branch(config, 1)

    def forward(self, fast, slow, portfolio, hidden=(None, None)):
        logits, actor = self.actor(fast, slow, portfolio, hidden[0])
        values, critic = self.critic(fast, slow, portfolio, hidden[1])
        return logits, values.squeeze(-1), (actor, critic)

    @torch.no_grad()
    def burn(self, fast, slow, portfolio):
        if fast.shape[1] == 0:
            return None, None
        return self(fast, slow, portfolio)[2]


def tensors(scaler, market, minute, portfolio, config):
    fast, slow = scaler.observations(market, minute, config.fast_minutes)
    return torch.from_numpy(fast), torch.from_numpy(slow), torch.from_numpy(portfolio)
