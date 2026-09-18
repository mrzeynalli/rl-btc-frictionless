"""Separate actor and critic: exogenous market memory, then a Markov account read-out."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import FEATURE_NAMES
from .ledger import ACTION_NAMES, PORTFOLIO_WIDTH


class Branch(nn.Module):
    """market features -> LSTM memory -> [memory, account] -> two hidden layers.

    The account never enters the recurrent memory. It is Markov-sufficient for
    reward and termination (ledger.py), so the memory is spent on the market
    alone, and one market pass serves every account - which is what lets
    evaluation cache it across sampling seeds. Recurrence rather than a stack of
    past observations follows Hausknecht & Stone (2015).

    Layer normalisation follows every hidden layer, the intervention Lyle et al.
    (ICML 2023) found best at preserving plasticity and added "after each hidden
    layer" of DQN for a robust Atari improvement. Its gain and bias act after
    normalisation and before the non-linearity, as Ba, Kiros & Hinton (2016)
    define it. The recurrent layer is normalised at its output; the in-cell
    LN-LSTM of Ba et al. measured 13 times slower than the fused kernel here.

    Separate policy and value networks, tanh, orthogonal initialisation and a
    final policy layer 100 times smaller follow Andrychowicz et al. (ICLR 2021)
    and CleanRL ppo_atari_lstm; the forget-gate bias of one follows Jozefowicz,
    Zaremba & Sutskever (ICML 2015).
    """

    def __init__(self, config, actor):
        super().__init__()
        widths = config.branch_widths(actor)
        self.hidden, self.head_width, self.is_actor = widths['hidden'], widths['head_width'], actor
        self.encoder = nn.Linear(len(FEATURE_NAMES), self.hidden)
        self.encoder_norm = nn.LayerNorm(self.hidden)
        self.memory = nn.LSTM(self.hidden, self.hidden, batch_first=True)
        self.memory_norm = nn.LayerNorm(self.hidden)
        self.first = nn.Linear(self.hidden + PORTFOLIO_WIDTH, self.head_width)
        self.first_norm = nn.LayerNorm(self.head_width)
        self.second = nn.Linear(self.head_width, self.head_width)
        self.second_norm = nn.LayerNorm(self.head_width)
        self.output = nn.Linear(self.head_width, len(ACTION_NAMES) if actor else 1)
        for layer in (self.encoder, self.first, self.second):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.output.weight, 0.01 if actor else 1.0)
        nn.init.zeros_(self.output.bias)
        for name, parameter in self.memory.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(parameter, 1.0)
            else:
                nn.init.zeros_(parameter)
        with torch.no_grad():
            # One unit of forget bias in total, not once in both bias tensors.
            self.memory.bias_ih_l0[self.hidden:2 * self.hidden].fill_(1.0)

    def encode(self, market, hidden=None):
        memory, hidden = self.memory(torch.tanh(self.encoder_norm(self.encoder(market))), hidden)
        return self.memory_norm(memory), hidden

    def read(self, memory, account):
        """Log-probabilities for the actor; the state value for the critic."""
        x = torch.tanh(self.first_norm(self.first(torch.cat([memory, account], -1))))
        x = torch.tanh(self.second_norm(self.second(x)))
        output = self.output(x)
        return F.log_softmax(output, -1) if self.is_actor else output

    def forward(self, market, account, hidden=None):
        memory, hidden = self.encode(market, hidden)
        return self.read(memory, account), hidden


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.actor, self.critic = Branch(config, True), Branch(config, False)

    def forward(self, market, account, hidden=None):
        actor_hidden, critic_hidden = (None, None) if hidden is None else hidden
        logp, actor_hidden = self.actor(market, account, actor_hidden)
        estimate, critic_hidden = self.critic(market, account, critic_hidden)
        return logp, estimate[..., 0], (actor_hidden, critic_hidden)

    @torch.no_grad()
    def burn(self, market):
        """Recurrent states after a market prefix; no account is needed for it."""
        if not market.shape[1]:
            return None
        return self.actor.encode(market)[1], self.critic.encode(market)[1]


def market_tensor(scaler, market, minutes, config):
    return torch.from_numpy(scaler.observations(market, minutes, config.observation_clip))


def policy_entropy(logp):
    return -(logp.exp() * logp).sum(-1)
