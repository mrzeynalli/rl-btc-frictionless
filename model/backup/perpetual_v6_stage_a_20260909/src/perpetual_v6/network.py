from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from perpetual.data import FAST_NAMES, SLOW_NAMES
from perpetual.env import PORTFOLIO_WIDTH


def map_hidden(hidden, function):
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(map_hidden(x, function) for x in hidden)
    return function(hidden)


class CalibratedBatchNorm(nn.BatchNorm1d):
    """Fixed training-only moments prevent batch/time leakage and PPO drift."""
    def __init__(self, width):
        super().__init__(width, eps=1e-5, momentum=1., affine=True)
        self.calibrating = False

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        return F.batch_norm(flat, self.running_mean, self.running_var, self.weight,
                            self.bias, self.calibrating, 1., self.eps).reshape(shape)


class Branch(nn.Module):
    def __init__(self, config, actor):
        super().__init__()
        self.config, self.is_actor = config, actor
        recurrent = nn.GRU if config.cell == 'gru' else nn.LSTM
        self.minute = recurrent(len(FAST_NAMES), config.local_width, batch_first=True)
        self.attention = (nn.MultiheadAttention(config.local_width, config.attention_heads,
                                              dropout=0., batch_first=True) if config.attention else None)
        self.market_input = nn.Sequential(nn.Linear(config.local_width + len(SLOW_NAMES), config.hidden), nn.Tanh())
        self.market = recurrent(config.hidden, config.hidden, batch_first=True)
        self.position = recurrent(PORTFOLIO_WIDTH, config.position_hidden, batch_first=True)
        width = config.hidden + config.local_width + len(SLOW_NAMES) + PORTFOLIO_WIDTH + config.position_hidden
        self.fusion = nn.Linear(width, config.head_width)
        self.norm = (nn.LayerNorm(config.head_width) if config.normalization == 'layer' else
                     CalibratedBatchNorm(config.head_width) if config.normalization == 'batch' else nn.Identity())
        self.head = nn.Sequential(nn.Linear(config.head_width, 32), nn.Tanh(), nn.Linear(32, 4))
        for module in (self.minute, self.market, self.position):
            gates = 3 if config.cell == 'gru' else 4
            for name, parameter in module.named_parameters():
                if 'weight_hh' in name:
                    for gate in parameter.chunk(gates):
                        nn.init.orthogonal_(gate)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.zeros_(parameter)
            if config.cell == 'lstm':
                with torch.no_grad():
                    # One unit forget bias in total, not once in both bias tensors.
                    module.bias_ih_l0[module.hidden_size:2*module.hidden_size].fill_(1.)
        nn.init.orthogonal_(self.head[-1].weight, gain=.01 if actor else 1.)
        nn.init.zeros_(self.head[-1].bias)
        if actor:
            probabilities = (config.initial_persistence, config.initial_flat_persistence, config.initial_persistence)
            self.continuation_bias = nn.Parameter(torch.tensor([math.log(p/(1-p)) for p in probabilities]))

    def local(self, fast):
        """Arbitrary leading dimensions; every completed minute is included."""
        shape = fast.shape[:-2]
        sequence, _ = self.minute(fast.reshape(-1, fast.shape[-2], fast.shape[-1]))
        final = sequence[:, -1:]
        if self.attention is not None:
            attended, _ = self.attention(final, sequence, sequence, need_weights=False)
            final = final + attended
        return final[:, 0].reshape(*shape, self.config.local_width)

    def encode_market(self, local, slow, hidden=None):
        return self.market(self.market_input(torch.cat([local, slow], -1)), hidden)

    def decision(self, local, slow, account, market, position_hidden=None, mask=None):
        position, position_hidden = self.position(account, position_hidden)
        joined = torch.cat([market, local, slow, account, position], -1)
        encoded = torch.tanh(self.norm(self.fusion(joined)))
        if mask is not None:
            encoded = encoded * mask
        output = self.head(encoded)
        if self.is_actor:
            current = (account[..., 0].round().long() + 1).clamp(0, 2)
            keep = output[..., :1] + self.continuation_bias[current][..., None]
            alternatives = output[..., 1:].scatter(-1, current[..., None], -torch.inf)
            log_switch = F.logsigmoid(-keep) + F.log_softmax(alternatives, -1)
            output = log_switch.scatter(-1, current[..., None], F.logsigmoid(keep))
        return output, position_hidden

    def forward(self, fast, slow, account, hidden=None, mask=None):
        market_hidden, position_hidden = (None, None) if hidden is None else hidden
        local = self.local(fast)
        market, market_hidden = self.encode_market(local, slow, market_hidden)
        output, position_hidden = self.decision(local, slow, account, market, position_hidden, mask)
        return output, (market_hidden, position_hidden)


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.actor, self.critic = Branch(config, True), Branch(config, False)

    def forward(self, fast, slow, account, hidden=None, masks=None):
        ah, ch = (None, None) if hidden is None else hidden
        am, cm = (None, None) if masks is None else masks
        logp, ah = self.actor(fast, slow, account, ah, am)
        value_q, ch = self.critic(fast, slow, account, ch, cm)
        return logp, value_q[..., 0], value_q[..., 1:], (ah, ch)

    @torch.no_grad()
    def burn(self, fast, slow, account):
        if not fast.shape[1]:
            return None
        # Chunking caps intermediate local-RNN memory during long prefix replay.
        hidden = None
        for start in range(0, fast.shape[1], 64):
            hidden = self(fast[:, start:start+64], slow[:, start:start+64],
                          account[:, start:start+64], hidden)[-1]
        return hidden

    def masks(self, size, generator):
        p = self.config.dropout
        return tuple(((torch.rand(size, 1, self.config.head_width, generator=generator) >= p).float() / (1-p))
                     for _ in range(2))

    @torch.no_grad()
    def calibrate(self, fast, slow, account):
        for branch in (self.actor, self.critic):
            if isinstance(branch.norm, CalibratedBatchNorm):
                branch.norm.calibrating = True
                try:
                    branch(fast, slow, account)
                finally:
                    branch.norm.calibrating = False
