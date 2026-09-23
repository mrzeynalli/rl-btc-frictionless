"""Separate LSTM actor and LSTM critic.

Each branch:  input (120 market features + 3 position one-hot) -> LSTM(64) -> Linear(64, 64) + tanh -> head
  actor head:  Linear(64, 3) -> logits; pi(a | s) = softmax(logits) over short / flat / long
  critic head: Linear(64, 1) -> V(s), the expected discounted sum of future rewards
The LSTM carries a memory (h, c) from hour to hour; it is zeroed where `done` marks the start of a new path.
"""
import torch
from torch import nn


class Branch(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden)
        self.body = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, outputs)
        self.hidden = hidden

    def initial_state(self, n):
        """Zero memory for n walkers: (h, c), each of shape (1, n, hidden)."""
        return (torch.zeros(1, n, self.hidden), torch.zeros(1, n, self.hidden))

    def forward(self, x, state, done):
        """x: (T, n, inputs), done: (T, n) with 1.0 where a new path starts. Returns (outputs (T, n, k), state)."""
        outputs = []
        for t in range(x.shape[0]):
            keep = (1.0 - done[t]).view(1, -1, 1)                 # forget the memory at a path start
            state = (state[0] * keep, state[1] * keep)
            out, state = self.lstm(x[t:t + 1], state)
            outputs.append(out)
        memory = torch.cat(outputs)                                # (T, n, hidden)
        return self.head(self.body(memory)), state


class ActorCritic(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.actor = Branch(inputs, hidden, 3)
        self.critic = Branch(inputs, hidden, 1)
        # CleanRL initialisation: small policy head, so the first policy is close to uniform.
        nn.init.orthogonal_(self.actor.head.weight, 0.01)
        nn.init.zeros_(self.actor.head.bias)
        nn.init.orthogonal_(self.critic.head.weight, 1.0)
        nn.init.zeros_(self.critic.head.bias)
