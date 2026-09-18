"""Inspectable execution and ensemble rules applied after market scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from env import BUY, SELL, valid_action_mask


def executable_action_logits(
    market_scores: torch.Tensor,
    current_position: torch.Tensor,
    cost_rate: float = 0.0,
    kappa: float = 0.0,
) -> torch.Tensor:
    """Apply Spot inventory validity and optional explicit switching cost."""
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    mask = torch.from_numpy(
        valid_action_mask(current_position.detach().cpu().numpy())
    ).to(device=market_scores.device)
    adjusted = market_scores.clone()
    if cost_rate and kappa:
        transition = torch.zeros_like(adjusted)
        transition[..., BUY] = (current_position == 0).to(adjusted.dtype)
        transition[..., SELL] = (current_position == 1).to(adjusted.dtype)
        adjusted = adjusted - float(kappa) * float(cost_rate) * transition
    return adjusted.masked_fill(~mask, -torch.inf)


def select_actions(
    market_scores: torch.Tensor,
    current_position: torch.Tensor,
    cost_rate: float = 0.0,
    kappa: float = 0.0,
) -> torch.Tensor:
    return torch.argmax(
        executable_action_logits(market_scores, current_position, cost_rate, kappa),
        dim=-1,
    )


def aggregate_market_scores(seed_scores: torch.Tensor, method: str) -> torch.Tensor:
    """Combine seed opinions before any account-position term is introduced."""
    if seed_scores.ndim < 2:
        raise ValueError("seed_scores must have seed and action dimensions")
    if method == "mean":
        return seed_scores.mean(dim=0)
    if method != "majority":
        raise KeyError(f"unknown aggregation method {method!r}")
    preferred = seed_scores.argmax(dim=-1)
    action_count = seed_scores.shape[-1]
    votes = torch.nn.functional.one_hot(preferred, action_count).to(seed_scores.dtype)
    vote_share = votes.mean(dim=0)
    # A tiny, deterministic tie-break preserves the ordering of mean market
    # opinion while leaving actual majority decisions unchanged.
    mean = seed_scores.mean(dim=0)
    scale = mean.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    return vote_share + 1e-7 * mean / scale


def ensemble_agreement(seed_scores: np.ndarray) -> dict:
    """Agreement and score dispersion, based only on seed market opinions."""
    preferred = np.argmax(seed_scores, axis=-1)
    seeds = preferred.shape[0]
    flattened = preferred.reshape(seeds, -1)
    counts = np.stack(
        [(flattened == action).sum(axis=0) for action in range(seed_scores.shape[-1])],
        axis=-1,
    )
    maximum = counts.max(axis=-1)
    pairwise = [
        np.mean(flattened[left] == flattened[right])
        for left in range(seeds)
        for right in range(left + 1, seeds)
    ]
    return {
        "all_five_agreement": float(np.mean(maximum == seeds)),
        "four_of_five_or_better": float(np.mean(maximum >= seeds - 1)),
        "mean_pairwise_agreement": float(np.mean(pairwise)),
        "mean_seed_score_std": float(np.std(seed_scores, axis=0).mean()),
    }


def legacy_probability_hysteresis(
    probabilities: torch.Tensor,
    current_index: torch.Tensor,
    margin: float = 0.10,
) -> torch.Tensor:
    """Old comparison rule; intentionally excluded from the primary policy."""
    best = probabilities.argmax(dim=-1)
    current_probability = probabilities.gather(-1, current_index.unsqueeze(-1)).squeeze(-1)
    improvement = probabilities.gather(-1, best.unsqueeze(-1)).squeeze(-1) - current_probability
    return torch.where((best != current_index) & (improvement <= margin), current_index, best)


@dataclass
class CausalExpertWeights:
    """Optional daily-updated, shrunk exponential weighting of shadow seeds."""

    seeds: int
    decay: float = 0.99
    temperature: float = 10.0
    shrinkage: float = 0.25
    weight_floor: float = 0.05
    update_interval_hours: int = 24

    def __post_init__(self) -> None:
        if not 0 <= self.decay < 1:
            raise ValueError("decay must lie in [0, 1)")
        if self.temperature <= 0 or not 0 <= self.shrinkage <= 1:
            raise ValueError("invalid expert-weight parameters")
        if self.weight_floor < 0 or self.weight_floor * self.seeds >= 1:
            raise ValueError("weight floor leaves no probability mass")
        self.scores = np.zeros(self.seeds, dtype=np.float64)
        self._hours = 0
        self.weights = np.full(self.seeds, 1.0 / self.seeds)

    def observe_previous_returns(self, realised_returns: np.ndarray) -> np.ndarray:
        """Consume only already-realised shadow returns and update on schedule."""
        realised = np.asarray(realised_returns, dtype=np.float64)
        if realised.shape != (self.seeds,):
            raise ValueError(f"expected one realised return for each of {self.seeds} seeds")
        # Update S_t = rho*S_(t-1) + r_(t-1) every hour. Only publishing
        # new weights is delayed to the configured daily-style schedule.
        self.scores = self.decay * self.scores + realised
        self._hours += 1
        if self._hours % self.update_interval_hours:
            return self.weights.copy()
        centered = self.temperature * (self.scores - self.scores.max())
        raw = np.exp(centered)
        raw /= raw.sum()
        equal = np.full(self.seeds, 1.0 / self.seeds)
        shrunk = (1.0 - self.shrinkage) * raw + self.shrinkage * equal
        residual = 1.0 - self.seeds * self.weight_floor
        self.weights = self.weight_floor + residual * shrunk
        self.weights /= self.weights.sum()
        return self.weights.copy()
