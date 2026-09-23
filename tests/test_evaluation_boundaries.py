"""Final reporting must cover year boundaries without observing the following year's features."""
import numpy as np
import torch

from core.evaluate import run_policy


class LongActor:
    def __init__(self):
        self.seen = []

    def initial_state(self, n):
        return None

    def __call__(self, observation, state, reset):
        self.seen.append(float(observation[0, 0, 0]))
        return torch.tensor([[[0.0, 0.0, 1.0]]]), None


def test_boundary_return_is_settled_without_reading_future_features():
    from types import SimpleNamespace
    model = SimpleNamespace(actor=LongActor())
    features = np.arange(4, dtype=np.float32)[:, None]
    returns = np.array([0.01, 0.02, -0.03, np.nan])
    _, positions, result, _ = run_policy(model, features, returns, np.array([0, 1]), 0,
                                         include_last=True)
    np.testing.assert_allclose(result, [0.01, 0.02])
    assert model.actor.seen == [0, 1]
    _, _, following, _ = run_policy(model, features, returns, np.array([2, 3]), 0,
                                    include_last=True)
    np.testing.assert_allclose(following, [-0.03])
    assert model.actor.seen == [0, 1, 2]
    assert len(result) + len(following) == len(returns) - 1


def test_selection_keeps_original_within_period_convention():
    from types import SimpleNamespace
    model = SimpleNamespace(actor=LongActor())
    _, _, result, _ = run_policy(model, np.zeros((3, 1), np.float32),
                                 np.array([0.01, 0.02, np.nan]), np.array([0, 1]), 0)
    np.testing.assert_allclose(result, [0.01])
