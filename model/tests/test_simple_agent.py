"""Checks of the simple agent's arithmetic and data timing."""
import numpy as np
import pandas as pd
import torch

from simple_agent.config import Config
from simple_agent.env import Env, differential_sharpe
from simple_agent.model import ActorCritic
from simple_agent.ppo import PPO


def test_differential_sharpe_is_the_derivative_of_the_sharpe_ratio():
    # S(A, B) = A / sqrt(B - A^2); after one step with rate eta, dS/deta at eta = 0 must equal D.
    A, B, R, eta = np.array([0.001]), np.array([0.0001]), np.array([0.02]), 1e-7
    D, dA, dB = differential_sharpe(R, A, B)
    sharpe = lambda a, b: a / np.sqrt(b - a * a)
    numeric = (sharpe(A + eta * dA, B + eta * dB) - sharpe(A, B)) / eta
    assert np.allclose(D, numeric, rtol=1e-4)


def test_env_pays_position_times_next_return_and_wraps():
    features = np.zeros((5, 14), np.float32)
    returns = np.array([0.01, -0.02, 0.03, 0.04, np.nan])
    env = Env(features, returns, 1, 0.01, 1e-4)
    _, R, done = env.step(np.array([2]))          # long at row 0 -> earns +1%
    assert R[0] == 0.01 and not done[0]
    _, R, _ = env.step(np.array([0]))             # short at row 1 -> earns +2%
    assert R[0] == 0.02
    env.step(np.array([1]))                       # flat at row 2
    _, R, done = env.step(np.array([2]))          # row 3 is the last usable row -> wrap to 0, flat
    assert R[0] == 0.04 and done[0] and env.index[0] == 0 and env.position[0] == 0


def test_features_use_no_future_rows():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location('hf', pathlib.Path(__file__).parents[1] / 'data_pipeline/hourly_features.py')
    hf = importlib.util.module_from_spec(spec); spec.loader.exec_module(hf)
    table = pd.read_parquet(hf.OUT)
    # Rebuild from bars that end at a cut: the rows before the cut must be unchanged.
    original = hf.hourly_bars
    cut = pd.Timestamp('2021-03-01', tz='UTC')
    hf.hourly_bars = lambda: original().loc[:cut]
    out, hf.OUT = hf.OUT, pathlib.Path('/tmp/simple_agent_cut.parquet')
    try:
        truncated = hf.build().reset_index()
    finally:
        hf.hourly_bars, hf.OUT = original, out
    full = table[table.time <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(full, truncated[full.columns], check_exact=False, rtol=1e-9)


def test_ppo_update_runs_and_stays_finite():
    torch.manual_seed(0)
    c = Config(envs=4, rollout=16, minibatches=2, epochs=2, hidden=8)
    rng = np.random.default_rng(0)
    env = Env(rng.normal(size=(200, 14)).astype(np.float32), np.r_[rng.normal(0, 0.01, 199), np.nan], 4, c.eta, 1e-4)
    ppo = PPO(ActorCritic(17, 8), env, c)
    for _ in range(3):
        summary = ppo.update()
    assert all(np.isfinite(v) for v in summary.values() if isinstance(v, float))


def test_every_simple_agent_constant_is_documented():
    """ARCHITECTURE.md must name every Config field of the simple agent."""
    from dataclasses import fields
    from pathlib import Path
    import pytest
    path = Path(__file__).parents[1] / 'ARCHITECTURE.md'
    if not path.exists():
        pytest.skip('ARCHITECTURE.md is a local design document and is not distributed')
    text = path.read_text()
    section = text[text.index('## 11. Provenance'):]
    missing = [f.name for f in fields(Config) if f'`{f.name}`' not in section]
    assert not missing, missing


def test_reinforce_update_is_finite():
    """One critic-free update on a synthetic path runs and returns finite statistics."""
    from simple_agent.reinforce import Reinforce
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 5)).astype(np.float32)
    returns = rng.normal(0, 0.01, 400)
    returns[-1] = np.nan
    config = Config(rollout=16, envs=4, hidden=8, algo='reinforce')
    env = Env(features, returns, config.envs, config.eta, float(np.nanvar(returns)))
    agent = Reinforce(ActorCritic(5 + 3, config.hidden), env, config)
    summary = agent.update()
    assert all(np.isfinite(v) for v in summary.values() if isinstance(v, float))
