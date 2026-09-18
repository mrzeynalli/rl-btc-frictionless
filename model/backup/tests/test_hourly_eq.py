"""Invariants for the hourly log-equity study.

These are the properties that, if they broke silently, would make every number
in the study wrong in a way no plot would reveal: look-ahead in the state,
funding paid by the wrong position, costs charged on decisions instead of
executed turnover, a normaliser that has seen the future, an evaluation that is
not causal, or a PPO update whose replay does not reproduce what it sampled.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cryptorl.hourly_eq import data as data_module
from cryptorl.hourly_eq.env import (
    ONE_WAY_COST, EnvironmentConfig, VectorHourlyEnvironment, realised_return,
)
from cryptorl.hourly_eq.evaluation import (
    always_flat, always_long, circular_shift_null, metrics, run_policy,
)
from cryptorl.hourly_eq.features import (
    ALL_SPLITS, EVALUATION_WARMUP_HOURS, FINAL_SPLIT, WALK_FORWARD_SPLITS,
    build_market, build_split, fit_normaliser,
)
from cryptorl.hourly_eq.model import GruActorCritic
from cryptorl.hourly_eq.ppo import PPOConfig, PPOTrainer


@pytest.fixture(scope="module")
def market():
    return build_market()


@pytest.fixture(scope="module")
def split(market):
    return build_split(market, WALK_FORWARD_SPLITS[0])


# -- data -------------------------------------------------------------------


def test_dataset_manifest_passed_its_own_audit():
    manifest = json.loads((data_module.OUTPUT_DIR / "manifest.json").read_text())
    assert manifest["audit_passed"] is True
    assert manifest["cross_asset_data_used"] is False
    assert manifest["symbol"] == "BTCUSDT"
    assert manifest["warmup_days"] >= 10.0
    assert manifest["native_archive_check"]["agrees_with_native_archive"] is True


def test_only_btcusdt_is_read():
    for path in (data_module.SOURCE_MINUTE, data_module.SOURCE_FUNDING):
        assert "BTCUSDT" in path.name


def test_five_minute_grid_is_contiguous():
    bars, _ = data_module.load()
    deltas = bars["open_time"].diff().dropna().dt.total_seconds().unique()
    assert list(deltas) == [300.0]


# -- state timing -----------------------------------------------------------


def test_features_close_strictly_before_execution(market):
    assert (market.decision_time < market.execution_time).all()
    # The order fills at the next five-minute open, one bar after the close.
    gap = (market.execution_time - market.decision_time).total_seconds()
    assert np.allclose(gap, 0.001)


def test_holding_interval_is_exactly_one_hour(market):
    step = market.execution_time.to_series().diff().dropna().dt.total_seconds()
    assert step.between(3599.0, 3601.0).all()


def test_state_carries_no_price_level_or_calendar_variable(market):
    from cryptorl.hourly_eq.features import FAST_FEATURE_NAMES, SLOW_FEATURE_NAMES

    names = set(FAST_FEATURE_NAMES) | set(SLOW_FEATURE_NAMES)
    for forbidden in ("price", "hour", "weekday", "month", "regime", "countdown",
                      "unrealised", "exposure", "rsi", "macd"):
        assert not any(forbidden in name for name in names)
    assert len(FAST_FEATURE_NAMES) == 4
    assert len(SLOW_FEATURE_NAMES) == 7


def test_last_settled_funding_rate_is_already_settled(market):
    _, funding = data_module.load()
    times = pd.DatetimeIndex(funding["funding_time"])
    rates = funding["funding_rate"].to_numpy()
    for index in (0, 5000, 20000, len(market) - 1):
        execution = market.execution_time[index]
        settled = rates[: int(times.searchsorted(execution, "left"))]
        # The seventh slow feature is the last settled rate, unnormalised here.
        assert market.slow[index, 6] == pytest.approx(settled[-1])


# -- funding attribution ----------------------------------------------------


def test_every_settlement_is_attributed_exactly_once(market):
    _, funding = data_module.load()
    covered = market.funding_on_previous.sum() + market.funding_on_current.sum()
    first, last = market.execution_time[0], market.execution_time[-1]
    inside = funding[(funding["funding_time"] >= first) & (funding["funding_time"] < last)]
    assert covered == pytest.approx(inside["funding_rate"].sum(), rel=1e-9, abs=1e-12)


def test_settlement_at_execution_instant_belongs_to_the_previous_position(market):
    index = int(np.flatnonzero(market.funding_on_previous != 0.0)[0])
    rate = market.funding_on_previous[index]
    # A long held into the settlement pays a positive rate; the new position does not.
    _, funding_previous, _, _ = realised_return(
        np.array([0.0]), np.array([1.0]), np.array([1.0]),
        np.array([rate]), np.array([0.0]), 0.0,
    )
    _, funding_new, _, _ = realised_return(
        np.array([1.0]), np.array([0.0]), np.array([1.0]),
        np.array([rate]), np.array([0.0]), 0.0,
    )
    assert funding_previous[0] == pytest.approx(-rate)
    assert funding_new[0] == 0.0


def test_no_settlement_in_the_interval_means_exactly_zero_funding(market):
    index = int(np.flatnonzero(
        (market.funding_on_previous == 0.0) & (market.funding_on_current == 0.0)
    )[0])
    _, funding, _, _ = realised_return(
        np.array([1.0]), np.array([-1.0]), np.array([1.01]),
        market.funding_on_previous[index : index + 1],
        market.funding_on_current[index : index + 1], 0.0,
    )
    assert funding[0] == 0.0


def test_positive_funding_costs_a_long_and_pays_a_short():
    rate = np.array([0.0001])
    _, long_funding, _, _ = realised_return(
        np.array([0.0]), np.array([1.0]), np.array([1.0]), rate, np.array([0.0]), 0.0
    )
    _, short_funding, _, _ = realised_return(
        np.array([0.0]), np.array([-1.0]), np.array([1.0]), rate, np.array([0.0]), 0.0
    )
    _, flat_funding, _, _ = realised_return(
        np.array([0.0]), np.array([0.0]), np.array([1.0]), rate, np.array([0.0]), 0.0
    )
    assert long_funding[0] < 0 < short_funding[0]
    assert flat_funding[0] == 0.0


# -- cost accounting --------------------------------------------------------


@pytest.mark.parametrize(
    "previous, target, expected",
    [
        (0.0, 1.0, ONE_WAY_COST), (1.0, 0.0, ONE_WAY_COST),
        (0.0, -1.0, ONE_WAY_COST), (-1.0, 0.0, ONE_WAY_COST),
        (1.0, -1.0, 2 * ONE_WAY_COST), (-1.0, 1.0, 2 * ONE_WAY_COST),
        (1.0, 1.0, 0.0), (-1.0, -1.0, 0.0), (0.0, 0.0, 0.0),
    ],
)
def test_cost_is_charged_on_executed_turnover(previous, target, expected):
    _, _, cost, _ = realised_return(
        np.array([target]), np.array([previous]), np.array([1.0]),
        np.array([0.0]), np.array([0.0]), ONE_WAY_COST,
    )
    assert cost[0] == pytest.approx(expected)


def test_holding_the_same_position_costs_nothing_however_long(market):
    positions = np.ones(500)
    trajectory = always_long(build_split(market, WALK_FORWARD_SPLITS[0]), 20000, 20500, ONE_WAY_COST)
    assert trajectory.cost_paid[0] == pytest.approx(ONE_WAY_COST)
    assert trajectory.cost_paid[1:].sum() == 0.0
    assert len(positions) == len(trajectory.positions)


# -- reward -----------------------------------------------------------------


def test_reward_is_exactly_the_log_equity_ratio(split, market):
    environment = VectorHourlyEnvironment(
        split.fast, split.slow, market.entry_price, market.exit_price,
        market.funding_on_previous, market.funding_on_current,
        split.train_range, 4, EnvironmentConfig(episode_hours=128), np.random.default_rng(0),
    )
    previous_equity = environment.equity.copy()
    for _ in range(50):
        actions = np.random.default_rng(0).integers(0, 3, size=4)
        reward, _, info = environment.step(actions)
        expected = np.log(info["equity"] / previous_equity) * EnvironmentConfig().reward_scale
        assert np.allclose(reward, expected, atol=1e-12)
        previous_equity = info["equity"]


def test_reward_scale_is_one_constant_for_every_step(split, market):
    def total(scale: float) -> np.ndarray:
        environment = VectorHourlyEnvironment(
            split.fast, split.slow, market.entry_price, market.exit_price,
            market.funding_on_previous, market.funding_on_current,
            split.train_range, 2,
            EnvironmentConfig(episode_hours=128, reward_scale=scale), np.random.default_rng(7),
        )
        rewards = []
        for step in range(30):
            rewards.append(environment.step(np.full(2, step % 3))[0])
        return np.array(rewards)

    assert np.allclose(total(1.0) * 100.0, total(100.0), atol=1e-9)


def test_zero_cost_arm_charges_nothing_but_still_records_the_hypothetical(split, market):
    environment = VectorHourlyEnvironment(
        split.fast, split.slow, market.entry_price, market.exit_price,
        market.funding_on_previous, market.funding_on_current,
        split.train_range, 2,
        EnvironmentConfig(episode_hours=128, cost_rate=0.0), np.random.default_rng(1),
    )
    _, _, info = environment.step(np.array([2, 0]))
    assert info["cost_paid"].sum() == 0.0
    assert info["hypothetical_cost"].sum() == pytest.approx(2 * ONE_WAY_COST)


# -- normalisation ----------------------------------------------------------


def test_normaliser_sees_only_its_own_training_window(market):
    for split_definition in ALL_SPLITS:
        normaliser = fit_normaliser(market, split_definition)
        first, last = market.index_range(split_definition.train_start, split_definition.train_end)
        expected = market.slow[first:last].mean(axis=0)
        assert np.allclose(normaliser.slow_mean, expected)
        assert normaliser.fitted_on["end_exclusive"] == split_definition.train_end.isoformat()


def test_each_fold_gets_its_own_normaliser(market):
    first = fit_normaliser(market, WALK_FORWARD_SPLITS[0])
    last = fit_normaliser(market, FINAL_SPLIT)
    assert not np.allclose(first.slow_mean, last.slow_mean)


def test_normalised_features_are_finite_and_bounded(split):
    assert np.isfinite(split.fast).all() and np.isfinite(split.slow).all()
    assert np.abs(split.fast).max() <= 8.0 + 1e-6
    assert np.abs(split.slow).max() <= 8.0 + 1e-6


# -- splits -----------------------------------------------------------------


def test_splits_are_chronological_and_never_overlap():
    for split_definition in ALL_SPLITS:
        assert split_definition.train_start < split_definition.train_end
        assert split_definition.train_end == split_definition.evaluation_start
        assert split_definition.evaluation_start < split_definition.evaluation_end


def test_evaluation_warmup_comes_out_of_the_training_window(market):
    for split_definition in ALL_SPLITS:
        split = build_split(market, split_definition)
        assert split.evaluation_warmup_start >= split.train_range[0]
        assert split.evaluation_range[0] - split.evaluation_warmup_start == EVALUATION_WARMUP_HOURS


def test_test_year_is_never_in_any_training_window():
    for split_definition in ALL_SPLITS:
        assert split_definition.train_end <= FINAL_SPLIT.evaluation_start


# -- model and evaluation ---------------------------------------------------


def test_actor_starts_close_to_uniform(split):
    model = GruActorCritic(slow_features=split.observation_size)
    fast = torch.from_numpy(split.fast[:4]).unsqueeze(0)
    slow = torch.zeros(1, 4, split.observation_size)
    logits, _, _ = model(fast, slow, model.initial_hidden(1))
    probabilities = torch.softmax(logits, dim=-1).detach()
    assert float(probabilities.max()) < 0.40


def test_hidden_state_carries_across_hours(split):
    model = GruActorCritic(slow_features=split.observation_size)
    fast = torch.from_numpy(split.fast[:2]).unsqueeze(0)
    joined, _ = model.trunk(fast, model.initial_hidden(1))
    isolated, _ = model.trunk(fast[:, 1:2], model.initial_hidden(1))
    assert not torch.allclose(joined[:, 1], isolated[:, 0], atol=1e-6)
    first_alone, _ = model.trunk(fast[:, :1], model.initial_hidden(1))
    assert torch.allclose(joined[:, 0], first_alone[:, 0], atol=1e-6)


def test_deterministic_evaluation_repeats_exactly(split):
    model = GruActorCritic(slow_features=split.observation_size)
    first, last = split.evaluation_range
    last = first + 300
    one = run_policy(model, split, first, last, ONE_WAY_COST)
    two = run_policy(model, split, first, last, ONE_WAY_COST)
    assert np.array_equal(one.positions, two.positions)
    assert np.allclose(one.equity, two.equity)


def test_evaluation_enters_the_window_flat(split):
    model = GruActorCritic(slow_features=split.observation_size)
    first, last = split.evaluation_range
    trajectory = run_policy(model, split, first, first + 200, ONE_WAY_COST)
    if trajectory.positions[0] != 0.0:
        assert trajectory.cost_paid[0] == pytest.approx(ONE_WAY_COST)
    else:
        assert trajectory.cost_paid[0] == 0.0


def test_evaluation_warmup_changes_the_state_it_starts_from(split):
    model = GruActorCritic(slow_features=split.observation_size)
    first, last = split.evaluation_range
    warm = run_policy(model, split, first, first + 200, 0.0)
    cold = run_policy(model, split, first, first + 200, 0.0, warmup_start=first)
    assert warm.positions.shape == cold.positions.shape


def test_metrics_of_always_flat_are_all_zero(split):
    first, last = split.evaluation_range
    result = metrics(always_flat(split, first, last, ONE_WAY_COST), "flat")
    assert result["total_return"] == pytest.approx(0.0)
    assert result["turnover_total"] == 0.0
    assert result["share_flat"] == 1.0


def test_matched_turnover_null_preserves_turnover(split):
    rng = np.random.default_rng(0)
    positions = rng.choice([-1.0, 0.0, 1.0], size=800)
    first = split.evaluation_range[0]
    result = circular_shift_null(positions, split, first, first + 800, ONE_WAY_COST, 20, rng)

    def circular_turnover(path: np.ndarray) -> float:
        return float(np.abs(np.diff(path)).sum() + abs(path[0] - path[-1]))

    for offset in range(1, 20):
        rolled = np.roll(positions, offset)
        assert circular_turnover(rolled) == pytest.approx(circular_turnover(positions))
        # The linear turnover can only move by the single wrapped transition.
        assert abs(np.abs(np.diff(rolled)).sum() - np.abs(np.diff(positions)).sum()) <= 2.0
    assert result["draws"] == 20


# -- PPO --------------------------------------------------------------------


def test_rollout_replays_exactly_before_any_optimiser_step(split, market):
    torch.manual_seed(0)
    config = PPOConfig(environments=4, rollout_steps=32, minibatches=2, epochs=1)
    environment = VectorHourlyEnvironment(
        split.fast, split.slow, market.entry_price, market.exit_price,
        market.funding_on_previous, market.funding_on_current,
        split.train_range, config.environments,
        EnvironmentConfig(episode_hours=64), np.random.default_rng(0),
    )
    model = GruActorCritic(slow_features=environment.slow_width)
    trainer = PPOTrainer(model, environment, config, torch.Generator().manual_seed(0))
    initial_hidden = trainer.hidden.clone()
    buffer, _, _ = trainer.collect()

    fast = torch.from_numpy(buffer.fast).permute(1, 0, 2, 3).contiguous()
    slow = torch.from_numpy(buffer.slow).permute(1, 0, 2).contiguous()
    actions = torch.from_numpy(buffer.actions).permute(1, 0).contiguous()
    with torch.no_grad():
        logits, values, _ = model(fast, slow, initial_hidden)
        replayed = torch.distributions.Categorical(logits=logits).log_prob(actions)
    assert torch.allclose(
        replayed, torch.from_numpy(buffer.log_probabilities).permute(1, 0), atol=1e-5
    )
    assert torch.allclose(values, torch.from_numpy(buffer.values).permute(1, 0), atol=1e-5)


def test_episode_boundaries_never_fall_inside_a_backpropagated_sequence(split, market):
    config = PPOConfig(environments=4, rollout_steps=32, minibatches=2, epochs=1)
    environment = VectorHourlyEnvironment(
        split.fast, split.slow, market.entry_price, market.exit_price,
        market.funding_on_previous, market.funding_on_current,
        split.train_range, config.environments,
        EnvironmentConfig(episode_hours=64), np.random.default_rng(0),
    )
    model = GruActorCritic(slow_features=environment.slow_width)
    trainer = PPOTrainer(model, environment, config, torch.Generator().manual_seed(0))
    for _ in range(4):
        buffer, _, _ = trainer.collect()
        if buffer.dones.any():
            assert buffer.dones[:-1].sum() == 0


def test_episode_length_must_divide_into_rollouts(split, market):
    config = PPOConfig(environments=2, rollout_steps=32)
    environment = VectorHourlyEnvironment(
        split.fast, split.slow, market.entry_price, market.exit_price,
        market.funding_on_previous, market.funding_on_current,
        split.train_range, 2, EnvironmentConfig(episode_hours=50), np.random.default_rng(0),
    )
    model = GruActorCritic(slow_features=environment.slow_width)
    with pytest.raises(ValueError):
        PPOTrainer(model, environment, config, torch.Generator().manual_seed(0))


def test_advantages_do_not_leak_across_an_episode_boundary(split, market):
    config = PPOConfig(environments=2, rollout_steps=8, minibatches=1, epochs=1, gamma=0.99)
    environment = VectorHourlyEnvironment(
        split.fast, split.slow, market.entry_price, market.exit_price,
        market.funding_on_previous, market.funding_on_current,
        split.train_range, 2, EnvironmentConfig(episode_hours=8), np.random.default_rng(0),
    )
    model = GruActorCritic(slow_features=environment.slow_width)
    trainer = PPOTrainer(model, environment, config, torch.Generator().manual_seed(0))
    buffer, bootstrap, _ = trainer.collect()
    buffer.dones[3] = True
    buffer.truncated_values[3] = 0.0
    advantages, _ = trainer.advantages(buffer, bootstrap)
    delta = buffer.rewards[3] + config.gamma * 0.0 - buffer.values[3]
    assert np.allclose(advantages[3], delta, atol=1e-5)
