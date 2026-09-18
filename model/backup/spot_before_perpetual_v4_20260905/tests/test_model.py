from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from data import OUTPUT_DIR, load
from env import (
    ACTION_NAMES,
    BUY,
    HOLD,
    PORTFOLIO_WIDTH,
    SELL,
    EnvironmentConfig,
    VectorHourlyEnvironment,
    action_targets,
    realised_return,
    valid_action_mask,
)
from execution import aggregate_market_scores, executable_action_logits
from evaluation import always_btc, relative_performance
from features import SLOW_FEATURE_NAMES, WALK_FORWARD_SPLITS, build_market, build_split
from model import MARKET_WIDTH, MarketActorCritic
from ppo import PPOConfig, PPOTrainer


def require_retired_spot_dataset() -> None:
    if not (OUTPUT_DIR / "manifest.json").exists():
        pytest.skip("Retired Spot integration dataset was explicitly removed")


def test_spot_dataset_has_no_funding() -> None:
    require_retired_spot_dataset()
    manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text())
    assert manifest["market"] == "BTCUSDT Spot"
    assert manifest["interval"] == "1m"
    assert manifest["funding_data_present"] is False
    bars = load()
    assert "funding_rate" not in bars.columns


def test_spot_actions_and_inventory_masks() -> None:
    assert ACTION_NAMES == ("BUY", "SELL", "HOLD")
    mask = valid_action_mask(np.array([0.0, 1.0]))
    np.testing.assert_array_equal(mask[0], [True, False, True])
    np.testing.assert_array_equal(mask[1], [False, True, True])
    targets = action_targets(
        np.array([BUY, SELL, HOLD, HOLD]), np.array([0.0, 1.0, 0.0, 1.0])
    )
    np.testing.assert_array_equal(targets, [1.0, 0.0, 0.0, 1.0])


def test_invalid_redundant_spot_orders_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        action_targets(np.array([BUY]), np.array([1.0]))
    with np.testing.assert_raises(ValueError):
        action_targets(np.array([SELL]), np.array([0.0]))


def test_cost_only_applies_when_inventory_changes() -> None:
    price, cost, total = realised_return(
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.ones(3),
        0.0004,
    )
    np.testing.assert_allclose(price, 0.0)
    np.testing.assert_allclose(cost, [0.0004, 0.0004, 0.0])
    np.testing.assert_allclose(total, -cost)


def test_execution_masks_inventory_after_position_independent_scores() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    executable = executable_action_logits(scores, torch.tensor([0.0, 1.0]))
    assert torch.isneginf(executable[0, SELL])
    assert torch.isneginf(executable[1, BUY])
    assert int(executable[0].argmax()) == BUY
    assert int(executable[1].argmax()) == SELL


def test_actor_logits_are_numerically_position_invariant() -> None:
    torch.manual_seed(7)
    model = MarketActorCritic(critic_mode="shared")
    fast = torch.randn(2, 3, 60, 4)
    slow = torch.randn(2, 3, 10)
    state = model.initial_state(2)
    cash = torch.zeros(2, 3, PORTFOLIO_WIDTH)
    btc = cash.clone()
    btc[..., 0] = 1.0
    cash_logits, cash_value, _ = model(fast, slow, cash, state)
    btc_logits, btc_value, _ = model(fast, slow, btc, state)
    torch.testing.assert_close(cash_logits, btc_logits, rtol=0, atol=0)
    assert not torch.equal(cash_value, btc_value)


def test_critic_modes_control_encoder_gradients() -> None:
    fast = torch.randn(2, 2, 60, 4)
    slow = torch.randn(2, 2, 10)
    portfolio = torch.randn(2, 2, PORTFOLIO_WIDTH)
    for mode in ("shared", "detached", "separate"):
        model = MarketActorCritic(critic_mode=mode)
        _, value, _ = model(fast, slow, portfolio, model.initial_state(2))
        value.square().mean().backward()
        actor_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.actor_encoder.parameters()
            if parameter.grad is not None
        )
        if mode == "shared":
            assert actor_gradient > 0
        else:
            assert actor_gradient == 0
        if mode == "separate":
            assert model.critic_encoder is not None
            assert any(parameter.grad is not None for parameter in model.critic_encoder.parameters())


def test_gru_and_lstm_have_required_market_width() -> None:
    assert MARKET_WIDTH == 42
    for encoder in ("gru", "lstm"):
        model = MarketActorCritic(encoder_type=encoder, critic_mode="separate")
        logits, values, state = model(
            torch.zeros(1, 2, 60, 4),
            torch.zeros(1, 2, 10),
            torch.zeros(1, 2, PORTFOLIO_WIDTH),
            model.initial_state(1),
        )
        assert logits.shape == (1, 2, 3)
        assert values.shape == (1, 2)
        assert state.critic is not None


def _small_environment(
    *, environments: int = 1, episode_hours: int = 8, burn_in_hours: int = 0
) -> VectorHourlyEnvironment:
    rows = 40
    fast = np.zeros((rows, 60, 4), dtype=np.float32)
    slow = np.zeros((rows, 10), dtype=np.float32)
    entry = np.full(rows, 100.0)
    exit_price = np.full(rows, 99.0)
    segment = np.zeros(rows, dtype=bool)
    segment[0] = True
    return VectorHourlyEnvironment(
        fast, slow, entry, exit_price, segment, (0, rows), environments,
        EnvironmentConfig(
            episode_hour_options=(episode_hours,), burn_in_hours=burn_in_hours,
            randomize_starting_position=False,
            regime_balanced_sampling=False,
        ),
        np.random.default_rng(9),
    )


def test_reward_is_net_return_including_each_position_change_cost() -> None:
    env = _small_environment()
    env.entry_price[:] = 100.0
    env.exit_price[:] = 100.0
    buy_reward, _, buy = env.step(np.array([BUY]))
    hold_reward, _, hold = env.step(np.array([HOLD]))
    sell_reward, _, sell = env.step(np.array([SELL]))
    expected = np.log(1.0 - env.config.cost_rate) * env.config.reward_scale
    np.testing.assert_allclose(buy_reward, expected)
    np.testing.assert_allclose(sell_reward, expected)
    np.testing.assert_allclose(hold_reward, 0.0)
    np.testing.assert_allclose(buy["cost_paid"], env.config.cost_rate)
    np.testing.assert_allclose(sell["cost_paid"], env.config.cost_rate)
    np.testing.assert_allclose(hold["cost_paid"], 0.0)


def test_critic_portfolio_features_track_open_loss_and_time() -> None:
    env = _small_environment()
    before = env.portfolio_features().copy()
    env.step(np.array([BUY]))
    after = env.portfolio_features()
    assert after[0, 0] == 1.0
    assert after[0, 2] > before[0, 2]
    assert after[0, 3] > before[0, 3]
    assert after[0, 4] > before[0, 4]


def test_random_episode_sampling_is_balanced_and_contiguous() -> None:
    require_retired_spot_dataset()
    market = build_market()
    split = build_split(market, WALK_FORWARD_SPLITS[2])
    environment = VectorHourlyEnvironment(
        split.fast, split.slow, market.entry_price, market.exit_price,
        market.segment_start, split.train_range, 12,
        EnvironmentConfig(), np.random.default_rng(3),
    )
    assert environment.start_position_counts.tolist() == [6, 6]
    assert max(environment.regime_sample_counts) - min(environment.regime_sample_counts) <= 1
    assert set(environment.available_episode_hours).issubset({720, 1440, 2160, 2880, 3600})
    for start, length in zip(environment.cursor, environment.episode_target):
        assert environment._segment_ids[start - environment.burn_in_hours] == environment._segment_ids[start + length]


def test_feature_timeline_is_causal_gap_aware_and_funding_free() -> None:
    require_retired_spot_dataset()
    market = build_market()
    assert market.slow_names == SLOW_FEATURE_NAMES
    assert len(market.slow_names) == 10
    assert market.audit["funding_features"] is False
    assert np.all(market.decision_time < market.execution_time)
    assert market.fast.shape[1:] == (60, 4)
    assert market.segment_start.sum() > 1


def test_buy_and_hold_accounting_carries_inventory_across_data_gaps() -> None:
    require_retired_spot_dataset()
    market = build_market()
    split = build_split(market, WALK_FORWARD_SPLITS[1])
    first, last = split.evaluation_range
    trajectory = always_btc(split, first, last, 0.0)
    expected = np.log(market.exit_price[last - 1] / market.entry_price[first])
    np.testing.assert_allclose(trajectory.rewards.sum(), expected, atol=1e-12)


def test_normaliser_is_fitted_on_training_period_only() -> None:
    require_retired_spot_dataset()
    market = build_market()
    split = build_split(market, WALK_FORWARD_SPLITS[2])
    first, last = split.train_range
    expected = market.fast[first:last].reshape(-1, 4).mean(axis=0)
    np.testing.assert_allclose(split.normaliser.fast_mean, expected)
    assert split.normaliser.fitted_on["end_exclusive"] == "2022-01-01T00:00:00+00:00"


def test_gamma_lambda_and_critic_rate_are_independent() -> None:
    config = PPOConfig(
        gamma=0.99, gae_lambda=0.95,
        actor_learning_rate=0.0003, critic_learning_rate=0.0001,
    )
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.critic_learning_rate == 0.0001


def test_recurrent_replay_reproduces_rollout_across_burned_resets() -> None:
    torch.manual_seed(19)
    environment = _small_environment(
        environments=2, episode_hours=4, burn_in_hours=2
    )
    model = MarketActorCritic(encoder_type="gru", critic_mode="detached")
    trainer = PPOTrainer(
        model,
        environment,
        PPOConfig(
            environments=2,
            rollout_steps=8,
            minibatches=1,
            actor_epochs=1,
            extra_critic_epochs=0,
            gamma=0.95,
            gae_lambda=0.95,
            burn_in_hours=2,
            adaptive_entropy=True,
        ),
        torch.Generator().manual_seed(19),
    )
    summary = trainer.run_update()
    assert summary["replay_ratio_max_abs_error"] < 2e-5
    assert summary["replay_value_max_abs_error"] < 2e-5
    assert summary["completed_episodes"] >= 2


def test_ensemble_aggregates_raw_market_scores() -> None:
    seeds = torch.tensor([
        [3.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0],
        [2.0, 0.0, 1.0], [0.0, 3.0, 1.0],
    ])
    torch.testing.assert_close(aggregate_market_scores(seeds, "mean"), seeds.mean(0))
    assert int(aggregate_market_scores(seeds, "majority").argmax()) == BUY


def test_relative_reporting_matches_bitcoin_identity() -> None:
    bitcoin = np.array([0.01, -0.02, 0.005, 0.002])
    report = relative_performance(bitcoin, bitcoin)
    np.testing.assert_allclose(report["beta_to_bitcoin"], 1.0)
    np.testing.assert_allclose(report["correlation_to_bitcoin"], 1.0)
    np.testing.assert_allclose(
        report["annualised_log_alpha_to_bitcoin"], 0.0, atol=1e-12
    )
