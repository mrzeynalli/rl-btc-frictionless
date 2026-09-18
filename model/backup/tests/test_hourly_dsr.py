"""Invariants for study 3: DSR reward, masking, persistence, counterfactuals."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cryptorl.hourly_dsr.env import (
    ACTION_COUNT,
    POSITION_LEVELS,
    HourlyDSREnvironment,
    TradingConfig,
    VectorHourlyDSREnvironment,
)
from cryptorl.hourly_dsr.evaluation import (
    benchmark_actions,
    cost_sweep,
    evaluate_policy,
    run_action_sequence,
)
from cryptorl.hourly_dsr.features import (
    FOLDS,
    build_decision_arrays,
    build_fold,
    build_hourly_market_data,
    observation_size,
)
from cryptorl.hourly_dsr.model import PersistentActorCritic, masked_categorical
from cryptorl.hourly_dsr.ppo import PPOConfig, collect_rollout, unchanged_policy_invariant


@pytest.fixture(scope="module")
def market():
    return build_hourly_market_data()


@pytest.fixture(scope="module")
def fold(market):
    return build_fold(market, FOLDS[-1])


@pytest.fixture(scope="module")
def decisions(market):
    return build_decision_arrays(market)


def test_three_actions_only():
    assert ACTION_COUNT == 3
    assert POSITION_LEVELS.tolist() == [-1.0, 0.0, 1.0]


def test_validation_windows_are_two_months(market):
    for definition in FOLDS:
        fold = build_fold(market, definition)
        start, end = fold.splits["validation"]
        assert 1300 < end - start < 1600, (definition["name"], end - start)


def test_sealed_test_months_match_study_two(market):
    """The five test months must be unchanged so results stay comparable."""
    import pandas as pd
    for definition in FOLDS:
        fold = build_fold(market, definition)
        start, end = fold.splits["test"]
        times = pd.to_datetime(market.open_times_ns[market.decision_bars[[start, end - 1]]], utc=True)
        assert times[0].day == 1
        assert times[0].month == times[1].month


def test_counterfactual_matches_the_realised_step(fold, decisions):
    """The heart of the all-action gradient: preview must equal reality."""
    rng = np.random.default_rng(0)
    environment = HourlyDSREnvironment(fold, *fold.splits["test"], TradingConfig(),
                                       decisions=decisions)
    environment.reset(index=fold.splits["test"][0])
    for _ in range(300):
        mask = environment.action_mask(environment.state)
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal))
        predicted_reward, predicted_observation, _ = environment.preview(action)
        observation, reward, done, _ = environment.step(action)
        assert reward == pytest.approx(predicted_reward, rel=1e-12, abs=1e-12)
        assert np.allclose(observation, predicted_observation, atol=1e-6)
        if done:
            break


def test_preview_does_not_mutate_state(fold, decisions):
    environment = HourlyDSREnvironment(fold, *fold.splits["test"], TradingConfig(),
                                       decisions=decisions)
    environment.reset(index=fold.splits["test"][0])
    before = environment.state
    for action in range(ACTION_COUNT):
        environment.preview(action)
    assert environment.state == before


def test_minimum_hold_is_enforced(fold, decisions):
    config = TradingConfig(minimum_hold_hours=8)
    rng = np.random.default_rng(1)
    environment = HourlyDSREnvironment(fold, *fold.splits["test"], config, decisions=decisions)
    environment.reset(index=fold.splits["test"][0])
    last_change = -99
    changes = 0
    for step in range(400):
        mask = environment.action_mask(environment.state)
        action = int(rng.choice(np.flatnonzero(mask)))
        _, _, done, info = environment.step(action)
        if info["held_level"] != info["previous_level"]:
            assert step - last_change >= config.minimum_hold_hours
            last_change = step
            changes += 1
        if done:
            break
    assert changes > 3, "the test needs some direction changes to be meaningful"


def test_masked_actions_are_never_selected(fold, decisions):
    config = TradingConfig(minimum_hold_hours=8)
    environment = HourlyDSREnvironment(fold, *fold.splits["test"], config, decisions=decisions)
    environment.reset(index=fold.splits["test"][0])
    for _ in range(100):
        mask = environment.action_mask(environment.state)
        if mask.sum() == 1:
            forbidden = int(np.flatnonzero(~mask)[0])
            previous = environment.state.level
            _, _, done, info = environment.step(forbidden)
            assert info["held_level"] == previous
            assert not info["traded"]
        else:
            _, _, done, _ = environment.step(1)
        if done:
            break


def test_volatility_targeting_caps_and_scales(fold, decisions):
    assert decisions.volatility_scale.max() <= 1.0 + 1e-12
    assert decisions.volatility_scale.min() > 0.0
    start, end = fold.splits["test"]
    performance, frame = run_action_sequence(
        fold, start, end, TradingConfig(), np.full(end - start, 2, dtype=np.int64), decisions)
    assert frame.exposure_ratio.abs().max() < 1.2
    assert performance.exposure_mean_absolute <= 1.0


def test_accounting_reconciles(fold, decisions):
    start, end = fold.splits["test"]
    rng = np.random.default_rng(7)
    actions = rng.integers(0, ACTION_COUNT, end - start)
    performance, frame = run_action_sequence(fold, start, end, TradingConfig(), actions, decisions)
    assert abs(performance.accounting_reconciliation_error) < 1e-6
    assert frame.accounting_reconciliation_error.abs().max() < 1e-6


def test_flat_policy_costs_nothing(fold, decisions):
    start, end = fold.splits["test"]
    performance, _ = run_action_sequence(
        fold, start, end, TradingConfig(), np.full(end - start, 1, dtype=np.int64), decisions)
    assert performance.total_transaction_cost == 0.0
    assert performance.net_return == 0.0


def test_dsr_reward_is_bounded_and_scale_free(fold, decisions):
    config = TradingConfig(dsr_clip=5.0)
    start, end = fold.splits["test"]
    rng = np.random.default_rng(3)
    actions = rng.integers(0, ACTION_COUNT, end - start)
    environment = HourlyDSREnvironment(fold, start, end, config, decisions=decisions)
    environment.reset(index=start)
    rewards = []
    for action in actions:
        _, reward, done, _ = environment.step(int(action))
        rewards.append(reward)
        if done:
            break
    rewards = np.asarray(rewards)
    assert np.isfinite(rewards).all()
    assert np.abs(rewards).max() <= config.dsr_clip + config.turnover_penalty * 2 + 1e-9


def test_dsr_ignores_absolute_wealth(fold, decisions):
    """Scale invariance: the reward stream must not depend on account size."""
    start, end = fold.splits["test"]
    rng = np.random.default_rng(5)
    actions = rng.integers(0, ACTION_COUNT, 200)
    streams = []
    for equity in (10_000.0, 1_000_000.0):
        environment = HourlyDSREnvironment(
            fold, start, end, TradingConfig(initial_equity=equity), decisions=decisions)
        environment.reset(index=start)
        stream = []
        for action in actions:
            _, reward, done, _ = environment.step(int(action))
            stream.append(reward)
            if done:
                break
        streams.append(np.asarray(stream))
    assert np.allclose(streams[0], streams[1], atol=1e-9)


def test_turnover_penalty_does_not_touch_equity(fold, decisions):
    start, end = fold.splits["test"]
    rng = np.random.default_rng(11)
    actions = rng.integers(0, ACTION_COUNT, end - start)
    plain, _ = run_action_sequence(
        fold, start, end, TradingConfig(turnover_penalty=0.0), actions, decisions)
    shaped, _ = run_action_sequence(
        fold, start, end, TradingConfig(turnover_penalty=1.0), actions, decisions)
    assert plain.final_equity == pytest.approx(shaped.final_equity, rel=1e-12)


def test_cost_sweep_is_monotone(fold, decisions):
    start, end = fold.splits["test"]
    rng = np.random.default_rng(13)
    actions = rng.integers(0, ACTION_COUNT, end - start)
    returns = [row["net_return"] for row in cost_sweep(fold, start, end, TradingConfig(), actions)]
    assert returns == sorted(returns, reverse=True)


def test_persistence_logit_favours_the_held_action():
    model = PersistentActorCritic(10, 8, ACTION_COUNT)
    with torch.no_grad():
        model.stickiness_raw.fill_(2.0)
    observation = torch.zeros(1, 1, 10)
    observation[..., model.position_feature] = 1.0        # currently LONG
    logits, _, _, _ = model(observation, *model.initial_hidden(1, torch.device("cpu")))
    assert logits[0, 0, 2] > logits[0, 0, 0]
    assert logits[0, 0, 2] > logits[0, 0, 1]


def test_persistence_logit_is_non_negative():
    model = PersistentActorCritic(10, 8, ACTION_COUNT)
    with torch.no_grad():
        model.stickiness_raw.fill_(-50.0)
    assert float(model.stickiness.detach()) >= 0.0


def test_masked_categorical_gives_zero_probability_to_illegal_actions():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    probabilities = masked_categorical(logits, mask).probs
    assert probabilities[0, 1].item() == pytest.approx(0.0, abs=1e-9)
    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_recurrent_replay_reproduces_the_rollout(fold):
    device = torch.device("cpu")
    config = PPOConfig(environments=4, rollout_steps=32, chunk_length=16, chunks_per_minibatch=4)
    vector = VectorHourlyDSREnvironment(fold, *fold.splits["train"], TradingConfig(),
                                        count=config.environments, episode_hours=64, seed=1)
    model = PersistentActorCritic(observation_size(fold), 16, ACTION_COUNT)
    observations, masks = vector.reset()
    state = (observations, masks, *model.initial_hidden(config.environments, device))
    rollout, _ = collect_rollout(model, vector, state, config, device)
    invariant = unchanged_policy_invariant(model, rollout, config, device)
    assert invariant["maximum_absolute_probability_difference"] < 1e-5
    assert invariant["maximum_absolute_value_difference"] < 1e-5


def test_all_action_advantages_are_centred_under_the_behaviour_policy(fold):
    device = torch.device("cpu")
    config = PPOConfig(environments=4, rollout_steps=16, chunk_length=16, chunks_per_minibatch=4)
    vector = VectorHourlyDSREnvironment(fold, *fold.splits["train"], TradingConfig(),
                                        count=config.environments, episode_hours=64, seed=2)
    model = PersistentActorCritic(observation_size(fold), 16, ACTION_COUNT)
    observations, masks = vector.reset()
    state = (observations, masks, *model.initial_hidden(config.environments, device))
    rollout, _ = collect_rollout(model, vector, state, config, device)
    weighted = (rollout.old_probabilities * rollout.action_advantages).sum(-1)
    assert np.abs(weighted).max() < 1e-4
    # Illegal actions must carry no advantage signal at all.
    assert np.abs(rollout.action_advantages[~rollout.masks]).max() == 0.0


def test_evaluation_does_not_disturb_the_global_torch_stream(fold, decisions):
    model = PersistentActorCritic(observation_size(fold), 16, ACTION_COUNT)
    start, end = fold.splits["validation"]
    torch.manual_seed(1234)
    before = torch.randn(3)
    torch.manual_seed(1234)
    evaluate_policy(model, fold, start, start + 200, TradingConfig(), torch.device("cpu"),
                    deterministic=False, seed=5, decisions=decisions)
    after = torch.randn(3)
    assert torch.allclose(before, after)


def test_deterministic_evaluation_is_reproducible(fold, decisions):
    model = PersistentActorCritic(observation_size(fold), 16, ACTION_COUNT)
    start, end = fold.splits["validation"]
    first, _ = evaluate_policy(model, fold, start, end, TradingConfig(),
                               torch.device("cpu"), True, decisions=decisions)
    second, _ = evaluate_policy(model, fold, start, end, TradingConfig(),
                                torch.device("cpu"), True, decisions=decisions)
    assert first.to_dict() == second.to_dict()


def test_carry_benchmark_only_uses_settled_rates(fold):
    start, end = fold.splits["test"]
    actions = benchmark_actions("carry", fold, start, end)
    assert set(np.unique(actions).tolist()) <= {0, 1, 2}
    assert len(actions) == end - start
