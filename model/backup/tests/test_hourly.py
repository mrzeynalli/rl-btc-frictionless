"""Invariants for the hourly study: accounting, leakage, and PPO replay."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cryptorl.hourly.env import (
    POSITION_LEVELS,
    HourlyTradingEnvironment,
    TradingConfig,
    VectorHourlyEnvironment,
)
from cryptorl.hourly.evaluation import cost_sweep, evaluate_policy, run_action_sequence
from cryptorl.hourly.data import OUTPUT_DIR
from cryptorl.hourly.features import (
    BARS_PER_HOUR,
    FOLDS,
    MARKET_FEATURE_NAMES,
    _ewma,
    _rolling_sum,
    build_fold,
    build_hourly_market_data,
)
from cryptorl.hourly.model import RecurrentActorCritic, run_gru
from cryptorl.hourly.ppo import PPOConfig, collect_rollout, unchanged_policy_invariant


def _truncated_market(cut: int):
    """Rebuild the state from a tape that simply stops at bar ``cut``."""
    import shutil
    import tempfile
    from pathlib import Path

    import pandas as pd

    directory = Path(tempfile.mkdtemp())
    try:
        for name in ("BTCUSDT-5m-contract", "BTCUSDT-5m-mark-price"):
            frame = pd.read_parquet(OUTPUT_DIR / f"{name}.parquet").iloc[:cut]
            frame.to_parquet(directory / f"{name}.parquet", index=False)
        contract = pd.read_parquet(directory / "BTCUSDT-5m-contract.parquet")
        funding = pd.read_parquet(OUTPUT_DIR / "BTCUSDT-funding-settlements.parquet")
        funding[funding.funding_time <= contract.close_time.iloc[-1]].to_parquet(
            directory / "BTCUSDT-funding-settlements.parquet", index=False
        )
        shutil.copy(OUTPUT_DIR / "manifest.json", directory / "manifest.json")
        return build_hourly_market_data(directory, minimum_decisions=100)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture(scope="module")
def market():
    return build_hourly_market_data()


@pytest.fixture(scope="module")
def fold(market):
    return build_fold(market, FOLDS[-1])


def test_decision_grid_is_hourly_and_inside_the_study(market):
    assert np.all(np.diff(market.decision_bars) == BARS_PER_HOUR)
    assert np.isfinite(market.raw_features).all()
    opens = market.open_times_ns[market.decision_bars]
    minutes = ((opens // 60_000_000_000) % 60).astype(int)
    assert set(np.unique(minutes).tolist()) == {55}


def test_normalization_is_fit_on_training_only(fold):
    train_start, train_end = fold.splits["train"]
    for name in MARKET_FEATURE_NAMES:
        parameters = fold.normalization["parameters"][name]
        assert parameters["fit_start_decision"] == train_start
        assert parameters["fit_end_exclusive_decision"] == train_end


def test_walk_forward_partitions_are_contiguous_and_ordered(market):
    for definition in FOLDS:
        fold = build_fold(market, definition)
        train, validation, test = (fold.splits[k] for k in ("train", "validation", "test"))
        assert train[0] == 0
        assert train[1] == validation[0]
        assert validation[1] == test[0]
        assert test[1] > test[0]


def test_test_partitions_do_not_overlap_across_folds(market):
    seen: list[tuple[int, int]] = []
    for definition in FOLDS:
        window = build_fold(market, definition).splits["test"]
        assert all(window[0] >= end or window[1] <= start for start, end in seen)
        seen.append(window)


def test_rolling_helpers_are_backward_looking():
    """Perturbing a future value must leave every earlier output untouched."""
    values = np.arange(1.0, 21.0)
    perturbed = values.copy()
    perturbed[15] += 1000.0
    for window in (3, 5):
        a, b = _rolling_sum(values, window), _rolling_sum(perturbed, window)
        assert np.allclose(a[:15], b[:15], equal_nan=True)
        assert not np.allclose(a[15:], b[15:], equal_nan=True)
    a, b = _ewma(values, 4.0), _ewma(perturbed, 4.0)
    assert np.allclose(a[:15], b[:15], equal_nan=True)


def test_decision_features_never_read_beyond_their_own_bar(market):
    """Rebuild from a truncated tape; shared decisions must be identical."""
    cut = int(market.decision_bars[len(market.decision_bars) // 2])
    truncated = _truncated_market(cut)
    shared = len(truncated.decision_bars)
    assert shared > 100
    assert np.array_equal(truncated.decision_bars, market.decision_bars[:shared])
    assert np.allclose(
        truncated.raw_features, market.raw_features[:shared], rtol=1e-9, atol=1e-12
    )


def test_always_long_matches_buy_and_hold(fold):
    start, end = fold.splits["test"]
    config = TradingConfig(fee_per_leg=0.0, slippage_per_leg=0.0)
    performance, _ = run_action_sequence(
        fold, start, end, config, np.full(end - start, 4, dtype=np.int64)
    )
    bars = fold.market.decision_bars
    expected = (
        fold.market.close_prices[bars[end - 1] + BARS_PER_HOUR]
        / fold.market.open_prices[bars[start] + 1] - 1.0
    )
    # Funding is the only difference between the two at zero cost.
    assert performance.net_return == pytest.approx(expected, abs=0.01)


def test_accounting_reconciles(fold):
    start, end = fold.splits["test"]
    rng = np.random.default_rng(7)
    actions = rng.integers(0, len(POSITION_LEVELS), end - start)
    performance, frame = run_action_sequence(fold, start, end, TradingConfig(), actions)
    assert abs(performance.accounting_reconciliation_error) < 1e-6
    assert frame.accounting_reconciliation_error.abs().max() < 1e-6


def test_flat_policy_never_pays_anything(fold):
    start, end = fold.splits["test"]
    performance, _ = run_action_sequence(
        fold, start, end, TradingConfig(), np.full(end - start, 2, dtype=np.int64)
    )
    assert performance.total_transaction_cost == 0.0
    assert performance.net_return == 0.0
    assert performance.trades == 0


def test_repeating_a_level_is_free(fold):
    """The no-trade band must make an unchanged target cost nothing."""
    start, end = fold.splits["test"]
    performance, frame = run_action_sequence(
        fold, start, end, TradingConfig(), np.full(end - start, 4, dtype=np.int64)
    )
    # One entry, at most a few drift rebalances, and one forced exit.
    assert performance.trades <= 6
    assert frame.traded.sum() <= 6


def test_no_trade_band_suppresses_small_rebalances(fold):
    start, end = fold.splits["test"]
    actions = np.full(end - start, 4, dtype=np.int64)
    wide, _ = run_action_sequence(
        fold, start, end, TradingConfig(no_trade_band=0.95), actions
    )
    narrow, _ = run_action_sequence(
        fold, start, end, TradingConfig(no_trade_band=0.0), actions
    )
    assert wide.trades <= narrow.trades


def test_leverage_never_exceeds_the_cap(fold):
    start, end = fold.splits["test"]
    rng = np.random.default_rng(3)
    actions = rng.integers(0, len(POSITION_LEVELS), end - start)
    _, frame = run_action_sequence(fold, start, end, TradingConfig(), actions)
    assert frame.exposure_ratio.abs().max() < 1.6


def test_turnover_penalty_changes_reward_but_not_equity(fold):
    start, end = fold.splits["test"]
    rng = np.random.default_rng(11)
    actions = rng.integers(0, len(POSITION_LEVELS), end - start)
    plain, _ = run_action_sequence(
        fold, start, end, TradingConfig(turnover_penalty=0.0), actions
    )
    shaped, _ = run_action_sequence(
        fold, start, end, TradingConfig(turnover_penalty=0.01), actions
    )
    assert plain.final_equity == pytest.approx(shaped.final_equity, rel=1e-12)


def test_cost_sweep_is_monotone_in_cost(fold):
    start, end = fold.splits["test"]
    rng = np.random.default_rng(5)
    actions = rng.integers(0, len(POSITION_LEVELS), end - start)
    sweep = cost_sweep(fold, start, end, TradingConfig(), actions)
    returns = [row["net_return"] for row in sweep]
    assert returns == sorted(returns, reverse=True)


def test_gru_reset_mask_matches_manual_restart():
    torch.manual_seed(0)
    gru = torch.nn.GRU(4, 6, batch_first=True)
    inputs = torch.randn(1, 5, 4)
    hidden = torch.randn(1, 1, 6)
    resets = torch.zeros(1, 5, dtype=torch.bool)
    resets[0, 3] = True
    masked, _ = run_gru(gru, inputs, hidden, resets)
    first, _ = gru(inputs[:, :3], hidden)
    second, _ = gru(inputs[:, 3:], torch.zeros_like(hidden))
    assert torch.allclose(masked, torch.cat([first, second], dim=1), atol=1e-6)


def test_gru_without_resets_matches_fused_call():
    torch.manual_seed(1)
    gru = torch.nn.GRU(4, 6, batch_first=True)
    inputs = torch.randn(2, 5, 4)
    hidden = torch.randn(1, 2, 6)
    a, _ = run_gru(gru, inputs, hidden, torch.zeros(2, 5, dtype=torch.bool))
    b, _ = gru(inputs, hidden)
    assert torch.allclose(a, b, atol=1e-6)


def test_recurrent_replay_reproduces_the_rollout(fold):
    device = torch.device("cpu")
    config = PPOConfig(environments=4, rollout_steps=32, chunk_length=16, chunks_per_minibatch=4)
    vector = VectorHourlyEnvironment(
        fold, *fold.splits["train"], TradingConfig(), count=config.environments,
        episode_hours=48, seed=1,
    )
    model = RecurrentActorCritic(vector.observation_size, 16, len(POSITION_LEVELS))
    state = (vector.reset(), *model.initial_hidden(config.environments, device))
    rollout, _ = collect_rollout(model, vector, state, config, device)
    invariant = unchanged_policy_invariant(model, rollout, config, device)
    assert invariant["maximum_absolute_log_probability_difference"] < 1e-5
    assert invariant["maximum_absolute_value_difference"] < 1e-5


def test_episodes_reset_equity_and_use_random_offsets(fold):
    vector = VectorHourlyEnvironment(
        fold, *fold.splits["train"], TradingConfig(), count=8, episode_hours=64, seed=2,
    )
    vector.reset()
    offsets = {environment.index for environment in vector.environments}
    assert len(offsets) > 1
    for environment in vector.environments:
        assert environment.equity == environment.config.initial_equity


def test_deterministic_evaluation_is_reproducible(fold):
    device = torch.device("cpu")
    start, end = fold.splits["validation"]
    model = RecurrentActorCritic(fold.observation_size, 16, len(POSITION_LEVELS))
    first, _ = evaluate_policy(model, fold, start, end, TradingConfig(), device, True)
    second, _ = evaluate_policy(model, fold, start, end, TradingConfig(), device, True)
    assert first.to_dict() == second.to_dict()


def test_observation_excludes_equity(fold):
    """Cost counterfactual replay is only exact if equity is not observed."""
    start, end = fold.splits["test"]
    rich = HourlyTradingEnvironment(
        fold, start, end, TradingConfig(initial_equity=10_000.0)
    )
    poor = HourlyTradingEnvironment(
        fold, start, end, TradingConfig(initial_equity=1_000_000.0)
    )
    a, b = rich.reset(index=start), poor.reset(index=start)
    assert np.allclose(a, b)
    for action in (4, 4, 0, 2, 3):
        a, *_ = rich.step(action)
        b, *_ = poor.step(action)
        assert np.allclose(a, b)
