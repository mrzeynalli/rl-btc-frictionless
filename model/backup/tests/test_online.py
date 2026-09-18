"""Invariants for the BTC volatility-event online-learning study.

Several of these are regression tests for specific failures of the previous
run; each one is named for the failure it would have caught.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from cryptorl.online.features import (
    INITIALISATION_END,
    INITIALISATION_START,
    MARKET_FEATURE_NAMES,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    CausalNormalizer,
    calibrate_threshold,
    build_event_market,
    _funding_at_decisions,
)
from cryptorl.online.ledger import (
    ConvictionAnchor,
    LedgerState,
    TradingConfig,
    anchored_level,
    apply_gate,
    apply_minimum_hold,
    apply_no_trade_band,
    liquidate,
    transition,
)
from cryptorl.online.models import DirectPositionGRU, SampledPPOGRU
from cryptorl.online.objectives import (
    ObjectiveConfig,
    RunningScale,
    UtilityState,
    soft_clip,
    torch_utility_rewards,
    utility_value,
)
from cryptorl.online.training import (
    RegimeBalancedSampler,
    TrainingConfig,
    _batched_transition,
    _initial_book,
    flush_online_updates,
    initialize_online,
    make_model,
    online_checkpoint,
    restore_online_checkpoint,
    run_online_period,
    train_offline,
)


@pytest.fixture(scope="module")
def market():
    return build_event_market(mark_path=None, require_basis=False)


# --------------------------------------------------------------------------
# Protocol and data
# --------------------------------------------------------------------------


def test_data_cover_exact_requested_partitions(market):
    initialisation = market.transition_indices(INITIALISATION_START, INITIALISATION_END)
    train = market.transition_indices(TRAIN_START, TRAIN_END)
    validation = market.transition_indices(TRAIN_END, VALIDATION_END)
    test = market.transition_indices(VALIDATION_END, "2026-07-01")
    assert len(initialisation) > 800
    assert len(train) > 6_000
    assert len(validation) > 2_000
    assert len(test) > 1_000
    assert not set(initialisation) & set(train)
    assert not set(train) & set(validation)
    assert not set(validation) & set(test)
    assert market.outcome_timestamps_ns[train[-1]] < np.datetime64(
        "2026-01-01").astype("datetime64[ns]").astype(int)


def test_initialisation_precedes_training_chronologically(market):
    initialisation = market.transition_indices(INITIALISATION_START, INITIALISATION_END)
    train = market.transition_indices(TRAIN_START, TRAIN_END)
    assert market.outcome_timestamps_ns[initialisation[-1]] <= market.timestamps_ns[train[0]]


def test_threshold_calibration_is_independent_of_future_returns():
    rng = np.random.default_rng(1)
    training = rng.lognormal(-12, 1, 20_000)
    future = rng.lognormal(-8, 2, 5_000)
    first = calibrate_threshold(training, 360, 500)
    second = calibrate_threshold(np.r_[training], 360, 500)
    assert first == second
    assert len(future)  # Future exists but is intentionally never passed to calibration.


def test_no_calendar_or_order_book_features_are_claimed():
    names = set(MARKET_FEATURE_NAMES)
    assert not names & {"hour_sin", "hour_cos", "weekday_sin", "weekday_cos"}
    assert not any("bid" in name or "ask" in name or "depth" in name for name in names)
    assert "taker_imbalance_1e" in names
    assert "funding_hours_to_next_settlement" in names


def test_funding_countdown_does_not_read_the_future_settlement():
    import pandas as pd
    funding = pd.DataFrame({
        "funding_time": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T08:00Z"]),
        "funding_rate": [0.001, 0.9], "funding_interval_hours": [8, 8],
    })
    decisions = pd.Series(pd.to_datetime(["2026-01-01T04:00Z"]))
    before = _funding_at_decisions(funding, decisions)
    changed = funding.copy()
    changed.loc[1, "funding_time"] = pd.Timestamp("2026-01-01T09:00Z")
    changed.loc[1, "funding_rate"] = -0.9
    after = _funding_at_decisions(changed, decisions)
    assert all(np.allclose(a, b) for a, b in zip(before, after))


def test_event_clock_is_not_fixed_hourly(market):
    durations = np.diff(market.timestamps_ns) / 60e9
    assert len(np.unique(durations.astype(int))) > 20
    assert durations.max() <= 361


def test_causal_normalizer_round_trip_and_update_order():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(100, 4))
    normalizer = CausalNormalizer.fit(values)
    before = normalizer.transform(values[-1])
    state = normalizer.state_dict()
    restored = CausalNormalizer.from_state_dict(state)
    assert np.allclose(before, restored.transform(values[-1]))
    restored.update(np.full(4, 100.0))
    assert not np.allclose(before, restored.transform(values[-1]))


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------


def test_differential_sharpe_is_scale_free_and_quadratic_is_not():
    """The reason the conviction anchor exists.

    A ratio objective is exactly indifferent to leverage, transaction costs
    included, so it cannot be asked to choose a position size.
    """
    torch.manual_seed(0)
    returns = torch.randn(200) * 0.5 + 0.01
    # Moments scale with the stream, which is the state a policy running at a
    # steady size actually reaches.
    warm = lambda k: UtilityState(downside_second=0.25 * k * k, second=0.25 * k * k)
    sharpe = [float(torch_utility_rewards(
        returns * k, ObjectiveConfig("differential_sharpe"), warm(k))[0].mean())
        for k in (0.1, 1.0, 10.0)]
    quadratic = [float(torch_utility_rewards(
        returns * k, ObjectiveConfig("quadratic"), warm(k))[0].mean())
        for k in (0.1, 1.0, 10.0)]
    assert max(sharpe) - min(sharpe) < 1e-6
    assert max(quadratic) - min(quadratic) > 1e-2


def test_only_ratio_objectives_declare_themselves_scale_free():
    assert ObjectiveConfig("differential_sharpe").is_scale_free
    assert ObjectiveConfig("differential_sortino").is_scale_free
    assert not ObjectiveConfig("quadratic").is_scale_free


def test_soft_clip_bounds_the_reward_without_deleting_its_gradient():
    assert abs(soft_clip(500.0, 5.0)) <= 5.0
    assert soft_clip(0.4, 5.0) == pytest.approx(0.4, rel=1e-2)  # near-identity in range
    beyond = torch.tensor(15.0, requires_grad=True)  # three times the bound
    soft_clip(beyond, 5.0).backward()
    assert float(beyond.grad) > 1e-3  # A hard clamp would give exactly zero here.


def test_warm_moments_remove_the_utility_warm_up_transient():
    """At eta=0.01 a cold recurrence needs ~100 events; a sequence is 64."""
    config = ObjectiveConfig("differential_sharpe")
    torch.manual_seed(1)
    returns = torch.randn(64) * 0.3
    cold = torch_utility_rewards(returns, config, None)[0]
    warm = torch_utility_rewards(
        returns, config, UtilityState(downside_second=0.045, second=0.09))[0]
    assert cold[:8].abs().mean() > warm[:8].abs().mean()


def test_differential_sortino_large_gain_does_not_increase_downside_moment():
    config = ObjectiveConfig("differential_sortino", adaptation_rate=0.02)
    state = UtilityState(downside_second=0.5, second=0.5)
    before = state.downside_second
    reward = state.update(20.0, config)
    assert state.downside_second < before
    assert np.isfinite(reward)
    assert abs(reward) <= config.reward_clip


def test_quadratic_utility_tracks_total_second_moment():
    config = ObjectiveConfig("quadratic", adaptation_rate=0.02)
    state = UtilityState(second=1e-4)
    state.update(20.0, config)
    assert state.second > 1.0


def test_running_scale_tracks_reward_magnitude():
    scale = RunningScale()
    for _ in range(500):
        scale.update(np.full(32, 3.0))
    assert 2.5 < scale.scale < 3.5
    assert RunningScale.from_state_dict(scale.state_dict()).scale == scale.scale


# --------------------------------------------------------------------------
# Sizing, hysteresis, and the gate
# --------------------------------------------------------------------------


def test_conviction_anchor_pins_size_for_scale_free_objectives_only():
    anchor = ConvictionAnchor(target=0.5, adaptation_rate=0.2, enabled=True)
    for _ in range(200):
        anchor.advance(0.02)
    assert anchored_level(0.02, anchor) == pytest.approx(0.4, abs=0.15)
    disabled = ConvictionAnchor(target=0.5, adaptation_rate=0.2, enabled=False)
    for _ in range(200):
        disabled.advance(0.02)
    assert disabled.gain() == 1.0
    assert anchored_level(0.02, disabled) == pytest.approx(0.02)


def test_conviction_anchor_cannot_amplify_without_bound():
    anchor = ConvictionAnchor(target=0.5, adaptation_rate=0.5, maximum_gain=20.0)
    for _ in range(200):
        anchor.advance(0.0)
    assert anchor.gain() <= 20.0
    assert abs(anchored_level(1.0, anchor)) <= 1.0


def test_minimum_hold_bounds_turnover_under_adversarial_sampling():
    """Regression: sampled execution previously burned ~30% of equity per quarter.

    The action stream was indistinguishable from independent draws; the hold is
    what makes that churn structurally impossible rather than merely undesirable.
    """
    rng = np.random.default_rng(5)
    levels = rng.choice([-1.0, 0.0, 1.0], size=2000)
    turnovers = {}
    for hold in (1, 4, 12):
        config = TradingConfig(minimum_hold_events=hold, gate_warmup_events=10**9)
        state = LedgerState()
        total = 0.0
        for level in levels:
            state, row = transition(state, float(level), 0.6, 0.0, 0.0, config)
            total += row["shadow_turnover"]
        turnovers[hold] = total
    assert turnovers[4] < 0.45 * turnovers[1]
    assert turnovers[12] < 0.45 * turnovers[4]


def test_no_trade_band_and_hold_are_reported_as_suppressed_requests():
    config = TradingConfig(minimum_hold_events=6, gate_warmup_events=10**9)
    state = LedgerState()
    state, first = transition(state, 1.0, 0.6, 0.0, 0.0, config)
    assert first["shadow_position"] == pytest.approx(0.6)
    state, second = transition(state, -1.0, 0.6, 0.0, 0.0, config)
    assert second["requested_position"] == pytest.approx(-0.6)
    assert second["shadow_position"] == pytest.approx(0.6)  # held


def test_straight_through_rules_are_exact_forward_and_transparent_backward():
    """The band and the gate are bypassed in the backward pass; the hold is not.

    A banded or gated event is a decision the policy made that the execution
    layer overrode, so the policy should still learn from it.  A held event is
    not a decision at all, so its exact zero gradient is the right one.
    """
    for rule, args, expected_gradient in (
        (apply_no_trade_band, (torch.tensor([0.02], requires_grad=True),
                               torch.tensor([0.0]), 0.10), 1.0),
        (apply_gate, (torch.tensor([0.9], requires_grad=True), False), 1.0),
        (apply_minimum_hold, (torch.tensor([0.9], requires_grad=True),
                              torch.tensor([0.0]), 1.0, 6), 0.0),
    ):
        target = args[0]
        out = rule(*args)
        out.backward()
        assert float(out.detach()) == 0.0                # exact in the forward pass
        assert float(target.grad) == pytest.approx(expected_gradient)


def test_no_op_events_carry_no_policy_gradient():
    """Regression: with a minimum hold of four, three in four sampled actions
    cannot move the book, and must not be credited with what followed."""
    from cryptorl.online.ledger import may_decide
    assert may_decide(0, 4) is False and may_decide(4, 4) is True
    assert may_decide(0, 1) is True  # no hold means every event is a decision
    elapsed = torch.tensor([0.0, 3.0, 4.0, 9.0])
    assert torch.equal(may_decide(elapsed, 4),
                       torch.tensor([False, False, True, True]))


def test_shadow_and_live_accounting_reconciles_and_gate_can_flatten():
    config = TradingConfig(gate_warmup_events=0, minimum_hold_events=1)
    state = LedgerState(shadow_edge=-0.01)
    next_state, row = transition(state, 1.0, 0.5, 0.02, -0.001, config)
    assert row["shadow_position"] == pytest.approx(0.5)
    assert row["live_position"] == 0.0
    expected_shadow = 10_000 * (1 + 0.5 * 0.02 + 0.5 * -0.001 - 0.5 * 0.0006)
    assert next_state.shadow_equity == pytest.approx(expected_shadow)
    assert next_state.live_equity == 10_000


def test_terminal_liquidation_charges_one_leg_and_flattens():
    config = TradingConfig()
    state = LedgerState(shadow_position=-0.5, live_position=0.25)
    closed, row = liquidate(state, config)
    assert closed.shadow_position == closed.live_position == 0.0
    assert closed.shadow_equity == pytest.approx(10_000 * (1 - 0.5 * 0.0006))
    assert closed.live_equity == pytest.approx(10_000 * (1 - 0.25 * 0.0006))
    assert row["shadow_closing_turnover"] == 0.5


# --------------------------------------------------------------------------
# The rule that the trained object equals the executed object
# --------------------------------------------------------------------------


def test_training_rollout_books_exactly_what_execution_books():
    """Regression: training previously optimised an unbanded, ungated,
    unheld path that the ledger never executed."""
    rng = np.random.default_rng(11)
    length = 120
    levels = rng.uniform(-1, 1, length)
    gross = rng.normal(0, 0.005, length)
    funding = rng.normal(0, 1e-5, length)
    scales = rng.uniform(0.3, 1.0, length)
    trading = TradingConfig(minimum_hold_events=5)

    state = LedgerState(events_since_change=trading.minimum_hold_events)
    state.observations = trading.gate_warmup_events
    scalar = []
    for i in range(length):
        state, row = transition(state, float(levels[i]), float(scales[i]),
                                float(gross[i]), float(funding[i]), trading)
        scalar.append((row["shadow_position"], row["live_position"], row["live_net_return"]))

    # Double precision throughout so the comparison isolates the rules rather
    # than float32 rounding against the scalar path's Python floats.
    book = {key: value.double()
            for key, value in _initial_book(1, trading, torch.device("cpu")).items()}
    double = lambda value: torch.tensor([value], dtype=torch.float64)
    batched = []
    for i in range(length):
        book, accounting = _batched_transition(
            book, double(levels[i]), 1.0, double(scales[i]), double(gross[i]),
            double(funding[i]), trading)
        batched.append((float(book["shadow"]), float(book["live"]),
                        float(accounting["live_net"])))
    assert np.allclose(np.asarray(scalar), np.asarray(batched), atol=1e-9)


def test_training_rollout_keeps_the_recurrent_position_gradient():
    trading = TradingConfig(minimum_hold_events=1, gate_warmup_events=10**9)
    book = _initial_book(1, trading, torch.device("cpu"))
    level = torch.tensor([0.8], requires_grad=True)
    for _ in range(3):
        book, accounting = _batched_transition(
            book, level, 1.0, torch.tensor([0.8]), torch.tensor([0.004]),
            torch.tensor([0.0]), trading)
    accounting["live_net"].backward()
    assert level.grad is not None and abs(float(level.grad)) > 0.0


# --------------------------------------------------------------------------
# Models and execution
# --------------------------------------------------------------------------


def test_models_have_no_deterministic_execution_method():
    direct = DirectPositionGRU(5, 8)
    ppo = SampledPPOGRU(5, 8)
    assert not hasattr(direct, "act_deterministic")
    assert not hasattr(ppo, "act_deterministic")
    assert not hasattr(ppo, "argmax_action")


def test_sampled_ppo_uses_reproducible_local_generator():
    logits = torch.tensor([[0.1, 0.2, 0.3]]).repeat(20, 1)
    one = torch.Generator(device="cpu").manual_seed(7)
    two = torch.Generator(device="cpu").manual_seed(7)
    first = SampledPPOGRU.sample(logits, one)[0]
    second = SampledPPOGRU.sample(logits, two)[0]
    assert torch.equal(first, second)
    assert len(torch.unique(first)) > 1


def test_regime_sampler_returns_contiguous_sequences(market):
    indices = market.transition_indices(TRAIN_START, TRAIN_END)
    sampler = RegimeBalancedSampler(market, indices, 32, 180)
    selected, correction = sampler.sample(20, np.random.default_rng(3))
    assert selected.shape == (20, 32)
    assert np.all(np.diff(selected, axis=1) == 1)
    assert np.isfinite(correction).all() and np.all(correction > 0)
    regimes = market.raw_features[selected[:, 0], MARKET_FEATURE_NAMES.index("trend_regime")] >= 0
    assert len(np.unique(regimes)) == 2


@pytest.mark.parametrize("family", ["direct_gru", "sampled_ppo"])
def test_model_forward_shapes(family):
    model = make_model(family, 6, 8)
    hidden = model.initial_hidden(3, torch.device("cpu"))
    features = torch.zeros(3, 6)
    previous = torch.zeros(3)
    if family == "direct_gru":
        level, hidden = model.step(features, previous, hidden)
        assert level.shape == (3,)
        assert torch.all(level.abs() <= 1)
    else:
        logits, value, hidden = model.step(features, previous, hidden)
        assert logits.shape == (3, 3)
        assert value.shape == (3,)
    assert hidden.shape == (1, 3, 8)


# --------------------------------------------------------------------------
# Two-stage training and online continuation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("objective_name", ["differential_sharpe", "quadratic"])
def test_second_stage_continues_from_the_first(market, objective_name):
    objective = ObjectiveConfig(objective_name)
    config = TrainingConfig(hidden_size=8, sequence_length=16, batch_size=2)
    trading = TradingConfig()
    initialisation = market.transition_indices(INITIALISATION_START, INITIALISATION_END)
    train = market.transition_indices(TRAIN_START, "2025-05-01")
    torch.manual_seed(17)
    model = make_model("direct_gru", market.raw_features.shape[1], 8)
    first = train_offline(model, market, initialisation, objective, config, trading,
                          17, torch.device("cpu"), updates=3)
    second = train_offline(model, market, train, objective, config, trading,
                           17, torch.device("cpu"), updates=3, warm=first)
    # The normaliser advanced over the new window rather than being refit.
    assert second.normalization.count > first.normalization.count
    assert second.utility.observations > first.utility.observations
    assert second.anchor.enabled == objective.is_scale_free
    # The first stage's terminal state is left intact for inspection.
    assert first.normalization.count == len(initialisation)


@pytest.mark.parametrize("family", ["direct_gru", "sampled_ppo"])
def test_online_checkpoint_continuation_is_exact(market, family):
    objective = ObjectiveConfig("quadratic")
    config = TrainingConfig(hidden_size=8, online_batch_events=4, online_epochs=1)
    trading = TradingConfig()
    indices = market.transition_indices(TRAIN_END, "2026-02-01")[:14]
    train = market.transition_indices(TRAIN_START, TRAIN_END)
    torch.manual_seed(17)
    model = make_model(family, market.raw_features.shape[1], 8)
    state = initialize_online(
        model, CausalNormalizer.fit(market.raw_features[train]), objective,
        config, trading, 17, torch.device("cpu"))
    _, state = run_online_period(
        state, market, indices[:8], objective, config, trading, torch.device("cpu"))
    flush_online_updates(state, objective, config, trading, torch.device("cpu"))
    payload = online_checkpoint(state, family, objective, config, trading, 17)
    restored = restore_online_checkpoint(payload, torch.device("cpu"))
    first, _ = run_online_period(
        state, market, indices[8:], objective, config, trading, torch.device("cpu"))
    second, _ = run_online_period(
        restored, market, indices[8:], objective, config, trading, torch.device("cpu"))
    assert np.allclose(first.target_level, second.target_level)
    assert np.allclose(first.shadow_equity, second.shadow_equity)


@pytest.mark.parametrize("family", ["direct_gru", "sampled_ppo"])
def test_partial_buffer_is_flushed_for_both_families(market, family):
    """The final partial batch must actually train, not just clear."""
    objective = ObjectiveConfig("differential_sharpe")
    config = TrainingConfig(hidden_size=8, online_batch_events=1000, online_epochs=1)
    trading = TradingConfig()
    indices = market.transition_indices(TRAIN_END, "2026-02-01")[:20]
    train = market.transition_indices(TRAIN_START, TRAIN_END)
    torch.manual_seed(23)
    model = make_model(family, market.raw_features.shape[1], 8)
    state = initialize_online(
        model, CausalNormalizer.fit(market.raw_features[train]), objective,
        config, trading, 23, torch.device("cpu"))
    _, state = run_online_period(
        state, market, indices, objective, config, trading, torch.device("cpu"))
    assert len(state.buffer) == len(indices)  # nothing was flushed mid-period
    before = {name: value.clone() for name, value in state.model.state_dict().items()}
    diagnostics = flush_online_updates(state, objective, config, trading, torch.device("cpu"))
    assert diagnostics and not state.buffer
    assert any(not torch.equal(before[name], value)
               for name, value in state.model.state_dict().items())


def test_online_frame_records_what_the_replay_needs(market):
    """Regression: the online update used to propagate its own recomputed
    exposure instead of the position the ledger actually held."""
    objective = ObjectiveConfig("differential_sharpe")
    config = TrainingConfig(hidden_size=8, online_batch_events=100)
    trading = TradingConfig()
    indices = market.transition_indices(TRAIN_END, "2026-02-01")[:20]
    train = market.transition_indices(TRAIN_START, TRAIN_END)
    torch.manual_seed(3)
    model = make_model("direct_gru", market.raw_features.shape[1], 8)
    state = initialize_online(
        model, CausalNormalizer.fit(market.raw_features[train]), objective,
        config, trading, 3, torch.device("cpu"))
    frame, state = run_online_period(
        state, market, indices, objective, config, trading, torch.device("cpu"))
    for column in ("previous_position", "previous_shadow_position",
                   "previous_events_since_change", "gate_open", "requested_position"):
        assert column in frame
    # The agent's self-state is the position it actually holds.
    assert np.allclose(frame.previous_position.to_numpy()[1:],
                       frame.live_position.to_numpy()[:-1])
    assert np.allclose(frame.previous_shadow_position.to_numpy()[1:],
                       frame.shadow_position.to_numpy()[:-1])
