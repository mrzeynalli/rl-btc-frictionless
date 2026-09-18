"""Self-contained tests for the agent: data, reward, ledger, network, PPO, evaluation, scoring, synthetic markets."""
from dataclasses import replace
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from agent.config import (FOLDS, QUARTER_BARS, SEARCH_FOLDS, SEEDS, TEST_START, YEARS, Config, TestLockedError, check_horizon, checkpoint_updates,
                          cost_break_even_hours, fold_config, protocol, quarter_episode)
from agent.data import BAR_MINUTES, CANDIDATE_FEATURES, DAY_MINUTES, FEATURE_NAMES, Market, Scaler, causal_arrays
from agent.evaluate import evaluate, metrics, quarter_windows
from agent.ledger import (Environment, Ledger, TrainingEnvironment, FLAT, LONG, SHORT, PORTFOLIO_NAMES,
                          PORTFOLIO_WIDTH)
from agent.network import ActorCritic, market_tensor, policy_entropy
from agent.ppo import Trainer, advantages, clipped_surrogate, state_action_mutual_information
from agent.reward import (differential_ratio, first_order_differential_sharpe, first_order_downside_ratio,
                          performance_ratio)
from agent.scoring import interquartile_mean, probability_of_backtest_overfitting, stratified_bootstrap_interval

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
KINDS = ('sharpe', 'sortino')
# MEASURED by Scaler.fit in v1: five-minute log-return standard deviation of BTC, 2017-2023.
BAR_SD = 0.002655


def full(**kwargs):
    """The production configuration with the reward prior a fold would measure."""
    return replace(Config(), dsr_initial_second_moment=BAR_SD ** 2, **kwargs)


def market(size=4000, drift=0.0, seed=0):
    generator = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(generator.normal(drift, 1e-4, size)))
    rows = np.repeat(price[:, None], 4, axis=1)
    features = generator.normal(0, .1, (size, len(CANDIDATE_FEATURES))).astype(np.float32)
    return Market(rows, np.ones(size, bool), np.zeros(size), features, np.full(size, 0.02, np.float32),
                  int(pd.Timestamp('2020-01-01', tz='UTC').value // 60_000_000_000), {}, history_minutes=1)


def scaler():
    return Scaler(np.zeros(len(FEATURE_NAMES), np.float32), np.ones(len(FEATURE_NAMES), np.float32),
                  '2021-01-01T00:00:00Z', BAR_SD)


def config(**kwargs):
    # Reduced sizes need a reduced horizon and reward window, since Config
    # enforces that rollout, recurrent span and reward window fit the horizon.
    # Mechanics are tested one bar per decision; the hourly production interval has its own tests.
    base = dict(decision_bars=1, environments=4, rollout=16, sequence=8, burn_in=8, episode_steps=64, minibatch_sequences=2,
                hidden=8, head_width=8, gamma=.9, gae_lambda=.85, dsr_eta=1/32, epochs=1, updates=2,
                evaluation_seeds=2, bootstrap_replications=200, dsr_initial_second_moment=BAR_SD ** 2)
    return replace(Config(), **{**base, **kwargs})


def trainer(c, seed=11, size=4000, drift=0.0):
    torch.manual_seed(seed)
    env = TrainingEnvironment(market(size, drift=drift), (100, size), c, np.random.default_rng(seed))
    return Trainer(ActorCritic(c), env, scaler(), c, seed)


def minute_bars(minutes=3 * DAY_MINUTES + 200, seed=0, scale=1.0, volume_scale=1):
    """OHLCV built from a log-price path, so scaling the path scales every log move."""
    generator = np.random.default_rng(seed)
    start = pd.Timestamp('2020-03-01', tz='UTC')
    log_close = np.cumsum(generator.normal(0, 5e-4, minutes)) * scale
    log_open = np.r_[0.0, log_close[:-1]]
    high = np.maximum(log_open, log_close) + np.abs(generator.normal(0, 3e-4, minutes)) * scale
    low = np.minimum(log_open, log_close) - np.abs(generator.normal(0, 3e-4, minutes)) * scale
    frame = pd.DataFrame({'open_time': pd.date_range(start, periods=minutes, freq='min'),
                          'open': 100 * np.exp(log_open), 'high': 100 * np.exp(high), 'low': 100 * np.exp(low),
                          'close': 100 * np.exp(log_close),
                          'volume': generator.integers(100, 1000, minutes).astype(np.int64) * volume_scale})
    missing = generator.choice(np.arange(1, minutes), 25, replace=False)
    frame = frame.drop(index=missing).reset_index(drop=True)
    stamps = pd.date_range(start + pd.Timedelta(hours=4), start + pd.Timedelta(minutes=minutes), freq='8h',
                           inclusive='left')
    funding = pd.DataFrame({'timestamp': stamps, 'fundingRate': generator.normal(1e-4, 1e-4, len(stamps)),
                            'funding_interval_hours': 8.0})
    return frame, funding, start, start + pd.Timedelta(minutes=minutes)


def total_reward(path, c):
    mean, second, total = np.zeros(1), np.full(1, c.initial_reward_second_moment), 0.0
    for value in path:
        d, mean, second = differential_ratio(np.array([value]), mean, second, c)
        total += float(d[0])
    return total


# --- specification -------------------------------------------------------------

def test_credit_window_clears_the_cost_break_even_measured_on_training_data():
    c = full()
    assert c.gamma == c.gae_lambda == 0.99825
    break_even = check_horizon(c, BAR_SD)
    assert 12 < break_even < 13 and c.credit_window_hours > break_even
    assert c.rollout / 2 >= c.credit_window and c.burn_in + c.sequence >= c.credit_window
    # The validation years' volatility is what once put the break-even at 22 h.
    assert cost_break_even_hours(0.00148, c) == pytest.approx(21.9, abs=0.1)


def test_the_predecessor_horizon_is_refused():
    short = replace(full(), decision_bars=1, rollout=1024, sequence=256, burn_in=128,
                    episode_steps=quarter_episode(1, 1024), gamma=.9975961328836203, gae_lambda=.9738730024397007)
    assert short.credit_window_hours < 3
    with pytest.raises(ValueError, match='below the cost break-even'):
        check_horizon(short, BAR_SD)


def test_cost_is_realistic_and_fixed_with_no_curriculum():
    c = Config()
    assert (c.fee, c.slippage) == (0.0005, 0.0001)
    assert c.round_trip_cost == pytest.approx(0.0012)
    assert not any('curriculum' in name or 'stress' in name for name in c.as_dict())


def test_rollout_must_carry_its_own_credit_window():
    with pytest.raises(ValueError, match='truncates the credit window'):
        replace(full(), rollout=32, sequence=32)
    with pytest.raises(ValueError, match='truncates the credit window'):
        replace(full(), decision_bars=1, rollout=512, sequence=128, burn_in=64, episode_steps=quarter_episode(1, 512))


def test_recurrent_span_must_cover_the_credit_window():
    with pytest.raises(ValueError, match='recurrent span'):
        replace(full(), sequence=8, burn_in=8)


def test_reward_window_lies_between_credit_window_and_episode():
    with pytest.raises(ValueError, match='differential Sharpe window'):
        replace(full(), dsr_eta=1/200)
    with pytest.raises(ValueError, match='differential Sharpe window'):
        replace(full(), dsr_eta=1/30000)


def test_reward_must_be_sharpe_or_sortino():
    with pytest.raises(ValueError, match='reward must be one of'):
        replace(Config(), reward='sortino_ratio')
    assert Config().reward == 'sharpe'


def test_decisions_are_hourly_and_keep_the_horizon_in_time():
    hourly = full()
    five = replace(hourly, decision_bars=1, rollout=1024, sequence=256, burn_in=128, episode_steps=quarter_episode(1, 1024))
    assert hourly.decision_bars * hourly.decision_minutes == 60
    assert hourly.decision_gamma == pytest.approx(hourly.gamma ** 12)
    assert hourly.credit_window_hours == pytest.approx(five.credit_window_hours, rel=0.03)


def test_outer_folds_inner_search_folds_and_five_seeds():
    assert FOLDS == {2024: 4, 2025: 4} and YEARS == (2024, 2025)
    assert SEARCH_FOLDS == {2022: 4, 2023: 4} and max(SEARCH_FOLDS) < min(FOLDS)
    assert len(SEEDS) == len(set(SEEDS)) == 5
    # USER: the whole of 2026 is the test window; the last validation quarter ends where it starts.
    assert TEST_START == pd.Timestamp('2026-01-01', tz='UTC') == quarter_windows(2025, FOLDS[2025])[-1][1]
    assert dict(Config().folds) == FOLDS and Config().seeds == SEEDS and Config().enforce_horizon


def test_config_round_trips_through_the_frozen_protocol():
    c = replace(Config(), folds=tuple(SEARCH_FOLDS.items()), seeds=(11,), enforce_horizon=False, gamma=0.999)
    assert Config.from_dict(json.loads(json.dumps(c.as_dict()))) == c


def test_training_episode_is_a_quarter_to_the_nearest_rollout():
    c = Config()
    days = c.episode_steps * c.decision_bars * c.decision_minutes / DAY_MINUTES
    assert 60 <= days <= 93 and c.episode_steps % c.rollout == 0
    assert QUARTER_BARS == 26_298
    assert c.episode_steps == quarter_episode(c.decision_bars) == 2_176
    assert quarter_episode(1, 1024) == 26 * 1024


def test_update_shape_is_ppo_atari_as_cleanrl_runs_it():
    """Schulman et al. 2017 Table 5 and CleanRL ppo_atari_lstm: 8 actors x 128 steps, 4 minibatches, 4 epochs."""
    c = Config()
    assert (c.environments, c.rollout, c.epochs) == (8, 128, 4)
    assert c.samples_per_update == 1024
    assert c.samples_per_update // (c.environments // c.minibatch_sequences) == 32 * 8
    assert c.gradient_steps_per_update == 16
    assert c.sequence == c.rollout and c.burn_in / c.sequence == 0.5


def test_literature_defaults():
    c = Config()
    assert c.clip == 0.1 and c.actor_lr == c.critic_lr == 2.5e-4 and c.anneal_lr
    assert c.entropy_coefficient == 0.01 and c.gradient_clip == 0.5 and c.recompute_advantages_per_pass
    assert c.training_passes == 60 and c.validation_passes == 5
    assert c.training_cost_multiplier == 2.0 and c.maintenance == 0.004 and c.fee == 0.0005
    assert c.observation_clip == 10.0
    assert c.bootstrap_replications == 50_000 and c.confidence_level == 0.95


def test_the_budget_is_counted_in_passes_over_each_folds_training_span():
    base = Config()
    shorter, longer = fold_config(base, BAR_SD, 61_368), fold_config(base, BAR_SD, 70_128)
    for fold, decisions in ((shorter, 61_368), (longer, 70_128)):
        assert fold.updates * fold.samples_per_update / decisions == pytest.approx(base.training_passes, rel=1e-3)
        points = checkpoint_updates(fold)
        assert len(points) == base.training_passes // base.validation_passes and points[-1] == fold.updates
        passes = np.array(points) * fold.samples_per_update / decisions
        np.testing.assert_allclose(np.diff(passes), base.validation_passes, rtol=0.01)
    assert longer.updates > shorter.updates
    assert shorter.dsr_initial_second_moment == pytest.approx(BAR_SD ** 2)
    with pytest.raises(ValueError, match='measured on the training span'):
        fold_config(base, None, 61_368)


def test_the_reward_prior_is_never_a_constant():
    assert Config().dsr_initial_second_moment is None and Config().updates is None
    with pytest.raises(ValueError, match='measured on the training span'):
        Ledger(1, Config())


def test_scaler_measures_volatility_on_the_training_span_only():
    m = market(30_000, seed=3)
    end = m.timestamp(20_000)
    fitted = Scaler.fit(m, end)
    first, last = m.bounds(pd.Timestamp('2017-01-01', tz='UTC'), end)
    closes = np.log(np.asarray(m.prices[first - 1:last - 1:BAR_MINUTES, 3]))
    assert fitted.bar_return_sd == pytest.approx(float(np.diff(closes).std()))
    m.prices[25_000:] *= 3.0
    assert Scaler.fit(m, end).bar_return_sd == fitted.bar_return_sd
    assert Scaler.from_dict(fitted.as_dict()).bar_return_sd == fitted.bar_return_sd


def test_every_constant_has_a_documented_provenance():
    """ARCHITECTURE.md §12 must name every Config field; nothing may be undocumented."""
    text = (ROOT / 'ARCHITECTURE.md').read_text()
    section = text[text.index('## 12.'):]
    missing = [name for name in Config().as_dict() if f'`{name}`' not in section]
    assert not missing, missing


def test_config_survives_the_json_round_trip():
    c = full(updates=7)
    assert c.as_dict() == json.loads(json.dumps(c.as_dict()))
    assert json.loads(json.dumps(Config().as_dict()))['dsr_initial_second_moment'] is None


def test_reward_scale_keeps_the_value_target_at_unit_scale():
    c = full()
    assert c.reward_scale == pytest.approx(math.sqrt((1 - c.gamma ** 24) / 12))
    assert replace(c, reward='sortino').reward_scale == pytest.approx(math.sqrt(0.5) * c.reward_scale)
    five = replace(c, decision_bars=1, rollout=1024, sequence=256, burn_in=128, episode_steps=quarter_episode(1, 1024))
    assert five.reward_scale == pytest.approx(math.sqrt(1 - c.gamma ** 2))
    assert 'reward_scale' not in c.as_dict()


def test_scaled_rewards_of_both_ratios_have_the_same_spread():
    generator = np.random.default_rng(0)
    spreads = []
    for kind in KINDS:
        c = full(reward=kind)
        mean, second, values = np.zeros(1), np.full(1, c.initial_reward_second_moment), []
        for value in generator.normal(0, BAR_SD, 30_000):
            d, mean, second = differential_ratio(np.array([value]), mean, second, c)
            values.append(float(d[0]) * c.reward_scale)
        spreads.append(np.std(values[5000:]))
    assert spreads[1] == pytest.approx(spreads[0], rel=0.05)


def test_protocol_records_the_reasoning():
    spec = protocol()
    assert spec['test_locked'] and not spec['test_performance_accessed']
    assert 'no curriculum' in spec['cost_policy']
    assert 'argmax' in spec['evaluation'] and 'training still samples' in spec['evaluation']
    assert spec['signal_sharpe_assumption'] == 1.0 and 'passes' in spec['budget']
    assert spec['validation_folds'] == {'2024': 4, '2025': 4}
    assert 'USDT' in spec['objective'] and 'BitMEX' in spec['instrument'] and 'linear' in spec['instrument']
    assert 'evaluation charges the real cost' in spec['cost_policy']
    assert 'Moskowitz' in spec['references']['volatility-normalised returns over several horizons']
    assert spec['state']['market'] == list(FEATURE_NAMES)
    assert spec['state']['account'] == list(PORTFOLIO_NAMES)
    assert 'Lyle' in spec['references']['layer normalisation']
    assert 'Hsu' in spec['references']['PPO discrete-action failure mode']
    assert 'Moody' in spec['references']['training budget in passes, stopping by validation']
    assert 'CleanRL' in spec['references']['PPO discrete-action settings']
    assert 'Sortino' in protocol(replace(Config(), reward='sortino'))['objective']


# --- market features -------------------------------------------------------------

def test_state_is_the_preregistered_survivors_without_clock_or_single_bar_features():
    assert len(FEATURE_NAMES) == 13 and set(FEATURE_NAMES) <= set(CANDIDATE_FEATURES)
    assert list(FEATURE_NAMES) == [name for name in CANDIDATE_FEATURES if name in FEATURE_NAMES]
    for removed in ('drawdown_4w', 'trend_efficiency_1w'):
        assert removed not in FEATURE_NAMES
    assert not any(token in name for name in FEATURE_NAMES for token in ('hour', 'phase', 'bar_'))


def test_features_are_causal():
    """Changing minute k may change the observation from the next five-minute boundary on, never before."""
    frame, funding, start, end = minute_bars()
    base = causal_arrays(frame, funding, start, end)
    present = set(((frame.open_time - start) // pd.Timedelta(minutes=1)).tolist())
    k = next(m for m in range(2 * DAY_MINUTES + 30, 3 * DAY_MINUTES) if m % BAR_MINUTES == 4 and m in present)
    boundary = k + 1
    changed = frame.copy()
    row = changed.index[changed.open_time == start + pd.Timedelta(minutes=k)][0]
    changed.loc[row, ['close', 'high']] = changed.loc[row, 'high'] * 1.02
    changed.loc[row, 'volume'] *= 50
    other = causal_arrays(changed, funding, start, end)
    np.testing.assert_array_equal(base.features[:boundary], other.features[:boundary])
    np.testing.assert_array_equal(base.daily_volatility[:boundary], other.daily_volatility[:boundary])
    # The volume regime compares a week with the four before it; three days of bars cannot move it.
    for name in ('trend_efficiency_1w', 'volatility_regime', 'return_1h', 'return_1d'):
        column = CANDIDATE_FEATURES.index(name)
        assert base.features[boundary, column] != other.features[boundary, column], name
    settled = funding.copy()
    settled.loc[3, 'fundingRate'] += 0.01
    event = int((settled.timestamp[3] - start) / pd.Timedelta(minutes=1))
    shifted = causal_arrays(frame, settled, start, end)
    visible = (event // BAR_MINUTES + 1) * BAR_MINUTES
    np.testing.assert_array_equal(base.features[:visible], shifted.features[:visible])
    column = CANDIDATE_FEATURES.index('funding_rate')
    assert shifted.features[visible, column] == pytest.approx(settled.fundingRate[3], rel=1e-5)


def test_minutes_between_boundaries_carry_the_last_boundary_observation():
    frame, funding, start, end = minute_bars()
    m = causal_arrays(frame, funding, start, end)
    boundary = 2 * DAY_MINUTES
    for offset in range(1, BAR_MINUTES):
        np.testing.assert_array_equal(m.features[boundary + offset], m.features[boundary])
        assert m.daily_volatility[boundary + offset] == m.daily_volatility[boundary]


def test_features_are_scale_free_except_the_volatility_level():
    """Twice the volatility and a thousand times the volume look the same; only the level moves, by log 2."""
    frame, funding, start, end = minute_bars()
    wild, _, _, _ = minute_bars(scale=2.0, volume_scale=1000)
    calm = causal_arrays(frame, funding, start, end)
    loud = causal_arrays(wild, funding, start, end)
    rows = slice(DAY_MINUTES + 10, None)
    for name in ('trend_efficiency_1w', 'drawdown_4w', 'volatility_regime', 'return_1h', 'return_1d', 'return_1w'):
        i = CANDIDATE_FEATURES.index(name)
        np.testing.assert_allclose(calm.features[rows, i], loud.features[rows, i], atol=2e-3, err_msg=name)
    i = CANDIDATE_FEATURES.index('volume_regime')
    np.testing.assert_allclose(calm.features[rows, i], loud.features[rows, i], atol=2e-2)
    i = CANDIDATE_FEATURES.index('volatility_level')
    np.testing.assert_allclose(loud.features[rows, i] - calm.features[rows, i], math.log(2), atol=2e-4)
    np.testing.assert_allclose(loud.daily_volatility[rows], 2 * calm.daily_volatility[rows], rtol=1e-3)


def test_trend_efficiency_is_one_on_a_straight_rise_and_bounded_everywhere():
    minutes = 3 * DAY_MINUTES
    start = pd.Timestamp('2020-03-01', tz='UTC')
    close = 100 * np.exp(1e-4 * np.arange(1, minutes + 1))
    frame = pd.DataFrame({'open_time': pd.date_range(start, periods=minutes, freq='min'), 'open': close,
                          'high': close, 'low': close, 'close': close, 'volume': 500.0})
    funding = pd.DataFrame({'timestamp': pd.DatetimeIndex([], tz='UTC'), 'fundingRate': np.zeros(0),
                            'funding_interval_hours': np.zeros(0)})
    m = causal_arrays(frame, funding, start, start + pd.Timedelta(minutes=minutes))
    rows = np.arange(2 * BAR_MINUTES, minutes, BAR_MINUTES)
    np.testing.assert_allclose(m.features[rows, CANDIDATE_FEATURES.index('trend_efficiency_1w')], 1.0, atol=1e-6)
    np.testing.assert_allclose(m.features[rows, CANDIDATE_FEATURES.index('drawdown_4w')], 0.0, atol=1e-6)
    noisy, noisy_funding, s, e = minute_bars()
    column = CANDIDATE_FEATURES.index('trend_efficiency_1w')
    assert np.abs(causal_arrays(noisy, noisy_funding, s, e).features[:, column]).max() <= 1.0


def test_scaler_clips_and_refuses_the_first_row():
    m = market(200)
    m.features[50] = 1e6
    s = scaler()
    assert np.abs(s.observations(m, [50], 10.0)).max() == 10.0
    with pytest.raises(ValueError):
        s.observations(m, [0], 10.0)


def test_market_bounds_respect_the_feature_history():
    m = market(60_000)
    m.history_minutes = 30_000
    first, _ = m.bounds(m.timestamp(10), m.timestamp(50_000))
    assert first >= 30_000 and (m.epoch_start + first) % 5 == 0


# --- reward -------------------------------------------------------------------------

@pytest.mark.parametrize('kind', KINDS)
def test_exact_increment_reduces_to_the_first_order_term_for_ordinary_returns(kind):
    c = full(reward=kind)
    sd = math.sqrt(c.dsr_initial_second_moment)
    mean = np.full(7, 0.1 * sd)
    second = np.full(7, c.initial_reward_second_moment)
    returns = sd * np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    exact, _, _ = differential_ratio(returns, mean, second, c)
    first_order = first_order_differential_sharpe if kind == 'sharpe' else first_order_downside_ratio
    np.testing.assert_allclose(exact, first_order(returns, mean, second), rtol=0.01, atol=0.02)


@pytest.mark.parametrize('kind', KINDS)
def test_exact_increment_is_bounded_and_graded_for_catastrophic_losses(kind):
    """The old clip paid -10 for a 20-sd loss and a wipeout alike; the exact change grades them."""
    c = full(reward=kind)
    losses = np.array([-1 * BAR_SD, -5 * BAR_SD, -20 * BAR_SD, math.log(1e-12)])
    values, _, _ = differential_ratio(losses, np.zeros(4), np.full(4, c.initial_reward_second_moment), c)
    assert (np.diff(values) < 0).all()
    bound = 1 / math.sqrt(c.dsr_eta * (1 - c.dsr_eta)) if kind == 'sharpe' else 1 / math.sqrt(c.dsr_eta)
    assert np.abs(values).max() <= bound + 1e-6
    assert values[2] < -10


@pytest.mark.parametrize('kind', KINDS)
def test_rewards_telescope_to_the_change_in_the_ratio(kind):
    c = config(reward=kind)
    generator = np.random.default_rng(3)
    mean, second = np.zeros(1), np.full(1, c.initial_reward_second_moment)
    start, total = float(performance_ratio(mean, second, kind)[0]), 0.0
    for value in generator.normal(1e-4, 1.5e-3, 300):
        d, mean, second = differential_ratio(np.array([value]), mean, second, c)
        total += float(d[0])
    assert total * c.dsr_eta == pytest.approx(float(performance_ratio(mean, second, kind)[0]) - start, abs=1e-9)


@pytest.mark.parametrize('kind', KINDS)
def test_differential_ratio_is_scale_invariant(kind):
    c = config(reward=kind)
    r = np.array([0.001, -0.002, 0.0005])
    a, b = np.zeros(3), np.full(3, c.initial_reward_second_moment)
    plain, _, _ = differential_ratio(r, a, b, c)
    scaled, _, _ = differential_ratio(100 * r, 100 * a, 10000 * b, c)
    np.testing.assert_allclose(plain, scaled, rtol=1e-9)


def test_differential_sharpe_prefers_steadiness_to_magnitude():
    c = config()
    steady, lumpy = np.full(200, 0.001), np.tile([0.021, -0.019], 100)
    assert steady.sum() == pytest.approx(lumpy.sum())
    assert total_reward(steady, c) > total_reward(lumpy, c)


def test_downside_ratio_does_not_penalise_upside_dispersion():
    """Moody & Saffell: the downside deviation ratio has "no penalty for large positive returns"."""
    even = np.tile([0.002, -0.001, 0.002, -0.001], 50)
    spread = np.tile([0.0035, -0.001, 0.0005, -0.001], 50)
    assert even.sum() == pytest.approx(spread.sum())
    changes = {}
    for kind in KINDS:
        c = config(reward=kind)
        smooth, dispersed = total_reward(even, c), total_reward(spread, c)
        changes[kind] = (dispersed - smooth) / abs(smooth)
    assert changes['sharpe'] < -0.2
    assert abs(changes['sortino']) < abs(changes['sharpe']) / 4


@pytest.mark.parametrize('kind', KINDS)
def test_leverage_ratchet_does_not_pay(kind):
    c = config(reward=kind)
    base = np.random.default_rng(5).normal(2e-4, 1.5e-3, 400)
    assert total_reward(2 * base, c) == pytest.approx(total_reward(base, c), rel=0.2)


@pytest.mark.parametrize('kind', KINDS)
def test_terminated_accounts_freeze_their_reward_statistics(kind):
    c = config(reward=kind)
    a, b = np.array([0.5, 0.5]), np.array([1.0, 1.0])
    d, a2, b2 = differential_ratio(np.array([-0.01, -0.01]), a, b, c, active=np.array([True, False]))
    assert d[1] == 0.0 and a2[1] == a[1] and b2[1] == b[1]
    assert d[0] != 0.0 and a2[0] != a[0]


def test_downside_prior_is_half_the_variance():
    c = config(reward='sortino')
    ledger = Ledger(2, c)
    assert ledger.dsr_second[0] == pytest.approx(c.dsr_initial_second_moment / 2)
    assert ledger.portfolio(np.full(2, 100.), 0.02)[0, PORTFOLIO_NAMES.index('relative_risk')] == pytest.approx(0.0)


def test_environment_reports_both_reward_signals():
    c = config()
    env = Environment(market(600), (100, 600), c, np.random.default_rng(0), training=False)
    _, _, info = env.step(np.array([LONG]))
    assert 'log_return' in info and 'differential_ratio' in info


# --- account and ledger ---------------------------------------------------------------

def test_account_observation_carries_side_trade_and_reward_state():
    assert PORTFOLIO_WIDTH == len(PORTFOLIO_NAMES) == 8
    assert PORTFOLIO_NAMES[:3] == ('short', 'flat', 'long')
    assert Ledger(4, config()).portfolio(np.full(4, 100.), 0.02).shape == (4, PORTFOLIO_WIDTH)


def test_reward_statistics_are_visible_to_the_policy():
    c = config()
    ledger = Ledger(2, c)
    before = ledger.portfolio(np.full(2, 100.), 0.02).copy()
    ledger.dsr_mean[:] = 0.0002
    ledger.dsr_second[:] = 4 * c.dsr_initial_second_moment
    after = ledger.portfolio(np.full(2, 100.), 0.02)
    changed = set(np.flatnonzero(np.abs(after - before).sum(0) > 0))
    assert changed == {PORTFOLIO_NAMES.index('running_ratio'), PORTFOLIO_NAMES.index('relative_risk')}


def test_trade_age_counts_bars_held_in_credit_windows():
    c = config()
    env = Environment(market(3000), (200, 3000), c, np.random.default_rng(0), training=False)
    column = PORTFOLIO_NAMES.index('trade_age')
    assert env.observation()[1][0, column] == 0.0
    for held in range(1, 6):
        env.step(np.array([LONG]))
        assert env.observation()[1][0, column] == pytest.approx(math.log1p(held / c.credit_window), rel=1e-5)
    env.step(np.array([FLAT]))
    assert env.observation()[1][0, column] == 0.0


def test_a_decision_of_several_bars_equals_that_many_one_bar_decisions():
    m = market(3000, drift=2e-5, seed=4)
    one, three = config(), config(decision_bars=3)
    fine = Environment(m, (200, 3000), one, np.random.default_rng(0), training=False)
    coarse = Environment(m, (200, 3000), three, np.random.default_rng(0), training=False)
    for action in (LONG, SHORT, FLAT):
        increments, returns = 0.0, 0.0
        for _ in range(3):
            _, _, info = fine.step(np.array([action]))
            increments += float(info['differential_ratio'][0])
            returns += float(info['log_return'][0])
        reward, _, info = coarse.step(np.array([action]))
        assert info['differential_ratio'][0] == pytest.approx(increments, rel=1e-9)
        assert info['log_return'][0] == pytest.approx(returns, rel=1e-9, abs=1e-15)
        assert reward[0] == pytest.approx(increments * three.reward_scale, rel=1e-6)
        assert coarse.cursor[0] == fine.cursor[0]
        for name in ('wallet', 'contracts', 'dsr_mean', 'dsr_second', 'age'):
            np.testing.assert_allclose(getattr(coarse.ledger, name), getattr(fine.ledger, name), rtol=1e-12)


def test_linear_contract_settles_profit_in_usdt():
    """A 1x long gains the price return and a 1x short loses it; flat keeps its USDT."""
    c = replace(full(), fee=0.0, slippage=0.0)
    for action, sign in ((LONG, 1), (SHORT, -1), (FLAT, 0)):
        for move in (1.1, 0.9):
            ledger = Ledger(1, c)
            ledger.execute(np.array([action]), np.array([50_000.0]), np.array([True]))
            ledger.mark(np.array([50_000.0 * move]))
            assert ledger.equity[0] == pytest.approx(1 + sign * (move - 1), rel=1e-12)


def test_exposure_feature_is_notional_over_equity_and_a_one_x_long_is_never_liquidated():
    c = full()
    ledger = Ledger(1, c)
    ledger.execute(np.array([LONG]), np.array([100.0]), np.array([True]))
    column = PORTFOLIO_NAMES.index('exposure')
    for price in np.geomspace(100, 0.5, 4000):
        mark = np.array([price])
        ratio = float(np.abs(ledger.contracts[0]) * price / ledger.value(mark)[0])
        feature = float(ledger.portfolio(mark, 0.02)[0, column])
        assert feature == pytest.approx(math.log1p(min(ratio, math.e ** c.observation_clip - 1)), rel=1e-5)
        ledger.mark(mark, mark)
        assert not ledger.terminated[0], price


def test_a_one_x_short_is_liquidated_near_a_doubling():
    """Equity W + Q(P - entry) meets |Q| P (m + f) at P = (W - Q entry) / (|Q| (m + f) - Q): about 199 here."""
    c = full()
    ledger = Ledger(1, c)
    ledger.execute(np.array([SHORT]), np.array([100.0]), np.array([True]))
    q, w, entry = ledger.contracts[0], ledger.wallet[0], ledger.entry[0]
    expected = (w - q * entry) / (abs(q) * (c.maintenance + c.fee) - q)
    for price in np.linspace(100, 250, 150_001):
        ledger.mark(np.array([price]), np.array([price]))
        if ledger.terminated[0]:
            assert price == pytest.approx(expected, abs=2e-3) and 198 < price < 200
            break
    else:
        pytest.fail('the short was never liquidated')


def test_turnover_is_not_inflated_by_a_liquidation():
    c = full()
    ledger = Ledger(1, c)
    ledger.execute(np.array([SHORT]), np.array([100.0]), np.array([True]))
    ledger.mark(np.array([250.0]), np.array([250.0]))
    assert ledger.terminated[0]
    assert ledger.turnover_total[0] < 5


def test_training_prefix_produces_all_three_sides_in_balance():
    c = replace(config(), environments=300, burn_in=64, rollout=64, sequence=32, minibatch_sequences=3,
                episode_steps=128, dsr_eta=1/100)
    env = TrainingEnvironment(market(6000), (100, 6000), c, np.random.default_rng(3))
    sides = np.sign(env.ledger.contracts)
    for value in (-1.0, 0.0, 1.0):
        assert 0.2 < np.mean(sides == value) < 0.5
    ages = env.ledger.age[env.ledger.contracts != 0]
    assert ages.min() < ages.max() <= c.burn_in * 5
    assert env.ledger.fees_total.sum() > 0


def test_first_episodes_are_staggered_by_whole_rollouts():
    c = replace(config(), environments=40, minibatch_sequences=4)
    env = TrainingEnvironment(market(6000), (100, 6000), c, np.random.default_rng(1))
    assert (env.steps % c.rollout == 0).all() and (env.steps < c.episode_steps).all()
    assert len(np.unique(env.steps)) > 1


def test_training_states_are_reproducible():
    c = replace(config(), environments=30, burn_in=64, minibatch_sequences=3, rollout=64, sequence=32,
                episode_steps=128, dsr_eta=1/100)
    env = TrainingEnvironment(market(6000), (100, 6000), c, np.random.default_rng(11))
    saved = copy.deepcopy(env.state_dict())
    mask = np.arange(env.size) % 2 == 0
    env.reset(mask)
    expected = env.state_dict()
    env.load_state_dict(saved)
    env.reset(mask)
    for key, value in expected['ledger'].items():
        np.testing.assert_array_equal(env.ledger.state_dict()[key], value)
    np.testing.assert_array_equal(env.warm_minutes, expected['warm_minutes'])


# --- network -----------------------------------------------------------------------------

def test_layer_normalisation_follows_every_hidden_layer():
    model = ActorCritic(config(hidden=16, head_width=12))
    for branch in (model.actor, model.critic):
        norms = [m for m in branch.modules() if isinstance(m, torch.nn.LayerNorm)]
        assert len(norms) == 4
        assert branch.encoder_norm.normalized_shape == branch.memory_norm.normalized_shape == (branch.hidden,)
        assert branch.first_norm.normalized_shape == branch.second_norm.normalized_shape == (branch.head_width,)
        memory, _ = branch.encode(torch.randn(3, 20, len(FEATURE_NAMES)) * 50)
        torch.testing.assert_close(memory.mean(-1), torch.zeros(3, 20), atol=1e-5, rtol=0)
        # Unit spread up to LayerNorm's own epsilon inside the variance.
        torch.testing.assert_close(memory.std(-1, unbiased=False), torch.ones(3, 20), atol=1e-2, rtol=0)


def test_initial_policy_is_near_uniform():
    """Andrychowicz et al.: a 100x smaller final policy layer."""
    model = ActorCritic(Config())
    logp, _, _ = model(torch.randn(5, 7, len(FEATURE_NAMES)), torch.randn(5, 7, PORTFOLIO_WIDTH))
    torch.testing.assert_close(logp.exp(), torch.full((5, 7, 3), 1/3), atol=0.02, rtol=0)


def test_account_never_enters_the_recurrent_memory():
    model = ActorCritic(config())
    features = torch.randn(2, 6, len(FEATURE_NAMES))
    first = model(features, torch.zeros(2, 6, PORTFOLIO_WIDTH))
    second = model(features, torch.ones(2, 6, PORTFOLIO_WIDTH))
    for a, b in zip(first[2], second[2]):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
    assert not torch.allclose(first[1], second[1])


def test_critic_cannot_update_the_actor():
    t = trainer(config())
    minute, account = t.env.observation()
    _, value, _ = t.model(market_tensor(t.scaler, t.env.market, minute[:, None], t.config),
                          torch.from_numpy(account[:, None]))
    value.sum().backward()
    assert all(p.grad is None for p in t.model.actor.parameters())


def test_critic_outputs_one_state_value():
    model = ActorCritic(config())
    assert model.critic.output.out_features == 1 and model.actor.output.out_features == 3


def test_value_network_is_wider_than_the_policy():
    model = ActorCritic(Config())
    assert model.critic.hidden == 2 * model.actor.hidden and model.critic.head_width == 2 * model.actor.head_width
    assert sum(p.numel() for p in model.critic.parameters()) > sum(p.numel() for p in model.actor.parameters())


# --- PPO ----------------------------------------------------------------------------------

def test_replay_and_resume_are_exact():
    c = config()
    t = trainer(c)
    first = t.run_update()
    assert max(first['replay_probability_error'], first['replay_value_error'], first['replay_bootstrap_error']) < 5e-5
    saved = copy.deepcopy(t.state_dict())
    expected = t.run_update()
    restored = trainer(c)
    restored.load_state_dict(saved)
    assert restored.run_update() == expected


def test_advantage_recomputation_preserves_replay_exactness():
    for flag in (True, False):
        summary = trainer(config(recompute_advantages_per_pass=flag, epochs=2)).run_update()
        assert summary['replay_probability_error'] < 5e-5


def test_clipped_surrogate_stops_pushing_outside_the_trust_region():
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0])
    log_ratio = torch.log(torch.tensor([1.1, 1.4, 0.9, 0.6])).requires_grad_()
    clipped_surrogate(log_ratio, advantage, 0.25).sum().backward()
    gradient = log_ratio.grad
    # Inside the region the step follows the advantage; beyond it, in the advantage's direction, it stops.
    assert gradient[0] > 0 and gradient[2] < 0
    assert gradient[1] == 0 and gradient[3] == 0
    # Pessimistic: a ratio that moved against the advantage is never clipped in its favour.
    assert float(clipped_surrogate(torch.log(torch.tensor([0.5])), torch.tensor([1.0]), 0.25)) == pytest.approx(0.5)


def test_ppo_learns_the_side_of_a_strong_drift():
    """No costs and a drift five times the noise: holding long is plainly best, and PPO must find it."""
    c = config(fee=0.0, slippage=0.0, epochs=4, updates=40, actor_lr=3e-3, critic_lr=3e-3)
    t = trainer(c, seed=5, size=20_000, drift=1e-4)
    first = t.run_update()['sampled_action_share'][LONG]
    for _ in range(c.updates - 1):
        summary = t.run_update()
    assert summary['sampled_action_share'][LONG] > max(0.7, first + 0.3)


def test_a_side_locked_policy_shows_no_state_action_information():
    t = trainer(config())
    with torch.no_grad():
        t.model.actor.output.bias[SHORT] += 8.0
    summary = t.run_update()
    assert summary['sampled_action_share'][SHORT] > 0.9
    assert summary['state_action_mutual_information'] < 1e-3


def test_a_liquidated_account_restarts_at_the_next_rollout():
    t = trainer(config())
    t.env.ledger.terminated[0] = True
    rows, _ = t.collect()
    assert rows['valid'][0].all() and not t.env.ledger.terminated[0]


def test_the_recurrent_prefix_rolls_forward_with_the_rollout():
    c = config()
    t = trainer(c)
    t.env.steps[:] = 0
    rows, _ = t.collect()
    np.testing.assert_array_equal(t.prefix_minutes, rows['minutes'][:, -c.burn_in:])
    assert (np.diff(rows['minutes'], axis=1) == BAR_MINUTES).all()


def test_entropy_bonus_raises_entropy_against_no_bonus():
    entropies = []
    for coefficient in (0.0, 5.0):
        t = trainer(config(entropy_coefficient=coefficient, epochs=3))
        with torch.no_grad():
            t.model.actor.output.bias[LONG] += 4.0
        entropies.append(t.run_update()['policy_entropy'])
    assert entropies[1] > entropies[0]


def test_gae_masks_terminations_but_bootstraps_time_limits():
    rewards = np.array([[1.0, 1.0, 1.0]], np.float32)
    values = np.zeros((1, 3), np.float32)
    valid = np.ones((1, 3), bool)
    bootstrap = np.array([10.0], np.float32)
    open_ended, _ = advantages(rewards, values, bootstrap, np.zeros((1, 3), bool), valid, 0.5, 1.0)
    terminated, _ = advantages(rewards, values, bootstrap, np.array([[False, False, True]]), valid, 0.5, 1.0)
    assert open_ended[0, 2] == pytest.approx(1 + 0.5 * 10)
    assert terminated[0, 2] == pytest.approx(1.0)


def test_learning_rate_anneals_linearly_to_zero():
    c = replace(config(), updates=100)
    t = trainer(c)
    assert t.learning_rates()[0] == pytest.approx(c.actor_lr)
    t.updates = 50
    assert t.learning_rates()[0] == pytest.approx(c.actor_lr / 2)
    t.updates = 100
    assert t.learning_rates()[0] == pytest.approx(0.0)
    off = trainer(replace(c, anneal_lr=False))
    off.updates = 50
    assert off.learning_rates()[0] == pytest.approx(c.actor_lr)


def test_diagnostics_are_reported_every_update():
    summary = trainer(config()).run_update()
    for key in ('policy_entropy', 'long_entry_share', 'entry_decisions', 'exit_decisions', 'mean_age_at_exit_hours',
                'state_action_mutual_information', 'state_action_mutual_information_flat',
                'policy_implied_holding_hours', 'account_side_share', 'explained_variance_post',
                'liquidations_in_rollout', 'approx_kl', 'clip_fraction', 'replay_probability_error'):
        assert key in summary, key
    json.dumps(summary, allow_nan=False)


def test_mutual_information_is_zero_for_a_constant_policy():
    mask = torch.ones(200, dtype=torch.bool)
    assert state_action_mutual_information(torch.full((200, 3), 1/3), mask) == pytest.approx(0, abs=1e-9)
    torch.manual_seed(0)
    varied = torch.softmax(torch.randn(200, 3), -1)
    assert state_action_mutual_information(varied, mask) > 0
    assert state_action_mutual_information(varied, torch.zeros(200, dtype=torch.bool)) == 0.


def test_policy_entropy_is_ln3_for_a_uniform_policy():
    assert float(policy_entropy(torch.log(torch.full((1, 3), 1/3)))[0]) == pytest.approx(math.log(3))


# --- evaluation -------------------------------------------------------------------------------

def test_evaluation_takes_the_most_probable_action_by_default():
    c = config()
    assert c.evaluation_policy == 'argmax'
    t = trainer(c, size=20000)
    m = t.env.market
    result, frame = evaluate([t.model], m, t.scaler, c, m.timestamp(15000), m.timestamp(16000), seeds=3)
    assert result['seeds'] == [0] and len(result['per_seed']) == 1, 'a deterministic policy has one path'
    chosen = frame[['p_short', 'p_flat', 'p_long']].to_numpy().argmax(1)
    assert (frame.action.to_numpy() == chosen).all()
    again, frame_again = evaluate([t.model], m, t.scaler, c, m.timestamp(15000), m.timestamp(16000))
    assert (frame_again.action.to_numpy() == frame.action.to_numpy()).all()
    assert again['total_log_return'] == result['total_log_return']


def test_unknown_evaluation_policy_is_rejected():
    with pytest.raises(ValueError, match='evaluation_policy'):
        full(evaluation_policy='greedy')


def test_evaluation_samples_and_reports_a_distribution():
    c = config(evaluation_policy='sampled')
    t = trainer(c, size=20000)
    m = t.env.market
    result, frame = evaluate([t.model], m, t.scaler, c, m.timestamp(15000), m.timestamp(16000))
    assert result['orders_max'] > 0, 'a near-uniform sampled policy must trade'
    assert len(result['per_seed']) == c.evaluation_seeds
    assert set(frame.seed.unique()) == set(result['seeds'])
    assert result['total_log_return_min'] <= result['total_log_return'] <= result['total_log_return_max']
    assert 0 <= result['long_share'] <= 1


def test_evaluation_steps_one_decision_interval_at_a_time():
    c = config(decision_bars=3)
    t = trainer(c, size=20000)
    m = t.env.market
    result, frame = evaluate([t.model], m, t.scaler, c, m.timestamp(15000), m.timestamp(16000), seeds=1)
    assert (frame.time.diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert result['per_seed'][0]['decisions'] == len(frame) == 1000 // 15


def test_quarter_windows_tile_the_year():
    windows = quarter_windows(2024)
    assert len(windows) == 4
    assert windows[0][0] == pd.Timestamp('2024-01-01', tz='UTC')
    assert windows[-1][1] == pd.Timestamp('2025-01-01', tz='UTC')
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))


def test_extra_slippage_costs_turnover_times_slippage():
    """The first-order identity behind the break-even slippage report."""
    m = market(3000, seed=2)
    totals = {}
    for slippage in (0.0001, 0.0003):
        env = Environment(m, (200, 3000), config(slippage=slippage), np.random.default_rng(0), training=False)
        total = sum(float(env.step(np.array([action]))[2]['log_return'][0])
                    for action in [LONG, FLAT, SHORT, SHORT, FLAT, LONG] * 20)
        totals[slippage] = (total, float(env.ledger.turnover_total[0]))
    assert totals[0.0001][0] - totals[0.0003][0] == pytest.approx(0.0002 * totals[0.0001][1], rel=0.05)


def test_break_even_slippage_is_return_over_turnover():
    from agent.experiment import break_even_slippage
    paths = [{'total_log_return': 0.02, 'turnover': 40.0}, {'total_log_return': -0.01, 'turnover': 10.0},
             {'total_log_return': 0.5, 'turnover': 0.0}]
    report = break_even_slippage(paths)
    assert report['paths'] == 2 and report['median_bp'] == pytest.approx(-2.5)
    assert report['share_above_zero'] == 0.5


def test_metrics_count_flickers_entries_and_their_direction():
    frame = pd.DataFrame({'time': pd.date_range('2023-01-01', periods=8, freq='5min'),
                          'log_return': 0.0, 'differential_ratio': 0.0, 'equity': 1.0, 'mark': 100.0,
                          'contracts': [0, .01, 0, 0, -.01, -.01, .01, 0]})
    result = metrics(frame, Ledger(1, config()), 100.0, 5 / 60)
    assert result['single_bar_round_trips'] == 2
    assert result['entries'] == 2 and result['long_entry_share'] == pytest.approx(0.5)
    assert result['reversals'] == 1
    assert result['median_holding_spell_hours'] == pytest.approx(5 / 60)


def test_evaluator_refuses_locked_test_data():
    with pytest.raises(TestLockedError):
        evaluate([], None, None, config(), '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z')


# --- scoring ------------------------------------------------------------------------------------

def test_interquartile_mean_is_the_trimmed_mean():
    assert interquartile_mean(np.arange(1, 9)) == pytest.approx(4.5)
    assert interquartile_mean([[1, 100], [2, 3]]) == pytest.approx(2.5)


def test_stratified_bootstrap_interval_brackets_the_point():
    scores = np.random.default_rng(0).normal(0.1, 0.2, (3, 8))
    interval = stratified_bootstrap_interval(scores, interquartile_mean, 2000, 0.95)
    assert interval['lower'] <= interval['point'] <= interval['upper']
    assert interval['runs'] == 3 and interval['tasks'] == 8


def test_backtest_overfitting_is_high_on_noise_and_zero_for_a_real_edge():
    generator = np.random.default_rng(4)
    noise = generator.normal(0, 1, (8, 3, 10))
    assert 0.25 < probability_of_backtest_overfitting(noise)['probability'] < 0.9
    edge = generator.normal(0, 0.1, (8, 3, 10))
    edge[:, :, 0] += 5
    assert probability_of_backtest_overfitting(edge)['probability'] == 0.0


# --- synthetic markets --------------------------------------------------------------------------

def test_planted_market_signal_is_causal():
    from agent.synthetic import REGIME_FEATURE, planted_market
    m, regime = planted_market(minutes=20_000, seed=1, noise=0.0)
    column = FEATURE_NAMES.index(REGIME_FEATURE)
    np.testing.assert_allclose(m.features[1:, column], regime[:-1], atol=1e-5)
    assert m.features[0, column] == 0.0


def test_planted_edge_pays_several_times_its_own_cost():
    from agent.synthetic import edge_economics, planted_market
    m, _ = planted_market(minutes=20_000, seed=1)
    assert edge_economics(m, Config())['edge_to_cost_ratio'] > 3.0


def test_oracle_beats_buy_and_hold_on_the_planted_market():
    from agent.synthetic import decision_minutes, oracle_actions, planted_market
    m, regime = planted_market(minutes=120_000, seed=0)
    bounds = m.bounds(m.timestamp(20_000), m.timestamp(110_000))
    c = full()
    totals = {}
    for name, actions in (('oracle', oracle_actions(m, regime, bounds, c)),
                          ('long', np.full(len(decision_minutes(bounds, c)), LONG))):
        env = Environment(m, bounds, c, np.random.default_rng(0), training=False)
        totals[name] = sum(float(env.step(np.array([a]))[2]['log_return'][0]) for a in actions)
    assert totals['oracle'] > totals['long'] + 0.1 and totals['oracle'] > 0.2


def test_latent_regimes_are_hidden_but_leave_a_trace_in_the_trend_state():
    from agent.synthetic import RANGE, decision_minutes, latent_regime_market, trend_rule_actions
    m, regime = latent_regime_market(years=1.0, seed=1)
    first, last = m.bounds(m.timestamp(0), m.timestamp(len(m)))
    at = np.arange(first, last, BAR_MINUTES)
    features = np.asarray(m.features[at], dtype=np.float64)
    direction = (regime[at - 1] - RANGE).astype(float)
    for name in FEATURE_NAMES:
        column = features[:, CANDIDATE_FEATURES.index(name)]
        assert np.isfinite(column).all()
        assert abs(np.corrcoef(column, direction)[0, 1]) < 0.9, name
    trend = features[:, CANDIDATE_FEATURES.index('return_1w')]
    assert np.corrcoef(trend, direction)[0, 1] > 0.1
    rule = trend_rule_actions(m, (first, last), Config())
    assert len(rule) == len(decision_minutes((first, last), Config())) and set(np.unique(rule)) <= {SHORT, LONG}


def test_momentum_reference_holds_the_sign_and_revises_only_at_midnight():
    c = config(decision_bars=12)
    size = 5 * DAY_MINUTES
    m = market(size=size, seed=3)
    column = CANDIDATE_FEATURES.index('return_1w')
    features = np.array(m.features)
    # Positive until noon of the evaluation's first day, negative afterwards.
    features[:, column] = np.where(np.arange(size) < int(1.5 * DAY_MINUTES), 1.0, -1.0)
    m = Market(m.prices, m.present, m.funding, features, m.daily_volatility, m.epoch_start, {}, history_minutes=1)
    result, frame = evaluate([], m, None, c, m.timestamp(DAY_MINUTES), m.timestamp(4 * DAY_MINUTES),
                             baseline='momentum_1w')
    hours = frame.time.dt.hour.to_numpy()
    days = (frame.time - frame.time.iloc[0]).dt.days.to_numpy()
    assert (frame.action[days == 0] == LONG).all(), 'a switch at noon waits for the next midnight'
    assert (frame.action[days >= 1] == SHORT).all()
    assert hours[np.flatnonzero(np.diff(frame.action.to_numpy()))[0] + 1] == 0
    with pytest.raises(ValueError, match='unknown baseline'):
        evaluate([], m, None, c, m.timestamp(DAY_MINUTES), m.timestamp(2 * DAY_MINUTES), baseline='magic')


def test_evaluation_records_every_input_and_the_critic_value_of_every_decision():
    c = config()
    t = trainer(c, size=20000)
    m = t.env.market
    _, frame = evaluate([t.model], m, t.scaler, c, m.timestamp(15000), m.timestamp(15400))
    for name in FEATURE_NAMES:
        assert f'z_{name}' in frame
    for name in PORTFOLIO_NAMES:
        assert f'account_{name}' in frame
    assert np.isfinite(frame.value).all() and frame.value.abs().sum() > 0
    np.testing.assert_allclose(frame[[f'z_{n}' for n in FEATURE_NAMES]].to_numpy(),
                               t.scaler.observations(m, m.index(frame.time.iloc[0]) + np.arange(len(frame)) * c.decision_bars * BAR_MINUTES,
                                                     c.observation_clip),
                               rtol=1e-6)
    # A reference rule has no network inputs to record.
    _, passive = evaluate([], m, None, c, m.timestamp(15000), m.timestamp(15400), baseline='long')
    assert 'z_return_1h' not in passive


def test_update_logs_gradient_norms_and_keeps_the_rollout_it_learned_from():
    c = config()
    t = trainer(c)
    summary = t.run_update()
    for key in ('actor_grad_norm', 'critic_grad_norm', 'mean_value', 'advantage_std', 'reward_p05', 'reward_p95'):
        assert np.isfinite(summary[key]), key
    snapshot = t.last_rollout
    assert snapshot['advantages_after_update'].shape == snapshot['rewards'].shape == (c.environments, c.rollout)
    assert snapshot['account'].shape == (c.environments, c.rollout, PORTFOLIO_WIDTH)
