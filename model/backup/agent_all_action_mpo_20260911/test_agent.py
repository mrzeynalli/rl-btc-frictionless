"""Self-contained tests for the agent: data, reward, ledger, network, PPO, evaluation, scoring."""
from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
import torch

from agent.config import COST_BREAK_EVEN_HOURS, SEEDS, YEARS, Config, TestLockedError, protocol
from agent.data import DAY_BARS, DAY_MINUTES, FEATURE_NAMES, Market, Scaler, causal_arrays
from agent.evaluate import evaluate, metrics, quarter_windows
from agent.ledger import (Environment, Ledger, TrainingEnvironment, FLAT, LONG, SHORT, PORTFOLIO_NAMES,
                          PORTFOLIO_WIDTH)
from agent.network import ActorCritic, market_tensor, policy_entropy
from agent.ppo import Trainer, advantages, state_action_mutual_information
from agent.reward import differential_sharpe, first_order_differential_sharpe, sharpe_ratio
from agent.scoring import interquartile_mean, probability_of_backtest_overfitting, stratified_bootstrap_interval

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def market(size=4000, drift=0.0, seed=0):
    generator = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(generator.normal(drift, 1e-4, size)))
    rows = np.repeat(price[:, None], 4, axis=1)
    features = generator.normal(0, .1, (size, len(FEATURE_NAMES))).astype(np.float32)
    return Market(rows, np.ones(size, bool), np.zeros(size), features, np.full(size, 0.02, np.float32),
                  int(pd.Timestamp('2020-01-01', tz='UTC').value // 60_000_000_000), {}, history_minutes=1)


def scaler():
    return Scaler(np.zeros(len(FEATURE_NAMES), np.float32), np.ones(len(FEATURE_NAMES), np.float32),
                  '2021-01-01T00:00:00Z')


def config(**kwargs):
    # Reduced sizes need a reduced horizon and reward window, since Config
    # enforces that rollout, recurrent span and reward window fit the horizon.
    base = dict(environments=4, rollout=16, sequence=8, burn_in=8, episode_steps=64, minibatch_sequences=2,
                hidden=8, head_width=8, gamma=.9, gae_lambda=.85, dsr_eta=1/32, epochs=1, updates=2,
                evaluation_seeds=2, bootstrap_replications=200)
    return replace(Config(), **{**base, **kwargs})


def trainer(c, seed=11, size=4000):
    torch.manual_seed(seed)
    env = TrainingEnvironment(market(size), (100, size), c, np.random.default_rng(seed))
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


# --- specification -------------------------------------------------------------

def test_credit_window_clears_the_cost_break_even():
    c = Config()
    assert c.gamma == c.gae_lambda == 0.99825
    assert 23 < c.credit_window_hours < 25 and c.credit_window_hours > COST_BREAK_EVEN_HOURS
    assert c.rollout / 2 >= c.credit_window
    assert c.burn_in + c.sequence >= c.credit_window


def test_cost_is_realistic_and_fixed_with_no_curriculum():
    c = Config()
    assert (c.fee, c.slippage) == (0.0005, 0.0001)
    assert c.round_trip_cost == pytest.approx(0.0012)
    assert not any('curriculum' in name for name in c.as_dict())


def test_rollout_must_carry_its_own_credit_window():
    with pytest.raises(ValueError, match='truncates the credit window'):
        replace(Config(), rollout=512)


def test_recurrent_span_must_cover_the_credit_window():
    with pytest.raises(ValueError, match='recurrent span'):
        replace(Config(), sequence=64, burn_in=64)


def test_reward_window_lies_between_credit_window_and_episode():
    with pytest.raises(ValueError, match='differential Sharpe window'):
        replace(Config(), dsr_eta=1/200)
    with pytest.raises(ValueError, match='differential Sharpe window'):
        replace(Config(), dsr_eta=1/30000)


def test_experiment_refuses_the_predecessor_horizon(tmp_path):
    from agent import experiment
    short = replace(Config(), gamma=.9975961328836203, gae_lambda=.9738730024397007,
                    rollout=1024, dsr_eta=1/1440)
    assert short.credit_window_hours < 3
    with pytest.raises(ValueError, match='below the cost break-even'):
        experiment.run(tmp_path, short, workers=1)


def test_folds_are_2024_and_2025_with_three_seeds():
    assert YEARS == (2024, 2025)
    assert len(SEEDS) == 3


def test_training_episode_is_one_quarter():
    c = Config()
    days = c.episode_steps * c.decision_minutes / DAY_MINUTES
    assert 90 <= days <= 93
    assert c.episode_steps % c.rollout == 0


def test_gradient_step_budget_matches_the_recurrent_ppo_reference():
    """CleanRL ppo_atari_lstm: 10M steps / 1,024 batch x 4 epochs x 4 minibatches."""
    c = Config()
    reference = (10_000_000 // (8 * 128)) * 4 * 4
    per_update = c.epochs * (c.rollout // c.sequence) * (c.environments // c.minibatch_sequences)
    assert per_update == 32
    assert 0.9 * reference <= c.updates * per_update <= 1.2 * reference


def test_literature_defaults():
    c = Config()
    assert c.policy_kl_step == 0.1 and c.actor_lr == c.critic_lr == 3e-4 and c.anneal_lr
    assert c.entropy_coefficient == 0.01 and c.epochs == 4 and c.environments == 8
    assert c.burn_in / c.sequence == 0.5
    assert c.observation_clip == 10.0
    assert c.bootstrap_replications == 50_000 and c.confidence_level == 0.95


def test_every_constant_has_a_documented_provenance():
    """ARCHITECTURE.md §12 must name every Config field; nothing may be undocumented."""
    text = (ROOT / 'ARCHITECTURE.md').read_text()
    section = text[text.index('## 12.'):]
    missing = [name for name in Config().as_dict() if f'`{name}`' not in section]
    assert not missing, missing


def test_config_survives_the_json_round_trip():
    c = Config()
    assert c.as_dict() == json.loads(json.dumps(c.as_dict()))
    assert isinstance(c.stress_slippage, tuple)


def test_reward_scale_keeps_the_value_target_at_unit_scale():
    c = Config()
    assert c.reward_scale == pytest.approx(math.sqrt(1 - c.gamma ** 2))
    assert 0.5 < c.reward_scale * math.sqrt(1 / (1 - c.gamma ** 2)) < 2.0
    assert 'reward_scale' not in c.as_dict()


def test_protocol_records_the_reasoning():
    spec = protocol()
    assert spec['test_locked'] and not spec['test_performance_accessed']
    assert 'no curriculum' in spec['cost_policy']
    assert 'argmax is never used' in spec['evaluation']
    assert spec['credit_window_hours'] > spec['cost_break_even_hours']
    assert spec['validation_years'] == [2024, 2025]
    assert 'Lyle' in spec['references']['layer normalisation']


# --- market features -------------------------------------------------------------

def test_features_are_causal():
    """Changing minute k may change the observation at boundary k+1 onward, never before."""
    frame, funding, start, end = minute_bars()
    base = causal_arrays(frame, funding, start, end)
    k = 2 * DAY_MINUTES + 37
    changed = frame.copy()
    row = changed.index[changed.open_time == start + pd.Timedelta(minutes=k)][0]
    changed.loc[row, ['close', 'high']] = changed.loc[row, 'high'] * 1.02
    changed.loc[row, 'volume'] *= 50
    other = causal_arrays(changed, funding, start, end)
    np.testing.assert_array_equal(base.features[:k + 1], other.features[:k + 1])
    np.testing.assert_array_equal(base.daily_volatility[:k + 1], other.daily_volatility[:k + 1])
    assert not np.array_equal(base.features[k + 1], other.features[k + 1])
    settled = funding.copy()
    settled.loc[3, 'fundingRate'] += 0.01
    event = int((settled.timestamp[3] - start) / pd.Timedelta(minutes=1))
    shifted = causal_arrays(frame, settled, start, end)
    np.testing.assert_array_equal(base.features[:event + 1], shifted.features[:event + 1])
    column = FEATURE_NAMES.index('funding_rate')
    assert shifted.features[event + 1, column] != base.features[event + 1, column]


def test_price_features_are_scale_free():
    """A year with twice the volatility and ten times the volume must look the same."""
    frame, funding, start, end = minute_bars()
    wild, _, _, _ = minute_bars(scale=2.0, volume_scale=1000)
    calm = causal_arrays(frame, funding, start, end).features
    loud = causal_arrays(wild, funding, start, end).features
    rows = slice(DAY_MINUTES + 10, None)
    for name in ('bar_return', 'bar_range', 'momentum_1d', 'momentum_1w', 'volatility_regime'):
        i = FEATURE_NAMES.index(name)
        np.testing.assert_allclose(calm[rows, i], loud[rows, i], atol=2e-4, err_msg=name)
    for name in ('bar_volume', 'volume_regime'):
        i = FEATURE_NAMES.index(name)
        np.testing.assert_allclose(calm[rows, i], loud[rows, i], atol=2e-2, err_msg=name)


def test_bar_return_and_daily_momentum_are_bounded_by_construction():
    """Cauchy-Schwarz: a return inside the variance window cannot exceed sqrt(bars) sd."""
    frame, funding, start, end = minute_bars()
    features = causal_arrays(frame, funding, start, end).features
    rows = slice(DAY_MINUTES + 10, None)
    for name in ('bar_return', 'momentum_1d'):
        assert np.abs(features[rows, FEATURE_NAMES.index(name)]).max() <= math.sqrt(DAY_BARS) + 1e-4


def test_funding_phase_and_hour_are_encoded_on_the_clock():
    frame, funding, start, end = minute_bars()
    m = causal_arrays(frame, funding, start, end)
    settlement = m.index(pd.Timestamp('2020-03-01 12:00', tz='UTC'))
    sine, cosine = FEATURE_NAMES.index('funding_phase_sin'), FEATURE_NAMES.index('funding_phase_cos')
    assert m.features[settlement + 1, sine] == pytest.approx(math.sin(2 * math.pi / 480), abs=1e-6)
    assert m.features[settlement + 480, cosine] == pytest.approx(1.0, abs=1e-6)
    six = m.index(pd.Timestamp('2020-03-02 06:00', tz='UTC'))
    assert m.features[six, FEATURE_NAMES.index('hour_sin')] == pytest.approx(1.0, abs=1e-6)
    assert m.features[six, FEATURE_NAMES.index('hour_cos')] == pytest.approx(0.0, abs=1e-6)


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

def test_exact_increment_reduces_to_the_first_order_term_for_ordinary_returns():
    c = Config()
    sd = math.sqrt(c.dsr_initial_second_moment)
    mean = np.full(7, 0.1 * sd)
    second = np.full(7, c.dsr_initial_second_moment)
    returns = sd * np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    exact, _, _ = differential_sharpe(returns, mean, second, c)
    first = first_order_differential_sharpe(returns, mean, second)
    np.testing.assert_allclose(exact, first, atol=0.02)


def test_exact_increment_is_bounded_and_graded_for_catastrophic_losses():
    """The clip paid -10 for a 5% loss and a wipeout alike; the exact change grades them."""
    c = Config()
    losses = np.log(np.array([0.995, 0.95, 0.5, 1e-12]))
    values, _, _ = differential_sharpe(losses, np.zeros(4), np.full(4, c.dsr_initial_second_moment), c)
    assert (np.diff(values) < 0).all()
    bound = 1 / math.sqrt(c.dsr_eta * (1 - c.dsr_eta))
    assert np.abs(values).max() <= bound + 1e-6
    # A 5% single-step loss already costs more than the old clip allowed anything to.
    assert values[1] < -20


def test_rewards_telescope_to_the_change_in_the_sharpe_ratio():
    c = config()
    generator = np.random.default_rng(3)
    mean, second = np.zeros(1), np.full(1, c.dsr_initial_second_moment)
    start, total = float(sharpe_ratio(mean, second)[0]), 0.0
    for value in generator.normal(1e-4, 1.5e-3, 300):
        d, mean, second = differential_sharpe(np.array([value]), mean, second, c)
        total += float(d[0])
    assert total * c.dsr_eta == pytest.approx(float(sharpe_ratio(mean, second)[0]) - start, abs=1e-9)


def test_differential_sharpe_is_scale_invariant():
    c = config()
    r = np.array([0.001, -0.002, 0.0005])
    a, b = np.zeros(3), np.full(3, c.dsr_initial_second_moment)
    plain, _, _ = differential_sharpe(r, a, b, c)
    scaled, _, _ = differential_sharpe(100 * r, 100 * a, 10000 * b, c)
    np.testing.assert_allclose(plain, scaled, rtol=1e-9)


def test_differential_sharpe_prefers_steadiness_to_magnitude():
    c = config()
    steady, lumpy = np.full(200, 0.001), np.tile([0.021, -0.019], 100)
    assert steady.sum() == pytest.approx(lumpy.sum())
    totals = []
    for path in (steady, lumpy):
        a, b, total = np.zeros(1), np.full(1, c.dsr_initial_second_moment), 0.0
        for value in path:
            d, a, b = differential_sharpe(np.array([value]), a, b, c)
            total += float(d[0])
        totals.append(total)
    assert totals[0] > totals[1]


def test_leverage_ratchet_does_not_pay_under_differential_sharpe():
    c = config()
    base = np.random.default_rng(5).normal(2e-4, 1.5e-3, 400)
    totals = []
    for multiple in (1.0, 2.0):
        a, b, total = np.zeros(1), np.full(1, c.dsr_initial_second_moment), 0.0
        for value in base * multiple:
            d, a, b = differential_sharpe(np.array([value]), a, b, c)
            total += float(d[0])
        totals.append(total)
    assert totals[1] == pytest.approx(totals[0], rel=0.2)


def test_terminated_accounts_freeze_their_reward_statistics():
    c = config()
    a, b = np.array([0.5, 0.5]), np.array([1.0, 1.0])
    d, a2, b2 = differential_sharpe(np.array([0.01, 0.01]), a, b, c, active=np.array([True, False]))
    assert d[1] == 0.0 and a2[1] == a[1] and b2[1] == b[1]
    assert d[0] != 0.0 and a2[0] != a[0]


def test_environment_reports_both_reward_signals():
    c = config()
    env = Environment(market(600), (100, 600), c, np.random.default_rng(0), training=False)
    _, _, info = env.step(np.array([LONG]))
    assert 'log_return' in info and 'differential_sharpe' in info


# --- account and ledger ---------------------------------------------------------------

def test_account_observation_is_markov_minimal():
    assert PORTFOLIO_WIDTH == len(PORTFOLIO_NAMES) == 7
    assert PORTFOLIO_NAMES[:3] == ('short', 'flat', 'long')
    c = config()
    assert Ledger(4, c).portfolio(np.full(4, 100.), 0.02).shape == (4, PORTFOLIO_WIDTH)


def test_reward_statistics_are_visible_to_the_policy():
    c = config()
    ledger = Ledger(2, c)
    before = ledger.portfolio(np.full(2, 100.), 0.02).copy()
    ledger.dsr_mean[:] = 0.0002
    ledger.dsr_second[:] = 4 * c.dsr_initial_second_moment
    after = ledger.portfolio(np.full(2, 100.), 0.02)
    changed = set(np.flatnonzero(np.abs(after - before).sum(0) > 0))
    assert changed == {PORTFOLIO_NAMES.index('running_sharpe'), PORTFOLIO_NAMES.index('relative_volatility')}


def test_exposure_feature_is_the_liquidation_distance():
    """A 1x inverse long is liquidated exactly when |contracts|/price/equity reaches 1/(m+f)."""
    c = Config()
    ledger = Ledger(1, c)
    ledger.execute(np.array([LONG]), np.array([100.0]), np.array([True]))
    threshold = 1 / (c.maintenance + c.fee)
    column = PORTFOLIO_NAMES.index('exposure')
    for price in np.linspace(100, 40, 6001):
        mark = np.array([price])
        ratio = float(np.abs(ledger.contracts[0]) / price / ledger.value(mark)[0])
        feature = float(ledger.portfolio(mark, 0.02)[0, column])
        ledger.mark(mark, mark)
        if ledger.terminated[0]:
            assert ratio >= threshold * (1 - 1e-9)
            assert 45 < price < 55
            break
        assert ratio < threshold
        assert feature == pytest.approx(math.log1p(ratio), rel=1e-5)
    else:
        pytest.fail('the long was never liquidated')


def test_a_one_x_short_survives_a_hundredfold_rally():
    c = Config()
    ledger = Ledger(1, c)
    ledger.execute(np.array([SHORT]), np.array([100.0]), np.array([True]))
    for price in np.geomspace(100, 10_000, 500):
        ledger.mark(np.array([price]), np.array([price]))
    assert not ledger.terminated[0]


def test_turnover_is_not_inflated_by_a_liquidation():
    c = Config()
    ledger = Ledger(1, c)
    ledger.execute(np.array([LONG]), np.array([100.0]), np.array([True]))
    ledger.mark(np.array([40.0]), np.array([40.0]))
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


def test_critic_carries_a_state_value_and_an_action_value_head():
    """The policy step reads a learned Q, not one bar's outcome (MPO's E-step)."""
    model = ActorCritic(config())
    assert model.critic.output.out_features == 1 + 3 and model.actor.output.out_features == 3
    estimate = model.critic.read(torch.randn(2, 4, model.critic.hidden), torch.randn(2, 4, PORTFOLIO_WIDTH))
    assert estimate.shape == (2, 4, 4)


def test_value_network_is_wider_than_the_policy():
    model = ActorCritic(Config())
    assert model.critic.hidden == 2 * model.actor.hidden and model.critic.head_width == 2 * model.actor.head_width
    assert sum(p.numel() for p in model.critic.parameters()) > sum(p.numel() for p in model.actor.parameters())


# --- PPO ----------------------------------------------------------------------------------

def test_replay_and_resume_are_exact():
    c = config()
    t = trainer(c)
    first = t.run_update()
    assert first['replay_ratio_error'] < 5e-5 and first['replay_value_error'] < 5e-5
    saved = copy.deepcopy(t.state_dict())
    expected = t.run_update()
    restored = trainer(c)
    restored.load_state_dict(saved)
    assert restored.run_update() == expected


def test_advantage_recomputation_preserves_replay_exactness():
    for flag in (True, False):
        summary = trainer(config(recompute_advantages_per_pass=flag, epochs=2)).run_update()
        assert summary['replay_ratio_error'] < 5e-5


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
    for key in ('policy_entropy', 'long_entry_share', 'entry_decisions', 'state_action_mutual_information',
                'state_action_mutual_information_flat', 'policy_implied_holding_hours', 'account_side_share',
                'explained_variance_post', 'liquidations_in_rollout', 'approx_kl',
                'policy_temperature', 'policy_target_kl', 'policy_target_entropy'):
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


# --- counterfactual branches and all-action updates ------------------------------------------

def test_branches_reproduce_the_environment_for_every_action():
    """The same accounting on copies: each branch equals the environment stepped with that action."""
    from agent.ledger import branch_every_action, replicate
    c = config()
    m = market(3000)
    for action in (SHORT, FLAT, LONG):
        env = Environment(m, (200, 3000), c, np.random.default_rng(0), training=False)
        env.step(np.array([LONG]))
        branches, rewards, _ = branch_every_action(replicate(env.ledger, [0], 1), m, env.cursor, c, env.bounds[1])
        reward, _, _ = env.step(np.array([action]))
        assert rewards[0, action] == pytest.approx(float(reward[0]), rel=1e-9, abs=1e-12)
        for name in ('wallet', 'contracts', 'dsr_mean', 'dsr_second'):
            np.testing.assert_allclose(getattr(branches, name)[action], getattr(env.ledger, name)[0], rtol=1e-12)


def test_starting_siblings_contain_the_visited_account_and_all_three_sides():
    c = replace(config(), environments=12, minibatch_sequences=3)
    env = TrainingEnvironment(market(6000), (100, 6000), c, np.random.default_rng(2))
    siblings, actual = env.starting_siblings(np.ones(env.size, bool))
    chosen = np.arange(env.size) * 3 + actual
    for name in ('wallet', 'contracts', 'entry', 'dsr_mean', 'dsr_second'):
        np.testing.assert_allclose(getattr(siblings, name)[chosen], getattr(env.ledger, name), rtol=1e-12, atol=0)
    sides = np.sign(siblings.contracts).reshape(env.size, 3)
    assert (np.sort(sides, axis=1) == [-1, 0, 1]).all()


def test_collection_tracks_siblings_exactly_and_covers_every_side():
    t = trainer(config())
    rows, _ = t.collect()
    assert rows['sibling_gap'] < 1e-9
    sides = np.sort(rows['sibling_account'][..., :3].argmax(-1), axis=-1)
    assert (sides[rows['valid']] == [0, 1, 2]).all()
    index = rows['actual_index'][..., None, None].repeat(PORTFOLIO_WIDTH, -1)
    visited = np.take_along_axis(rows['sibling_account'], index, axis=2)[:, :, 0]
    np.testing.assert_array_equal(visited, rows['account'])


def test_action_values_are_exact_rewards_plus_the_critic_and_targets_split_by_visit():
    t = trainer(config())
    rows, _ = t.collect()
    critic = t.evaluate_critic(rows, t.rollout_market(rows))
    e, s = np.nonzero(rows['valid'])
    chosen = rows['branch_reward'][e, s, rows['actual_index'][e, s], rows['actions'][e, s]]
    np.testing.assert_allclose(chosen, rows['rewards'][e, s], rtol=1e-6, atol=1e-7)
    q = critic['one_step_values'].numpy()
    terminal = rows['branch_terminal']
    np.testing.assert_allclose(q[terminal], rows['branch_reward'][terminal], rtol=1e-6, atol=1e-7)
    expected = (rows['sibling_probabilities'] * q).sum(-1)
    targets = critic['targets'].numpy()
    visited = np.eye(3, dtype=bool)[rows['actual_index']]
    np.testing.assert_allclose(targets[~visited], expected[~visited], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(targets[visited].reshape(rows['rewards'].shape), critic['returns'], rtol=1e-5, atol=1e-6)


def test_all_action_advantage_is_centred_under_the_collection_policy():
    t = trainer(config())
    rows, _ = t.collect()
    q = t.evaluate_critic(rows, t.rollout_market(rows))['action_values']
    old = torch.from_numpy(rows['sibling_probabilities'])
    advantage = q - (old * q).sum(-1, keepdim=True)
    torch.testing.assert_close((old * advantage).sum(-1), torch.zeros_like(advantage[..., 0]), atol=1e-5, rtol=0)


def test_policy_target_takes_the_configured_kl_step_and_revives_a_starved_action():
    """MPO's E-step: a bounded KL move toward higher action values, whose pull on a
    near-zero-probability action does not vanish the way PPO's ratio gradient does."""
    t = trainer(config())
    old = torch.tensor([[[[1e-6, 1 - 2e-6, 1e-6]]]])
    values = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    target, temperature, step = t.policy_target(values, old, mask)
    assert step == pytest.approx(t.config.policy_kl_step, rel=1e-3)
    assert temperature > 0
    assert float(target[0, 0, 0, 0]) > 1000 * float(old[0, 0, 0, 0])
    assert float(target[0, 0, 0, 0]) > 1e-3
    assert float(target.sum()) == pytest.approx(1.0, abs=1e-6)


def test_policy_target_is_the_old_policy_when_the_actions_are_equal():
    t = trainer(config())
    old = torch.full((2, 3, 3, 3), 1/3)
    target, temperature, step = t.policy_target(torch.zeros(2, 3, 3, 3), old, torch.ones(2, 3, 3, dtype=torch.bool))
    torch.testing.assert_close(target, old)
    assert step == 0. and temperature == 0.


def test_a_policy_locked_on_one_side_is_still_trained_at_the_other_accounts():
    """Exploring starts over accounts: while holding short, flat and long accounts are in every update."""
    t = trainer(config())
    with torch.no_grad():
        t.model.actor.output.bias[SHORT] += 8.0
    rows, bootstrap = t.collect()
    sides = rows['sibling_account'][..., :3].argmax(-1)
    active = rows['sibling_active'] & rows['valid'][..., None]
    assert np.mean(rows['sides'][rows['valid']] == SHORT) > 0.9
    assert ((sides == LONG) & active).any(-1)[rows['valid']].all()
    assert ((sides == FLAT) & active).any(-1)[rows['valid']].all()
    summary = t.update(rows, bootstrap)
    for key in ('counterfactual_consistency_gap', 'sibling_side_coverage', 'action_value_spread',
                'sibling_value_error', 'sibling_policy_entropy'):
        assert key in summary, key
    assert summary['sibling_side_coverage'] == 1.0


# --- evaluation -------------------------------------------------------------------------------

def test_evaluation_samples_and_reports_a_distribution():
    c = config()
    t = trainer(c, size=20000)
    m = t.env.market
    result, frame = evaluate([t.model], m, t.scaler, c, m.timestamp(15000), m.timestamp(16000))
    assert result['orders_max'] > 0, 'a near-uniform sampled policy must trade'
    assert len(result['per_seed']) == c.evaluation_seeds
    assert set(frame.seed.unique()) == set(result['seeds'])
    assert result['total_log_return_btc_min'] <= result['total_log_return_btc'] <= result['total_log_return_btc_max']
    assert 0 <= result['long_share'] <= 1


def test_quarter_windows_tile_the_year():
    windows = quarter_windows(2024)
    assert len(windows) == 4
    assert windows[0][0] == pd.Timestamp('2024-01-01', tz='UTC')
    assert windows[-1][1] == pd.Timestamp('2025-01-01', tz='UTC')
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))


def test_metrics_count_flickers_entries_and_their_direction():
    frame = pd.DataFrame({'time': pd.date_range('2023-01-01', periods=8, freq='5min'),
                          'log_return': 0.0, 'differential_sharpe': 0.0, 'equity_btc': 1.0,
                          'equity_usd': 100.0, 'mark': 100.0,
                          'contracts': [0, 100, 0, 0, -100, -100, 100, 0]})
    result = metrics(frame, Ledger(1, config()), 100.0)
    assert result['single_bar_round_trips'] == 2
    assert result['entries'] == 2 and result['long_entry_share'] == pytest.approx(0.5)
    assert result['reversals'] == 1


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


# --- planted-edge markets -----------------------------------------------------------------------

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


def passive_totals(m, regime, bounds):
    from agent.synthetic import oracle_actions
    c = Config()
    steps = (bounds[1] - bounds[0]) // 5
    totals = {}
    for name, actions in (('oracle', oracle_actions(m, regime, bounds)), ('long', np.full(steps, LONG))):
        env = Environment(m, bounds, c, np.random.default_rng(0), training=False)
        totals[name] = sum(float(env.step(np.array([a]))[2]['log_return'][0]) for a in actions)
    return totals


def test_oracle_beats_buy_and_hold_on_the_planted_market():
    from agent.synthetic import planted_market
    m, regime = planted_market(minutes=120_000, seed=0)
    totals = passive_totals(m, regime, m.bounds(m.timestamp(20_000), m.timestamp(110_000)))
    assert totals['oracle'] > totals['long'] + 0.1 and totals['oracle'] > 0.2


def test_drift_market_carries_the_long_attractor_over_seven_years():
    """The verification script's drift construction: most training episodes rise, as BTC's did,
    and on the held-out window long pays while the regime pays more.

    Over a 299-day span the same drift was invisible - constant long was liquidated -
    which is why the span is seven years, the length of BTC's 2017-2023 training data.
    """
    from agent.synthetic import BTC_IN_SAMPLE_SHARPE, planted_market
    seven_years = 20_000 + 7 * 525_600
    m, regime = planted_market(minutes=seven_years + 150_000, seed=7, start='2010-01-01',
                               unconditional_sharpe=BTC_IN_SAMPLE_SHARPE)
    price = np.asarray(m.prices[:, 3])
    window = Config().episode_steps * 5
    starts = np.arange(20_000, seven_years - window, 1440)
    assert np.mean(np.log(price[starts + window] / price[starts]) > 0) > 0.5
    totals = passive_totals(m, regime, m.bounds(m.timestamp(seven_years + 10_000), m.timestamp(seven_years + 140_000)))
    assert totals['long'] > 0
    assert totals['oracle'] > totals['long'] + 0.2
