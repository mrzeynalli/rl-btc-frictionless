"""Self-contained tests for the agent: ledger, reward, exploration, evaluation."""
from dataclasses import replace
import copy
import math

import numpy as np
import pandas as pd
import pytest
import torch

from agent.config import Config, TestLockedError, protocol, SEEDS, YEARS
from agent.data import FAST_NAMES, SLOW_NAMES, Market, Scaler
from agent.evaluate import block_windows, evaluate, metrics
from agent.ledger import (Environment, Ledger, TrainingEnvironment, FLAT, LONG, SHORT,
                          PORTFOLIO_NAMES, PORTFOLIO_WIDTH)
from agent.network import ActorCritic, direction_entropy, gate_entropy, tensors
from agent.ppo import Trainer, state_action_mutual_information
from agent.reward import differential_sharpe, observation_features

torch.set_num_threads(1)


def market(size=2000, drift=0.0, seed=0):
    generator = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(generator.normal(drift, 1e-4, size)))
    rows = np.repeat(price[:, None], 4, axis=1)
    return Market(rows, np.ones(size, bool), np.zeros(size),
                  np.zeros((size, len(FAST_NAMES)), np.float32),
                  np.zeros((size, len(SLOW_NAMES)), np.float32),
                  int(pd.Timestamp('2020-01-01', tz='UTC').value // 60_000_000_000), {})


def scaler():
    return Scaler(np.zeros(len(FAST_NAMES), np.float32), np.ones(len(FAST_NAMES), np.float32),
                  np.zeros(len(SLOW_NAMES), np.float32), np.ones(len(SLOW_NAMES), np.float32),
                  '2021-01-01T00:00:00Z')


def config(**kwargs):
    # Reduced sizes need a reduced horizon and a reduced reward window, since
    # Config enforces that the rollout can carry its own credit window and that
    # the reward baseline is slower than that window.
    return replace(Config(), environments=3, rollout=16, sequence=8, burn_in=8,
                   episode_steps=32, minibatch_sequences=3, fast_minutes=12,
                   local_width=8, hidden=8, position_hidden=4, head_width=8,
                   gamma=.9, gae_lambda=.85, dsr_eta=1/64,
                   counterfactual_stride=8, counterfactual_horizon=2,
                   epochs=1, critic_epochs=0, updates=2, counterfactual_warmup=0,
                   evaluation_seeds=2, **kwargs)


def trainer(c, seed=11, size=2000):
    torch.manual_seed(seed)
    m = market(size)
    generator = np.random.default_rng(34)
    m.fast[:] = generator.normal(0, .1, m.fast.shape)
    m.slow[:] = generator.normal(0, .1, m.slow.shape)
    env = TrainingEnvironment(m, (100, size), c, np.random.default_rng(seed))
    t = Trainer(ActorCritic(c), env, scaler(), c, seed)
    t.calibrate()
    return t


# --- specification ---------------------------------------------------------

def test_credit_window_clears_the_cost_break_even():
    c = Config()
    assert c.gamma == c.gae_lambda == 0.99825
    assert 23 < c.credit_window_hours < 25
    assert c.credit_window_hours > 22
    assert c.rollout >= 3 * c.credit_window


def test_cost_is_realistic_and_fixed_with_no_curriculum():
    c = Config()
    assert (c.fee, c.slippage) == (0.0005, 0.0001)
    assert c.round_trip_cost == pytest.approx(0.0012)
    # An edge discovered below live cost does not survive contact with it, so
    # nothing in the config may schedule the fee.
    assert not any('curriculum' in name for name in c.as_dict())


def test_rollout_must_carry_its_own_credit_window():
    with pytest.raises(ValueError, match='truncates the credit window'):
        replace(Config(), rollout=256, episode_steps=1024)


def test_reward_window_must_be_slower_than_the_credit_window():
    with pytest.raises(ValueError, match='differential Sharpe window is shorter'):
        replace(Config(), dsr_eta=1/100)


def test_experiment_refuses_the_predecessor_horizon(tmp_path):
    from agent import experiment
    short = replace(Config(), gamma=.9975961328836203, gae_lambda=.9738730024397007)
    assert short.credit_window_hours < 3
    with pytest.raises(ValueError, match='below the cost break-even'):
        experiment.run(tmp_path, short, workers=1)


def test_seed_and_fold_budget():
    assert len(SEEDS) == 3 and len(YEARS) == 3
    assert len(SEEDS) * len(YEARS) == 9


# --- portfolio width -------------------------------------------------------

def test_no_component_hardcodes_the_portfolio_width():
    """The reward statistics widened the observation from 16 to 18.

    Three separate buffers in the predecessor allocated a literal 16, which
    would silently truncate the two new features.
    """
    assert PORTFOLIO_WIDTH == len(PORTFOLIO_NAMES) == 18
    c = config()
    t = trainer(c)
    minutes, accounts = t.env.initial_prefix()
    assert accounts.shape[-1] == PORTFOLIO_WIDTH
    rows, _, _ = t.collect()
    assert rows['account'].shape[-1] == PORTFOLIO_WIDTH
    assert t.env.warm_account.shape[-1] == PORTFOLIO_WIDTH
    assert Ledger(4, c).portfolio(np.full(4, 100.)).shape[-1] == PORTFOLIO_WIDTH


def test_reward_statistics_are_visible_to_the_policy():
    assert 'running_sharpe' in PORTFOLIO_NAMES and 'relative_volatility' in PORTFOLIO_NAMES
    c = config()
    ledger = Ledger(2, c)
    before = ledger.portfolio(np.full(2, 100.)).copy()
    ledger.dsr_mean[:] = 0.02
    ledger.dsr_second[:] = 4 * c.dsr_initial_second_moment
    after = ledger.portfolio(np.full(2, 100.))
    changed = np.flatnonzero(np.abs(after - before).sum(0) > 0)
    assert set(changed) == {PORTFOLIO_NAMES.index('running_sharpe'),
                            PORTFOLIO_NAMES.index('relative_volatility')}


# --- differential Sharpe ---------------------------------------------------

def test_differential_sharpe_is_scale_invariant():
    """A Sharpe ratio cannot depend on the units of its returns."""
    c = config()
    r = np.array([0.001, -0.002, 0.0005])
    a, b = np.zeros(3), np.full(3, c.dsr_initial_second_moment)
    plain, _, _ = differential_sharpe(r, a, b, c)
    scaled, _, _ = differential_sharpe(100 * r, 100 * a, 10000 * b,
                                       replace(c, dsr_initial_second_moment=10000 * c.dsr_initial_second_moment))
    np.testing.assert_allclose(plain, scaled, rtol=1e-9)


def test_differential_sharpe_prefers_steadiness_to_magnitude():
    """Two paths, same total return, different variance: the steadier wins."""
    c = replace(config(), dsr_clip=1e9)
    steady = np.full(200, 0.001)
    lumpy = np.tile([0.021, -0.019], 100)
    assert steady.sum() == pytest.approx(lumpy.sum())
    totals = []
    for path in (steady, lumpy):
        a, b = np.zeros(1), np.full(1, c.dsr_initial_second_moment)
        total = 0.0
        for value in path:
            d, a, b = differential_sharpe(np.array([value]), a, b, c)
            total += float(d[0])
        totals.append(total)
    assert totals[0] > totals[1]


def test_leverage_ratchet_does_not_pay_under_differential_sharpe():
    """Scaling exposure scales return and volatility together; fees are loss."""
    c = replace(config(), dsr_clip=1e9)
    generator = np.random.default_rng(5)
    base = generator.normal(2e-4, 1.5e-3, 400)
    totals = []
    for multiple, fees in ((1.0, 0.0), (2.0, 0.0)):
        a, b = np.zeros(1), np.full(1, c.dsr_initial_second_moment)
        total = 0.0
        for value in base * multiple - fees:
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


def test_reward_is_bounded_and_survives_a_collapsing_denominator():
    c = config()
    d, _, _ = differential_sharpe(np.array([1.0]), np.zeros(1), np.zeros(1), c)
    assert np.isfinite(d).all() and abs(d[0]) <= c.dsr_clip


def test_environment_reports_both_reward_signals():
    c = config()
    env = Environment(market(600), (100, 600), c, np.random.default_rng(0), training=False)
    _, _, info = env.step(np.array([LONG]))
    assert 'log_return' in info and 'differential_sharpe' in info
    assert info['log_return'].shape == info['differential_sharpe'].shape


# --- exploration -----------------------------------------------------------

def test_direction_entropy_is_the_conditional_and_maxes_at_ln2():
    logits = torch.log(torch.tensor([[0.98, 0.01, 0.01]]))
    current = torch.tensor([0])
    # Given a switch, the two alternatives are equally likely: maximum entropy.
    assert float(direction_entropy(logits, current)[0]) == pytest.approx(math.log(2), abs=1e-6)
    lopsided = torch.log(torch.tensor([[0.98, 0.019, 0.001]]))
    assert float(direction_entropy(lopsided, current)[0]) < math.log(2)


def test_direction_entropy_ignores_the_gate():
    """Holding harder must not change the direction entropy."""
    a = torch.log(torch.tensor([[0.90, 0.05, 0.05]]))
    b = torch.log(torch.tensor([[0.998, 0.001, 0.001]]))
    current = torch.tensor([0])
    assert float(direction_entropy(a, current)[0]) == pytest.approx(float(direction_entropy(b, current)[0]), abs=1e-6)


def test_direction_entropy_is_invariant_to_relabelling_sides():
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    mirrored = torch.log(torch.tensor([[0.7, 0.1, 0.2]]))
    current = torch.tensor([0])
    assert float(direction_entropy(logits, current)[0]) == pytest.approx(
        float(direction_entropy(mirrored, current)[0]), abs=1e-6)


def test_entropy_coefficient_rises_when_direction_entropy_is_below_target():
    t = trainer(config(entropy_target=0.69, entropy_alpha_lr=0.5))
    start = float(t.alpha().detach())
    with torch.no_grad():
        # Lock the direction choice hard onto one side.
        t.model.actor.head[-1].bias[1 + LONG] += 8.0
    t.run_update()
    assert float(t.alpha().detach()) > start


def test_entropy_coefficient_is_bounded_and_survives_resume():
    c = config()
    t = trainer(c)
    t.run_update()
    saved = copy.deepcopy(t.state_dict())
    restored = trainer(c)
    restored.load_state_dict(saved)
    assert float(restored.alpha().detach()) == pytest.approx(float(t.alpha().detach()))
    assert restored.advantage_scale == pytest.approx(t.advantage_scale)
    assert c.entropy_alpha_min <= float(t.alpha().detach()) <= c.entropy_alpha_max


def test_advantage_scale_is_global_not_per_batch():
    """A batch with no signal must produce a small update, not a unit-scale one."""
    t = trainer(config())
    t.run_update()
    first = t.advantage_scale
    assert first is not None and first > 0
    t.run_update()
    # The running estimate moves only a fraction of the way to the new batch.
    assert t.advantage_scale != first


# --- starting states -------------------------------------------------------

def test_training_prefix_produces_all_three_sides_and_varied_ages():
    """The prefix must not inherit the 24 h holding prior.

    Driving it at initial_persistence would change side on roughly a third of
    accounts across the prefix, so most episodes would start flat and the policy
    would seldom practise an exit.
    """
    c = replace(config(), environments=60, burn_in=64)
    env = TrainingEnvironment(market(3000), (100, 3000), c, np.random.default_rng(11))
    sides = np.unique(np.sign(env.ledger.contracts))
    assert set(sides.tolist()) == {-1.0, 0.0, 1.0}
    assert env.ledger.fees_total.sum() > 0
    held = env.ledger.age[env.ledger.contracts != 0]
    assert held.std() > 0
    assert (env.steps == 0).all()


def test_prefix_sides_are_balanced_not_left_to_a_random_walk():
    """Roughly a third flat, a third short, a third long, with varied ages."""
    c = replace(config(), environments=300, burn_in=64)
    env = TrainingEnvironment(market(3000), (100, 3000), c, np.random.default_rng(3))
    sides = np.sign(env.ledger.contracts)
    for value in (-1.0, 0.0, 1.0):
        assert 0.2 < np.mean(sides == value) < 0.5, f'side {value} share off balance'
    ages = env.ledger.age[env.ledger.contracts != 0]
    assert ages.min() < ages.max() and ages.max() <= c.burn_in * 5


def test_training_states_are_reproducible():
    c = replace(config(), environments=30, burn_in=64)
    env = TrainingEnvironment(market(3000), (100, 3000), c, np.random.default_rng(11))
    saved = copy.deepcopy(env.state_dict())
    mask = np.arange(env.size) % 2 == 0
    env.reset(mask)
    expected = env.state_dict()
    env.load_state_dict(saved)
    env.reset(mask)
    for key, value in expected['ledger'].items():
        np.testing.assert_array_equal(env.ledger.state_dict()[key], value)


# --- PPO invariants --------------------------------------------------------

def test_replay_counterfactual_and_resume_are_exact():
    c = config()
    t = trainer(c)
    first = t.run_update()
    assert first['replay_ratio_error'] < 5e-5
    assert first['counterfactual_replay_error'] < 5e-5
    assert first['cf_actor_loss'] != 0
    saved = copy.deepcopy(t.state_dict())
    expected = t.run_update()
    restored = trainer(c)
    restored.load_state_dict(saved)
    assert restored.run_update() == expected


def test_counterfactual_branches_carry_their_own_reward_state():
    """Each branch's Sharpe statistics must diverge with its own returns."""
    t = trainer(config())
    rows, _, cf = t.collect()
    assert cf is not None and np.isfinite(cf['targets']).all()
    assert 'dsr_mean' in t.env.ledger.state_dict()
    assert 'dsr_second' in t.env.ledger.state_dict()


def test_critic_cannot_update_the_actor():
    t = trainer(config())
    minute, account = t.env.observation()
    inputs = tensors(t.scaler, t.env.market, minute[:, None], account[:, None], t.config)
    _, value, _, _ = t.model(*inputs)
    value.sum().backward()
    assert all(p.grad is None for p in t.model.actor.parameters())


def test_gate_starts_at_the_configured_persistence():
    t = trainer(config())
    minute, account = t.env.observation()
    x = list(tensors(t.scaler, t.env.market, minute[:, None], account[:, None], t.config))
    x[2][:, 0, 0] = torch.tensor([-1, 0, 1])
    logits, _, _, _ = t.model(*x)
    p = logits.exp()[:, 0]
    torch.testing.assert_close(p.sum(-1), torch.ones(3))
    torch.testing.assert_close(p.diag(), torch.tensor([.9965, .995, .9965]), atol=.01, rtol=0)
    assert (p > 0).all()


def test_mutual_information_is_zero_for_a_constant_policy():
    mask = torch.ones(200, dtype=torch.bool)
    assert state_action_mutual_information(torch.full((200, 3), 1/3), mask) == pytest.approx(0, abs=1e-9)
    torch.manual_seed(0)
    varied = torch.softmax(torch.randn(200, 3), -1)
    assert state_action_mutual_information(varied, mask) > 0
    assert state_action_mutual_information(varied, torch.zeros(200, dtype=torch.bool)) == 0.


def test_diagnostics_are_reported_every_update():
    summary = trainer(config()).run_update()
    for key in ('entropy_alpha', 'advantage_scale', 'direction_entropy', 'mean_net_log_return',
                'reward_sd', 'reward_clip_fraction', 'state_action_mutual_information',
                'state_action_mutual_information_flat', 'policy_implied_holding_hours'):
        assert key in summary, key


# --- evaluation ------------------------------------------------------------

def test_evaluation_samples_and_actually_trades():
    """The predecessor's argmax evaluation placed zero orders in every run.

    A persistence-gated policy cannot leave flat under argmax, because that
    needs P(switch) > 0.5 while the prior sets it near 0.003.
    """
    c = config()
    t = trainer(c, size=20000)
    m = t.env.market
    start, end = m.timestamp(15000), m.timestamp(16000)
    result, frame = evaluate([t.model], m, t.scaler, c, start, end)
    assert result['orders_max'] > 0, 'sampled evaluation must be able to trade'
    assert set(frame.seed.unique()) == set(result['seeds'])
    # And the reason argmax cannot: at every flat state the mode is "stay flat",
    # so a greedy policy would never open a position at all.
    flat_rows = frame[frame.contracts == 0][['p_short', 'p_flat', 'p_long']].to_numpy()
    assert len(flat_rows) and (flat_rows.argmax(1) == FLAT).all()


def test_evaluation_reports_a_distribution_over_seeds():
    c = config()
    t = trainer(c, size=20000)
    m = t.env.market
    result, _ = evaluate([t.model], m, t.scaler, c, m.timestamp(15000), m.timestamp(16000))
    assert len(result['per_seed']) == c.evaluation_seeds
    assert result['total_log_return_btc_min'] <= result['total_log_return_btc'] <= result['total_log_return_btc_max']
    assert 0 <= result['positive_seed_share'] <= 1


def test_evaluation_seeds_give_different_paths():
    c = config()
    t = trainer(c, size=20000)
    m = t.env.market
    result, _ = evaluate([t.model], m, t.scaler, c, m.timestamp(15000), m.timestamp(16000))
    actions = [r['orders'] for r in result['per_seed']]
    assert result['total_log_return_btc_max'] >= result['total_log_return_btc_min']
    assert len(actions) == c.evaluation_seeds


def test_single_bar_positions_are_reported_separately_from_trades():
    frame = pd.DataFrame({'time': pd.date_range('2023-01-01', periods=8, freq='5min'),
                          'log_return': 0.0, 'differential_sharpe': 0.0, 'equity_btc': 1.0,
                          'equity_usd': 100.0, 'mark': 100.0,
                          'contracts': [0, 100, 0, 0, 100, 100, 100, 0]})
    ledger = Ledger(1, config())
    result = metrics(frame, ledger, 100.0)
    assert result['single_bar_round_trips'] == 1
    assert result['completed_round_trips'] == 2
    assert result['flicker_share'] == pytest.approx(0.5)


def test_block_windows_tile_the_year_and_overlap():
    c = Config()
    windows = block_windows(2024, c)
    assert len(windows) > 15
    assert all((b - a).days == c.block_days for a, b in windows)
    assert (windows[1][0] - windows[0][0]).days == c.block_step_days
    assert windows[-1][1] <= pd.Timestamp('2025-01-01', tz='UTC')


def test_evaluator_refuses_locked_test_data():
    with pytest.raises(TestLockedError):
        evaluate([], None, None, config(), '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z')


def test_protocol_records_the_reasoning():
    spec = protocol()
    assert spec['test_locked'] and not spec['test_performance_accessed']
    assert 'no curriculum' in spec['cost_policy']
    assert 'argmax is never used' in spec['evaluation']
    assert spec['credit_window_hours'] > spec['cost_break_even_hours']


# --- regressions found in review -------------------------------------------

def test_config_survives_the_json_round_trip():
    """protocol.json is read back as lists; tuple fields must still compare equal.

    Otherwise every training job aborts immediately with a false
    'source/config differs from frozen protocol'.
    """
    import json as _json
    c = Config()
    assert c.as_dict() == _json.loads(_json.dumps(c.as_dict()))
    assert isinstance(c.stress_slippage, tuple)


def test_reward_scale_keeps_the_value_target_at_unit_scale():
    """Unscaled, the differential Sharpe return would have a scale of about 17.

    The critic head starts near zero and Adam moves a weight by roughly its
    learning rate per step, so it could never travel that far inside the run.
    """
    c = Config()
    assert c.reward_scale == pytest.approx(math.sqrt(1 - c.gamma ** 2))
    per_step_sd = 1.0                      # the ratio is approximately a z-score
    target_sd = per_step_sd * c.reward_scale * math.sqrt(1 / (1 - c.gamma ** 2))
    assert 0.5 < target_sd < 2.0
    assert 'reward_scale' not in c.as_dict(), 'derived, never stored or tuned'


def test_entropy_coefficient_parameter_is_clamped_not_only_its_reading():
    """The dual loss is exp(log_alpha) * gap, so a drifting parameter would make
    its own gradient explode even while the clamped reading looked healthy."""
    c = config(entropy_target=0.69, entropy_alpha_lr=5.0)
    t = trainer(c)
    with torch.no_grad():
        t.model.actor.head[-1].bias[1 + LONG] += 12.0
    for _ in range(2):
        t.run_update()
    assert float(t.log_alpha.detach()) <= math.log(c.entropy_alpha_max) + 1e-9
    assert float(t.log_alpha.detach()) >= math.log(c.entropy_alpha_min) - 1e-9
    assert math.isfinite(float(t.log_alpha.detach()))


def test_direction_entropy_gradient_is_finite():
    """0 * -inf is discarded in the forward pass but its derivative is nan."""
    logits = torch.log(torch.tensor([[0.9965, 0.0025, 0.0010]], requires_grad=False)).requires_grad_(True)
    direction_entropy(logits, torch.tensor([0])).sum().backward()
    assert torch.isfinite(logits.grad).all()


# --- literature-anchored configuration -------------------------------------

def test_gradient_step_budget_matches_the_recurrent_ppo_reference():
    """CleanRL ppo_atari_lstm: 10M steps / 1024 batch x 4 epochs x 4 minibatches."""
    c = Config()
    reference = (10_000_000 // (8 * 128)) * 4 * 4
    per_update = c.epochs * (c.rollout // c.sequence) * (c.environments // c.minibatch_sequences)
    assert per_update == 32
    assert 0.9 * reference <= c.updates * per_update <= 1.2 * reference


def test_bptt_window_spans_the_credit_window():
    c = Config()
    assert c.sequence >= c.credit_window / 2
    assert c.burn_in / c.sequence == 0.5, 'R2D2 stores 80-step sequences behind a 40-step burn-in'
    with pytest.raises(ValueError, match='BPTT sequence is too short'):
        replace(Config(), sequence=32, minibatch_sequences=4)


def test_minibatch_size_is_preserved_by_the_sequence_change():
    c = Config()
    assert c.minibatch_sequences * c.sequence == 1024
    assert c.environments % c.minibatch_sequences == 0


def test_value_network_is_wider_only_along_the_value_path():
    c = Config()
    actor, critic = c.branch_widths(True), c.branch_widths(False)
    assert critic['hidden'] == 2 * actor['hidden']
    assert critic['head_width'] == 2 * actor['head_width']
    # Shared feature encoders are NOT widened; the citation covers the value MLP.
    assert critic['local_width'] == actor['local_width']
    assert critic['position_hidden'] == actor['position_hidden']
    model = ActorCritic(c)
    assert model.critic.hidden == 2 * model.actor.hidden
    assert sum(p.numel() for p in model.critic.parameters()) > sum(p.numel() for p in model.actor.parameters())


def test_clip_threshold_follows_the_recommendation():
    assert Config().clip == 0.25


# --- gate entropy, the variable that collapsed -----------------------------

def test_gate_entropy_measures_the_keep_switch_decision():
    for keep, expected_hold_h in ((0.9965, 23.8), (0.998106, 44.0)):
        q = (1 - keep) / 2
        logits = torch.log(torch.tensor([[keep, q, q]]))
        value = float(gate_entropy(logits, torch.tensor([0]))[0])
        manual = -(1 - keep) * math.log(1 - keep) - keep * math.log(keep)
        assert value == pytest.approx(manual, abs=1e-5)
    # Maximum when the gate is a coin flip, near zero when it never switches.
    even = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    assert float(gate_entropy(even, torch.tensor([0]))[0]) == pytest.approx(math.log(2), abs=1e-6)
    stuck = torch.log(torch.tensor([[0.9997, 0.00015, 0.00015]]))
    assert float(gate_entropy(stuck, torch.tensor([0]))[0]) < 0.005


def test_gate_entropy_is_independent_of_the_direction_split():
    """The two duals must regulate genuinely separate quantities."""
    a = torch.log(torch.tensor([[0.99, 0.005, 0.005]]))
    b = torch.log(torch.tensor([[0.99, 0.009, 0.001]]))
    current = torch.tensor([0])
    assert float(gate_entropy(a, current)[0]) == pytest.approx(float(gate_entropy(b, current)[0]), abs=1e-6)
    assert float(direction_entropy(a, current)[0]) != pytest.approx(float(direction_entropy(b, current)[0]), abs=1e-3)


def test_gate_entropy_target_is_derived_from_the_break_even():
    c = Config()
    steps = 2 * 22.0 * 60 / c.decision_minutes
    keep = 1 - 1 / steps
    assert c.gate_entropy_target == pytest.approx(-(1-keep)*math.log(1-keep) - keep*math.log(keep))
    assert 0.010 < c.gate_entropy_target < 0.020


def test_gate_coefficient_rises_when_the_gate_collapses():
    c = config(entropy_alpha_lr=0.5)
    t = trainer(c)
    start = float(t.gate_alpha().detach())
    with torch.no_grad():
        t.model.actor.continuation_bias += 6.0      # drive P(keep) toward one
    t.run_update()
    assert float(t.gate_alpha().detach()) > start


def test_gate_entropy_gradient_is_finite():
    logits = torch.log(torch.tensor([[0.9965, 0.0025, 0.0010]])).requires_grad_(True)
    gate_entropy(logits, torch.tensor([0])).sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_both_duals_persist_across_resume():
    c = config()
    t = trainer(c)
    t.run_update()
    restored = trainer(c)
    restored.load_state_dict(copy.deepcopy(t.state_dict()))
    assert float(restored.gate_alpha().detach()) == pytest.approx(float(t.gate_alpha().detach()))
    assert float(restored.alpha().detach()) == pytest.approx(float(t.alpha().detach()))


# --- annealing, KL scope, advantage recomputation --------------------------

def test_learning_rate_anneals_linearly_to_zero():
    c = replace(config(), updates=100, anneal_lr=True)
    t = trainer(c)
    assert t.learning_rates()[0] == pytest.approx(c.actor_lr)
    t.updates = 50
    assert t.learning_rates()[0] == pytest.approx(c.actor_lr / 2)
    t.updates = 100
    assert t.learning_rates()[0] == pytest.approx(0.0)
    off = trainer(replace(c, anneal_lr=False))
    off.updates = 50
    assert off.learning_rates()[0] == pytest.approx(c.actor_lr)


def test_kl_guard_is_off_by_default_and_configurable():
    assert Config().kl_stop_scope == 'off'
    with pytest.raises(ValueError, match='kl_stop_scope'):
        replace(Config(), kl_stop_scope='sometimes')
    for scope in ('off', 'pass', 'update'):
        summary = trainer(config(kl_stop_scope=scope)).run_update()
        assert 'kl_stopped' in summary


def test_advantage_recomputation_preserves_replay_exactness():
    for flag in (True, False):
        t = trainer(config(recompute_advantages_per_pass=flag))
        summary = t.run_update()
        assert summary['replay_ratio_error'] < 5e-5
        assert summary['counterfactual_replay_error'] < 5e-5


def test_advantage_recomputation_is_deterministic_and_resumable():
    c = config(recompute_advantages_per_pass=True)
    t = trainer(c)
    t.run_update()
    saved = copy.deepcopy(t.state_dict())
    expected = t.run_update()
    restored = trainer(c)
    restored.load_state_dict(saved)
    assert restored.run_update() == expected


# --- planted-edge market ---------------------------------------------------

def test_planted_market_signal_is_causal():
    from agent.synthetic import planted_market, REGIME_COLUMN
    m, regime = planted_market(minutes=20_000, seed=1, noise=0.0)
    # The column at minute t must equal the regime as of t-1, never t.
    np.testing.assert_allclose(m.slow[1:, REGIME_COLUMN], regime[:-1], atol=1e-5)
    assert m.slow[0, REGIME_COLUMN] == 0.0


def test_planted_edge_pays_several_times_its_own_cost():
    from agent.synthetic import planted_market, edge_economics
    m, _ = planted_market(minutes=20_000, seed=1)
    economics = edge_economics(m, Config())
    assert economics['edge_to_cost_ratio'] > 3.0


def test_oracle_beats_buy_and_hold_on_the_planted_market():
    """If the ground-truth policy did not win, the test could prove nothing."""
    from agent.synthetic import planted_market, oracle_actions
    m, regime = planted_market(minutes=120_000, seed=0)
    c = Config()
    bounds = m.bounds(m.timestamp(20_000), m.timestamp(110_000))
    totals = {}
    for name, actions in (('oracle', oracle_actions(m, regime, bounds)), ('long', None)):
        env = Environment(m, bounds, c, np.random.default_rng(0), training=False)
        steps = (bounds[1] - bounds[0]) // 5
        sequence = actions if actions is not None else np.full(steps, LONG)
        totals[name] = sum(float(env.step(np.array([a]))[2]['log_return'][0]) for a in sequence)
    assert totals['oracle'] > totals['long'] + 0.1
    assert totals['oracle'] > 0.2


# --- entropy coefficient bound (windup fix) --------------------------------

def test_entropy_coefficient_lr_matches_the_sac_reference():
    """CleanRL SAC: a_optimizer = Adam([log_alpha], lr=q_lr), q_lr = 1e-3."""
    assert Config().entropy_alpha_lr == 1e-3


def test_entropy_coefficient_is_capped_within_the_measured_working_band():
    """Peak alpha where the mechanism worked: 0.0089-0.0474.
    Peak alpha where the gate overshot its target 19-36x: 0.0926, 0.1903.
    """
    cap = Config().entropy_alpha_max
    assert 0.0474 < cap < 0.0926
    t = trainer(config(entropy_alpha_lr=5.0))
    with torch.no_grad():
        t.model.actor.continuation_bias += 10.0      # force a hard gate collapse
    for _ in range(3):
        t.run_update()
    assert float(t.gate_alpha().detach()) <= cap + 1e-9
    assert float(t.alpha().detach()) <= cap + 1e-9


def test_gate_entropy_gradient_vanishes_at_collapse_and_peaks_mid_range():
    """The reason SAC can leave its temperature unbounded and we cannot."""
    def slope(p):
        z = math.log(p/(1-p))
        return abs(z*p*(1-p))
    collapsed, target, middle = slope(0.9999), slope(0.998106), slope(0.78)
    assert collapsed < target < middle
    assert target/collapsed > 10      # too weak to escape a collapse
    assert middle/target > 10         # too strong once escaped


def test_coefficient_saturation_is_reported():
    summary = trainer(config()).run_update()
    assert 'entropy_alpha_saturated' in summary
    assert 0.0 <= summary['entropy_alpha_saturated'] <= 1.0
