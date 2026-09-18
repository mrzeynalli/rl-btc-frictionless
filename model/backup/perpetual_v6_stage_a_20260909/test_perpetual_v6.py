"""Stage A: horizon, holding-time prior, single architecture, new diagnostics."""
from dataclasses import replace
import copy
import numpy as np
import pytest
import torch

from perpetual.config import TestLockedError as LockedError
from perpetual.env import Environment, FLAT, LONG, SHORT
from perpetual.network import tensors
from perpetual_v6.config import Config, variants
from perpetual_v6.network import ActorCritic
from perpetual_v6.ppo import Trainer
from test_perpetual import market, scaler

torch.set_num_threads(1)


def config(**kwargs):
    # Reduced sizes need a reduced horizon too: the rollout must still be able
    # to carry its own credit window, which is exactly what Config enforces.
    return replace(Config(), environments=3, rollout=16, sequence=8, burn_in=8,
                   episode_steps=32, minibatch_sequences=3, fast_minutes=12,
                   local_width=8, hidden=8, position_hidden=4, head_width=8,
                   gamma=.9, gae_lambda=.85, counterfactual_stride=8, counterfactual_horizon=2,
                   epochs=1, critic_epochs=0, updates=2, counterfactual_warmup=0, **kwargs)


def trainer(c, seed=11):
    torch.manual_seed(seed)
    m = market(2000)
    rng = np.random.default_rng(34)
    m.fast[:] = rng.normal(0, .1, m.fast.shape)
    m.slow[:] = rng.normal(0, .1, m.slow.shape)
    from perpetual_v6.environment import TrainingEnvironment
    e = TrainingEnvironment(m, (100, 2000), c, np.random.default_rng(seed))
    t = Trainer(ActorCritic(c), e, scaler(), c, seed)
    t.calibrate()
    return t


# --- stage A specification -------------------------------------------------

def test_credit_window_clears_the_cost_break_even():
    c = Config()
    assert c.gamma == c.gae_lambda == 0.99825
    # 12 bp round trip needs about 22 h in position to pay for itself.
    assert 23 < c.credit_window_hours < 25
    assert c.credit_window_hours > 22


def test_rollout_must_be_able_to_carry_its_own_credit_window():
    assert Config().rollout >= 3*Config().credit_window
    with pytest.raises(ValueError, match='truncates the credit window'):
        replace(Config(), rollout=256, episode_steps=1024)


def test_experiment_refuses_a_horizon_that_would_repeat_v5(tmp_path):
    from perpetual_v6 import experiment
    short = replace(Config(), gamma=.9975961328836203, gae_lambda=.9738730024397007)
    assert short.credit_window_hours < 3          # the v5 setting
    with pytest.raises(ValueError, match='below the cost break-even'):
        experiment.run(tmp_path, short, workers=1)


def test_holding_priors_match_the_intended_economics():
    c = Config()
    hold = 5/(60*(1-c.initial_persistence))
    flat = 5/(60*(1-c.initial_flat_persistence))
    assert 23 < hold < 25          # v5 encoded 50 minutes
    assert 16 < flat < 18          # v5 encoded 10 minutes


def test_single_architecture_and_full_seed_budget():
    from perpetual_v6.config import SEEDS, YEARS
    assert list(variants()) == ['lstm_plain']
    assert variants()['lstm_plain'].cell == 'lstm'
    assert variants()['lstm_plain'].normalization == 'none'
    assert variants()['lstm_plain'].dropout == 0
    assert variants()['lstm_plain'].attention is False
    assert len(variants())*len(YEARS)*len(SEEDS) == 21


def test_stage_a_leaves_reward_costs_and_entropy_untouched():
    c, v5 = Config(), None
    assert (c.reward_scale, c.fee, c.slippage, c.entry_leverage) == (100., .0005, .0001, 1.)
    assert c.entropy_start == c.entropy_end == .005
    assert (c.actor_lr, c.critic_lr) == (1e-4, 3e-4)


# --- new diagnostics -------------------------------------------------------

def test_mutual_information_is_zero_for_a_state_independent_policy():
    # The diagnostics describe the post-update policy, so freeze the actor to
    # keep the constructed policy constant through the update.
    t = trainer(config(actor_lr=0.))
    # Kill every state-dependent path into the actor head. The overall policy
    # still varies with the held side, because the current action is masked out
    # of the alternatives, and the position is genuinely part of the state. But
    # among flat states the side is fixed, so there the policy is constant.
    with torch.no_grad():
        t.model.actor.head[-1].weight.zero_()
        t.model.actor.continuation_bias.zero_()
    summary = t.run_update()
    assert summary['state_action_mutual_information_flat'] == pytest.approx(0, abs=1e-6)
    assert summary['switch_probability_sd'] == pytest.approx(0, abs=1e-6)


def test_mutual_information_is_exactly_zero_for_a_constant_distribution():
    from perpetual_v6.ppo import state_action_mutual_information
    mask = torch.ones(200, dtype=torch.bool)
    constant = torch.full((200, 3), 1/3)
    assert state_action_mutual_information(constant, mask) == pytest.approx(0, abs=1e-9)
    assert state_action_mutual_information(constant, torch.zeros(200, dtype=torch.bool)) == 0.


def test_mutual_information_is_positive_for_a_state_dependent_policy():
    t = trainer(config())
    with torch.no_grad():
        for parameter in t.model.actor.head[-1].parameters():
            parameter.mul_(50)
    summary = t.run_update()
    assert summary['state_action_mutual_information'] > 0
    assert summary['switch_probability_sd'] > 0


def test_mutual_information_is_invariant_to_relabelling_long_and_short():
    """The diagnostic must measure state dependence, never prefer a direction."""
    from perpetual_v6.ppo import state_action_mutual_information
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(500, 3), -1)
    mask = torch.ones(500, dtype=torch.bool)
    baseline = state_action_mutual_information(p, mask)
    mirrored = state_action_mutual_information(p[:, [LONG, FLAT, SHORT]], mask)
    assert baseline > 0
    assert baseline == pytest.approx(mirrored, rel=1e-9)


def test_mutual_information_is_never_negative():
    from perpetual_v6.ppo import state_action_mutual_information
    torch.manual_seed(1)
    almost_constant = torch.full((300, 3), 1/3)+1e-9*torch.randn(300, 3)
    assert state_action_mutual_information(almost_constant, torch.ones(300, dtype=torch.bool)) >= 0


def test_policy_implied_holding_time_tracks_the_switch_probability():
    t = trainer(config())
    summary = t.run_update()
    expected = 5/(60*max(summary['switch_probability_mean'], 1e-9))
    assert summary['policy_implied_holding_hours'] == pytest.approx(expected)


# --- invariants carried over from v5 ---------------------------------------

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
    for key, value in t.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)


def test_every_minute_including_latest_has_influence():
    model = ActorCritic(config())
    fast = torch.randn(2, 12, 6, requires_grad=True)
    model.actor.local(fast).square().sum().backward()
    assert (fast.grad.abs().sum((0, 2)) > 0).all()


def test_current_account_reaches_head_and_critic_cannot_update_actor():
    t = trainer(config())
    minute, account = t.env.observation()
    inputs = tensors(t.scaler, t.env.market, minute[:, None], account[:, None], t.config)
    a, _, _, _ = t.model(*inputs)
    altered = inputs[2].clone()
    altered[..., 2] = .4
    altered[..., 3] = .5
    b, value, _, _ = t.model(inputs[0], inputs[1], altered)
    assert not torch.equal(a, b)
    value.sum().backward()
    assert all(p.grad is None for p in t.model.actor.parameters())


def test_gate_is_trainable_and_starts_at_the_new_persistence():
    t = trainer(config())
    minute, account = t.env.observation()
    x = list(tensors(t.scaler, t.env.market, minute[:, None], account[:, None], t.config))
    x[2][:, 0, 0] = torch.tensor([-1, 0, 1])
    logits, _, _, _ = t.model(*x)
    p = logits.exp()[:, 0]
    torch.testing.assert_close(p.sum(-1), torch.ones(3))
    torch.testing.assert_close(p.diag(), torch.tensor([.9965, .995, .9965]), atol=.01, rtol=0)
    assert (p > 0).all()
    (-logits[:, 0, 0].mean()).backward()
    assert t.model.actor.continuation_bias.grad.abs().sum() > 0


def test_counterfactual_accounts_pay_costs_and_do_not_mutate_actual_account():
    t = trainer(config(fee=.0005, slippage=.0001))
    t.env.ledger.reset(np.ones(t.env.size, bool))
    t.prefix_minutes, t.prefix_account = Environment.initial_prefix(t.env)
    with torch.no_grad():
        t.model.critic.head[-1].weight.zero_()
        t.model.critic.head[-1].bias.zero_()
        t.model.actor.head[-1].weight.zero_()
        t.model.actor.head[-1].bias[0] = 20
    rows, _, cf = t.collect()
    assert (rows['actions'] == FLAT).all()
    np.testing.assert_allclose(cf['targets'][:, FLAT], 0, atol=1e-8)
    assert (cf['targets'][:, SHORT] < 0).all()
    assert (cf['targets'][:, LONG] < 0).all()
    assert t.env.ledger.orders.sum() == 0


def test_training_initial_histories_have_real_positions_and_longer_holds():
    from perpetual_v6.environment import TrainingEnvironment
    c = replace(config(), environments=30, burn_in=64)
    env = TrainingEnvironment(market(3000), (100, 3000), c, np.random.default_rng(11))
    minutes, accounts = env.initial_prefix()
    assert np.array_equal(minutes[:, -1]+5, env.cursor)
    assert (env.steps == 0).all()
    assert (minutes[:, 0] >= env.bounds[0]).all()
    saved = copy.deepcopy(env.state_dict())
    mask = np.arange(env.size) % 2 == 0
    env.reset(mask)
    expected = env.state_dict()
    env.load_state_dict(saved)
    env.reset(mask)
    for key, value in expected['ledger'].items():
        np.testing.assert_array_equal(env.ledger.state_dict()[key], value)


def test_cached_evaluation_matches_full_recurrent_policy():
    from perpetual_v6.evaluate import evaluate
    c = config()
    t = trainer(c)
    m = market(17000)
    rng = np.random.default_rng(44)
    m.fast[:] = rng.normal(0, .1, m.fast.shape)
    m.slow[:] = rng.normal(0, .1, m.slow.shape)
    price = 100*np.exp(np.cumsum(rng.normal(0, .0001, len(m))))
    m.prices[:] = price[:, None]
    start, end = m.timestamp(15000), m.timestamp(15200)
    result, cached = evaluate([t.model], m, t.scaler, c, start, end)
    env = Environment(m, m.bounds(start, end), c, np.random.default_rng(0), training=False)
    minute_history, account_history = env.initial_prefix()
    hidden = None
    with torch.no_grad():
        for step in range(len(cached)):
            minute, account = env.observation()
            if step % c.rollout == 0:
                hidden = t.model.burn(*tensors(t.scaler, m, minute_history[:, -c.burn_in:], account_history[:, -c.burn_in:], c))
            logits, _, _, hidden = t.model(*tensors(t.scaler, m, minute[:, None], account[:, None], c), hidden)
            p = logits[0, 0].exp().numpy()
            np.testing.assert_allclose(p, cached[['p_short', 'p_flat', 'p_long']].iloc[step].to_numpy(), atol=5e-6)
            action = int(p.argmax())
            _, _, info = env.step(np.array([action]))
            assert action == cached.action.iloc[step]
            assert info['equity_btc'][0] == cached.equity_btc.iloc[step]
            minute_history = np.concatenate([minute_history, minute[:, None]], 1)
            account_history = np.concatenate([account_history, account[:, None]], 1)
    assert result['test_locked']


def test_v6_evaluator_rejects_test_before_data_access():
    from perpetual_v6.evaluate import evaluate
    with pytest.raises(LockedError):
        evaluate([], None, None, config(), '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z')
