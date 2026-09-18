from dataclasses import replace
import copy
import numpy as np
import pytest
import torch

from perpetual.config import TestLockedError as LockedError
from perpetual.env import Environment, FLAT, LONG, SHORT
from perpetual.network import tensors
from perpetual_v5.config import Config, variants
from perpetual_v5.network import ActorCritic, CalibratedBatchNorm
from perpetual_v5.ppo import Trainer
from test_perpetual import market, scaler

torch.set_num_threads(1)


def config(**kwargs):
    return replace(Config(), environments=3, rollout=16, sequence=8, burn_in=8,
                   episode_steps=32, minibatch_sequences=3, fast_minutes=12,
                   local_width=8, hidden=8, position_hidden=4, head_width=8,
                   counterfactual_stride=8, counterfactual_horizon=2,
                   epochs=1, critic_epochs=0, updates=2, counterfactual_warmup=0, **kwargs)


def trainer(c, seed=11):
    torch.manual_seed(seed)
    m = market(2000)
    rng = np.random.default_rng(34)
    m.fast[:] = rng.normal(0, .1, m.fast.shape)
    m.slow[:] = rng.normal(0, .1, m.slow.shape)
    from perpetual_v5.environment import TrainingEnvironment
    e = TrainingEnvironment(m, (100, 2000), c, np.random.default_rng(seed))
    t = Trainer(ActorCritic(c), e, scaler(), c, seed)
    t.calibrate()
    return t


@pytest.mark.parametrize('cell', ['gru', 'lstm'])
@pytest.mark.parametrize('norm', ['none', 'layer', 'batch'])
def test_dropout_replay_counterfactual_and_resume(cell, norm):
    c = config(cell=cell, normalization=norm, attention=True)
    t = trainer(c)
    first = t.run_update()
    assert first['replay_ratio_error'] < 5e-5
    assert first['counterfactual_replay_error'] < 5e-5
    assert first['counterfactual_branches'] == 18
    assert first['cf_actor_loss'] != 0
    saved = copy.deepcopy(t.state_dict())
    expected = t.run_update()
    restored = trainer(c)
    restored.load_state_dict(saved)
    actual = restored.run_update()
    assert actual == expected
    for key, value in t.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('cell', ['gru', 'lstm'])
@pytest.mark.parametrize('attention', [False, True])
def test_every_minute_including_latest_has_influence(cell, attention):
    model = ActorCritic(config(cell=cell, attention=attention))
    fast = torch.randn(2, 12, 6, requires_grad=True)
    model.actor.local(fast).square().sum().backward()
    assert (fast.grad.abs().sum((0, 2)) > 0).all()


def test_current_account_reaches_head_and_critic_cannot_update_actor():
    t = trainer(config())
    minute, account = t.env.observation()
    inputs = tensors(t.scaler, t.env.market, minute[:, None], account[:, None], t.config)
    a, _, _, _ = t.model(*inputs)
    altered = inputs[2].clone()
    # Same side; alter current P&L and age rather than the continuation mask.
    altered[..., 2] = .4
    altered[..., 3] = .5
    b, value, _, _ = t.model(inputs[0], inputs[1], altered)
    assert not torch.equal(a, b)
    value.sum().backward()
    assert all(p.grad is None for p in t.model.actor.parameters())


def test_gate_is_trainable_and_preserves_all_three_actions():
    t = trainer(config())
    minute, account = t.env.observation()
    x = list(tensors(t.scaler, t.env.market, minute[:, None], account[:, None], t.config))
    x[2][:, 0, 0] = torch.tensor([-1, 0, 1])
    logits, _, _, _ = t.model(*x)
    p = logits.exp()[:, 0]
    torch.testing.assert_close(p.sum(-1), torch.ones(3))
    torch.testing.assert_close(p.diag(), torch.tensor([.9, .5, .9]), atol=.01, rtol=0)
    assert (p > 0).all()
    (-logits[:, 0, 0].mean()).backward()
    assert t.model.actor.head[-1].bias.grad[0].abs() > 0
    assert t.model.actor.continuation_bias.grad.abs().sum() > 0


def test_batchnorm_is_batch_independent_and_does_not_change_in_ppo():
    t = trainer(config(normalization='batch'))
    bn = t.model.actor.norm
    assert isinstance(bn, CalibratedBatchNorm)
    before = (bn.running_mean.clone(), bn.running_var.clone())
    x = torch.randn(3, 4, 8)
    torch.testing.assert_close(bn(x)[:1], bn(x[:1]), rtol=0, atol=0)
    t.run_update()
    torch.testing.assert_close(before[0], bn.running_mean, rtol=0, atol=0)
    torch.testing.assert_close(before[1], bn.running_var, rtol=0, atol=0)


def test_recorded_dropout_masks_are_required_for_replay():
    t = trainer(config(dropout=.5))
    rows, bootstrap, cf = t.collect()
    rows['masks'] = tuple(torch.ones_like(m) for m in rows['masks'])
    with pytest.raises(RuntimeError, match='replay mismatch'):
        t.update(rows, bootstrap, cf)


def test_counterfactual_accounts_pay_costs_and_do_not_mutate_actual_account():
    t = trainer(config(fee=.0005, slippage=.0001))
    # Begin this accounting diagnostic flat after the general training warmup.
    t.env.ledger.reset(np.ones(t.env.size, bool))
    t.prefix_minutes, t.prefix_account = Environment.initial_prefix(t.env)
    # Zero continuation value and policy that almost always retains its side.
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


def test_training_initial_histories_have_real_positions_and_consistent_final_state():
    from perpetual_v5.environment import TrainingEnvironment
    c = replace(config(), environments=30, burn_in=64)
    env = TrainingEnvironment(market(3000), (100, 3000), c, np.random.default_rng(11))
    minutes, accounts = env.initial_prefix()
    assert np.array_equal(minutes[:, -1]+5, env.cursor)
    assert np.unique(np.sign(env.ledger.contracts)).tolist() == [-1., 0., 1.]
    assert env.ledger.fees_total.sum() > 0
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


def test_matrix_has_matched_cells_and_separate_components():
    configs = variants()
    assert len(configs) == 18
    assert len({k.removeprefix('gru_').removeprefix('lstm_') for k in configs}) == 9
    assert configs['gru_plain'].dropout == 0
    assert configs['lstm_attention'].normalization == 'none'


@pytest.mark.parametrize('cell', ['gru', 'lstm'])
def test_cached_evaluation_matches_full_recurrent_policy(cell):
    from perpetual_v5.evaluate import evaluate
    from perpetual_v5.network import map_hidden
    c = config(cell=cell, attention=True, normalization='batch')
    t = trainer(c)
    m = t.env.market
    # Market.bounds reserves long feature warmup; use synthetic timestamps directly.
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


def test_v5_evaluator_rejects_test_before_data_access():
    from perpetual_v5.evaluate import evaluate
    with pytest.raises(LockedError):
        evaluate([], None, None, config(), '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z')


def test_ppo_only_control_has_no_counterfactual_update():
    t = trainer(config(counterfactual_enabled=False))
    result = t.run_update()
    assert result['counterfactual_branches'] == 0
    assert result['cf_actor_loss'] == 0
