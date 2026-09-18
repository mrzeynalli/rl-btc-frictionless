from dataclasses import replace
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from perpetual.config import Config, TEST_START, TestLockedError as LockedError, development_range
from perpetual.data import FAST_NAMES, SLOW_NAMES, Market, Scaler, causal_arrays
from perpetual.env import Environment, Ledger, SHORT, FLAT, LONG, PORTFOLIO_WIDTH
from perpetual.evaluate import evaluate
from perpetual.network import ActorCritic, tensors
from perpetual.ppo import Trainer, advantages

torch.set_num_threads(1)


def market(size=2000):
    rows = np.full((size, 4), 100.)
    return Market(rows, np.ones(size, bool), np.zeros(size),
                  np.zeros((size, len(FAST_NAMES)), np.float32), np.zeros((size, len(SLOW_NAMES)), np.float32),
                  int(pd.Timestamp("2020-01-01", tz="UTC").value // 60_000_000_000), {})


def config(**kwargs):
    return replace(Config(environments=2, fast_minutes=16, hidden=16, local_width=16,
                          rollout=16, sequence=8, burn_in=8, episode_steps=64,
                          epochs=1, critic_epochs=1, minibatch_sequences=2, updates=2, fee=0, slippage=0), **kwargs)


def scaler():
    return Scaler(np.zeros(len(FAST_NAMES), np.float32), np.ones(len(FAST_NAMES), np.float32),
                  np.zeros(len(SLOW_NAMES), np.float32), np.ones(len(SLOW_NAMES), np.float32), "2021-01-01T00:00:00Z")


@pytest.mark.parametrize("side,exit_price,expected", [(LONG, 110, 1 + 100 * (1/100-1/110)),
                                                     (SHORT, 90, 1 - 100 * (1/100-1/90))])
def test_inverse_contract_pnl(side, exit_price, expected):
    ledger = Ledger(1, config())
    ledger.execute(np.array([side]), np.array([100.]), np.array([True]))
    ledger.mark(np.array([exit_price]))
    assert ledger.equity[0] == pytest.approx(expected)
    ledger.execute(np.array([FLAT]), np.array([float(exit_price)]), np.array([True]))
    assert ledger.wallet[0] == pytest.approx(expected)
    assert ledger.contracts[0] == 0


def test_hold_keeps_quantity_and_reversal_pays_both_legs():
    ledger = Ledger(1, config(fee=0.001))
    ledger.execute(np.array([LONG]), np.array([100.]), np.array([True]))
    assert ledger.contracts[0] == 100
    initial_fee = ledger.fees_total[0]
    ledger.execute(np.array([LONG]), np.array([110.]), np.array([True]))
    assert ledger.contracts[0] == 100 and ledger.fees_total[0] == initial_fee
    ledger.execute(np.array([SHORT]), np.array([100.]), np.array([True]))
    assert ledger.orders[0] == 3
    assert ledger.fees_total[0] == pytest.approx(0.00299)


def test_slippage_adverse_and_flat_wallet_is_btc():
    for action in (SHORT, LONG):
        ledger = Ledger(1, config(slippage=0.001))
        ledger.execute(np.array([action]), np.array([100.]), np.array([True]))
        assert ledger.value(np.array([100.]))[0] < 1
    ledger = Ledger(1, config(fee=0.001))
    ledger.execute(np.array([FLAT]), np.array([100.]), np.array([True]))
    assert ledger.value(np.array([110.]))[0] == 1


def test_funding_precedes_entry_and_same_minute_exit():
    m = market()
    m.funding[100], m.funding[105] = 0.01, 0.02
    env = Environment(m, (100, 120), config(), np.random.default_rng(0), training=False)
    _, _, info = env.step(np.array([LONG]))
    assert info["funding_btc"][0] == 0
    _, _, info = env.step(np.array([FLAT]))
    assert info["funding_btc"][0] == pytest.approx(-0.02)
    assert env.ledger.wallet[0] == pytest.approx(0.98)


def test_short_receives_positive_funding():
    ledger = Ledger(1, config())
    ledger.execute(np.array([SHORT]), np.array([100.]), np.array([True]))
    ledger.funding_event(np.array([0.01]), np.array([100.]))
    assert ledger.wallet[0] == pytest.approx(1.01)


def test_no_trade_does_not_fill_but_position_and_funding_persist():
    m = market()
    m.present[100:105] = False
    m.prices[100:105, :3] = np.nan
    env = Environment(m, (100, 120), config(), np.random.default_rng(0), training=False)
    _, _, info = env.step(np.array([LONG]))
    assert env.ledger.contracts[0] == 0 and not info["filled"][0]
    env.step(np.array([LONG]))
    m.present[110:115] = False
    m.prices[110:115, :3] = np.nan
    m.funding[112] = 0.01
    env.step(np.array([FLAT]))
    assert env.ledger.contracts[0] == 100 and env.ledger.wallet[0] == pytest.approx(0.99)


def test_first_eligible_trade_controls_fill_price():
    m = market()
    m.present[100:102] = False
    m.prices[100:102, :3] = np.nan
    m.prices[102:, :] = 200
    env = Environment(m, (100, 120), config(), np.random.default_rng(0), training=False)
    env.step(np.array([LONG]))
    assert env.ledger.entry[0] == 200 and env.ledger.contracts[0] == 200
    assert env.ledger.equity[0] == pytest.approx(1)


def test_intrabar_liquidation_is_terminal_and_prevents_reopening():
    m = market()
    m.prices[102, 2] = 40
    env = Environment(m, (100, 120), config(), np.random.default_rng(0), training=False)
    reward, terminal, _ = env.step(np.array([LONG]))
    assert terminal[0] and reward[0] < -100
    reward, _, info = env.step(np.array([LONG]))
    assert reward[0] == 0 and info["contracts"][0] == 0


def test_gae_time_limit_bootstrap_and_economic_termination():
    r, v, valid = np.array([[1.]], np.float32), np.array([[10.]], np.float32), np.array([[True]])
    adv, _ = advantages(r, v, np.array([10.]), np.array([[False]]), valid, 0.9, 0.95)
    assert adv[0, 0] == pytest.approx(0)
    adv, _ = advantages(r, v, np.array([10.]), np.array([[True]]), valid, 0.9, 0.95)
    assert adv[0, 0] == -9


def bars_fixture():
    times = pd.date_range("2020-01-01", periods=200, freq="min", tz="UTC")
    bars = pd.DataFrame({"open_time": times, "open": 100., "high": 100., "low": 100., "close": 100., "volume": 10})
    funding = pd.DataFrame({"timestamp": [times[100]], "fundingRate": [0.01], "funding_interval_hours": [8.]})
    return times, bars, funding


def test_observations_ignore_future_price_and_next_funding():
    times, bars, funding = bars_fixture()
    a = causal_arrays(bars, funding, times[0], times[-1] + pd.Timedelta(minutes=1))
    bars.loc[100:, ["open", "high", "low", "close"]] = 500
    funding.loc[0, "fundingRate"] = -0.05
    b = causal_arrays(bars, funding, times[0], times[-1] + pd.Timedelta(minutes=1))
    np.testing.assert_array_equal(a.fast[:100], b.fast[:100])
    np.testing.assert_array_equal(a.slow[:101], b.slow[:101])
    assert a.slow[100, 10] == 0 and a.slow[101, 10] == pytest.approx(0.01)
    assert a.funding[100] == 0.01


def test_missing_observations_are_flagged_and_not_backfilled():
    times, bars, funding = bars_fixture()
    bars = bars.drop(index=[99, 100])
    bars.loc[101:, ["open", "high", "low", "close"]] = 200
    a = causal_arrays(bars, funding, times[0], times[-1] + pd.Timedelta(minutes=1))
    assert a.prices[100, 3] == 100 and np.isnan(a.prices[100, 0])
    assert a.fast[100, 4] == 0 and a.fast[100, 5] > 0


def test_test_period_rejected_before_evaluation_and_features():
    with pytest.raises(LockedError):
        development_range(TEST_START, TEST_START + pd.Timedelta(days=1))
    with pytest.raises(LockedError):
        evaluate([], None, None, config(), "2025-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    times, bars, funding = bars_fixture()
    with pytest.raises(LockedError):
        causal_arrays(bars, funding, times[0], TEST_START + pd.Timedelta(minutes=1))
    development_range("2025-01-01T00:00:00Z", TEST_START)


def test_actor_uses_portfolio_and_critic_has_separate_gradients():
    model = ActorCritic(config())
    fast, slow = torch.zeros(1, 2, 16, len(FAST_NAMES)), torch.zeros(1, 2, len(SLOW_NAMES))
    portfolio = torch.zeros(1, 2, PORTFOLIO_WIDTH)
    a, _, _ = model(fast, slow, portfolio)
    portfolio[..., 0] = 1
    b, value, _ = model(fast, slow, portfolio)
    assert not torch.equal(a, b)
    value.sum().backward()
    assert all(p.grad is None for p in model.actor.parameters())


def test_replay_and_checkpoint_resume_reproduce_next_update():
    c, m = config(), market()
    def create():
        torch.manual_seed(11)
        env = Environment(m, (100, 2000), c, np.random.default_rng(11))
        return Trainer(ActorCritic(c), env, scaler(), c, 11)
    trainer = create()
    assert trainer.run_update()["replay_ratio_error"] < 5e-5
    saved = copy.deepcopy(trainer.state_dict())
    expected = trainer.run_update()
    restored = create()
    restored.load_state_dict(saved)
    assert restored.run_update() == expected
    for name, parameter in trainer.model.state_dict().items():
        torch.testing.assert_close(parameter, restored.model.state_dict()[name], rtol=0, atol=0)


def test_cached_ensemble_matches_full_models_on_one_actual_account():
    c, m = config(), market(16000)
    rng = np.random.default_rng(44)
    m.fast[:] = rng.normal(0, 0.1, m.fast.shape)
    m.slow[:] = rng.normal(0, 0.1, m.slow.shape)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.0001, len(m))))
    m.prices[:] = price[:, None]
    torch.manual_seed(44)
    models = [ActorCritic(c), ActorCritic(c)]
    start, end = m.timestamp(15000), m.timestamp(15120)
    result, cached = evaluate(models, m, scaler(), c, start, end)
    env = Environment(m, m.bounds(start, end), c, np.random.default_rng(0), training=False)
    minute_history, portfolio_history = env.initial_prefix()
    hidden = [None] * len(models)
    with torch.no_grad():
        for step in range(len(cached)):
            minutes, portfolio = env.observation()
            opinions = []
            for i, model in enumerate(models):
                if step % c.rollout == 0:
                    hidden[i] = model.burn(*tensors(scaler(), m, minute_history[:, -c.burn_in:],
                                                   portfolio_history[:, -c.burn_in:], c))
                logits, _, hidden[i] = model(*tensors(scaler(), m, minutes[:, None], portfolio[:, None], c), hidden[i])
                opinions.append(logits[0, 0].softmax(-1))
            action = int(torch.stack(opinions).mean(0).argmax())
            _, _, info = env.step(np.array([action]))
            assert cached.action.iloc[step] == action
            assert cached.equity_btc.iloc[step] == info["equity_btc"][0]
            minute_history = np.concatenate([minute_history, minutes[:, None]], axis=1)
            portfolio_history = np.concatenate([portfolio_history, portfolio[:, None]], axis=1)
    assert result["total_log_return_btc"] == pytest.approx(np.log(result["total_return_btc"]+1))


def test_scaler_cannot_observe_validation_values():
    m = market(16000)
    before = Scaler.fit(m, m.timestamp(15000))
    m.fast[15000:] = 10000
    m.slow[15000:] = -10000
    after = Scaler.fit(m, m.timestamp(15000))
    assert before.as_dict() == after.as_dict()


def test_raw_cache_preparation_pushes_locked_cutoff_into_both_parquet_reads(tmp_path, monkeypatch):
    import perpetual.data as data
    import json
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": "prices", "funding": {"sha256": "funding"}}))
    monkeypatch.setattr(data, "DATA", tmp_path)
    reads = []
    def read(path, *, filters):
        reads.append((str(path), filters))
        return object()
    def stop(bars, funding, start, end):
        assert end == TEST_START
        raise RuntimeError("intentional stop after checking read predicates")
    monkeypatch.setattr(data.pd, "read_parquet", read)
    monkeypatch.setattr(data, "causal_arrays", stop)
    with pytest.raises(RuntimeError, match="intentional stop"):
        data.prepare(tmp_path / "cache")
    assert len(reads) == 2
    for _, filters in reads:
        assert any(operator == "<" and date == TEST_START for _, operator, date in filters)


def test_age_counts_minutes_once():
    env = Environment(market(), (100, 120), config(), np.random.default_rng(0), training=False)
    env.step(np.array([LONG]))
    assert env.ledger.age[0] == 5
    env.step(np.array([LONG]))
    assert env.ledger.age[0] == 10


def test_decisions_align_to_utc_when_first_print_is_off_grid():
    m = market(17000)
    m.epoch_start += 8
    start = pd.Timestamp("2020-01-11", tz="UTC")
    end = start + pd.Timedelta(days=1)
    first, last = m.bounds(start, end, warmup=100)
    assert m.timestamp(first) == start
    assert (last-first) // 5 == 288
    env = Environment(m, (100, 17000), config(), np.random.default_rng(11))
    assert all(m.timestamp(x).minute % 5 == 0 for x in env.cursor)
