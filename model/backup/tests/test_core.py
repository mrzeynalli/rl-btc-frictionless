"""Small deterministic tests for the fixed five-minute implementation."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from cryptorl.env import TradingConfig, TradingEnvironment
from cryptorl.features import FEATURE_NAMES, MarketData, build_market_data
from cryptorl.model import RecurrentActorCritic
from cryptorl.ppo import PPOConfig, collect_rollout, sequence_ranges, unchanged_policy_invariant


def synthetic_market(settlement_rate: float = 0.0) -> MarketData:
    length = 8
    opens = np.asarray([100, 100, 101, 102, 101, 103, 104, 105], dtype=float)
    closes = np.asarray([100, 101, 102, 101, 103, 104, 105, 106], dtype=float)
    times = np.arange(length, dtype=np.int64) * 300_000_000_000
    settlement_rates = np.zeros(length)
    settlement_rates[1] = settlement_rate
    return MarketData(
        open_times_ns=times,
        close_times_ns=times + 299_999_000_000,
        market_features=np.zeros((length, 43), dtype=np.float32),
        open_prices=opens,
        high_prices=np.maximum(opens, closes),
        low_prices=np.minimum(opens, closes),
        close_prices=closes,
        mark_open_prices=opens.copy(),
        settlement_rates=settlement_rates,
        settlement_times_ns=np.where(settlement_rates != 0, times + 1_000_000, -1),
        study_start=0,
        study_end=length,
        splits={"train": (0, length), "validation": (0, length), "test": (0, length)},
        normalization={}, feature_audit={}, data_manifest={"audit_passed": True},
    )


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_market_data()

    def test_requested_state_only(self):
        self.assertEqual(self.data.market_features.shape[1], 43)
        self.assertEqual(len(FEATURE_NAMES), 46)
        self.assertFalse(any("price" in name for name in FEATURE_NAMES))
        self.assertFalse(any(name in FEATURE_NAMES for name in ("rsi", "macd", "funding_phase")))

    def test_boundaries_and_training_only_normalization(self):
        self.assertEqual(self.data.splits["train"], (3456, 77040))
        self.assertEqual(self.data.splits["validation"], (77040, 92808))
        self.assertEqual(self.data.splits["test"], (92808, 108576))
        self.assertEqual(self.data.data_manifest["warmup_days"], 12.0)
        self.assertTrue(self.data.normalization["warmup_excluded_from_fit"])
        for parameter in self.data.normalization["parameters"].values():
            self.assertEqual(parameter["fit_start_index"], 3456)
            self.assertEqual(parameter["fit_end_exclusive_index"], 77040)

    def test_observation_position_is_one_hot(self):
        observation = self.data.observation(self.data.study_start, -1)
        np.testing.assert_array_equal(observation[-3:], [1, 0, 0])
        observation = self.data.observation(self.data.study_start, 1)
        np.testing.assert_array_equal(observation[-3:], [0, 0, 1])


class AccountingTests(unittest.TestCase):
    def test_holding_keeps_quantity_and_charges_no_second_entry(self):
        environment = TradingEnvironment(synthetic_market(), 0, 5, TradingConfig())
        environment.reset()
        _, _, _, first = environment.step(2)
        quantity = environment.quantity
        _, _, _, second = environment.step(2)
        self.assertEqual(first["execution_legs"], 1)
        self.assertEqual(second["execution_legs"], 0)
        self.assertEqual(environment.quantity, quantity)
        self.assertEqual(second["transaction_cost"], 0.0)

    def test_reversal_has_two_execution_legs(self):
        environment = TradingEnvironment(synthetic_market(), 0, 5, TradingConfig())
        environment.reset()
        environment.step(2)
        _, _, _, reversal = environment.step(0)
        self.assertEqual(reversal["change_type"], "reversal")
        self.assertEqual(reversal["execution_legs"], 2)
        self.assertGreater(reversal["transaction_cost"], 0.0)

    def test_recorded_funding_applies_after_next_open_execution(self):
        environment = TradingEnvironment(
            synthetic_market(settlement_rate=0.001), 0, 4, TradingConfig()
        )
        environment.reset()
        _, _, _, info = environment.step(2)
        self.assertLess(info["funding_pnl"], 0.0)
        self.assertNotEqual(info["funding_time_ns"], -1)

    def test_accounting_reconciles(self):
        environment = TradingEnvironment(synthetic_market(), 0, 6, TradingConfig())
        environment.reset()
        for action in (2, 2, 0, 1, 2):
            _, _, done, info = environment.step(action)
            self.assertAlmostEqual(info["accounting_reconciliation_error"], 0.0, places=8)
            if done:
                break


class RecurrentPPOTests(unittest.TestCase):
    def test_independent_actor_and_critic(self):
        model = RecurrentActorCritic(46, 16)
        actor_ids = {id(parameter) for parameter in model.actor_gru.parameters()}
        critic_ids = {id(parameter) for parameter in model.critic_gru.parameters()}
        self.assertTrue(actor_ids.isdisjoint(critic_ids))

    def test_unchanged_policy_invariant_and_value_targets(self):
        torch.manual_seed(7)
        np.random.seed(7)
        device = torch.device("cpu")
        data = synthetic_market()
        environment = TradingEnvironment(data, 0, len(data), TradingConfig())
        model = RecurrentActorCritic(environment.observation_size, 8)
        config = PPOConfig(
            rollout_steps=12, sequence_length=3, sequences_per_minibatch=2
        )
        actor_hidden, critic_hidden = model.initial_hidden(1, device)
        rollout, *_ = collect_rollout(
            model, environment, environment.reset(), actor_hidden, critic_hidden,
            config, device, steps=12,
        )
        metrics = unchanged_policy_invariant(model, rollout, config, device)
        self.assertLess(metrics["mean_absolute_log_probability_difference"], 1e-6)
        self.assertLess(metrics["approximate_kl"], 1e-6)
        self.assertAlmostEqual(metrics["probability_ratio_mean"], 1.0, places=6)
        self.assertEqual(metrics["clipped_ratio_fraction"], 0.0)
        np.testing.assert_allclose(
            rollout.value_targets, rollout.old_values + rollout.advantages
        )
        for start, end in sequence_ranges(rollout, config.sequence_length):
            self.assertFalse(rollout.episode_starts[start + 1 : end].any())


if __name__ == "__main__":
    unittest.main()
