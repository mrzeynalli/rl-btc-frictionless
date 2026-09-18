"""Prelaunch diagnostics on synthetic data, plus optional pre-2023 timing.

These are software/learning checks, never financial holdout evaluations.
Run from model/: PYTHONPATH=src .venv/bin/python scripts/verify_perpetual.py
"""
from dataclasses import replace
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from perpetual.config import Config, OUTPUT, TRAIN_START, write_json
from perpetual.data import Scaler, causal_arrays, prepare
from perpetual.env import Environment
from perpetual.evaluate import evaluate
from perpetual.network import ActorCritic
from perpetual.ppo import Trainer


def synthetic():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    rng = np.random.default_rng(92)
    size = 40000
    times = pd.date_range("2020-01-01", periods=size, freq="min", tz="UTC")
    # Strong observable, persistent trends with independent small disturbances.
    direction = np.where((np.arange(size) // 500) % 2 == 0, 1., -1.)
    change = direction * 0.0003 + rng.normal(0, 0.000025, size)
    close = 10000 * np.exp(np.cumsum(change))
    opening = np.r_[10000., close[:-1]]
    bars = pd.DataFrame({"open_time": times, "open": opening,
                         "high": np.maximum(opening, close), "low": np.minimum(opening, close),
                         "close": close, "volume": np.full(size, 10000.)})
    funding = pd.DataFrame({"timestamp": times[:0], "fundingRate": [], "funding_interval_hours": []})
    market = causal_arrays(bars, funding, times[0], times[-1] + pd.Timedelta(minutes=1))
    config = replace(Config(), environments=8, fast_minutes=16, hidden=16, local_width=16,
                     rollout=64, sequence=32, burn_in=16, episode_steps=512,
                     epochs=3, critic_epochs=1, minibatch_sequences=8, updates=150,
                     actor_lr=0.001, critic_lr=0.001, gamma=0.98, gae_lambda=0.95,
                     entropy_decay_updates=50, threads=1)
    scaler = Scaler.fit(market, times[28000])
    env = Environment(market, (15000, 28000), config, np.random.default_rng(11))
    model = ActorCritic(config)
    trainer = Trainer(model, env, scaler, config, 11)
    start, end = times[30000], times[35000]
    before = evaluate([model], market, scaler, config, start, end)[0]
    began = time.monotonic()
    for update in range(config.updates):
        record = trainer.run_update()
        if (update + 1) % 25 == 0:
            print(json.dumps({"synthetic_update": update+1, "reward": record["mean_reward"],
                              "ev": record["explained_variance_post"]}), flush=True)
    after, frame = evaluate([model], market, scaler, config, start, end)
    expected = direction[30000:35000:5]
    directional_accuracy = float(np.mean(np.sign(frame.contracts) == expected))
    passed = after["total_return_btc"] > 0.1 and directional_accuracy > 0.8 and after["liquidations"] == 0
    result = {"check": "learn observable synthetic persistent trends", "passed": passed,
              "config": config.as_dict(), "before": before, "after": after,
              "directional_accuracy": directional_accuracy, "seconds": time.monotonic()-began,
              "financial_test_accessed": False,
              "interpretation": "Checks policy optimization on deliberately easy artificial dynamics; provides no evidence of a real-market edge"}
    write_json(OUTPUT / "verification/synthetic_learning.json", result)
    if not passed:
        raise AssertionError("Synthetic learning diagnostic failed; inspect report before launching")
    print(json.dumps({"synthetic_passed": passed, "directional_accuracy": directional_accuracy,
                      "return_btc": after["total_return_btc"]}), flush=True)


def development_smoke():
    config = Config()
    torch.set_num_threads(config.threads)
    torch.manual_seed(11)
    market = prepare()
    end = pd.Timestamp("2023-01-01", tz="UTC")
    scaler = Scaler.fit(market, end)
    env = Environment(market, market.bounds(TRAIN_START, end), config, np.random.default_rng(11))
    trainer = Trainer(ActorCritic(config), env, scaler, config, 11)
    began = time.monotonic()
    updates = [trainer.run_update() for _ in range(2)]
    seconds_per_update = (time.monotonic()-began) / len(updates)
    # Brief integration exercise entirely inside the training period.
    began = time.monotonic()
    result, _ = evaluate([trainer.model], market, scaler, config,
                         "2022-01-01T00:00:00Z", "2022-01-03T00:00:00Z")
    write_json(OUTPUT / "verification/development_smoke.json", {
        "training_period_only": True, "test_performance_accessed": False,
        "seconds_per_update": seconds_per_update, "updates": updates,
        "two_day_evaluation_seconds": time.monotonic()-began, "integration_evaluation": result,
        "cache": market.audit})
    print(json.dumps({"development_smoke_passed": True, "seconds_per_update": seconds_per_update}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-smoke", action="store_true")
    args = parser.parse_args()
    development_smoke() if args.development_smoke else synthetic()
