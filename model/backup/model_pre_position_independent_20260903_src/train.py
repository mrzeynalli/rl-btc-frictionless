"""One training run: a split, a cost arm, a configuration and a seed.

Walk-forward runs evaluate their own validation window on a fixed checkpoint
grid and record the whole curve, because the training budget is a thing the
protocol chooses on validation.  The final run does not evaluate anything: it
trains for exactly the budget selected on the folds, saves its weights, and
stops.  The sealed test year is opened later, once, by ``evaluate.py``.

Every run writes ``training.json`` atomically after every checkpoint and one
JSON line per update to ``updates.jsonl``, so a run that dies is visible and a
finished run can be re-read without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from env import ONE_WAY_COST, EnvironmentConfig, VectorHourlyEnvironment
from evaluation import account_positions, metrics, policy_diagnostics, run_policy
from features import MarketData, Split, SplitData, build_market, build_split, split_by_name
from model import GruActorCritic
from ppo import PPOConfig, PPOTrainer

ARMS = {"zero_cost": 0.0, "economic": ONE_WAY_COST}
DEFAULT_EVALUATION_INTERVAL = 25


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=float))
    os.replace(temporary, path)


@dataclass(frozen=True)
class RunSpec:
    split_name: str
    arm: str
    config_id: str
    seed: int
    updates: int
    ppo: PPOConfig
    episode_hours: int = 512
    evaluation_interval: int = DEFAULT_EVALUATION_INTERVAL
    evaluate_validation: bool = True
    position_ladder: str = "ternary"
    funding_features: bool = True

    @property
    def directory(self) -> Path:
        return Path(self.split_name) / self.arm / self.config_id / f"seed_{self.seed}"

    def as_dict(self) -> dict:
        return {
            "split": self.split_name,
            "arm": self.arm,
            "cost_rate": ARMS[self.arm],
            "config_id": self.config_id,
            "seed": self.seed,
            "updates": self.updates,
            "episode_hours": self.episode_hours,
            "evaluation_interval": self.evaluation_interval,
            "position_ladder": self.position_ladder,
            "funding_features": self.funding_features,
            "ppo": self.ppo.as_dict(),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**31 - 1))
    torch.manual_seed(seed)


def build_run(spec: RunSpec, split: SplitData) -> tuple[GruActorCritic, PPOTrainer]:
    market = split.market
    environment = VectorHourlyEnvironment(
        fast=split.fast,
        slow=split.slow,
        entry_price=market.entry_price,
        exit_price=market.exit_price,
        funding_on_previous=market.funding_on_previous,
        funding_on_current=market.funding_on_current,
        index_range=split.train_range,
        environments=spec.ppo.environments,
        config=EnvironmentConfig(
            episode_hours=spec.episode_hours,
            cost_rate=ARMS[spec.arm],
            position_ladder=spec.position_ladder,
            burn_in_hours=spec.ppo.burn_in_hours,
        ),
        rng=np.random.default_rng(spec.seed),
    )
    model = GruActorCritic(
        slow_features=environment.slow_width,
        action_count=environment.action_count,
        critic_encoder=spec.ppo.critic_encoder,
    )
    trainer = PPOTrainer(model, environment, spec.ppo, torch.Generator().manual_seed(spec.seed))
    return model, trainer


def evaluate_validation(
    model: GruActorCritic, split: SplitData, arm: str, ladder: str = "ternary"
) -> dict:
    """Deterministic argmax over the validation window at both cost levels."""
    first, last = split.evaluation_range
    own = run_policy(model, split, first, last, ARMS[arm], ladder=ladder)
    other_arm = "economic" if arm == "zero_cost" else "zero_cost"
    other = account_positions(own.positions, split.market, first, last, ARMS[other_arm])
    result = {arm: metrics(own, arm), other_arm: metrics(other, other_arm)}
    result["maximum_action_probability"] = float(own.action_probabilities.max(axis=1).mean())
    result["action_probability_std"] = float(own.action_probabilities.std())
    result["policy_diagnostics"] = policy_diagnostics(own)
    model.train()
    return result


def train_run(
    spec: RunSpec,
    output_dir: Path,
    market: MarketData | None = None,
    split_definition: Split | None = None,
) -> dict:
    torch.set_num_threads(1)
    seed_everything(spec.seed)
    market = market if market is not None else build_market(funding_features=spec.funding_features)
    definition = split_definition if split_definition is not None else split_by_name(spec.split_name)
    if definition.name != spec.split_name:
        raise ValueError(
            f"split override name {definition.name!r} does not match run specification "
            f"{spec.split_name!r}"
        )
    split = build_split(market, definition)

    run_directory = output_dir / spec.directory
    run_directory.mkdir(parents=True, exist_ok=True)
    updates_log = (run_directory / "updates.jsonl").open("w", buffering=1)

    model, trainer = build_run(spec, split)
    record = {
        "status": "running",
        "started_at_utc": now(),
        "specification": spec.as_dict(),
        "split": split.as_dict(),
        "model": {
            "parameters": model.parameter_count(),
            "hidden_size": model.hidden_size,
            "slow_features": model.slow_features,
            "critic_encoder": model.critic_encoder,
            "initialisation": model.initialisation_manifest,
        },
        "validation_curve": [],
        "test_performance_accessed": False,
    }
    write_json_atomic(run_directory / "training.json", record)

    started = time.time()
    for update in range(1, spec.updates + 1):
        summary = trainer.run_update()
        summary["elapsed_seconds"] = round(time.time() - started, 2)
        updates_log.write(json.dumps(summary, default=float) + "\n")

        checkpoint_due = update % spec.evaluation_interval == 0 or update == spec.updates
        if checkpoint_due and spec.evaluate_validation:
            # The budget is chosen on validation later, so every budget on the
            # grid has to be re-loadable exactly as it was.
            checkpoints = run_directory / "checkpoints"
            checkpoints.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoints / f"update_{update:04d}.pt")
            evaluation = evaluate_validation(model, split, spec.arm, spec.position_ladder)
            evaluation["update"] = update
            diagnostic_keys = (
                "entropy", "entropy_coefficient", "clip_fraction", "approximate_kl",
                "explained_variance", "recurrent_gradient_cosine", "mean_reward",
                "turnover_per_step", "sampled_action_share",
            )
            evaluation["training"] = {
                key: summary[key] for key in diagnostic_keys if key in summary
            }
            record["validation_curve"].append(evaluation)
            record["elapsed_seconds"] = round(time.time() - started, 2)
            record["updates_completed"] = update
            write_json_atomic(run_directory / "training.json", record)
        elif checkpoint_due:
            record["elapsed_seconds"] = round(time.time() - started, 2)
            record["updates_completed"] = update
            record["last_training_summary"] = summary
            write_json_atomic(run_directory / "training.json", record)

    torch.save(model.state_dict(), run_directory / "model_final.pt")
    updates_log.close()

    record["status"] = "completed"
    record["finished_at_utc"] = now()
    record["elapsed_seconds"] = round(time.time() - started, 2)
    record["updates_completed"] = spec.updates
    record["seconds_per_update"] = round((time.time() - started) / max(spec.updates, 1), 3)
    write_json_atomic(run_directory / "training.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-interval", type=int, default=DEFAULT_EVALUATION_INTERVAL)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--ppo-json", type=str, default="{}")
    parser.add_argument("--position-ladder", default="ternary", choices=["ternary", "ladder"])
    parser.add_argument("--no-funding-features", action="store_true")
    args = parser.parse_args()

    overrides = json.loads(args.ppo_json)
    episode_hours = int(overrides.pop("episode_hours", 512))
    spec = RunSpec(
        split_name=args.split,
        arm=args.arm,
        config_id=args.config_id,
        seed=args.seed,
        updates=args.updates,
        ppo=PPOConfig(**overrides),
        episode_hours=episode_hours,
        evaluation_interval=args.evaluation_interval,
        evaluate_validation=not args.no_validation,
        position_ladder=args.position_ladder,
        funding_features=not args.no_funding_features,
    )
    record = train_run(spec, args.output_dir)
    print(json.dumps({"status": record["status"], "elapsed": record["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
