"""Train one Spot PPO configuration on one expanding walk-forward fold."""

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

from env import ACTION_NAMES, EnvironmentConfig, ONE_WAY_COST, VectorHourlyEnvironment
from evaluation import metrics, policy_diagnostics, run_policy, starting_position_diagnostics
from features import MarketData, Split, SplitData, build_market, build_split, split_by_name
from model import MarketActorCritic
from ppo import PPOConfig, PPOTrainer


DEFAULT_EVALUATION_INTERVAL = 25


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=float))
    os.replace(temporary, path)


@dataclass(frozen=True)
class RunSpec:
    split_name: str
    config_id: str
    seed: int
    updates: int
    ppo: PPOConfig
    output_group: str = "tuning"
    evaluation_interval: int = DEFAULT_EVALUATION_INTERVAL
    evaluate_validation: bool = True
    encoder_type: str = "gru"
    critic_mode: str = "shared"
    episode_hour_options: tuple[int, ...] = (720, 1440, 2160, 2880, 3600)

    @property
    def directory(self) -> Path:
        return (
            Path(self.output_group) / self.split_name / self.config_id
            / f"seed_{self.seed}"
        )

    def as_dict(self) -> dict:
        return {
            "split": self.split_name,
            "config_id": self.config_id,
            "seed": self.seed,
            "updates": self.updates,
            "evaluation_interval": self.evaluation_interval,
            "encoder_type": self.encoder_type,
            "critic_mode": self.critic_mode,
            "reward": "scaled net log return including proportional position-change cost",
            "episode_hour_options": list(self.episode_hour_options),
            "market": "BTCUSDT Spot",
            "actions": list(ACTION_NAMES),
            "funding": False,
            "ppo": self.ppo.as_dict(),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**31 - 1))
    torch.manual_seed(seed)


def build_run(spec: RunSpec, split: SplitData) -> tuple[MarketActorCritic, PPOTrainer]:
    market = split.market
    environment = VectorHourlyEnvironment(
        fast=split.fast,
        slow=split.slow,
        entry_price=market.entry_price,
        exit_price=market.exit_price,
        segment_start=market.segment_start,
        index_range=split.train_range,
        environments=spec.ppo.environments,
        config=EnvironmentConfig(
            episode_hour_options=spec.episode_hour_options,
            cost_rate=ONE_WAY_COST,
            burn_in_hours=spec.ppo.burn_in_hours,
            randomize_starting_position=True,
            regime_balanced_sampling=True,
        ),
        rng=np.random.default_rng(spec.seed),
    )
    model = MarketActorCritic(
        slow_features=environment.slow_width,
        encoder_type=spec.encoder_type,
        critic_mode=spec.critic_mode,
    )
    trainer = PPOTrainer(
        model, environment, spec.ppo, torch.Generator().manual_seed(spec.seed)
    )
    return model, trainer


def evaluate_validation(model: MarketActorCritic, split: SplitData, spec: RunSpec) -> dict:
    first, last = split.evaluation_range

    def evaluate_start(initial: float):
        return run_policy(
            model, split, first, last, ONE_WAY_COST,
            execution_kappa=spec.ppo.execution_kappa,
            initial_position=initial,
        )

    cash = evaluate_start(0.0)
    result = {
        "economic": metrics(cash, "economic"),
        "policy_diagnostics": policy_diagnostics(cash),
        "starting_position_diagnostics": starting_position_diagnostics(evaluate_start),
    }
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
    market = market if market is not None else build_market()
    split = build_split(market, split_definition or split_by_name(spec.split_name))
    run_directory = output_dir / spec.directory
    run_directory.mkdir(parents=True, exist_ok=True)
    updates_log = (run_directory / "updates.jsonl").open("w", buffering=1)
    model, trainer = build_run(spec, split)
    record = {
        "status": "running",
        "started_at_utc": now(),
        "specification": spec.as_dict(),
        "split": split.as_dict(),
        "market_audit": market.audit,
        "model": {
            "class": "MarketActorCritic",
            "parameters": model.parameter_count(),
            "encoder_type": spec.encoder_type,
            "critic_mode": spec.critic_mode,
            "fast_input": list(split.fast.shape[1:]),
            "fast_hidden": 32,
            "slow_market_features": 10,
            "portfolio_critic_features": 7,
            "actor_position_independent": True,
            "actions": list(ACTION_NAMES),
            "funding": False,
        },
        "validation_curve": [],
        "locked_2026_accessed": spec.split_name == "locked_2026" and spec.evaluate_validation,
    }
    write_json_atomic(run_directory / "training.json", record)
    started = time.time()
    critic_curve: list[dict[str, float]] = []
    for update in range(1, spec.updates + 1):
        summary = trainer.run_update()
        summary["elapsed_seconds"] = round(time.time() - started, 2)
        critic_curve.append({
            "update": update,
            "explained_variance_pre_update": float(
                summary["explained_variance_pre_update"]
            ),
            "explained_variance_post_update": float(
                summary["explained_variance_post_update"]
            ),
            "prediction_std_post": float(summary["critic_prediction_std_post"]),
        })
        updates_log.write(json.dumps(summary, default=float) + "\n")
        checkpoint_due = update % spec.evaluation_interval == 0 or update == spec.updates
        if checkpoint_due:
            checkpoints = run_directory / "checkpoints"
            checkpoints.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoints / f"update_{update:04d}.pt")
            if spec.evaluate_validation:
                evaluation = evaluate_validation(model, split, spec)
                evaluation["update"] = update
                evaluation["training"] = {
                    key: summary.get(key)
                    for key in (
                        "explained_variance_pre_update",
                        "explained_variance_post_update",
                        "critic_prediction_std_post",
                        "entropy", "entropy_coefficient", "turnover_per_step",
                        "entropy_coefficient_next", "entropy_target", "approximate_kl",
                        "clip_fraction", "recurrent_gradient_cosine",
                        "replay_ratio_max_abs_error", "replay_value_max_abs_error",
                        "sampled_action_share", "episode_length_counts",
                        "regime_sample_counts",
                    )
                }
                record["validation_curve"].append(evaluation)
            record["updates_completed"] = update
            record["elapsed_seconds"] = round(time.time() - started, 2)
            record["last_training_summary"] = summary
            write_json_atomic(run_directory / "training.json", record)

    torch.save(model.state_dict(), run_directory / "model_final.pt")
    updates_log.close()
    elapsed = time.time() - started
    post_ev = np.asarray(
        [item["explained_variance_post_update"] for item in critic_curve],
        dtype=np.float64,
    )
    prediction_std = np.asarray(
        [item["prediction_std_post"] for item in critic_curve], dtype=np.float64
    )
    tail_size = max(5, len(post_ev) // 4)
    record.update({
        "status": "completed",
        "finished_at_utc": now(),
        "elapsed_seconds": round(elapsed, 2),
        "updates_completed": spec.updates,
        "seconds_per_update": round(elapsed / max(spec.updates, 1), 3),
        "critic_learning_diagnostics": {
            "definition": "explained variance of PPO value targets; 1 is perfect, 0 is no better than a constant",
            "post_update_final": float(post_ev[-1]),
            "post_update_max": float(post_ev.max()),
            "post_update_positive_share": float(np.mean(post_ev > 0.0)),
            "post_update_final_quarter_mean": float(post_ev[-tail_size:].mean()),
            "prediction_std_final": float(prediction_std[-1]),
            "prediction_nonconstant_share": float(np.mean(prediction_std > 1e-6)),
        },
    })
    write_json_atomic(run_directory / "training.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-group", default="tuning")
    parser.add_argument("--evaluation-interval", type=int, default=DEFAULT_EVALUATION_INTERVAL)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--ppo-json", default="{}")
    parser.add_argument("--encoder-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--critic-mode", choices=["shared", "detached", "separate"], default="shared")
    parser.add_argument("--episode-hours", default="720,1440,2160,2880,3600")
    args = parser.parse_args()
    spec = RunSpec(
        split_name=args.split,
        config_id=args.config_id,
        seed=args.seed,
        updates=args.updates,
        ppo=PPOConfig(**json.loads(args.ppo_json)),
        output_group=args.output_group,
        evaluation_interval=args.evaluation_interval,
        evaluate_validation=not args.no_validation,
        encoder_type=args.encoder_type,
        critic_mode=args.critic_mode,
        episode_hour_options=tuple(int(item) for item in args.episode_hours.split(",")),
    )
    train_run(spec, args.output_dir)


if __name__ == "__main__":
    main()
