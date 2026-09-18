"""Walk-forward ablations for the five recurrent-PPO design concerns.

The already-opened 2025-2026 test year is never read.  Every choice uses the
same four validation folds, five seeds, economic costs and fixed 225-update
budget as the original selection.  Stages are resumable through ``Runner``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from env import position_ladder
from evaluation import Trajectory, account_positions, metrics, policy_diagnostics, run_policy
from experiment import Runner, jobs_for
from features import WALK_FORWARD_SPLITS, build_market, build_split
from model import GruActorCritic
from ppo import PPOConfig
from train import ARMS, write_json_atomic


SEEDS = (11, 29, 47, 71, 101)
UPDATES = 225
INTERVAL = 25
BURN_IN_HOURS = 240
SWITCH_MARGINS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
STOCHASTIC_DRAWS = 8


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cell(
    gamma: float,
    burn_in_hours: int,
    entropy: float = 0.10,
    entropy_final: float | None = None,
    entropy_updates: int | None = None,
    critic_encoder: str = "shared",
) -> dict:
    return {
        "ppo": PPOConfig(
            gamma=gamma,
            gae_lambda=0.5,
            entropy_coefficient=entropy,
            entropy_final_coefficient=entropy_final,
            entropy_anneal_updates=entropy_updates,
            value_clip_range=None,
            normalise_returns=True,
            adaptive_learning_rate=False,
            critic_learning_rate_multiplier=1.0,
            burn_in_hours=burn_in_hours,
            critic_encoder=critic_encoder,
            measure_gradient_conflict=critic_encoder == "shared",
        ),
        "position_ladder": "ternary",
        "funding_features": True,
    }


def final_point(path: Path, updates: int) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    for point in record.get("validation_curve", []):
        if int(point["update"]) == updates:
            return point
    return None


def numeric_median(values: list[float | None]) -> float | None:
    clean = np.array([value for value in values if value is not None], dtype=float)
    clean = clean[np.isfinite(clean)]
    return float(np.median(clean)) if len(clean) else None


def summarise_config(
    output_dir: Path,
    config_id: str,
    folds,
    seeds,
    updates: int,
) -> dict:
    rows = []
    for split in folds:
        for seed in seeds:
            path = (
                output_dir / split.name / "economic" / config_id
                / f"seed_{seed}" / "training.json"
            )
            point = final_point(path, updates)
            if point is None:
                continue
            economic = point["economic"]
            training = point["training"]
            diagnostic = point.get("policy_diagnostics", {})
            rows.append({
                "fold": split.name,
                "seed": seed,
                "total_log_return": float(economic["total_log_return"]),
                "turnover_per_hour": float(economic["turnover_per_hour"]),
                "average_holding_hours": float(economic["average_holding_hours"]),
                "max_drawdown": float(economic["max_drawdown"]),
                "maximum_action_probability": float(point["maximum_action_probability"]),
                "policy_entropy": float(
                    diagnostic.get("action_probability_entropy_mean", np.nan)
                ),
                "top_two_margin": float(diagnostic.get("top_two_margin_median", np.nan)),
                "explained_variance": float(training["explained_variance"]),
                "gradient_cosine": training.get("recurrent_gradient_cosine"),
            })
    if len(rows) != len(folds) * len(seeds):
        raise RuntimeError(
            f"{config_id} has {len(rows)} final validation points; "
            f"expected {len(folds) * len(seeds)}"
        )

    returns = np.array([row["total_log_return"] for row in rows])
    return {
        "config_id": config_id,
        "runs": len(rows),
        "median_log_return": float(np.median(returns)),
        "mean_log_return": float(np.mean(returns)),
        "log_return_iqr": [
            float(np.percentile(returns, 25)), float(np.percentile(returns, 75))
        ],
        "share_positive": float(np.mean(returns > 0.0)),
        "median_turnover_per_hour": numeric_median(
            [row["turnover_per_hour"] for row in rows]
        ),
        "median_holding_hours": numeric_median(
            [row["average_holding_hours"] for row in rows]
        ),
        "median_max_drawdown": numeric_median([row["max_drawdown"] for row in rows]),
        "median_maximum_action_probability": numeric_median(
            [row["maximum_action_probability"] for row in rows]
        ),
        "median_policy_entropy": numeric_median([row["policy_entropy"] for row in rows]),
        "median_top_two_margin": numeric_median([row["top_two_margin"] for row in rows]),
        "median_explained_variance": numeric_median(
            [row["explained_variance"] for row in rows]
        ),
        "median_recurrent_gradient_cosine": numeric_median(
            [row["gradient_cosine"] for row in rows]
        ),
        "per_run": rows,
    }


def rank_configs(output_dir, ids, folds, seeds, updates) -> list[dict]:
    rows = [summarise_config(output_dir, config_id, folds, seeds, updates) for config_id in ids]
    return sorted(rows, key=lambda row: row["median_log_return"], reverse=True)


def trajectory_summary(trajectories: list[Trajectory]) -> dict:
    records = [metrics(trajectory) for trajectory in trajectories]
    keys = (
        "total_log_return", "total_return", "turnover_per_hour",
        "average_holding_hours", "max_drawdown", "sharpe",
    )
    return {
        f"median_{key}": float(np.median([record[key] for record in records]))
        for key in keys
    } | {
        "paths": len(records),
        "share_positive": float(np.mean([
            record["total_log_return"] > 0.0 for record in records
        ])),
    }


@torch.no_grad()
def run_ensemble(
    models: list[GruActorCritic],
    split,
    first: int,
    last: int,
    cost_rate: float,
    switch_margin: float,
) -> Trajectory:
    warmup_start = split.evaluation_warmup_start
    stream_first = min(warmup_start, first)
    fast = torch.from_numpy(split.fast[stream_first:last]).unsqueeze(0)
    hourly = [
        model.trunk(fast, model.initial_hidden(1))[0][0] for model in models
    ]
    slow = torch.from_numpy(split.slow[stream_first:last])
    levels = position_ladder("ternary")
    action_count = len(levels)
    flat = int(np.searchsorted(levels, 0.0))
    position_index = flat
    offset = first - stream_first
    length = last - first
    positions = np.zeros(length, dtype=np.float64)
    probabilities = np.zeros((length, action_count), dtype=np.float64)
    top_two = np.zeros(length, dtype=np.float64)
    switch = np.zeros(length, dtype=np.float64)
    one_hot = torch.zeros(action_count)
    for step in range(last - stream_first):
        if step == offset:
            position_index = flat
        one_hot.zero_()
        one_hot[position_index] = 1.0
        model_probabilities = []
        for model, representation in zip(models, hourly):
            joined = torch.cat([representation[step], slow[step], one_hot]).unsqueeze(0)
            logits = model.actor_output(model.activation(model.actor_hidden(joined)))[0]
            model_probabilities.append(torch.softmax(logits, dim=-1))
        mean_probability = torch.stack(model_probabilities).mean(dim=0)
        best = int(torch.argmax(mean_probability))
        improvement = float(mean_probability[best] - mean_probability[position_index])
        action = (
            position_index
            if best != position_index and improvement <= switch_margin
            else best
        )
        if step >= offset:
            target = step - offset
            positions[target] = levels[action]
            probabilities[target] = mean_probability.numpy()
            ordered = torch.topk(mean_probability, 2).values
            top_two[target] = float(ordered[0] - ordered[1])
            switch[target] = improvement
        position_index = action
    accounted = account_positions(positions, split.market, first, last, cost_rate)
    return Trajectory(
        positions=positions,
        rewards=accounted.rewards,
        price_return=accounted.price_return,
        funding_return=accounted.funding_return,
        cost_paid=accounted.cost_paid,
        hypothetical_cost=accounted.hypothetical_cost,
        equity=accounted.equity,
        action_probabilities=probabilities,
        action_top_two_margins=top_two,
        action_switch_margins=switch,
    )


def load_model(cell_spec: dict, split, weights: Path) -> GruActorCritic:
    actions = len(position_ladder(cell_spec["position_ladder"]))
    model = GruActorCritic(
        slow_features=split.observation_size(actions),
        action_count=actions,
        critic_encoder=cell_spec["ppo"].critic_encoder,
    )
    model.load_state_dict(torch.load(weights, weights_only=True))
    model.eval()
    return model


def posthoc_diagnostics(
    output_dir: Path,
    config_id: str,
    cell_spec: dict,
    folds,
    seeds,
    updates: int,
    stochastic_draws: int,
) -> dict:
    market = build_market(funding_features=True)
    by_margin: dict[float, list[Trajectory]] = {margin: [] for margin in SWITCH_MARGINS}
    stochastic: list[Trajectory] = []
    margin_values = []
    warmup: dict[int, list[dict]] = {hours: [] for hours in (0, 24, 72, 240, 512)}
    ensemble: dict[float, list[Trajectory]] = {margin: [] for margin in SWITCH_MARGINS}

    for fold_index, split_definition in enumerate(folds):
        split = build_split(market, split_definition)
        first, last = split.evaluation_range
        models = []
        for seed in seeds:
            weights = (
                output_dir / split_definition.name / "economic" / config_id
                / f"seed_{seed}" / "checkpoints" / f"update_{updates:04d}.pt"
            )
            model = load_model(cell_spec, split, weights)
            models.append(model)
            deterministic = None
            for threshold in SWITCH_MARGINS:
                trajectory = run_policy(
                    model, split, first, last, ARMS["economic"],
                    ladder="ternary", switch_margin=threshold,
                )
                by_margin[threshold].append(trajectory)
                if threshold == 0.0:
                    deterministic = trajectory
                    margin_values.append(trajectory.action_top_two_margins)
            for draw in range(stochastic_draws):
                generator = torch.Generator().manual_seed(
                    1_000_000 + 10_000 * fold_index + 100 * seed + draw
                )
                stochastic.append(run_policy(
                    model, split, first, last, ARMS["economic"],
                    deterministic=False, generator=generator, ladder="ternary",
                ))

            references = {}
            for hours in (0, 24, 72, 240, 512):
                references[hours] = run_policy(
                    model, split, first, last, ARMS["economic"],
                    warmup_start=first - hours, ladder="ternary",
                )
            reference = references[512]
            for hours, trajectory in references.items():
                warmup[hours].append({
                    "action_agreement_with_512h": float(
                        np.mean(trajectory.positions == reference.positions)
                    ),
                    "first_24h_agreement_with_512h": float(
                        np.mean(trajectory.positions[:24] == reference.positions[:24])
                    ),
                    "total_log_return": float(trajectory.rewards.sum()),
                })

        for threshold in SWITCH_MARGINS:
            ensemble[threshold].append(run_ensemble(
                models, split, first, last, ARMS["economic"], threshold
            ))

    margins = np.concatenate(margin_values)
    hysteresis = {
        f"{threshold:.2f}": trajectory_summary(paths)
        for threshold, paths in by_margin.items()
    }
    ensemble_summary = {
        f"{threshold:.2f}": trajectory_summary(paths)
        for threshold, paths in ensemble.items()
    }
    best_hysteresis = max(
        hysteresis, key=lambda key: hysteresis[key]["median_total_log_return"]
    )
    best_ensemble = max(
        ensemble_summary,
        key=lambda key: ensemble_summary[key]["median_total_log_return"],
    )
    return {
        "deterministic_argmax": hysteresis["0.00"],
        "stochastic_sampling": trajectory_summary(stochastic),
        "action_top_two_margin": {
            "observations": int(len(margins)),
            "mean": float(np.mean(margins)),
            "median": float(np.median(margins)),
            "p10": float(np.percentile(margins, 10)),
            "share_below_0.01": float(np.mean(margins < 0.01)),
            "share_below_0.05": float(np.mean(margins < 0.05)),
            "share_below_0.10": float(np.mean(margins < 0.10)),
        },
        "hysteresis": hysteresis,
        "best_hysteresis_margin": float(best_hysteresis),
        "ensemble": ensemble_summary,
        "best_ensemble_margin": float(best_ensemble),
        "warmup_sensitivity": {
            str(hours): {
                "runs": len(rows),
                "median_action_agreement_with_512h": numeric_median([
                    row["action_agreement_with_512h"] for row in rows
                ]),
                "median_first_24h_agreement_with_512h": numeric_median([
                    row["first_24h_agreement_with_512h"] for row in rows
                ]),
                "median_total_log_return": numeric_median([
                    row["total_log_return"] for row in rows
                ]),
            }
            for hours, rows in warmup.items()
        },
    }


def render_table(rows: list[dict], label: str) -> list[str]:
    lines = [
        f"### {label}", "",
        "| rank | configuration | median log return | positive | turnover/h | "
        "holding h | max probability | policy entropy | top-2 margin | critic EV | grad cosine |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        def f(key: str, digits: int = 4) -> str:
            value = row.get(key)
            return "n/a" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {rank} | `{row['config_id']}` | {f('median_log_return')} | "
            f"{row['share_positive']:.0%} | {f('median_turnover_per_hour')} | "
            f"{f('median_holding_hours', 1)} | {f('median_maximum_action_probability')} | "
            f"{f('median_policy_entropy')} | {f('median_top_two_margin')} | "
            f"{f('median_explained_variance')} | {f('median_recurrent_gradient_cosine')} |"
        )
    lines.append("")
    return lines


def write_report(output_dir: Path, results: dict) -> None:
    lines = [
        "# Recurrent PPO concern study", "",
        "All choices use economic costs and walk-forward validation only. The existing "
        "2025-2026 test year and the untouched July 2026 holdout were not read.", "",
        f"Fixed budget: {results['protocol']['updates']} PPO updates; "
        f"{len(results['protocol']['folds'])} folds × {len(results['protocol']['seeds'])} seeds.", "",
    ]
    lines += render_table(results["horizon_and_burnin"]["ranking"], "Burn-in and discount")
    lines += render_table(results["entropy"]["ranking"], "Entropy schedule")
    lines += render_table(results["encoder"]["ranking"], "Actor/critic encoder")
    posthoc = results["posthoc"]
    lines += [
        "### Deployment-policy diagnostics", "",
        f"Selected training configuration: `{results['selected_config_id']}`.", "",
        f"Median top-two action-probability margin: "
        f"{posthoc['action_top_two_margin']['median']:.4f}; "
        f"{posthoc['action_top_two_margin']['share_below_0.05']:.1%} of decisions are below 0.05.", "",
        "| policy | median log return | turnover/h | holding h | positive |", 
        "|---|---:|---:|---:|---:|",
    ]
    deterministic = posthoc["deterministic_argmax"]
    stochastic = posthoc["stochastic_sampling"]
    lines.append(
        f"| deterministic argmax | {deterministic['median_total_log_return']:+.4f} | "
        f"{deterministic['median_turnover_per_hour']:.4f} | "
        f"{deterministic['median_average_holding_hours']:.1f} | "
        f"{deterministic['share_positive']:.0%} |"
    )
    lines.append(
        f"| stochastic sampling | {stochastic['median_total_log_return']:+.4f} | "
        f"{stochastic['median_turnover_per_hour']:.4f} | "
        f"{stochastic['median_average_holding_hours']:.1f} | "
        f"{stochastic['share_positive']:.0%} |"
    )
    for name, group, best_key in (
        ("best hysteresis", posthoc["hysteresis"], "best_hysteresis_margin"),
        ("best five-seed ensemble", posthoc["ensemble"], "best_ensemble_margin"),
    ):
        threshold = posthoc[best_key]
        row = group[f"{threshold:.2f}"]
        lines.append(
            f"| {name} (margin {threshold:.2f}) | {row['median_total_log_return']:+.4f} | "
            f"{row['median_turnover_per_hour']:.4f} | "
            f"{row['median_average_holding_hours']:.1f} | {row['share_positive']:.0%} |"
        )
    lines += ["", "### Warm-up sensitivity", "", "| hours | agreement with 512 h | first-24 h agreement | median log return |", "|---:|---:|---:|---:|"]
    for hours, row in posthoc["warmup_sensitivity"].items():
        lines.append(
            f"| {hours} | {row['median_action_agreement_with_512h']:.1%} | "
            f"{row['median_first_24h_agreement_with_512h']:.1%} | "
            f"{row['median_total_log_return']:+.4f} |"
        )
    lines += ["", f"Completed at {results['finished_at_utc']}.", ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/concern_study"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "study.json"
    folds = list(WALK_FORWARD_SPLITS)
    seeds = list(SEEDS)
    updates = UPDATES
    interval = INTERVAL
    burn_in = BURN_IN_HOURS
    stochastic_draws = STOCHASTIC_DRAWS
    if args.smoke_test:
        folds = folds[:1]
        seeds = seeds[:1]
        updates, interval, burn_in, stochastic_draws = 4, 2, 24, 2

    results = {
        "status": "running",
        "started_at_utc": now(),
        "protocol": {
            "selection_data": "four walk-forward validation folds only",
            "test_year_accessed": False,
            "untouched_2026_07_accessed": False,
            "arm": "economic",
            "cost_rate": ARMS["economic"],
            "folds": [fold.name for fold in folds],
            "seeds": seeds,
            "updates": updates,
            "evaluation_interval": interval,
            "gae_lambda": 0.5,
        },
        "stages": {},
    }
    write_json_atomic(manifest_path, results)
    runner = Runner(output_dir, args.workers, updates)

    stage1 = {
        "control_g050_b000": cell(0.50, 0),
        "burn_g050_b240": cell(0.50, burn_in),
        "burn_g090_b240": cell(0.90, burn_in),
        "burn_g095_b240": cell(0.95, burn_in),
        "burn_g099_b240": cell(0.99, burn_in),
    }
    if args.smoke_test:
        stage1 = dict(list(stage1.items())[:2])
    runner.run_all(
        jobs_for(folds, ["economic"], list(stage1), seeds, stage1, updates, interval=interval),
        "horizon_and_burnin", results, manifest_path,
    )
    ranking1 = rank_configs(output_dir, list(stage1), folds, seeds, updates)
    eligible_gamma = [row for row in ranking1 if "b240" in row["config_id"]]
    best_gamma_row = eligible_gamma[0] if eligible_gamma else ranking1[0]
    best_gamma = stage1[best_gamma_row["config_id"]]["ppo"].gamma
    results["horizon_and_burnin"] = {
        "ranking": ranking1,
        "selected_gamma": best_gamma,
        "selected_config_id": best_gamma_row["config_id"],
    }
    write_json_atomic(manifest_path, results)

    entropy_candidates = {
        best_gamma_row["config_id"]: stage1[best_gamma_row["config_id"]],
        f"entropy_g{best_gamma:.2f}_const003": cell(best_gamma, burn_in, 0.03),
        f"entropy_g{best_gamma:.2f}_const001": cell(best_gamma, burn_in, 0.01),
        f"entropy_g{best_gamma:.2f}_010to001": cell(
            best_gamma, burn_in, 0.10, 0.01, updates
        ),
        f"entropy_g{best_gamma:.2f}_010to000": cell(
            best_gamma, burn_in, 0.10, 0.00, updates
        ),
        f"entropy_g{best_gamma:.2f}_003to000": cell(
            best_gamma, burn_in, 0.03, 0.00, updates
        ),
    }
    if args.smoke_test:
        entropy_candidates = dict(list(entropy_candidates.items())[:2])
    new_entropy = {
        key: value for key, value in entropy_candidates.items() if key not in stage1
    }
    if new_entropy:
        runner.run_all(
            jobs_for(
                folds, ["economic"], list(new_entropy), seeds, new_entropy,
                updates, interval=interval,
            ),
            "entropy", results, manifest_path,
        )
    ranking2 = rank_configs(output_dir, list(entropy_candidates), folds, seeds, updates)
    best_entropy_row = ranking2[0]
    best_entropy_id = best_entropy_row["config_id"]
    best_entropy_cell = entropy_candidates[best_entropy_id]
    results["entropy"] = {
        "ranking": ranking2,
        "selected_config_id": best_entropy_id,
    }
    write_json_atomic(manifest_path, results)

    base = best_entropy_cell["ppo"]
    encoder_candidates = {best_entropy_id: best_entropy_cell}
    for encoder in ("detached", "separate"):
        identifier = f"encoder_{encoder}_{best_entropy_id}"
        encoder_candidates[identifier] = cell(
            base.gamma,
            base.burn_in_hours,
            base.entropy_coefficient,
            base.entropy_final_coefficient,
            base.entropy_anneal_updates,
            encoder,
        )
    if args.smoke_test:
        encoder_candidates = dict(list(encoder_candidates.items())[:2])
    new_encoders = {
        key: value for key, value in encoder_candidates.items()
        if key != best_entropy_id
    }
    if new_encoders:
        runner.run_all(
            jobs_for(
                folds, ["economic"], list(new_encoders), seeds, new_encoders,
                updates, interval=interval,
            ),
            "encoder", results, manifest_path,
        )
    ranking3 = rank_configs(output_dir, list(encoder_candidates), folds, seeds, updates)
    selected = ranking3[0]["config_id"]
    selected_cell = encoder_candidates[selected]
    results["encoder"] = {"ranking": ranking3, "selected_config_id": selected}
    results["selected_config_id"] = selected
    results["selected_config"] = {
        **selected_cell["ppo"].as_dict(),
        "position_ladder": selected_cell["position_ladder"],
        "funding_features": selected_cell["funding_features"],
    }
    write_json_atomic(manifest_path, results)

    results["posthoc"] = posthoc_diagnostics(
        output_dir, selected, selected_cell, folds, seeds, updates, stochastic_draws
    )
    results["status"] = "completed"
    results["finished_at_utc"] = now()
    write_json_atomic(manifest_path, results)
    write_json_atomic(output_dir / "results.json", results)
    write_report(output_dir, results)
    print(f"[{now()}] concern study complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
