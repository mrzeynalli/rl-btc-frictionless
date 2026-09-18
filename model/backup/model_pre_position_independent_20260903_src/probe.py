"""A small ablation that asks whether the ``hourly_eq`` pathologies are gone.

``hourly_eq`` failed with three signatures that are all *optimisation*
signatures rather than signal ones: the critic's explained variance never left
zero, the PPO clip fraction was exactly 0.0000 at every update of all 320 runs,
and the entropy coefficient split the grid into permanent-flat or fair-die with
nothing in between.  This probe changes one thing at a time and measures those
three numbers, so each fix can be credited or dismissed on its own.

It is deliberately cheap -- one fold, one cost arm, a few seeds, a fraction of
the training budget.  It cannot say whether the agent makes money; it can only
say whether the optimiser has started working.  Nothing here is a study and
nothing here touches the sealed test year.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from ppo import PPOConfig
from train import RunSpec, train_run

# The ``hourly_eq`` winner, reproduced exactly: long discount, fixed rate,
# value clipping on, no return scaling, three-way action set.
BASELINE = PPOConfig(
    environments=32, rollout_steps=128, minibatches=2, epochs=4,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, value_clip_range=0.2,
    value_coefficient=0.5, entropy_coefficient=0.1, learning_rate=3e-4,
    max_gradient_norm=0.5, normalise_advantages=True,
    normalise_returns=False, adaptive_learning_rate=False,
    critic_learning_rate_multiplier=1.0,
)

# Discount sized to the two-step causal reach of an action.
GAMMA_FIX = dict(gamma=0.5, gae_lambda=0.5)
# Drive the policy to a measurable KL instead of letting it crawl.
RATE_FIX = dict(adaptive_learning_rate=True, target_kl=0.02)
# Unclip the value head, scale its target, and let it run faster than the actor.
CRITIC_FIX = dict(value_clip_range=None, normalise_returns=True,
                  critic_learning_rate_multiplier=3.0)


def arms() -> dict[str, dict]:
    """Each arm is the one before it plus exactly one change."""
    a1 = {**GAMMA_FIX}
    a2 = {**a1, **RATE_FIX}
    a3 = {**a2, **CRITIC_FIX}
    return {
        # control: what hourly_eq actually ran
        "0_baseline": dict(ppo={}, ladder="ternary", funding=False),
        "1_gamma": dict(ppo=a1, ladder="ternary", funding=False),
        "2_gamma_rate": dict(ppo=a2, ladder="ternary", funding=False),
        "3_gamma_rate_critic": dict(ppo=a3, ladder="ternary", funding=False),
        "4_funding": dict(ppo=a3, ladder="ternary", funding=True),
        "5_ladder": dict(ppo=a3, ladder="ladder", funding=True),
        # the bifurcation probe: the same two arms at the low entropy setting
        # that collapsed hourly_eq to permanent flat.
        "6_baseline_ent01": dict(ppo=dict(entropy_coefficient=0.01),
                                 ladder="ternary", funding=False),
        "7_ladder_ent01": dict(ppo={**a3, "entropy_coefficient": 0.01},
                               ladder="ladder", funding=True),
    }


def specifications(updates: int, seeds: tuple[int, ...]) -> list[RunSpec]:
    out: list[RunSpec] = []
    for name, arm in arms().items():
        config = replace(BASELINE, **arm["ppo"])
        for seed in seeds:
            out.append(RunSpec(
                split_name="fold_1", arm="economic", config_id=name, seed=seed,
                updates=updates, ppo=config, evaluation_interval=max(updates // 3, 1),
                position_ladder=arm["ladder"], funding_features=arm["funding"],
            ))
    return out


def _run(payload: tuple[RunSpec, str]) -> str:
    spec, output = payload
    started = time.time()
    train_run(spec, Path(output))
    return f"{spec.config_id} seed {spec.seed} done in {time.time() - started:.0f}s"


def summarise(output: Path, updates: int) -> list[dict]:
    """Median over seeds of the three diagnostics that defined the failure."""
    rows = []
    for name in arms():
        actions = 7 if arms()[name]["ladder"] == "ladder" else 3
        ceiling = float(np.log(actions))
        per_seed: dict[str, list[float]] = {
            k: [] for k in ("ev", "clip", "kl", "entropy", "turnover", "lr", "validation")
        }
        for path in sorted((output / "fold_1" / "economic" / name).glob("seed_*/updates.jsonl")):
            records = [json.loads(line) for line in path.open()]
            if len(records) < updates // 2:
                continue
            tail = records[-max(len(records) // 4, 1):]
            column = lambda key: np.array([r[key] for r in tail], dtype=float)
            per_seed["ev"].append(float(np.median(column("explained_variance"))))
            per_seed["clip"].append(float(np.median(column("clip_fraction"))))
            per_seed["kl"].append(float(np.median(column("approximate_kl"))))
            per_seed["entropy"].append(float(np.median(column("entropy"))) / ceiling)
            per_seed["turnover"].append(float(np.median(column("turnover_per_step"))))
            per_seed["lr"].append(float(np.median(column("learning_rate"))))
            training = json.loads((path.parent / "training.json").read_text())
            curve = training.get("validation_curve", [])
            if curve:
                economic = curve[-1]["economic"]
                per_seed["validation"].append(float(economic["total_log_return"]))
        if not per_seed["ev"]:
            continue
        rows.append({
            "arm": name, "seeds": len(per_seed["ev"]),
            **{k: float(np.median(v)) if v else float("nan") for k, v in per_seed.items()},
        })
    return rows


def render(rows: list[dict]) -> str:
    header = (f"{'arm':22s} {'n':>2s} {'EV':>9s} {'clipfrac':>9s} {'KL':>10s} "
              f"{'H/Hmax':>7s} {'turn':>6s} {'lr':>9s} {'val logret':>10s}")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['arm']:22s} {r['seeds']:2d} {r['ev']:+9.4f} {r['clip']:9.4f} "
            f"{r['kl']:10.2e} {r['entropy']:7.3f} {r['turnover']:6.3f} "
            f"{r['lr']:9.2e} {r['validation']:+10.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/probe")
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 47, 137])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--summarise-only", action="store_true")
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    if not arguments.summarise_only:
        specs = specifications(arguments.updates, tuple(arguments.seeds))
        print(f"{len(specs)} runs at {arguments.updates} updates on {arguments.workers} workers",
              flush=True)
        started = time.time()
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            for message in pool.map(_run, [(s, str(output)) for s in specs]):
                print("  " + message, flush=True)
        print(f"all runs finished in {(time.time() - started) / 60:.1f} min", flush=True)

    rows = summarise(output, arguments.updates)
    table = render(rows)
    print("\n" + table)
    (output / "probe_summary.json").write_text(json.dumps(rows, indent=2))
    (output / "probe_summary.txt").write_text(table + "\n")


if __name__ == "__main__":
    main()
