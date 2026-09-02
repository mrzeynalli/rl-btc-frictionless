"""Supervised information coefficient against the break-even cost hurdle.

This runs before any RL result exists and is purely diagnostic: a ridge probe
is fitted on each fold's training window and scored on that fold's validation
window, using exactly the state the agent sees.  It answers the only question
that matters first -- is there enough predictable variation in this state to
pay 8 bp a round trip -- and it answers it out of sample.

Break-even.  For a jointly normal signal and return, a unit long-short position
taken on the sign of a prediction with information coefficient ``rho`` earns
``rho * sigma_h * sqrt(2/pi)`` per holding period.  Setting that equal to the
round-trip cost ``2c`` gives the hurdle ``rho* = 2c / (sigma_h * sqrt(2/pi))``.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from env import ONE_WAY_COST
from features import MarketData, WALK_FORWARD_SPLITS, build_market, build_split

HORIZONS = (1, 3, 6, 12, 24, 72)
RIDGE_PENALTY = 1.0


def _forward_log_return(market: MarketData, horizon: int) -> np.ndarray:
    """Log return of the price actually traded, over ``horizon`` decisions."""
    entry = market.entry_price
    forward = np.full(len(entry), np.nan)
    forward[: len(entry) - horizon] = np.log(entry[horizon:] / entry[: len(entry) - horizon])
    return forward


def _ridge(design: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    centred = design - design.mean(axis=0)
    scale = centred.std(axis=0)
    scale[scale <= 0] = 1.0
    centred /= scale
    gram = centred.T @ centred + penalty * np.eye(centred.shape[1])
    coefficients = np.linalg.solve(gram, centred.T @ (target - target.mean()))
    return np.concatenate([coefficients / scale, [target.mean() - (design.mean(axis=0) / scale * coefficients).sum()]])


def _predict(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return design @ weights[:-1] + weights[-1]


def run_signal_check(market: MarketData | None = None) -> dict:
    market = market if market is not None else build_market()
    results = {"horizons": {}, "round_trip_cost": 2 * ONE_WAY_COST, "folds": []}
    for split_definition in WALK_FORWARD_SPLITS:
        split = build_split(market, split_definition)
        design = np.concatenate(
            [split.fast.reshape(len(split.fast), -1), split.slow], axis=1
        ).astype(np.float64)
        train_first, train_last = split.train_range
        eval_first, eval_last = split.evaluation_range
        fold = {"fold": split_definition.name, "horizons": {}}
        for horizon in HORIZONS:
            target = _forward_log_return(market, horizon)
            train = slice(train_first, train_last - horizon)
            evaluate = slice(eval_first, eval_last - horizon)
            weights = _ridge(design[train], target[train], RIDGE_PENALTY)
            prediction = _predict(design[evaluate], weights)
            truth = target[evaluate]
            valid = np.isfinite(prediction) & np.isfinite(truth)
            information_coefficient = float(np.corrcoef(prediction[valid], truth[valid])[0, 1])
            sigma = float(np.std(truth[valid]))
            hurdle = (2 * ONE_WAY_COST) / (sigma * math.sqrt(2.0 / math.pi)) if sigma > 0 else float("inf")
            samples = int(valid.sum())
            fold["horizons"][str(horizon)] = {
                "information_coefficient": information_coefficient,
                "standard_error": 1.0 / math.sqrt(max(samples, 2)),
                "t_statistic": information_coefficient * math.sqrt(max(samples, 2)),
                "return_sigma": sigma,
                "break_even_information_coefficient": hurdle,
                "clears_hurdle": bool(information_coefficient > hurdle),
                "samples": samples,
            }
        results["folds"].append(fold)

    for horizon in HORIZONS:
        values = [fold["horizons"][str(horizon)]["information_coefficient"] for fold in results["folds"]]
        hurdles = [fold["horizons"][str(horizon)]["break_even_information_coefficient"] for fold in results["folds"]]
        results["horizons"][str(horizon)] = {
            "median_information_coefficient": float(np.median(values)),
            "per_fold": values,
            "median_break_even": float(np.median(hurdles)),
            "folds_clearing_hurdle": int(sum(v > h for v, h in zip(values, hurdles))),
            "folds": len(values),
        }
    results["any_horizon_clears_hurdle_on_median"] = bool(
        any(
            results["horizons"][str(horizon)]["median_information_coefficient"]
            > results["horizons"][str(horizon)]["median_break_even"]
            for horizon in HORIZONS
        )
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    report = run_signal_check()
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
