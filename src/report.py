"""Turn the recorded study into a report a reader can check line by line.

Writes ``FINAL_REPORT.md``, the per-seed CSV tables behind every number in it,
and self-contained SVG charts.  Nothing is recomputed from the models here --
every figure comes from the JSON the run itself wrote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PALETTE = {
    "economic": "#2f6f4f",
    "zero_cost": "#1f4e79",
    "always_long": "#a34b2a",
    "momentum": "#7a5c9e",
    "always_flat": "#6b6b6b",
    "axis": "#444444",
    "grid": "#dddddd",
}


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:+.{digits}f}" if abs(value) < 1000 else f"{value:.1f}"


def _percent(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value * 100:+.{digits}f}%"


def _median_row(aggregate: dict, key: str, formatter=_fmt) -> str:
    entry = aggregate.get(key)
    if not entry:
        return "n/a"
    low, high = entry["interquartile_range"]
    return f"{formatter(entry['median'])} [{formatter(low)}, {formatter(high)}]"


# -- charts ----------------------------------------------------------------


def _line_chart(
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    y_label: str,
    width: int = 900,
    height: int = 380,
) -> str:
    margin_left, margin_right, margin_top, margin_bottom = 70, 190, 40, 45
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    finite = np.concatenate([values[np.isfinite(values)] for _, values, _ in series if values.size])
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-9:
        low, high = low - 0.5, high + 0.5
    pad = (high - low) * 0.08
    low, high = low - pad, high + pad
    length = max(len(values) for _, values, _ in series)

    def x(index: int) -> float:
        return margin_left + plot_width * (index / max(length - 1, 1))

    def y(value: float) -> float:
        return margin_top + plot_height * (1.0 - (value - low) / (high - low))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="ui-sans-serif, system-ui, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="24" font-size="15" font-weight="600" '
        f'fill="{PALETTE["axis"]}">{title}</text>',
    ]
    for fraction in np.linspace(0, 1, 5):
        value = low + fraction * (high - low)
        position = y(value)
        parts.append(
            f'<line x1="{margin_left}" y1="{position:.1f}" x2="{margin_left + plot_width}" '
            f'y2="{position:.1f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 8}" y="{position + 4:.1f}" font-size="11" '
            f'text-anchor="end" fill="{PALETTE["axis"]}">{value:.3f}</text>'
        )
    parts.append(
        f'<text x="18" y="{margin_top + plot_height / 2:.0f}" font-size="12" '
        f'fill="{PALETTE["axis"]}" transform="rotate(-90 18 {margin_top + plot_height / 2:.0f})" '
        f'text-anchor="middle">{y_label}</text>'
    )
    for offset, (name, values, colour) in enumerate(series):
        points = " ".join(
            f"{x(index):.1f},{y(float(value)):.1f}"
            for index, value in enumerate(values) if np.isfinite(value)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="1.6" '
            f'stroke-linejoin="round"/>'
        )
        legend_y = margin_top + 14 + offset * 18
        parts.append(
            f'<line x1="{margin_left + plot_width + 16}" y1="{legend_y}" '
            f'x2="{margin_left + plot_width + 40}" y2="{legend_y}" stroke="{colour}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{margin_left + plot_width + 46}" y="{legend_y + 4}" font-size="11" '
            f'fill="{PALETTE["axis"]}">{name}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


CONFIG_COLOURS = [
    "#1f4e79", "#2f6f4f", "#a34b2a", "#7a5c9e", "#0f766e", "#b45309",
    "#9d174d", "#3f6212", "#1e3a8a", "#7c2d12", "#4c1d95", "#155e75",
]


def _updates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def _column(records: list[dict], key: str) -> np.ndarray:
    return np.array([record.get(key, np.nan) for record in records], dtype=float)


def _rolling_median(values: np.ndarray, window: int = 25) -> np.ndarray:
    if values.size <= window:
        return values
    out = np.empty_like(values)
    for index in range(values.size):
        out[index] = np.median(values[max(0, index - window + 1) : index + 1])
    return out


def _write_training_charts(output_dir: Path) -> list[str]:
    """Charts of the optimiser's own behaviour, not of the money it made.

    These are the three numbers that diagnosed ``hourly_eq``: the entropy the
    policy settles at, the fraction of samples the PPO clip actually touches,
    and how much of the return variance the critic explains.  They are drawn
    here so a reader can see at a glance whether the optimiser was working,
    separately from whether the strategy was profitable.
    """
    written: list[str] = []
    manifest = _read(output_dir / "experiment.json") or {}
    grid = (manifest.get("protocol") or {}).get("configuration_grid") or {}
    selection = _read(output_dir / "selection.json") or {}
    chosen = selection.get("selected_config_id")

    # 1. Entropy against its own ceiling for every configuration on fold 1.
    #    A ceiling-hugging line is a fair die; a line at zero is a collapsed,
    #    permanently-flat policy.  hourly_eq produced only those two shapes.
    series = []
    for offset, (config_id, cell) in enumerate(sorted(grid.items())):
        records = _updates(
            output_dir / "fold_1" / "economic" / config_id / "seed_11" / "updates.jsonl"
        )
        if not records:
            continue
        ladder = cell.get("position_ladder", "ternary")
        ceiling = float(np.log(7 if ladder == "ladder" else 3))
        label = f"{config_id} ent={cell.get('entropy_coefficient')} {ladder}"
        if config_id == chosen:
            label += "  *"
        series.append((label, _column(records, "entropy") / ceiling,
                       CONFIG_COLOURS[offset % len(CONFIG_COLOURS)]))
    if series:
        (output_dir / "diagnostic_entropy.svg").write_text(_line_chart(
            series, "Policy entropy as a fraction of its ceiling (fold 1, seed 11)",
            "H / log(actions)", height=460,
        ))
        written.append("diagnostic_entropy.svg")

    # 2. Clip fraction and explained variance for the final runs of the
    #    selected configuration, both arms.
    for key, label, filename in (
        ("clip_fraction", "PPO clip fraction", "diagnostic_clip_fraction.svg"),
        ("explained_variance", "Critic explained variance", "diagnostic_explained_variance.svg"),
        ("approximate_kl", "Approximate KL per update", "diagnostic_kl.svg"),
    ):
        series = []
        for arm, colour in (("economic", PALETTE["economic"]), ("zero_cost", PALETTE["zero_cost"])):
            paths = sorted((output_dir / "final" / arm).glob(f"*/seed_*/updates.jsonl"))
            runs = [_updates(path) for path in paths]
            runs = [run for run in runs if run]
            if not runs:
                continue
            length = min(len(run) for run in runs)
            stacked = np.stack([_column(run, key)[:length] for run in runs])
            series.append((f"{arm} (median of {len(runs)} seeds)",
                           _rolling_median(np.median(stacked, axis=0)), colour))
        if series:
            (output_dir / filename).write_text(_line_chart(
                series, f"{label} over training (25-update rolling median)", label,
            ))
            written.append(filename)

    # 3. Where the test year's P&L actually came from.  Funding is the one
    #    component with a persistent sign, so its cumulative line is the
    #    clearest evidence for or against a carry contribution.
    trajectory_dir = output_dir / "test_trajectories"
    if trajectory_dir.exists():
        for arm in ("economic", "zero_cost"):
            paths = sorted(trajectory_dir.glob(f"test_{arm}_seed_*.csv"))
            if not paths:
                continue
            frames = [pd.read_csv(path) for path in paths]
            length = min(len(frame) for frame in frames)
            def cumulative(column: str, sign: float = 1.0) -> np.ndarray:
                stacked = np.stack([frame[column].to_numpy()[:length] for frame in frames])
                return np.cumsum(sign * np.median(stacked, axis=0))
            series = [
                ("price", cumulative("price_return"), PALETTE["always_long"]),
                ("funding", cumulative("funding_return"), PALETTE["economic"]),
                ("cost paid", cumulative("cost_paid", -1.0), PALETTE["momentum"]),
            ]
            (output_dir / f"test_pnl_decomposition_{arm}.svg").write_text(_line_chart(
                series, f"Sealed test year: cumulative P&L by source ({arm}, median seed)",
                "cumulative fraction of equity",
            ))
            written.append(f"test_pnl_decomposition_{arm}.svg")

    return written


def _write_charts(output_dir: Path, test: dict | None) -> list[str]:
    written = []
    trajectory_dir = output_dir / "test_trajectories"
    if test and trajectory_dir.exists():
        def thin(values: np.ndarray, points: int = 1500) -> np.ndarray:
            step = max(len(values) // points, 1)
            return values[::step]

        series = []
        for arm, colour in (("zero_cost", PALETTE["zero_cost"]), ("economic", PALETTE["economic"])):
            curves = []
            for path in sorted(trajectory_dir.glob(f"test_{arm}_seed_*.csv")):
                curves.append(pd.read_csv(path)["equity"].to_numpy())
            if curves:
                length = min(len(curve) for curve in curves)
                stacked = np.stack([curve[:length] for curve in curves])
                series.append((f"PPO {arm} (median of {len(curves)} seeds)",
                               thin(np.median(stacked, axis=0)), colour))
        for name, colour in (("always_long", PALETTE["always_long"]),
                             ("momentum", PALETTE["momentum"])):
            path = next(iter(sorted(trajectory_dir.glob(f"benchmark_{name}*_economic.csv"))), None)
            if path is not None:
                series.append((path.stem.replace("benchmark_", "").replace("_economic", ""),
                               thin(pd.read_csv(path)["equity"].to_numpy()), colour))
        if series:
            chart = _line_chart(series, "Sealed test year: equity", "equity (start = 1)")
            (output_dir / "test_equity.svg").write_text(chart)
            written.append("test_equity.svg")

    experiment = _read(output_dir / "experiment.json")
    selection = _read(output_dir / "selection.json")
    if experiment and selection:
        config_id = selection["selected_config_id"]
        series = []
        for fold in ("fold_1", "fold_2", "fold_3", "fold_4"):
            values, budgets = [], []
            for seed_dir in sorted((output_dir / fold / "economic" / config_id).glob("seed_*")):
                record = _read(seed_dir / "training.json")
                if not record:
                    continue
                curve = record.get("validation_curve", [])
                budgets = [point["update"] for point in curve]
                values.append([point["economic"]["total_log_return"] for point in curve])
            if values:
                length = min(len(row) for row in values)
                stacked = np.stack([row[:length] for row in values])
                series.append((f"{fold} median", np.median(stacked, axis=0), None))
        colours = [PALETTE["economic"], PALETTE["zero_cost"], PALETTE["momentum"], PALETTE["always_long"]]
        series = [(name, values, colours[index % len(colours)])
                  for index, (name, values, _) in enumerate(series)]
        if series:
            chart = _line_chart(
                series, f"Validation log return by training budget ({config_id}, economic arm)",
                "validation total log return",
            )
            (output_dir / "validation_budget_curve.svg").write_text(chart)
            written.append("validation_budget_curve.svg")
    return written


# -- tables ----------------------------------------------------------------


def _seed_table(test: dict) -> pd.DataFrame:
    rows = []
    for arm, payload in test["arms"].items():
        for entry in payload["per_seed"]:
            own = entry[arm]
            rows.append({
                "arm": arm,
                "seed": entry["seed"],
                "total_return": own["total_return"],
                "annualised_return": own["annualised_return"],
                "sharpe": own["sharpe"],
                "sortino": own["sortino"],
                "max_drawdown": own["max_drawdown"],
                "turnover_per_hour": own["turnover_per_hour"],
                "position_changes": own["position_changes"],
                "average_holding_hours": own["average_holding_hours"],
                "share_long": own["share_long"],
                "share_flat": own["share_flat"],
                "share_short": own["share_short"],
                "price_pnl": own["price_pnl"],
                "funding_pnl": own["funding_pnl"],
                "cost_pnl": own["cost_pnl"],
                "null_percentile": entry["circular_shift_null"].get("percentile_of_actual"),
                "max_action_probability": entry["maximum_action_probability_mean"],
            })
    return pd.DataFrame(rows)


def write_report(output_dir: Path) -> Path:
    experiment = _read(output_dir / "experiment.json") or {}
    selection = _read(output_dir / "selection.json")
    gate = _read(output_dir / "zero_cost_gate.json")
    test = _read(output_dir / "test_results.json")
    charts = _write_charts(output_dir, test) + _write_training_charts(output_dir)

    lines: list[str] = []
    lines.append("# Hourly GRU-PPO on BTCUSDT, five-minute state, log-equity reward\n")
    protocol = experiment.get("protocol", {})
    dataset = experiment.get("dataset", {})
    lines.append(
        f"Study window {protocol.get('study_window', 'n/a')}.  "
        f"{dataset.get('hourly_decisions_in_study', 'n/a')} hourly decisions built from "
        f"{dataset.get('total_bars', 'n/a')} five-minute bars with "
        f"{dataset.get('warmup_days', 'n/a')} days of feature warm-up, and "
        f"{dataset.get('funding_settlements_total', 'n/a')} funding settlements applied at "
        "their exact timestamps.  BTCUSDT USD-M perpetual futures only.\n"
    )
    lines.append("## Protocol\n")
    lines.append("| item | value |\n|---|---|")
    for key in ("decision_interval_hours", "observation", "actions", "exposure", "reward",
                "funding", "selection_rule", "maximum_updates", "evaluation_interval"):
        if key in protocol:
            lines.append(f"| {key.replace('_', ' ')} | {protocol[key]} |")
    lines.append(f"| cost arms | {protocol.get('cost_arms')} |")
    lines.append(f"| selection seeds | {protocol.get('selection_seeds')} |")
    lines.append(f"| final seeds | {protocol.get('final_seeds')} |")
    lines.append("")

    signal = experiment.get("signal_check")
    if signal:
        lines.append("## Supervised signal check, before any RL\n")
        lines.append(
            "Ridge probe on exactly the agent's state, fitted on each fold's training window and "
            "scored on that fold's validation window.  The hurdle is the information coefficient "
            "at which a sign-following position pays the 8 bp round trip at that holding period.\n"
        )
        lines.append("| horizon (h) | median IC | break-even IC | folds clearing |\n|---|---|---|---|")
        for horizon, payload in signal["horizons"].items():
            lines.append(
                f"| {horizon} | {payload['median_information_coefficient']:+.4f} | "
                f"{payload['median_break_even']:.4f} | "
                f"{payload['folds_clearing_hurdle']}/{payload['folds']} |"
            )
        lines.append("")

    if selection:
        lines.append("## Configuration and budget, selected on validation folds only\n")
        lines.append(
            f"Rule declared before the search: **{selection['rule']}**.  "
            f"{selection['full_ranking_size']} (configuration, budget) cells were scored.\n"
        )
        lines.append(
            f"Selected **{selection['selected_config_id']}** at "
            f"**{selection['selected_budget_updates']} updates**, median validation log return "
            f"{selection['selected_median_validation_log_return']:+.4f}.\n"
        )
        lines.append("```json")
        lines.append(json.dumps(selection["selected_config"], indent=2))
        lines.append("```\n")
        lines.append("| rank | config | budget | median | IQR |\n|---|---|---|---|---|")
        for rank, row in enumerate(selection["ranking_top_20"][:10], start=1):
            low, high = row["interquartile_range"]
            lines.append(
                f"| {rank} | {row['config_id']} | {row['budget_updates']} | "
                f"{row['median']:+.4f} | [{low:+.4f}, {high:+.4f}] |"
            )
        lines.append("")
        momentum = selection.get("momentum_benchmark", {})
        if momentum:
            lines.append(
                f"The momentum benchmark's lookback was chosen the same way and on the same "
                f"folds: **{momentum['selected_lookback_hours']} hours**.\n"
            )

    if gate and gate.get("status") == "evaluated":
        lines.append("## Zero-cost diagnostic gate\n")
        lines.append(f"Criteria: {gate['criteria']}.\n")
        lines.append("| quantity | value |\n|---|---|")
        lines.append(f"| runs | {gate['runs']} |")
        lines.append(f"| policy median validation log return | {gate['policy_median_log_return']:+.4f} |")
        lines.append(
            f"| policy IQR | [{gate['policy_interquartile_range'][0]:+.4f}, "
            f"{gate['policy_interquartile_range'][1]:+.4f}] |"
        )
        lines.append(f"| best benchmark median | {gate['best_benchmark_median_log_return']:+.4f} |")
        lines.append(
            f"| runs beating the best benchmark | "
            f"{gate['share_of_runs_beating_best_benchmark'] * 100:.1f}% |"
        )
        share = gate.get("share_of_runs_above_95th_percentile_of_matched_turnover_null")
        lines.append(
            f"| runs above the 95th percentile of their matched-turnover null | "
            f"{'n/a' if share is None else f'{share * 100:.1f}%'} |"
        )
        lines.append(f"| **gate passed** | **{gate['gate_passed']}** |")
        lines.append("")
        if not gate["gate_passed"]:
            lines.append(
                "> The zero-cost policy does not consistently beat matched-turnover noise and the "
                "simple rules.  Under the predeclared reading of the protocol that is a "
                "**signal** result, not a cost result: the test numbers below should not be read "
                "as evidence that fees and funding are what stands in the way.\n"
            )

    if test:
        lines.append("## Sealed test year, 1 June 2025 to 1 June 2026\n")
        lines.append(
            "Opened once, after every final run reported complete.  Deterministic argmax, "
            "frozen weights and frozen normaliser, hidden state warmed causally over the "
            "preceding 240 hours.  Median across seeds with the interquartile range in "
            "brackets.\n"
        )
        header = "| strategy | total return | annualised | Sharpe | Sortino | max drawdown | turnover/h |"
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|")
        for cost_name in ("zero_cost", "economic"):
            for name, payload in test["benchmarks"].get(cost_name, {}).items():
                label = f"{name} ({cost_name})"
                lines.append(
                    f"| {label} | {_percent(payload['total_return'])} | "
                    f"{_percent(payload['annualised_return'])} | {payload['sharpe']:+.3f} | "
                    f"{payload['sortino']:+.3f} | {_percent(payload['max_drawdown'])} | "
                    f"{payload['turnover_per_hour']:.3f} |"
                )
        for arm, payload in test["arms"].items():
            aggregate = payload["aggregate"]
            lines.append(
                f"| **PPO {arm}** | {_median_row(aggregate, 'total_return', _percent)} | "
                f"{_median_row(aggregate, 'annualised_return', _percent)} | "
                f"{_median_row(aggregate, 'sharpe')} | {_median_row(aggregate, 'sortino')} | "
                f"{_median_row(aggregate, 'max_drawdown', _percent)} | "
                f"{_median_row(aggregate, 'turnover_per_hour')} |"
            )
        lines.append("")

        frame = _seed_table(test)
        frame.to_csv(output_dir / "test_seed_table.csv", index=False)
        lines.append("### Every seed\n")
        lines.append(
            "| arm | seed | total return | Sharpe | max drawdown | turnover/h | long/flat/short | "
            "price / funding / cost | null percentile |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in frame.itertuples():
            lines.append(
                f"| {row.arm} | {row.seed} | {_percent(row.total_return)} | {row.sharpe:+.3f} | "
                f"{_percent(row.max_drawdown)} | {row.turnover_per_hour:.3f} | "
                f"{row.share_long:.2f}/{row.share_flat:.2f}/{row.share_short:.2f} | "
                f"{row.price_pnl:+.4f} / {row.funding_pnl:+.4f} / {row.cost_pnl:+.4f} | "
                f"{'n/a' if row.null_percentile is None or not np.isfinite(row.null_percentile) else f'{row.null_percentile:.1f}'} |"
            )
        lines.append("")

        lines.append("### Matched-turnover null\n")
        lines.append(
            "The null rotates each policy's own action sequence against history, which preserves "
            "every holding length and the exact turnover while destroying alignment with the "
            "returns those actions were chosen for.  A percentile above 95 means the policy beat "
            "its own turnover-matched noise.\n"
        )
        lines.append("| arm | median percentile | seeds above 95 |\n|---|---|---|")
        for arm, payload in test["arms"].items():
            aggregate = payload["aggregate"]
            percentile = aggregate.get("circular_shift_null_percentile", {})
            lines.append(
                f"| {arm} | {percentile.get('median', float('nan')):.1f} | "
                f"{aggregate.get('seeds_above_95th_percentile_of_null')} of "
                f"{percentile.get('seeds', 'n/a')} |"
            )
        lines.append("")

        lines.append("### Post-hoc regimes\n")
        lines.append(
            "Regimes are assigned after the fact from the trailing 30-day return and are never "
            "in the state; the GRU is left to infer conditions from returns, momentum, "
            "volatility and order flow.\n"
        )
        lines.append("| arm | regime | hours | median total log return | long/flat/short |")
        lines.append("|---|---|---|---|---|")
        for arm, payload in test["arms"].items():
            for regime in ("bull", "bear", "chop"):
                values = [
                    entry["regimes"][regime].get("total_log_return")
                    for entry in payload["per_seed"] if entry["regimes"][regime]["hours"]
                ]
                if not values:
                    continue
                sample = payload["per_seed"][0]["regimes"][regime]
                shares = (
                    np.median([e["regimes"][regime]["share_long"] for e in payload["per_seed"]]),
                    np.median([e["regimes"][regime]["share_flat"] for e in payload["per_seed"]]),
                    np.median([e["regimes"][regime]["share_short"] for e in payload["per_seed"]]),
                )
                lines.append(
                    f"| {arm} | {regime} | {sample['hours']} | {np.median(values):+.4f} | "
                    f"{shares[0]:.2f}/{shares[1]:.2f}/{shares[2]:.2f} |"
                )
        lines.append("")

    if charts:
        lines.append("## Charts\n")
        for chart in charts:
            lines.append(f"![{chart}]({chart})\n")

    lines.append("## Files\n")
    lines.append("| file | what it holds |\n|---|---|")
    lines.append("| `experiment.json` | protocol, dataset audit, stage progress, selection, gate |")
    lines.append("| `selection.json` | the configuration and budget chosen on the folds |")
    lines.append("| `zero_cost_gate.json` | the diagnostic gate, run before the test was opened |")
    lines.append("| `test_results.json` | every seed, benchmark, null and breakdown on the test year |")
    lines.append("| `test_seed_table.csv` | the per-seed table above, machine readable |")
    lines.append("| `test_trajectories/` | hourly position, P&L decomposition and equity per seed |")
    lines.append("| `<split>/<arm>/<config>/seed_<n>/updates.jsonl` | one line per PPO update |")
    lines.append("| `<split>/<arm>/<config>/seed_<n>/training.json` | the validation curve for that run |")
    lines.append("")

    path = output_dir / "FINAL_REPORT.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(write_report(args.output_dir))


if __name__ == "__main__":
    main()
