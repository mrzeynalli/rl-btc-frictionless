"""Reproduce the review's artifact counts and diagnostics; never train a model.

Run from anywhere with model/.venv/bin/python <this file>. Inputs are read-only;
small summaries are written beside this script. Archived copies are counted as
records, not asserted to be independent experiments.
This historical audit reads old, already-exposed evaluation artifacts. It is not
part of the v4 training runner and must not be rerun while the test lock is active.
"""

from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import json
import math
import sys

import numpy as np
import pandas as pd

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
ARCHIVE = ROOT / "backup/spot_before_perpetual_v4_20260905"


def write(name, value):
    def finite(item):
        if isinstance(item, dict):
            return {key: finite(val) for key, val in item.items()}
        if isinstance(item, list):
            return [finite(val) for val in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    # Some archived reports encode undefined diagnostics as NaN. Preserve
    # their absence as JSON null rather than emitting nonstandard JSON.
    (DEST / name).write_text(json.dumps(finite(value), indent=2, allow_nan=False) + "\n")


def family(path):
    parts = path.relative_to(ROOT).parts
    if parts[:2] == ("backup", ARCHIVE.name):
        parts = parts[2:]
    if parts[0] == "results":
        base = "/".join(parts[:2])
        if parts[1] in ("spot_v2", "spot_v3_1m"):
            return base + ("/" + parts[3] if parts[2] == "runs" else "/smoke")
        return base
    if parts[1] in ("artifacts", "results_pre_redesign_20260903"):
        return "/".join(parts[:3])
    return "/".join(parts[:2])


def training():
    keys = ["explained_variance", "explained_variance_pre_update",
            "explained_variance_post_update", "critic_prediction_std_post",
            "return_std", "entropy", "entropy_coefficient", "approximate_kl",
            "clip_fraction", "gradient_norm", "turnover_per_step", "mean_reward"]
    records, errors = [], []
    legacy_clips = Counter()
    for folder in ("results", "backup"):
        for path in sorted((ROOT / folder).rglob("training.json")):
            if "perpetual_v4" in path.parts:
                continue
            try:
                saved = json.loads(path.read_text())
                rows = []
                log = path.with_name("updates.jsonl")
                if log.exists():
                    for number, line in enumerate(log.open(), 1):
                        try:
                            rows.append(json.loads(line))
                        except ValueError as error:
                            errors.append([str(log.relative_to(ROOT)), number, str(error)])
                tail = rows[-max(5, len(rows) // 4):]
                record = {"path": str(path.relative_to(ROOT)), "family": family(path),
                          "status": saved.get("status"), "updates": len(rows)}
                for key in keys:
                    values = [x[key] for x in tail if isinstance(x.get(key), (int, float))
                              and math.isfinite(x[key])]
                    if values:
                        record[key] = float(np.mean(values))
                for key in ("replay_ratio_max_abs_error", "replay_value_max_abs_error"):
                    values = [x[key] for x in rows if isinstance(x.get(key), (int, float))
                              and math.isfinite(x[key])]
                    if values:
                        record[key + "_all_updates_max"] = float(max(values))
                if family(path) == "backup/artifacts/btc1h_gru_ppo_equity":
                    legacy_clips["rows"] += len(rows)
                    legacy_clips["positive_clip_rows"] += sum(x.get("clip_fraction", 0) > 0 for x in rows)
                records.append(record)
            except (OSError, ValueError) as error:
                errors.append([str(path.relative_to(ROOT)), str(error)])
    groups = defaultdict(list)
    for record in records:
        groups[record["family"]].append(record)
    summaries = {}
    for name, items in groups.items():
        summary = {"records": len(items), "updates": sum(x["updates"] for x in items),
                   "statuses": dict(Counter(x["status"] for x in items)),
                   "median_of_run_final_quarter_means": {}}
        for key in keys:
            values = [x[key] for x in items if key in x]
            if values:
                summary["median_of_run_final_quarter_means"][key] = float(np.median(values))
        summaries[name] = summary
    pd.DataFrame(records).to_csv(DEST / "training_records.csv", index=False)
    write("training_summary.json", {"training_records": len(records),
          "update_entries": sum(x["updates"] for x in records), "errors": errors,
          "legacy_hourly_ppo_clips": dict(legacy_clips), "families": summaries})


def economics():
    output = {}
    for version, name in (("spot_v2", "locked_2026"), ("spot_v3_1m", "test_2026")):
        base = ARCHIVE / "results" / version
        report = json.loads((base / (name + "_report.json")).read_text())
        frame = pd.read_csv(base / (name + "_equity.csv"))
        output[version] = {"report": report,
                           "non_hold_orders": frame[frame.action != "HOLD"].to_dict("records")}
    for name in ("btc_wf_1h", "btc_wf_4h"):
        result = {}
        for stage in ("development", "sealed"):
            paths = sorted((ROOT / "backup/artifacts" / name / stage).glob("*.json"))
            records = [json.loads(p.read_text()) for p in paths]
            result[stage] = {"records": len(records),
                             "retraining_blocks": sum(len(x["blocks"]) for x in records)}
            if stage == "sealed":
                result[stage]["positive_seeds"] = sum(x["metrics"]["total_return"] > 0 for x in records)
                result[stage]["median_metrics"] = {
                    key: float(np.median([x["metrics"][key] for x in records]))
                    for key in ("total_return", "sharpe", "price_pnl", "funding_pnl", "cost_pnl")}
        output[name] = result
    write("economics.json", output)


def market():
    base = ROOT / "data/bitmex_xbtusd_1m_2016_2026_08_01"
    manifest = json.loads((base / "manifest.json").read_text())
    bars = pd.read_parquet(base / "XBTUSD-1m-OHLCV.parquet")
    times = pd.DatetimeIndex(bars.open_time).as_unit("ns")
    delta = np.diff(times.asi8) // (60 * 10**9)
    output = {"rows": len(bars), "columns": list(bars), "first": str(times[0]),
              "last": str(times[-1]), "duplicate_timestamps": int(times.duplicated().sum()),
              "missing_minutes_between_first_and_last": int(np.maximum(delta - 1, 0).sum()),
              "missing_minutes_through_requested_end": manifest["minutes_without_regular_trades"],
              "gap_events": int((delta > 1).sum()), "by_year": {}}
    for year in sorted(set(times.year)):
        subset = times[times.year == year]
        grid = pd.date_range(subset[0], subset[-1], freq="min")
        missing = (~grid.isin(subset)).astype(np.int32)
        prefix = np.r_[0, np.cumsum(missing)]
        output["by_year"][str(year)] = {
            "rows": len(subset), "missing_share_within_observed_year": float(missing.mean()),
            "complete_window_share": {str(width): float(np.mean(prefix[width:] == prefix[:-width]))
                                      for width in (60, 1440, 14400)}}
    funding = pd.read_parquet(base / "funding/XBTUSD-funding.parquet")
    output["funding"] = {"rows": len(funding),
                         "interval_hours_counts": {str(k): int(v) for k, v in
                             funding.funding_interval_hours.value_counts().items()}}
    write("market_audit.json", output)


def scale_example():
    sys.path.insert(0, str(ARCHIVE / "src"))
    import torch
    from ppo import PPOConfig, PPOTrainer
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = PPOConfig(environments=1, gamma=0.9, gae_lambda=0.95)
    trainer.return_scale = SimpleNamespace(scale=2.0)
    buffer = SimpleNamespace(rewards=np.array([[1]], dtype=np.float32),
                             dones=np.array([[False]]),
                             truncated_values=np.array([[0]], dtype=np.float32),
                             values=np.array([[10]], dtype=np.float32))
    inconsistent, _ = trainer.advantages(buffer, torch.tensor([10.0]))
    buffer.values /= 2
    consistent, _ = trainer.advantages(buffer, torch.tensor([5.0]))
    write("scale_reproduction.json", {"raw_reward": 1, "gamma": 0.9,
          "perfect_raw_value": 10, "old_scale": 1, "new_scale": 2,
          "advantage_with_unchanged_old_values": float(inconsistent[0, 0]),
          "advantage_with_converted_values": float(consistent[0, 0]),
          "interpretation": "Controlled numerical example; does not estimate contribution to historical losses."})


if __name__ == "__main__":
    training()
    economics()
    market()
    scale_example()
    print("Wrote training, economics, market, and scaling evidence to", DEST)
