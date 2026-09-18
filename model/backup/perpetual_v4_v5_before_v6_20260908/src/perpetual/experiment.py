"""Resumable training/validation coordinator. There is no test stage."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from .config import Config, OUTPUT, ROOT, SEEDS, TEST_START, VALIDATION_YEARS, protocol, source_hash, write_json
from .data import Scaler, load_market, prepare
from .evaluate import evaluate
from .network import ActorCritic
from .train import now


def status(output, phase, jobs):
    entries = []
    for fold, seed, updates in jobs:
        path = output / "runs" / fold / f"seed_{seed}" / "training.json"
        item = {"fold": fold, "seed": seed, "planned_updates": updates, "status": "pending", "updates_completed": 0}
        if path.exists():
            saved = json.loads(path.read_text())
            item.update({key: saved.get(key) for key in ("status", "updates_completed", "last_update_at", "elapsed_seconds")})
            if saved.get("validation_curve"):
                item["last_validation"] = saved["validation_curve"][-1]
        entries.append(item)
    write_json(output / "status.json", {"phase": phase, "updated_at": now(), "jobs": entries,
               "completed_jobs": sum(x["status"] == "completed" for x in entries),
               "total_jobs": len(entries), "test_locked": True, "test_performance_accessed": False})


def dispatch(output, jobs, workers, threads, phase):
    expected_source = json.loads((output / "protocol.json").read_text())["source_sha256"]
    pending = []
    for job in jobs:
        fold, seed, _ = job
        path = output / "runs" / fold / f"seed_{seed}" / "training.json"
        if not path.exists() or json.loads(path.read_text()).get("status") != "completed":
            pending.append(job)
    active, attempts = [], {}
    log_dir = output / "logs"
    log_dir.mkdir(exist_ok=True)
    while pending or active:
        while pending and len(active) < workers:
            if source_hash() != expected_source:
                raise RuntimeError("source changed during the experiment; refusing to mix versions")
            job = pending.pop(0)
            fold, seed, updates = job
            attempts[job] = attempts.get(job, 0) + 1
            handle = (log_dir / f"{fold}_seed_{seed}.log").open("a", buffering=1)
            command = [sys.executable, "-u", "-m", "perpetual.train", "--fold", fold, "--seed", str(seed),
                       "--output", str(output), "--updates", str(updates), "--threads", str(threads)]
            env = dict(os.environ)
            env.update({"PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": str(threads),
                        "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads)})
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            active.append((job, process, handle))
            print(json.dumps({"event": "started", "job": job, "pid": process.pid, "at": now()}), flush=True)
        remaining = []
        for job, process, handle in active:
            code = process.poll()
            if code is None:
                remaining.append((job, process, handle))
                continue
            handle.close()
            outcome = {"job": job, "exit_code": code, "at": now(), "attempt": attempts[job]}
            with (log_dir / "jobs.jsonl").open("a") as log:
                log.write(json.dumps(outcome) + "\n")
            if code:
                if attempts[job] < 3:
                    pending.append(job)
                else:
                    for _, other, other_handle in active:
                        if other.poll() is None:
                            other.terminate()
                            other.wait(timeout=30)
                        other_handle.close()
                    raise RuntimeError(f"Job {job} failed three times; see {log_dir}")
        active = remaining
        status(output, phase, jobs)
        if pending or active:
            time.sleep(10)


def select_budget(output, config):
    curves = []
    for year in VALIDATION_YEARS:
        for seed in SEEDS:
            item = json.loads((output / "runs" / str(year) / f"seed_{seed}" / "training.json").read_text())
            if item["status"] != "completed" or item["test_performance_accessed"]:
                raise RuntimeError("development selection requires all clean completed runs")
            curves.append({x["update"]: x for x in item["validation_curve"]})
    common = sorted(set.intersection(*(set(x) for x in curves)))
    if not common:
        raise RuntimeError("no common development checkpoint")
    ranking = []
    for update in common:
        values = [x[update] for x in curves]
        ranking.append({"update": update,
                        "mean_validation_log_return": float(np.mean([x["total_log_return_btc"] for x in values])),
                        "median_validation_return": float(np.median([x["total_return_btc"] for x in values])),
                        "positive_run_share": float(np.mean([x["total_return_btc"] > 0 for x in values])),
                        "liquidations": sum(x["liquidations"] for x in values), "runs": len(values)})
    ranking.sort(key=lambda x: (-x["mean_validation_log_return"], x["update"]))
    selected = ranking[0]["update"]
    write_json(output / "selection.json", {"selected_updates": selected, "ranking": ranking,
               "rule": "mean net log BTC growth over all 15 seed/year pairs; ties prefer earlier budget",
               "hyperparameters": config.as_dict(), "test_locked": True, "test_performance_accessed": False})
    return selected


def ensemble_validation(output, config, update):
    torch.set_num_threads(config.threads)
    market = load_market(output / "cache")
    reports = []
    for year in VALIDATION_YEARS:
        report_path = output / f"ensemble_validation_{year}.json"
        if report_path.exists():
            item = json.loads(report_path.read_text())
            if item["selected_updates"] != update:
                raise ValueError("stored ensemble validation has a different selected budget")
            reports.append(item)
            continue
        start, end = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year+1}-01-01", tz="UTC")
        models, scaler = [], None
        for seed in SEEDS:
            checkpoint = torch.load(output / "runs" / str(year) / f"seed_{seed}" / f"policy_{update:04d}.pt",
                                    map_location="cpu", weights_only=False)
            model = ActorCritic(config)
            model.load_state_dict(checkpoint["model"])
            models.append(model)
            current_scaler = Scaler.from_dict(checkpoint["scaler"])
            if scaler is not None and current_scaler.as_dict() != scaler.as_dict():
                raise ValueError("ensemble members must use the same fold normalization")
            scaler = current_scaler
        item = {"year": year, "selected_updates": update, "cost_sensitivity": {}, "baselines": {},
                "test_locked": True, "test_performance_accessed": False}
        for slip in (0.0001, 0.0003, 0.0005):
            result, frame = evaluate(models, market, scaler, config, start, end, slippage=slip)
            label = str(round((config.fee + slip) * 10000))
            item["cost_sensitivity"][label] = result
            frame.to_csv(output / f"ensemble_validation_{year}_{label}bp.csv", index=False)
        for baseline in ("flat", "long", "short"):
            item["baselines"][baseline] = evaluate([], market, scaler, config, start, end, baseline=baseline)[0]
        write_json(report_path, item)
        reports.append(item)
    write_json(output / "development_report.json", {"selected_updates": update, "folds": reports,
               "selection_data_only": True, "test_locked": True, "test_performance_accessed": False})


def run(output=OUTPUT, workers=2, config=Config()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".experiment.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = protocol(config)
        spec_path = output / "protocol.json"
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError("source or protocol changed; refusing to mix experiment versions")
        write_json(spec_path, spec)
        write_json(output / "TEST_LOCK.json", {"locked": True, "start_inclusive": str(TEST_START),
                   "performance_accessed": False, "unlock_entrypoint": None,
                   "enforcement": "date guards in loader, split bounds and evaluator; raw Parquet read with pre-2026 filters"})
        prepare(output / "cache")
        jobs = [(str(year), seed, config.updates) for year in VALIDATION_YEARS for seed in SEEDS]
        dispatch(output, jobs, workers, config.threads, "training_and_validation")
        if source_hash() != spec["source_sha256"]:
            raise RuntimeError("source changed before development selection")
        selected = select_budget(output, config)
        status(output, "ensemble_validation", jobs)
        ensemble_validation(output, config, selected)
        final_jobs = [("final", seed, selected) for seed in SEEDS]
        dispatch(output, final_jobs, workers, config.threads, "final_training_without_test")
        status(output, "completed_test_locked", jobs + final_jobs)
        print(json.dumps({"event": "completed", "selected_updates": selected, "test_locked": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--updates", type=int, default=Config.updates)
    parser.add_argument("--threads", type=int, default=Config.threads)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("use one to three workers on this host")
    try:
        run(args.output, args.workers, replace(Config(), updates=args.updates, threads=args.threads))
    except Exception as error:
        write_json(args.output / "failure.json", {"at": now(), "error": str(error), "test_locked": True})
        raise


if __name__ == "__main__":
    main()
