from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .config import Config, OUTPUT, SEEDS, TEST_START, TRAIN_START, VALIDATION_YEARS, source_hash, write_json
from .data import Scaler, load_market
from .env import Environment
from .evaluate import evaluate
from .network import ActorCritic
from .ppo import Trainer


def now():
    return datetime.now(timezone.utc).isoformat()


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train(fold, seed, output=OUTPUT, config=Config()):
    if fold not in [str(x) for x in VALIDATION_YEARS] + ["final"]:
        raise ValueError("Only development folds 2023/2024/2025 or unevaluated pre-2026 retraining are permitted")
    protocol_path = Path(output) / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["source_sha256"] != source_hash():
        raise ValueError("source differs from frozen experiment protocol; use a new output directory")
    torch.set_num_threads(config.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    market = load_market(Path(output) / "cache")
    train_end = TEST_START if fold == "final" else pd.Timestamp(f"{fold}-01-01", tz="UTC")
    scaler = Scaler.fit(market, train_end)
    env = Environment(market, market.bounds(TRAIN_START, train_end), config, np.random.default_rng(seed))
    model = ActorCritic(config)
    trainer = Trainer(model, env, scaler, config, seed)
    run = Path(output) / "runs" / fold / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    fingerprint = {"source_sha256": source_hash(), "config": config.as_dict(), "fold": fold, "seed": seed}
    record_path = run / "training.json"
    record = {"status": "running", "started_at": now(), "specification": fingerprint,
              "scaler": scaler.as_dict(), "parameters": sum(x.numel() for x in model.parameters()),
              "validation_curve": [], "updates_completed": 0,
              "train_end_exclusive": str(train_end), "test_locked": True, "test_performance_accessed": False}
    latest = run / "resume.pt"
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old["specification"] != fingerprint:
            raise ValueError("run source/config changed; refusing to mix different experiments")
        if old["status"] == "completed":
            return old
        if latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=False)
            if state["specification"] != fingerprint:
                raise ValueError("checkpoint provenance mismatch")
            trainer.load_state_dict(state["trainer"])
            record = state["record"]
            # Remove only uncommitted log entries beyond the recovered state.
            log = run / "updates.jsonl"
            if log.exists():
                kept = []
                for line in log.open():
                    try:
                        if json.loads(line)["update"] <= trainer.updates:
                            kept.append(line)
                    except ValueError:
                        pass
                log.write_text("".join(kept))
        else:
            # An interruption before the first committed checkpoint restarts
            # deterministically at update zero, including its uncommitted log.
            (run / "updates.jsonl").write_text("")
    record["status"] = "running"
    write_json(record_path, record)
    start = time.monotonic()
    previous_elapsed = record.get("elapsed_seconds", 0.0)
    with (run / "updates.jsonl").open("a", buffering=1) as log:
        while trainer.updates < config.updates:
            summary = trainer.run_update()
            summary["elapsed_seconds"] = previous_elapsed + time.monotonic() - start
            log.write(json.dumps(summary, allow_nan=False) + "\n")
            record.update({"updates_completed": trainer.updates, "last_update_at": now(),
                           "elapsed_seconds": summary["elapsed_seconds"], "last_training_summary": summary})
            due = trainer.updates % config.validation_every == 0 or trainer.updates == config.updates
            if due:
                checkpoint = {"model": model.state_dict(), "scaler": scaler.as_dict(), "specification": fingerprint,
                              "update": trainer.updates, "test_locked": True}
                save_checkpoint(run / f"policy_{trainer.updates:04d}.pt", checkpoint)
                if fold != "final":
                    record["status"] = "validating"
                    write_json(record_path, record)
                    end = pd.Timestamp(f"{int(fold)+1}-01-01", tz="UTC")
                    result, frame = evaluate([model], market, scaler, config, train_end, end)
                    result["update"] = trainer.updates
                    record["validation_curve"].append(result)
                    write_json(run / f"validation_{trainer.updates:04d}.json", result)
                    frame.to_csv(run / f"validation_{trainer.updates:04d}.csv", index=False)
                    print(json.dumps({"fold": fold, "seed": seed, "update": trainer.updates,
                                      "validation_return_btc": result["total_return_btc"],
                                      "orders": result["orders"]}), flush=True)
                record["status"] = "running"
            if trainer.updates % 10 == 0 or due:
                record["elapsed_seconds"] = previous_elapsed + time.monotonic() - start
                save_checkpoint(latest, {"trainer": trainer.state_dict(), "record": record, "specification": fingerprint})
            write_json(record_path, record)
            if trainer.updates % 10 == 0:
                print(json.dumps({"fold": fold, "seed": seed, "update": trainer.updates,
                                  "reward": summary["mean_reward"], "critic_ev": summary["explained_variance_pre"],
                                  "elapsed_seconds": record["elapsed_seconds"], "test_locked": True}), flush=True)
    record.update({"status": "completed", "finished_at": now()})
    write_json(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description="Train/validate development data only; test access is not implemented")
    parser.add_argument("--fold", required=True, choices=[str(x) for x in VALIDATION_YEARS] + ["final"])
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--updates", type=int, default=Config.updates)
    parser.add_argument("--threads", type=int, default=Config.threads)
    args = parser.parse_args()
    train(args.fold, args.seed, args.output, replace(Config(), updates=args.updates, threads=args.threads))


if __name__ == "__main__":
    main()
