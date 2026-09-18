from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/bitmex_xbtusd_1m_2016_2026_08_01"
OUTPUT = ROOT / "results/perpetual_v4"
TEST_START = pd.Timestamp("2026-01-01", tz="UTC")
TRAIN_START = pd.Timestamp("2017-01-01", tz="UTC")
SEEDS = (11, 29, 47, 71, 101)
VALIDATION_YEARS = (2023, 2024, 2025)


class TestLockedError(ValueError):
    pass


def development_range(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("UTC-aware timestamps required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start >= TEST_START or end > TEST_START:
        raise TestLockedError("Test data from 2026-01-01 onward are LOCKED; no test entrypoint exists")
    if start >= end:
        raise ValueError("empty development interval")
    return start, end


@dataclass(frozen=True)
class Config:
    decision_minutes: int = 5
    fast_minutes: int = 60
    hidden: int = 64
    local_width: int = 32
    environments: int = 16
    rollout: int = 512
    sequence: int = 128
    burn_in: int = 128
    episode_steps: int = 4096
    minibatch_sequences: int = 16
    epochs: int = 4
    critic_epochs: int = 2
    gamma: float = 2 ** (-5 / 1440)
    gae_lambda: float = 2 ** (-5 / 120) / (2 ** (-5 / 1440))
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    clip: float = 0.2
    target_kl: float = 0.01
    entropy_start: float = 0.005
    entropy_end: float = 0.0
    entropy_decay_updates: int = 200
    gradient_clip: float = 0.5
    reward_scale: float = 100.0
    fee: float = 0.0005
    slippage: float = 0.0001
    entry_leverage: float = 1.0
    maintenance: float = 0.005
    updates: int = 400
    validation_every: int = 50
    threads: int = 2

    def __post_init__(self):
        if self.decision_minutes != 5:
            raise ValueError("v4 initial experiment fixes a five-minute decision interval")
        if self.rollout % self.sequence or self.episode_steps % self.rollout:
            raise ValueError("sequence must divide rollout and rollout must divide episode length")
        if min(self.environments, self.burn_in, self.sequence, self.updates) < 1:
            raise ValueError("positive training sizes required")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid discount")
        if not 0 <= self.fee < 0.01 or not 0 <= self.slippage < 0.01:
            raise ValueError("invalid execution cost")
        if not 0 < self.entry_leverage <= 1:
            raise ValueError("initial study permits at most 1x entry contract notional")

    def as_dict(self):
        return asdict(self)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol(config=Config()):
    return {"version": "perpetual_v4", "instrument": "BitMEX XBTUSD inverse perpetual",
            "objective": "net BTC equity log growth; USD equity also reported",
            "train_start": str(TRAIN_START), "validation_years": list(VALIDATION_YEARS),
            "test_start": str(TEST_START), "test_locked": True,
            "test_performance_accessed": False, "seeds": list(SEEDS),
            "source_sha256": source_hash(), "config": config.as_dict(),
            "cost_assumption": "fixed 5 bp taker commission plus 1 bp adverse fill slippage; not a reconstruction of historical fee tiers",
            "mark_assumption": "last regular-trade price proxy; intrabar adverse extreme for conservative margin checks; exchange mark/index unavailable",
            "funding_order": "settlement before orders at the same minute boundary, using previous observed trade mark",
            "missing_minutes": "stale observation with zero volume; no fills until a regular-trade bar; preserve holdings and funding",
            "selection": "mean validation net log growth across seeds and three years at one common checkpoint budget; all seeds retained",
            "ensemble": "equal probability mixture conditioned on one actual portfolio",
            "final_stage": "retrain all five seeds through 2025 at selected budget; never evaluate 2026"}
