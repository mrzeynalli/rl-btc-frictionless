"""Pre-registered redundancy rule for the v2 market state (results/agent_v2/PREREGISTRATION.md §2).

On the first fold's training span (2017-01-01 to 2024-01-01), sampled hourly, walk the candidates
in precedence order and keep one only if its Spearman |rho| with every feature already kept is
below 0.8. No relation to future returns is used. Since 2026-09-15 the data are BitMEX alone and
every candidate is eligible (PREREGISTRATION.md, addendum).

    PYTHONPATH=src .venv/bin/python scripts/feature_redundancy.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent.config import OUTPUT, TRAIN_START, write_json
from agent.data import CANDIDATE_FEATURES, prepare

THRESHOLD = 0.8
END = pd.Timestamp('2024-01-01', tz='UTC')


def main():
    market = prepare()
    removed = set()
    first, last = market.bounds(TRAIN_START, END)
    rows = np.asarray(market.features[first:last:60], dtype=np.float64)
    ranks = pd.DataFrame(rows, columns=CANDIDATE_FEATURES).rank()
    rho = ranks.corr().to_numpy()
    kept, decisions = [], []
    for i, name in enumerate(CANDIDATE_FEATURES):
        if name in removed:
            decisions.append({'feature': name, 'kept': False, 'reason': 'failed the proxy audit'})
            continue
        clashes = {CANDIDATE_FEATURES[j]: round(float(rho[i, j]), 3) for j in kept if abs(rho[i, j]) >= THRESHOLD}
        if clashes:
            decisions.append({'feature': name, 'kept': False, 'reason': 'redundant', 'with': clashes})
        else:
            kept.append(i)
            decisions.append({'feature': name, 'kept': True,
                              'max_abs_rho_with_kept': round(max([abs(float(rho[i, j])) for j in kept if j != i] or [0.0]), 3)})
    matrix = pd.DataFrame(rho, index=CANDIDATE_FEATURES, columns=CANDIDATE_FEATURES).round(3)
    report = {'span': [str(TRAIN_START), str(END)], 'hours': int(len(rows)), 'threshold': THRESHOLD,
              'decisions': decisions, 'selected': [CANDIDATE_FEATURES[i] for i in kept],
              'spearman': matrix.to_dict()}
    write_json(OUTPUT / 'audit/feature_redundancy.json', report)
    for item in decisions:
        print(item)
    print('selected:', report['selected'])
    print(matrix.loc[[CANDIDATE_FEATURES[i] for i in kept], [CANDIDATE_FEATURES[i] for i in kept]].to_string())


if __name__ == '__main__':
    main()
