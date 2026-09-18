"""Apply the pre-registered gamma/lambda decision rule (results/agent_v2/PREREGISTRATION.md §3).

Cell score: IQM of quarterly log return over 6 checkpoints x 8 quarters (argmax evaluation).
Main effects: each gamma level's mean cell score over the three lambda levels, and vice versa.
A level replaces v1's value (0.9792 per hour) only if its mean score exceeds v1's by more than
one standard error of the paired per-quarter difference (8 quarters, each a mean over the level's
cells and checkpoints). Writes results/agent_v2/search/decision.json.

    PYTHONPATH=src .venv/bin/python scripts/search_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agent.config import OUTPUT, write_json
from agent.scoring import interquartile_mean

SEARCH = OUTPUT / 'search'
GAMMAS = ('0.9583', '0.9792', '0.9900')
LAMBDAS = ('0.90', '0.95', '0.9792')
V1 = {'gamma': '0.9792', 'lambda': '0.9792'}
FOLDS = (2022, 2023)
SEED = 11
CHECKPOINTS = 6


def cell_matrix(gamma, lam):
    """(checkpoints, quarters) of quarterly log return, plus behaviour diagnostics."""
    folder = SEARCH / f'gamma_{gamma}_lambda_{lam}'
    rows, orders, holds, long_share, short_share = [], [], [], [], []
    for year in FOLDS:
        record = json.loads((folder / f'runs/{year}/seed_{SEED}/training.json').read_text())
        if record['status'] != 'completed':
            raise RuntimeError(f'{folder.name} {year} is {record["status"]}')
        curve = sorted(record['validation_curve'], key=lambda v: v['update'])[:CHECKPOINTS]
        if len(curve) != CHECKPOINTS:
            raise RuntimeError(f'{folder.name} {year} has {len(curve)} checkpoints')
        rows.append([[q['total_log_return_mean'] for q in c['quarters']] for c in curve])
        for c in curve:
            orders.extend(q['orders'] for q in c['quarters'])
            holds.extend(q['median_holding_spell_hours'] for q in c['quarters'])
            long_share.append(c['long_share'])
            short_share.append(c['short_share'])
    matrix = np.concatenate(rows, axis=1)          # checkpoints x 8 quarters
    return matrix, {'median_orders_per_quarter': float(np.median(orders)),
                    'median_holding_hours': float(np.median(holds)),
                    'mean_long_share': float(np.mean(long_share)), 'mean_short_share': float(np.mean(short_share))}


def main():
    cells, table = {}, []
    for g in GAMMAS:
        for l in LAMBDAS:
            matrix, behaviour = cell_matrix(g, l)
            cells[(g, l)] = matrix
            table.append({'gamma_hour': float(g), 'lambda_hour': float(l),
                          'credit_window_hours': 1 / (1 - float(g) * float(l)),
                          'score_iqm': interquartile_mean(matrix), 'mean': float(matrix.mean()),
                          'positive_quarter_checkpoints': float((matrix > 0).mean()),
                          'quarter_means': matrix.mean(0).round(5).tolist(),
                          'checkpoint_means': matrix.mean(1).round(5).tolist(), **behaviour})
    score = {(row['gamma_hour'], row['lambda_hour']): row['score_iqm'] for row in table}

    def effect(axis, levels, v1):
        means, quarters = {}, {}
        for level in levels:
            members = [key for key in cells if key[axis] == level]
            means[level] = float(np.mean([score[(float(k[0]), float(k[1]))] for k in members]))
            quarters[level] = np.mean([cells[k].mean(0) for k in members], axis=0)
        best = max(means, key=means.get)
        difference = quarters[best] - quarters[v1]
        standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference))) if best != v1 else 0.0
        gain = means[best] - means[v1]
        replace_v1 = best != v1 and gain > standard_error
        return {'level_means': means, 'best': best, 'gain_over_v1': gain,
                'standard_error_of_paired_quarter_difference': standard_error,
                'replaces_v1': bool(replace_v1), 'chosen': best if replace_v1 else v1}

    gamma = effect(0, GAMMAS, V1['gamma'])
    lam = effect(1, LAMBDAS, V1['lambda'])
    decision = {'rule': 'results/agent_v2/PREREGISTRATION.md §3', 'cells': table, 'gamma': gamma, 'lambda': lam,
                'chosen_per_hour': {'gamma': float(gamma['chosen']), 'lambda': float(lam['chosen'])},
                'chosen_per_bar': {'gamma': float(gamma['chosen']) ** (1 / 12), 'lambda': float(lam['chosen']) ** (1 / 12)}}
    write_json(SEARCH / 'decision.json', decision)
    print(f"{'gamma':>7} {'lambda':>7} {'window h':>9} {'IQM':>9} {'mean':>9} {'pos':>5} {'orders/q':>9} {'hold h':>7} {'long':>5} {'short':>5}")
    for row in table:
        print(f"{row['gamma_hour']:>7.4f} {row['lambda_hour']:>7.4f} {row['credit_window_hours']:>9.1f} {row['score_iqm']:>+9.4f} "
              f"{row['mean']:>+9.4f} {row['positive_quarter_checkpoints']:>5.2f} {row['median_orders_per_quarter']:>9.0f} "
              f"{row['median_holding_hours']:>7.1f} {row['mean_long_share']:>5.2f} {row['mean_short_share']:>5.2f}")
    print(json.dumps({'gamma': gamma, 'lambda': lam, 'chosen_per_hour': decision['chosen_per_hour']}, indent=1))


if __name__ == '__main__':
    main()
