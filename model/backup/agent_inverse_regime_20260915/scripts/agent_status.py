"""Queue progress and the diagnostics that decide whether learning is happening.

    .venv/bin/python scripts/agent_status.py [experiment directory ...]

Without arguments: every experiment under results/agent/ that has written a status.
"""
from pathlib import Path
import json
import math
import statistics
import sys

BASE = Path(__file__).resolve().parents[1]/'results/agent'


def fmt(value, pattern):
    return '-' if value is None else format(value, pattern)


def report(root):
    status = json.loads((root/'status.json').read_text())
    queue = json.loads((root/'queue.json').read_text())
    spec = json.loads((root/'protocol.json').read_text())
    print(f"\n=== {root.name}: reward {spec['config']['reward']}, {spec['decision_hours'] * 60:.0f}-minute decisions ===")
    print(f"Phase: {status['phase']} | completed {status['completed_jobs']}/{status['total_jobs']} | test LOCKED")
    folds = queue.get('folds', {})
    print(f"Credit window {spec['credit_window_hours']:.1f} h | cost break-even by fold "
          + ', '.join(f"{year} {fold['cost_break_even_hours']:.1f} h" for year, fold in folds.items())
          + f" | round trip {spec['round_trip_cost_bp']:.0f} bp, fixed | {spec['budget']}")
    speeds, running = [], []
    for job in status['jobs']:
        summary = job.get('last_training_summary')
        if not summary:
            continue
        if job['updates_completed'] >= 3:
            speeds.append(summary['update_seconds'])
        if 'pid' in job:
            running.append((job, summary))
    if running:
        print('Running (side shares short/flat/long in the last rollout):')
        for job, s in running:
            print(f"  {job['year']} seed {job['seed']:>3} | {job['updates_completed']:>5}/{folds[str(job['year'])]['updates']}"
                  f" | H {fmt(s.get('policy_entropy'), '.3f')}"
                  f" | side {'/'.join(f'{x:.2f}' for x in s['account_side_share'])}"
                  f" | long entries {fmt(s.get('long_entry_share'), '.2f')}"
                  f" | hold {fmt(s.get('policy_implied_holding_hours'), '6.1f')} h"
                  f" | exit age {fmt(s.get('mean_age_at_exit_hours'), '6.1f')} h"
                  f" | MI {s.get('state_action_mutual_information', 0):.4f}"
                  f" | EV {fmt(s.get('explained_variance_post'), '.2f')}"
                  f" | liq {s.get('liquidations_in_rollout', 0)} | pid {job['pid']}")
    for job in status['jobs']:
        v = job.get('last_validation')
        if not v:
            continue
        quarters = ' '.join(f"Q{q['quarter']} {q['total_log_return_btc_mean']:+.3f}" for q in v['quarters'])
        print(f"Validation {job['year']} seed {job['seed']} @ {v['update']}: {quarters}"
              f" | long {v['long_share']:.2f} short {v['short_share']:.2f} | liquidations {v['liquidations']}")
    if speeds:
        per_update = statistics.median(speeds)
        unfinished = [j for j in status['jobs'] if j['status'] != 'completed']
        waves = math.ceil(len(unfinished)/queue['workers']) if unfinished else 0
        slowest = max((folds[str(j['year'])]['updates']-j['updates_completed']) for j in unfinished) if unfinished else 0
        print(f"Provisional training-only time remaining: {waves*slowest*per_update/3600:.1f} h "
              f"({waves} wave(s) of {queue['workers']} at {per_update:.1f} s/update); validation is additional.")
    if status.get('error'):
        print(f"Failure: {status['error']}")


roots = [Path(p) for p in sys.argv[1:]] or sorted(p.parent for p in BASE.glob('*/status.json'))
if not roots:
    print('No experiment has written a status yet.')
for root in roots:
    report(root)
