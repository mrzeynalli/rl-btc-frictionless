"""Show v5 queue progress and a measured, provisional runtime estimate."""
from pathlib import Path
from datetime import datetime, timezone
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]/'results/perpetual_v5'
if not (ROOT/'status.json').exists():
    print('V5 status has not been written yet.')
    raise SystemExit(0)
status = json.loads((ROOT/'status.json').read_text())
queue = json.loads((ROOT/'queue.json').read_text())
print(f"Phase: {status['phase']} | completed {status['completed_jobs']}/{status['total_jobs']} | test LOCKED")
print(f"Status updated: {status['updated_at']}")
speed_by_cell = {'gru': [], 'lstm': []}
for job in status['jobs']:
    summary = job.get('last_training_summary')
    if summary and job['updates_completed'] >= 3:
        speed_by_cell[job['variant'].split('_')[0]].append(summary['update_seconds'])
    if 'pid' in job:
        flat = f"{100*summary['sampled_action_share'][1]:.2f}%" if summary else 'unknown'
        print(f"  {job['variant']} | {job['year']} seed {job['seed']} | {job['updates_completed']}/{queue['updates_per_job']} | sampled flat {flat} | pid {job['pid']}")
all_speeds = speed_by_cell['gru']+speed_by_cell['lstm']
if all_speeds:
    remaining_cpu = 0.
    for job in status['jobs']:
        speeds = speed_by_cell[job['variant'].split('_')[0]] or all_speeds
        remaining_cpu += max(0, queue['updates_per_job']-job['updates_completed'])*statistics.median(speeds)
    days = remaining_cpu/queue['workers']/86400
    print(f"Provisional training-only time remaining: {days:.1f} days; validation/ensemble overhead and future host load are additional.")
if status.get('error'):
    print(f"Failure: {status['error']}")
