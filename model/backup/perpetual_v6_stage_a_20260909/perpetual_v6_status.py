"""Stage A progress and its success criteria; reads files only, no evaluation."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]/'results/perpetual_v6'
if not (ROOT/'status.json').exists():
    print('V6 status has not been written yet.')
    raise SystemExit(0)
status = json.loads((ROOT/'status.json').read_text())
queue = json.loads((ROOT/'queue.json').read_text())
protocol = json.loads((ROOT/'protocol.json').read_text())

print(f"Phase: {status['phase']} | completed {status['completed_jobs']}/{status['total_jobs']} | test LOCKED")
print(f"Credit window: {protocol['credit_window_hours']:.1f} h against a "
      f"{protocol['cost_break_even_hours']:.0f} h cost break-even")
print(f"Status updated: {status['updated_at']}")

speeds, holding, information, running = [], [], [], []
for job in status['jobs']:
    summary = job.get('last_training_summary')
    if not summary:
        continue
    if job['updates_completed'] >= 3:
        speeds.append(summary['update_seconds'])
    if job['updates_completed'] >= 50:
        holding.append(summary.get('mean_open_position_age_minutes', 0)/60)
        information.append(summary.get('state_action_mutual_information_flat', 0))
    if 'pid' in job:
        running.append((job, summary))

if running:
    print('\nRunning:')
    for job, summary in running:
        print(f"  {job['year']} seed {job['seed']:>3} | {job['updates_completed']:>3}/{queue['updates_per_job']}"
              f" | hold {summary.get('mean_open_position_age_minutes', 0)/60:6.1f} h"
              f" | I(S;A|flat) {summary.get('state_action_mutual_information_flat', 0):.4f}"
              f" | switch sd {summary.get('switch_probability_sd', 0):.5f} | pid {job['pid']}")

if holding:
    median_hold = statistics.median(holding)
    median_information = statistics.median(information)
    in_band = sum(12 <= h <= 48 for h in holding)
    print(f"\nStage A criteria over {len(holding)} run(s) past update 50:")
    print(f"  median holding time      {median_hold:6.1f} h   (target 12-48 h; v5 collapsed to months)")
    print(f"  runs inside the band     {in_band}/{len(holding)}")
    print(f"  median I(S;A | flat)     {median_information:.4f}   (target > 0; exactly 0 means the entry"
          " rule ignores the market)")
    print("  Validation return is NOT the stage A criterion.")

if speeds:
    remaining = sum(max(0, queue['updates_per_job']-j['updates_completed']) for j in status['jobs'])
    hours = remaining*statistics.median(speeds)/queue['workers']/3600
    print(f"\nProvisional training-only time remaining: {hours:.1f} h "
          f"at {statistics.median(speeds):.1f} s/update; validation overhead is additional.")
if status.get('error'):
    print(f"Failure: {status['error']}")
