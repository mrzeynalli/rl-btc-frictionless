"""Queue progress and the diagnostics that decide whether learning is happening."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]/'results/agent'
if not (ROOT/'status.json').exists():
    print('The agent has not written a status yet.')
    raise SystemExit(0)
status = json.loads((ROOT/'status.json').read_text())
queue = json.loads((ROOT/'queue.json').read_text())
spec = json.loads((ROOT/'protocol.json').read_text())

print(f"Phase: {status['phase']} | completed {status['completed_jobs']}/{status['total_jobs']} | test LOCKED")
print(f"Credit window {spec['credit_window_hours']:.1f} h vs {spec['cost_break_even_hours']:.0f} h break-even"
      f" | round trip {spec['round_trip_cost_bp']:.0f} bp, fixed")

speeds, alive, holding, information, gate = [], [], [], [], []
for job in status['jobs']:
    summary = job.get('last_training_summary')
    if not summary:
        continue
    if job['updates_completed'] >= 3:
        speeds.append(summary['update_seconds'])
    if job['updates_completed'] >= 50:
        holding.append(summary.get('policy_implied_holding_hours', 0))
        information.append(summary.get('state_action_mutual_information_flat', 0))
        gate.append(summary.get('gate_entropy', 0))
    if 'pid' in job:
        alive.append((job, summary))

if alive:
    print('\nRunning:')
    for job, s in alive:
        print(f"  {job['year']} seed {job['seed']:>3} | {job['updates_completed']:>3}/{queue['updates_per_job']}"
              f" | hold {s.get('policy_implied_holding_hours', 0):6.1f} h"
              f" | gateH {s.get('gate_entropy', 0):.4f} a {s.get('gate_entropy_alpha', 0):.5f}"
              f" | dirH {s.get('direction_entropy', 0):.3f}"
              f" | MIflat {s.get('state_action_mutual_information_flat', 0):.4f}"
              f" | EV {s.get('explained_variance_post') or 0:.2f} | pid {job['pid']}")

if holding:
    print(f"\nAcross {len(holding)} run(s) past update 50:")
    print(f"  median policy-implied holding   {statistics.median(holding):6.1f} h"
          f"  (break-even {spec['cost_break_even_hours']:.0f} h)")
    print(f"  median I(S;A | flat)            {statistics.median(information):.4f}"
          "   (0 means the entry rule ignores the market)")
    target = spec['gate_entropy_target']
    below = sum(g < target for g in gate)
    print(f"  median gate entropy             {statistics.median(gate):.4f}"
          f"   (target {target:.4f}; 0.003 is the level every collapse died at)")
    print(f"  runs below the gate target      {below}/{len(gate)}"
          "   (the floor is meant to be working, not idle)")

for job in status['jobs']:
    v = job.get('last_validation')
    if v:
        print(f"\nLatest validation ({job['year']} seed {job['seed']}), {len(v['seeds'])} sampled paths:")
        print(f"  log return  median {v['total_log_return_btc']:+.3f}"
              f"  range [{v['total_log_return_btc_min']:+.3f}, {v['total_log_return_btc_max']:+.3f}]"
              f"  positive-seed share {v['positive_seed_share']:.2f}")
        print(f"  orders {v['orders']:.0f} | holding {v['median_holding_spell_hours']:.1f} h"
              f" | single-bar flickers {v['single_bar_round_trips']:.0f}")
        break

if speeds:
    import math
    per_update = statistics.median(speeds)
    unfinished = [j for j in status['jobs'] if j['status'] != 'completed']
    waves = math.ceil(len(unfinished)/queue['workers'])
    slowest = max((queue['updates_per_job']-j['updates_completed']) for j in unfinished) if unfinished else 0
    hours = waves*slowest*per_update/3600
    print(f"\nProvisional training-only time remaining: {hours:.1f} h "
          f"({waves} wave(s) of {queue['workers']} at {per_update:.1f} s/update); "
          "validation is additional.")
if status.get('error'):
    print(f"Failure: {status['error']}")
