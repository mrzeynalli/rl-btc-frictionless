"""Summarise smoke runs: references, validation curve, regime conditioning, holding diagnostics, verdict.

    .venv/bin/python scripts/smoke_summary.py [results/smoke]
"""
from pathlib import Path
import json
import sys


def number(value, pattern):
    return '-' if value is None else format(value, pattern)


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'results/smoke'
for run in sorted(p for p in root.iterdir() if (p / 'setup.json').exists()):
    setup = json.loads((run / 'setup.json').read_text())
    references = setup['references']
    print(f"\n=== {run.name} | pass above {setup['pass_if_last_three_mean_above']:+.3f} | oracle {references['oracle']:+.3f}"
          f" | trend rule {references['trend_rule']:+.3f} | long {references['constant_long']:+.3f}"
          f" | short {references['constant_short']:+.3f}")
    log = run / 'updates.jsonl'
    updates = [json.loads(line) for line in log.read_text().splitlines() if line] if log.exists() else []
    for path in sorted(run.glob('validation_*.json')):
        v = json.loads(path.read_text())
        u = updates[v['update'] - 1] if len(updates) >= v['update'] else {}
        print(f"  @{v['update']:>4}: log {v['total_log_return_btc_mean']:+.3f} | orders {v['orders']:6.0f}"
              f" | held spell {v['median_holding_spell_hours']:5.1f} h"
              f" | long up/down {v['long_when_up']:.2f}/{v['long_when_down']:.2f}"
              f" | short up/down {v['short_when_up']:.2f}/{v['short_when_down']:.2f}"
              f" || train entropy {number(u.get('policy_entropy'), '.3f')}"
              f" implied hold {number(u.get('policy_implied_holding_hours'), '.1f')} h"
              f" exit age {number(u.get('mean_age_at_exit_hours'), '.1f')} h"
              f" MI {number(u.get('state_action_mutual_information'), '.3f')}")
    verdict_path = run / 'verdict.json'
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text())
        print(f"  VERDICT {'PASS' if verdict['passed'] else 'FAIL'}: last three {verdict['last_three_mean_log_return']:+.3f}"
              f" against {verdict['criterion']:+.3f} | beats constants {verdict['beats_best_constant']}"
              f" | regime conditioned {verdict['regime_conditioned']}"
              f" | share of trend rule {number(verdict['share_of_trend_rule'], '.2f')} | {verdict['hours']:.2f} h")
    else:
        print(f"  running: {len(updates)} updates logged")
