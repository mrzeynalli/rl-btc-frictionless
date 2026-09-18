"""Summarise the pre-registered verification runs: curves by pass, the selected checkpoint, the verdict.

    .venv/bin/python scripts/verification_summary.py [results/verification]
"""
from pathlib import Path
import json
import sys


def number(value, pattern):
    return '-' if value is None else format(value, pattern)


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'results/verification'
for run in sorted(p for p in root.iterdir() if (p / 'setup.json').exists()):
    setup = json.loads((run / 'setup.json').read_text())
    test, validation = setup['references']['test'], setup['references']['validation']
    print(f"\n=== {run.name} | {setup['passes']} passes, checkpoint every {setup['evaluate_passes']} | "
          f"{setup['updates']} updates over {setup['training_decisions']} decisions")
    print(f"  validation window: oracle {validation['oracle']:+.3f} rule {validation['trend_rule']:+.3f} "
          f"long {validation['constant_long']:+.3f} short {validation['constant_short']:+.3f}")
    print(f"  test window:       oracle {test['oracle']:+.3f} rule {test['trend_rule']:+.3f} "
          f"long {test['constant_long']:+.3f} short {test['constant_short']:+.3f} | pass above "
          f"{setup['pass_if_selected_test_return_above']:+.3f}")
    book = run / 'checkpoints.jsonl'
    for line in (book.read_text().splitlines() if book.exists() else []):
        e = json.loads(line)
        v, t = e['validation'], e['test']
        print(f"  pass {e['passes']:5.1f} @{e['update']:>5}: validation {v['total_log_return_btc_mean']:+.3f} | test "
              f"{t['total_log_return_btc_mean']:+.3f} ({t['orders']:.0f} orders, {t['median_holding_spell_hours']:.1f} h) "
              f"| test long up/down {t['long_when_up']:.2f}/{t['long_when_down']:.2f} short down/up "
              f"{t['short_when_down']:.2f}/{t['short_when_up']:.2f}")
    verdict_path = run / 'verdict.json'
    if verdict_path.exists():
        d = json.loads(verdict_path.read_text())
        print(f"  VERDICT {'PASS' if d['passed'] else 'FAIL'}: selected pass {d['selected_passes']:.1f} | test "
              f"{d['selected_test_return']:+.3f} vs {d['criterion']:+.3f} | beats constants {d['beats_best_constant']} "
              f"| conditioned {d['regime_conditioned']} | share of trend rule {number(d['share_of_trend_rule'], '.2f')} "
              f"| final checkpoint test {d['final_checkpoint_test_return']:+.3f} | {d['hours']:.2f} h")
