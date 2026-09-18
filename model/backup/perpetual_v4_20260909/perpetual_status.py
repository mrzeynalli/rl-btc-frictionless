"""Read progress without loading market data or evaluating any model."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results/perpetual_v4"
path = root / "status.json"
if not path.exists():
    print("Coordinator has not published status yet.")
else:
    status = json.loads(path.read_text())
    print(f"Phase: {status['phase']}; completed jobs: {status['completed_jobs']}/{status['total_jobs']}")
    print(f"Test locked: {status['test_locked']} (2026-01-01 onward)")
    for job in status["jobs"]:
        if job["status"] != "pending":
            record = root / "runs" / job["fold"] / f"seed_{job['seed']}" / "training.json"
            current = json.loads(record.read_text())
            summary = current.get("last_training_summary", {})
            print(f"  {job['fold']} seed {job['seed']}: {current['status']}, "
                  f"{current['updates_completed']}/{job['planned_updates']} updates; "
                  f"reward={summary.get('mean_reward')}, EV={summary.get('explained_variance_pre')}")
    selection = root / "selection.json"
    if selection.exists():
        print(f"Selected budget: {json.loads(selection.read_text())['selected_updates']} updates")
