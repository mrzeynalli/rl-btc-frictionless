"""Audit completed v4 logs and development trajectories; no test data access."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
RESULTS = MODEL / 'results/perpetual_v4'


def save(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    rows, curves, runs = [], [], []
    for path in sorted((RESULTS/'runs').glob('*/*/training.json')):
        record = json.loads(path.read_text())
        spec = record['specification']
        fold, seed = spec['fold'], spec['seed']
        current = [json.loads(x) | {'fold': fold, 'seed': seed} for x in path.with_name('updates.jsonl').read_text().splitlines()]
        assert len(current) == spec['config']['updates'] and record['status'] == 'completed'
        for x in current:
            x['p_short'], x['p_flat'], x['p_long'] = x.pop('sampled_action_share')
        rows.extend(current)
        curves.extend([x | {'fold': fold, 'seed': seed} for x in record['validation_curve']])
        def collapse(threshold):
            for i in range(len(current)-9):
                if all(x['p_flat'] >= threshold for x in current[i:i+10]):
                    return current[i]['update']
        runs.append({'fold': fold, 'seed': seed, 'updates': len(current), 'started_at': record['started_at'],
                     'finished_at': record['finished_at'], 'collapse_99pct_10_updates': collapse(.99),
                     'collapse_999pct_10_updates': collapse(.999), 'source_sha256': spec['source_sha256']})
    df = pd.DataFrame(rows)
    df.to_csv(HERE/'all_updates.csv', index=False)
    pd.json_normalize(curves).to_csv(HERE/'validation_checkpoints.csv', index=False)
    save('runs.json', runs)
    numeric = ['p_flat', 'entropy', 'turnover_per_decision', 'mean_reward', 'explained_variance_pre',
               'explained_variance_post', 'target_std', 'prediction_std', 'critic_rmse_post',
               'actor_gradient_norm', 'critic_gradient_norm', 'clip_fraction', 'post_update_kl']
    windows = []
    for population in ['development', 'final']:
        pop = df[df.fold.ne('final') if population == 'development' else df.fold.eq('final')]
        for lo, hi in [(1,5),(1,25),(26,50),(51,75),(76,100),(176,200),(276,300),(376,400)]:
            window = pop[pop['update'].between(lo,hi)]
            if window.empty:
                continue
            windows.append({'population': population, 'updates': [lo,hi], 'rows': len(window),
                            'mean': window[numeric].mean().to_dict(), 'median': window[numeric].median().to_dict(),
                            'nonflat_actions_per_rollout_mean': float(((1-window.p_flat)*window.active_steps).mean()),
                            'all_flat_rollout_share': float(window.p_flat.eq(1).mean())})
    selected = [x for x in curves if x['update']==300]
    summary = {'runs': len(runs), 'update_rows': len(df), 'development_updates': int(df.fold.ne('final').sum()),
               'sampled_decisions': int(df.active_steps.sum()), 'validation_checkpoints': len(curves),
               'zero_order_validations': sum(x['orders']==0 for x in curves),
               'selected_zero_order_runs': sum(x['orders']==0 for x in selected),
               'selected_positive_runs': sum(x['total_return_btc']>0 for x in selected),
               'selected_total_orders': sum(x['orders'] for x in selected),
               'selected_total_round_trips': sum(x['completed_round_trips'] for x in selected),
               'max_replay_ratio_error': float(df.replay_ratio_error.max()),
               'max_replay_value_error': float(df.replay_value_error.max()),
               'liquidations_in_training': int(df.liquidations_in_rollout.sum()),
               'kl_stopped_updates': int(df.kl_stopped.sum()),
               'actor_gradient_clipped_updates': int(df.actor_gradient_norm.gt(.5).sum()),
               'critic_gradient_clipped_updates': int(df.critic_gradient_norm.gt(.5).sum()),
               'note_gradient_count': 'Counts update-mean preclip norm above threshold, not individual minibatches',
               'selected_validation': selected, 'windows': windows,
               'test_locked': True, 'test_performance_accessed': False}
    save('summary.json', summary)
    trades=[]
    for item in selected:
        if not item['orders']:
            continue
        path = RESULTS/'runs'/item['fold']/f"seed_{item['seed']}"/'validation_0300.csv'
        frame = pd.read_csv(path, parse_dates=['time'])
        assert frame.time.max() < pd.Timestamp('2026-01-01T00:00:00Z')
        side = np.sign(frame.contracts.to_numpy())
        entry = None
        for i in np.flatnonzero(side != np.r_[0,side[:-1]]):
            if entry is not None:
                first = max(0,entry-1)
                before = 1.0 if entry==0 else frame.equity_btc.iloc[first]
                segment = frame.iloc[entry:i+1]
                trades.append({'fold':item['fold'],'seed':item['seed'],'entry':str(frame.time.iloc[entry]),
                               'exit':str(frame.time.iloc[i]), 'minutes':int((i-entry)*5),
                               'side':int(side[entry]),'net_return':float(frame.equity_btc.iloc[i]/before-1),
                               'log_return':float(np.log(frame.equity_btc.iloc[i]/before)),
                               'fees_btc':float(segment.fees_btc.sum()),'funding_btc':float(segment.funding_btc.sum())})
                entry=None
            if side[i]:
                entry=i
    save('selected_trades.json',trades)
    if trades:
        pd.DataFrame(trades).to_csv(HERE/'selected_trades.csv',index=False)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['windows','selected_validation']}))
    print('WINDOWS',json.dumps([{'range':w['updates'],'population':w['population'],'flat':w['mean']['p_flat'],
                               'entropy':w['mean']['entropy'],'ev':w['mean']['explained_variance_pre'],
                               'nonflat':w['nonflat_actions_per_rollout_mean']} for w in windows]))
    print('COLLAPSE',json.dumps(runs))
    print('TRADES',json.dumps(trades))


if __name__=='__main__':
    main()
