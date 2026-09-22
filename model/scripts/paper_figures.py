"""Figures, benchmark numbers and tables for the paper (paper/figures, paper/tables).

Final agent: cfg_016, trained 2017-08 -> 2024, tested on one continuous account from 1 January 2025 to 1 August 2026
(results/simple_agent_final, results/simple_agent_hpo_multiyear). All accounts are cost-free and start at $10,000.
Returns are also split by calendar year.

Benchmarks, all on the same hourly account as the agent, with daily signals taken at 00:00 UTC from daily closes:
  buy-and-hold; 50- and 200-day moving average (long above, short below; Brock, Lakonishok & LeBaron 1992);
  time-series momentum (sign of the past 12-month return, rebalanced monthly; Moskowitz, Ooi & Pedersen 2012);
  MACD(12, 26, 9) crossover (Appel 2005); RSI(14) mean reversion (long below 30, short above 70; Wilder 1978);
  a uniformly random policy (median and 5-95% range over 1,000 paths).

    PYTHONPATH=src .venv/bin/python scripts/paper_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simple_agent.config import SEEDS, TEST
from simple_agent.data import load, next_returns, rows
from simple_agent.evaluate import POSITIONS, score

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / 'paper'
FIGURES, TABLES = PAPER / 'figures', PAPER / 'tables'
FINAL = ROOT / 'results/simple_agent_final'
SEARCH = ROOT / 'results/simple_agent_hpo_multiyear'

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight'})
SEED_COLOURS = ['#1f4e79', '#2e75b6', '#5b9bd5', '#9dc3e6', '#0b2e4f']


# --------------------------------------------------------------------------------------------------------------------
# Rule-based benchmarks on the test period
# --------------------------------------------------------------------------------------------------------------------
def daily_signal_positions(table, test_rows, rule):
    """Hourly positions for a rule evaluated on daily closes known at each 00:00 UTC and held until the next one."""
    daily = table.set_index('time').close
    daily = daily[daily.index.hour == 0]                       # the close of each completed day, known at midnight
    signal = rule(daily)                                       # position per midnight, in {-1, 0, 1}
    times = table.time.iloc[test_rows]
    return signal.reindex(times, method='ffill').fillna(0.0).to_numpy()


def moving_average(days):
    return lambda close: np.sign(close - close.rolling(days).mean())


def time_series_momentum(close):
    """Sign of the trailing 12-month return, fixed at each month's first midnight and held for the month."""
    raw = np.sign(close / close.shift(365) - 1)
    first_of_month = close.index.to_series().dt.day == 1
    return raw.where(first_of_month).ffill()


def macd(close):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return np.sign(line - line.ewm(span=9, adjust=False).mean())


def rsi(close):
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()      # Wilder's smoothing
    loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    value = 100 - 100 / (1 + gain / loss)
    return pd.Series(np.where(value < 30, 1.0, np.where(value > 70, -1.0, 0.0)), index=close.index)


RULES = {'Buy and hold': None, '50 day MA': moving_average(50), '200 day MA': moving_average(200),
         'Time series momentum': time_series_momentum, 'MACD': macd, 'RSI': rsi}




def calendar_years(times):
    """Hour masks of the test window for the two calendar years (the test is evaluated chronologically)."""
    return {'2025': (times < '2026-01-01').to_numpy(), '2026': (times >= '2026-01-01').to_numpy()}


def phase_returns(R, times, masks):
    """Compounded return in percent over the hours of each mask (one continuous account, restricted to those hours)."""
    return {name: float(np.expm1(np.log1p(R[mask]).sum()) * 100) for name, mask in masks.items()}


def win_rate(R, positions):
    """Share of trades that made money; a trade is a maximal run of hours with the same non-flat position."""
    change = np.flatnonzero(np.diff(positions) != 0) + 1
    wins = [np.prod(1 + R[a:b]) > 1 for a, b in zip(np.r_[0, change], np.r_[change, len(R)]) if positions[a] != 0]
    return float(np.mean(wins) * 100) if wins else float('nan')


def benchmark_scores():
    table = load(end=TEST[1])
    returns = next_returns(table)
    test_rows = rows(table, TEST)[:-1]
    r = returns[test_rows]
    times = table.time.iloc[test_rows].reset_index(drop=True)
    bounds = calendar_years(times)
    out, curves, by_phase = {}, {}, {}
    for name, rule in RULES.items():
        positions = np.ones(len(test_rows)) if rule is None else daily_signal_positions(table, test_rows, rule)
        out[name] = score(positions * r, positions)
        out[name]['win_rate_pct'] = win_rate(positions * r, positions)
        curves[name] = 10_000 * np.cumprod(1 + positions * r)
        by_phase[name] = phase_returns(positions * r, times, bounds)
    rng = np.random.default_rng(0)
    random_scores = [{**score(p * r, p), 'win_rate_pct': win_rate(p * r, p)}
                     for p in (POSITIONS[rng.integers(0, 3, len(r))] for _ in range(1000))]
    out['Random (median)'] = {k: float(np.median([s[k] for s in random_scores]))
                              for k in ('total_return_pct', 'equity_from_10k', 'sharpe', 'max_drawdown',
                                        'position_changes', 'win_rate_pct')}
    returns_random = np.array([s['total_return_pct'] for s in random_scores])
    out['Random (median)']['p05_return_pct'] = float(np.percentile(returns_random, 5))
    out['Random (median)']['p95_return_pct'] = float(np.percentile(returns_random, 95))
    return out, curves, times, table, r, bounds, by_phase, returns_random


def agent_analysis(r, times, bounds, returns_random):
    """Per seed: phase returns, the share of random paths beaten, and the timing test (own positions shuffled)."""
    equity = pd.read_csv(FINAL / 'equity_test.csv', parse_dates=['time'])
    rng = np.random.default_rng(0)
    out = {}
    for seed in SEEDS:
        positions = equity[f'position_{seed}'].to_numpy()
        R = positions * r
        total = float(np.expm1(np.log1p(R).sum()) * 100)
        shuffled = np.array([np.expm1(np.log1p(rng.permutation(positions) * r).sum()) * 100 for _ in range(2000)])
        out[seed] = {'phases': phase_returns(R, times, bounds), 'return_pct': total,
                     'beats_random_pct': float((returns_random < total).mean() * 100),
                     'beats_shuffled_pct': float((shuffled < total).mean() * 100),
                     'shuffled_median_pct': float(np.median(shuffled)),
                     'long_hours_btc_log': float(np.log1p(r[positions > 0]).sum()),
                     'short_hours_btc_log': float(np.log1p(r[positions < 0]).sum()),
                     'hit_rate': float((R[positions != 0] > 0).mean()),
                     'win_rate_pct': win_rate(R, positions)}
    mean_return = float(np.mean([out[s]['return_pct'] for s in SEEDS]))
    out['mean'] = {'phases': {k: float(np.mean([out[s]['phases'][k] for s in SEEDS])) for k in bounds},
                   'return_pct': mean_return, 'beats_random_pct': float((returns_random < mean_return).mean() * 100)}
    return out


# --------------------------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------------------------
def figure_equity(curves, times):
    """Equity of the PPO agent (the mean over five random initializations) against every rule based benchmark."""
    equity = pd.read_csv(FINAL / 'equity_test.csv', parse_dates=['time'])
    agent = equity[[f'seed_{s}' for s in SEEDS]]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(equity.time, agent.mean(axis=1), color='black', lw=1.8, label='PPO agent')
    styles = {'Buy and hold': ('#c00000', '-'), '50 day MA': ('#ed7d31', '-'),
              '200 day MA': ('#7f7f7f', '--'), 'Time series momentum': ('#2e75b6', ':'),
              'MACD': ('#70ad47', '-'), 'RSI': ('#7030a0', '-')}
    labels = {'50 day MA': '50 day moving average', '200 day MA': '200 day moving average'}
    for name, (colour, style) in styles.items():
        ax.plot(times, curves[name], color=colour, lw=1.1, ls=style, label=labels.get(name, name))
    ax.axhline(10_000, color='#bfbfbf', lw=0.6)
    ax.set_ylabel('Equity (USD)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=8)
    fig.savefig(FIGURES / 'equity_test.pdf')
    plt.close(fig)


def figure_test_price(table, times):
    """Hourly closing price of the XBTUSD perpetual over the test window, with the all time high marked."""
    close = table.set_index('time').close.reindex(times)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(times, close.to_numpy(), color='#1f4e79', lw=0.8)
    high = close.idxmax()
    ax.annotate(f'All time high\n{close[high]:,.0f} USD', xy=(high, close[high]), xytext=(25, -25),
                textcoords='offset points', fontsize=8, arrowprops={'arrowstyle': '->', 'lw': 0.6})
    ax.set_ylim(50_000, 133_000)
    ax.set_ylabel('XBTUSD price (USD)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b\n%Y'))
    fig.savefig(FIGURES / 'price_test.pdf')
    plt.close(fig)


def figure_walk_forward():
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    start = pd.Timestamp('2017-08-12')
    rows_ = [('Tuning fold 1', 2021), ('Tuning fold 2', 2022), ('Tuning fold 3', 2023), ('Tuning fold 4', 2024)]
    labels = []
    for i, (label, validation) in enumerate(rows_):
        y = len(rows_) - i + 0.5
        train_end = pd.Timestamp(f'{validation}-01-01')
        ax.barh(y, (train_end - start).days, left=start, height=0.6, color='#bfbfbf')
        ax.barh(y, 365, left=train_end, height=0.6, color='#2e75b6')
        labels.append((y, label))
    y = 0.5
    ax.barh(y, (pd.Timestamp('2025-01-01') - start).days, left=start, height=0.6, color='#bfbfbf', label='Training')
    ax.barh(y, 0, left=pd.Timestamp('2025-01-01'), height=0.6, color='#2e75b6', label='Validation (hyperparameter search)')
    ax.barh(y, (pd.Timestamp('2026-08-01') - pd.Timestamp('2025-01-01')).days, left=pd.Timestamp('2025-01-01'),
            height=0.6, color='#c00000', label='Held out test')
    labels.append((y, 'Final agent'))
    ax.set_yticks([l[0] for l in labels], [l[1] for l in labels])
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp('2017-06-01'), pd.Timestamp('2026-10-01'))
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.22))
    ax.tick_params(axis='y', length=0)
    fig.savefig(FIGURES / 'walk_forward.pdf')
    plt.close(fig)


def figure_training():
    runs = SEARCH / 'final/cfg_016/test_2026'
    logs = {s: pd.read_json(runs / f'seed_{s}' / 'updates.jsonl', lines=True) for s in SEEDS}
    panels = [('mean_reward', 'Mean reward per step'), ('entropy', 'Policy entropy (nats)'),
              ('value_loss', 'Critic loss'), ('explained_variance', 'Explained variance of returns')]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels):
        smooth = pd.DataFrame({s: log[key].rolling(25, min_periods=1).mean() for s, log in logs.items()})
        passes = logs[SEEDS[0]]['update'] / logs[SEEDS[0]]['update'].max() * 10
        ax.plot(passes, smooth.mean(axis=1), color='#1f4e79', lw=1.3)
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel('Training passes')
    fig.tight_layout()
    fig.savefig(FIGURES / 'training_curves.pdf')
    plt.close(fig)


def figure_tuning():
    rung1 = json.loads((SEARCH / 'rung1.json').read_text())
    rung2 = json.loads((SEARCH / 'rung2.json').read_text())
    rung3 = json.loads((SEARCH / 'rung3.json').read_text())
    winner = json.loads((SEARCH / 'winner.json').read_text())['config']
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={'width_ratios': [1, 1.35]})
    rng = np.random.default_rng(1)
    for x, rung, promoted in ((0, rung1, {r['config'] for r in rung2}), (1, rung2, {r['config'] for r in rung3}),
                              (2, rung3, {winner})):
        for r in rung:
            colour = '#c00000' if r['config'] == winner else ('#1f4e79' if r['config'] in promoted else '#9dc3e6')
            left.scatter(x + rng.uniform(-0.18, 0.18), r['mean_sharpe'], s=14, color=colour, edgecolor='none')
    left.set_xticks([0, 1, 2], ['Rung 1\n51 configs', 'Rung 2\n12 configs', 'Rung 3\n6 configs'])
    left.set_ylabel('Mean validation Sharpe ratio')
    left.axhline(0, color='black', lw=0.5)
    left.scatter([], [], s=14, color='#1f4e79', label='Promoted')
    left.scatter([], [], s=14, color='#c00000', label='Selected')
    left.legend(frameon=False, loc='lower left')
    order = [r['config'] for r in rung3]
    years = ['2021', '2022', '2023', '2024']
    width = 0.8 / len(order)
    for i, r in enumerate(rung3):
        values = [r['per_fold'][str(int(y) + 1)]['sharpe'] for y in years]
        name = 'initial' if r['config'] == 'current' else r['config'].replace('cfg_', 'C')
        right.bar(np.arange(4) + (i - (len(order) - 1) / 2) * width, values, width=width,
                  color='#c00000' if r['config'] == winner else ('#bfbfbf' if r['config'] == 'current'
                                                                  else SEED_COLOURS[i % 5]), label=name)
    right.set_xticks(np.arange(4), years)
    right.set_xlabel('Validation year')
    right.axhline(0, color='black', lw=0.5)
    right.legend(frameon=False, ncol=3, fontsize=7.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIGURES / 'tuning.pdf')
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    scores, curves, times, table, r, bounds, by_phase, returns_random = benchmark_scores()
    final = json.loads((FINAL / 'final_report.json').read_text())
    agent = {f'Agent, seed {s}': final['seeds'][str(s)]['test'] for s in SEEDS}
    agent['Agent, mean of five seeds'] = {k: float(np.mean([a[k] for a in agent.values()]))
                                          for k in ('total_return_pct', 'equity_from_10k', 'sharpe', 'max_drawdown',
                                                    'position_changes', 'long_share', 'short_share', 'flat_share')}
    analysis = agent_analysis(r, times, bounds, returns_random)
    for s in SEEDS:
        agent[f'Agent, seed {s}']['win_rate_pct'] = analysis[s]['win_rate_pct']
    agent['Agent, mean of five seeds']['win_rate_pct'] = float(np.mean([analysis[s]['win_rate_pct'] for s in SEEDS]))
    figure_equity(curves, times)
    figure_test_price(table, times)
    figure_walk_forward()
    figure_training()
    figure_tuning()
    close = table.set_index('time').close.reindex(times)
    describe = {'hours': {k: int(v.sum()) for k, v in bounds.items()},
                'btc_return_pct': phase_returns(r, times, bounds)}
    (TABLES / 'test.json').write_text(json.dumps({'agent': agent, 'benchmarks': scores, 'analysis': analysis,
                                                   'benchmark_phases': by_phase, 'phases': describe},
                                                  indent=1, default=str))
    for name, s in {**agent, **scores}.items():
        print(f"{name:28} {s['total_return_pct']:+7.1f}%  ${s['equity_from_10k']:>9,.0f}  Sharpe {s['sharpe']:+.2f}  "
              f"maxDD {s.get('max_drawdown', float('nan')) * 100:6.1f}%  changes {s.get('position_changes', '')}  "
              f"win {s.get('win_rate_pct', float('nan')):.1f}%")
    print(json.dumps(describe, indent=1))
    for k, v in analysis.items():
        print(k, {x: (round(y, 1) if isinstance(y, float) else {a: round(b, 1) for a, b in y.items()})
                  for x, y in v.items()})
    for k, v in by_phase.items():
        print(k, {a: round(b, 1) for a, b in v.items()})


if __name__ == '__main__':
    main()
