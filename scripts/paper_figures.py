"""Figures and tables for the paper (report/figures, report/tables) from the nested walk-forward of the PPO agent.

Test: 1 January 2021 -> 1 August 2026, each year traded by models trained on the data before it
(agents/ppo/results/walkforward, written by scripts/walkforward_nested.py and scripts/walkforward_report.py).
The PPO agent is the mean over ten random initializations; all accounts are frictionless and start at 10,000 USD.
Calendar-year equity curves are rebased to 10,000 USD; the PPO line is the mean of the ten runs.

    PYTHONPATH=.:scripts .venv/bin/python scripts/paper_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import HOURS_PER_YEAR
from core.data import load
import walkforward_nested as nested

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT.parents[1] / 'report'                 # research/report, next to research/implementation
FIGURES, TABLES = REPORT / 'figures', REPORT / 'tables'
RUN = ROOT / 'agents/ppo/results/walkforward'
YEARS = [str(y) for y in nested.TEST_YEARS]
BENCHMARKS = ['Buy and hold', '50 day MA', '200 day MA', 'Time series momentum', 'MACD', 'RSI']
LABELS = {'50 day MA': '50 day moving average', '200 day MA': '200 day moving average'}
STYLES = {'PPO agent': ('black', '-', 1.6), 'Buy and hold': ('#c00000', '-', 1.0),
          '50 day MA': ('#ed7d31', '-', 1.0), '200 day MA': ('#7f7f7f', '--', 1.0),
          'Time series momentum': ('#2e75b6', ':', 1.2), 'MACD': ('#70ad47', '-', 1.0), 'RSI': ('#7030a0', '-', 1.0)}

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight'})


def hourly_returns(equity):
    values = equity.to_numpy()
    return np.r_[values[0] / 10_000, values[1:] / values[:-1]] - 1


def sharpe(R):
    return float(R.mean() / R.std() * np.sqrt(HOURS_PER_YEAR)) if R.std() > 0 else 0.0


def max_drawdown(R):
    equity = np.cumprod(1 + R)
    return float((equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1).min())


def compound(R):
    return float(np.expm1(np.log1p(R).sum()))


def series_stats(R, year_of, months):
    years = len(R) / HOURS_PER_YEAR
    monthly = pd.Series(np.log1p(R)).groupby(months).sum().pipe(np.expm1)
    return {'annual_return_pct': ((1 + compound(R)) ** (1 / years) - 1) * 100, 'sharpe': sharpe(R),
            'max_drawdown_pct': max_drawdown(R) * 100,
            'by_year': {y: {'return_pct': compound(R[year_of == y]) * 100, 'sharpe': sharpe(R[year_of == y]),
                            'equity': 10_000 * (1 + compound(R[year_of == y]))} for y in YEARS},
            'monthly': {'mean_pct': monthly.mean() * 100, 'positive_pct': (monthly > 0).mean() * 100,
                        'worst_pct': monthly.min() * 100, 'best_pct': monthly.max() * 100, 'months': len(monthly)}}


def mean_stats(stats):
    out = {k: float(np.mean([s[k] for s in stats])) for k in ('annual_return_pct', 'sharpe', 'max_drawdown_pct')}
    out['by_year'] = {y: {k: float(np.mean([s['by_year'][y][k] for s in stats])) for k in ('return_pct', 'sharpe', 'equity')}
                      for y in YEARS}
    out['monthly'] = {k: float(np.mean([s['monthly'][k] for s in stats])) for k in stats[0]['monthly']}
    return out


def descriptive_by_year(table):
    close = table.set_index('time').close
    out = {}
    for y in range(2018, 2027):
        start = pd.Timestamp(f'{y}-01-01', tz='UTC')
        end = min(pd.Timestamp(f'{y + 1}-01-01', tz='UTC'), close.index[-1])
        part = close.loc[start:end]
        r = np.log(part).diff().dropna()
        out[str(y)] = {'first': float(part.iloc[0]), 'last': float(part.iloc[-1]), 'min': float(part.min()),
                       'max': float(part.max()), 'return_pct': float((part.iloc[-1] / part.iloc[0] - 1) * 100),
                       'volatility_pct': float(r.std() * np.sqrt(HOURS_PER_YEAR) * 100),
                       'max_drawdown_pct': float((part / part.cummax() - 1).min() * 100)}
    return out


# --------------------------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------------------------
def figure_price(table):
    close = table.set_index('time').close['2021-01-01':'2026-08-02 00:00']
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(close.index, close.to_numpy(), color='#1f4e79', lw=0.7)
    high = close.idxmax()
    ax.annotate(f'All time high\n{close[high]:,.0f} USD', xy=(high, close[high]), xytext=(-95, -10),
                textcoords='offset points', fontsize=8, arrowprops={'arrowstyle': '->', 'lw': 0.6})
    for y in range(2022, 2027):
        ax.axvline(pd.Timestamp(f'{y}-01-01', tz='UTC'), color='#d9d9d9', lw=0.6, zorder=0)
    ax.set_ylabel('XBTUSD price (USD)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    fig.savefig(FIGURES / 'price_test.pdf')
    plt.close(fig)


def figure_walk_forward():
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    start = pd.Timestamp('2017-08-12')
    for i, test_year in enumerate(nested.TEST_YEARS):
        y = len(nested.TEST_YEARS) - i
        end = pd.Timestamp(f'{test_year}-01-01')
        ax.barh(y, (end - start).days, left=start, color='#bfbfbf', height=0.55,
                label='Training' if i == 0 else None)
        for v in nested.selection_years(test_year):
            days = (pd.Timestamp(f'{v + 1}-01-01') - pd.Timestamp(f'{v}-01-01')).days
            ax.barh(y, days, left=pd.Timestamp(f'{v}-01-01'), color='#2e75b6', height=0.25,
                    label='Selection years' if i == 0 and v == nested.selection_years(test_year)[0] else None)
        test_end = pd.Timestamp('2026-08-02') if test_year == 2026 else pd.Timestamp(f'{test_year + 1}-01-01')
        ax.barh(y, (test_end - end).days, left=end, color='#c00000', height=0.55, label='Test year' if i == 0 else None)
    ax.set_yticks(range(len(nested.TEST_YEARS), 0, -1), [str(y) for y in nested.TEST_YEARS])
    ax.set_ylabel('Test year')
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    fig.savefig(FIGURES / 'walk_forward.pdf')
    plt.close(fig)


def figure_equity_by_year(frame, agent_columns):
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6), sharey=False)
    year_of = frame.time.dt.year.astype(str).to_numpy()
    for ax, y in zip(axes.flat, YEARS):
        mask = year_of == y
        decisions = frame.time[mask]
        times = pd.DatetimeIndex([decisions.iloc[0], *(decisions + pd.Timedelta(hours=1))])
        paths = np.array([10_000 * np.r_[1.0, np.cumprod(1 + hourly_returns(frame[c])[mask])]
                          for c in agent_columns])
        agent = paths.mean(axis=0)
        ax.plot(times, agent, color='black', lw=1.5, label='PPO agent')
        for name in BENCHMARKS:
            colour, style, width = STYLES[name]
            ax.plot(times, 10_000 * np.r_[1.0, np.cumprod(1 + hourly_returns(frame[name])[mask])], color=colour, ls=style, lw=width,
                    label=LABELS.get(name, name))
        ax.axhline(10_000, color='#bfbfbf', lw=0.6)
        ax.set_title(y if y != '2026' else '2026 (to 1 August)')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1000:,.0f}k'))
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
        ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel('Equity (USD)')
    axes[1, 0].set_ylabel('Equity (USD)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / 'equity_by_year.pdf')
    plt.close(fig)


def figure_training():
    runs = RUN / 'final' / 'test_2026'
    logs = {s: pd.read_json(runs / f'seed_{s}' / 'updates.jsonl', lines=True) for s in nested.FINAL_SEEDS}
    panels = [('mean_reward', 'Mean reward per step'), ('entropy', 'Policy entropy (nats)'),
              ('value_loss', 'Critic loss'), ('explained_variance', 'Explained variance of returns')]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex=True)
    first = logs[nested.FINAL_SEEDS[0]]
    passes = first['update'] / first['update'].max() * first['pass'].max()
    for ax, (key, title) in zip(axes.flat, panels):
        smooth = pd.DataFrame({s: log[key].rolling(25, min_periods=1).mean() for s, log in logs.items()})
        ax.plot(passes, smooth.mean(axis=1), color='#1f4e79', lw=1.3)
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel('Training passes')
    fig.tight_layout()
    fig.savefig(FIGURES / 'training_curves.pdf')
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    report = json.loads((RUN / 'report.json').read_text())
    frame = pd.read_csv(RUN / 'equity_test.csv', parse_dates=['time'])
    agent_columns = [f'seed_{s}' for s in nested.FINAL_SEEDS]
    year_of = frame.time.dt.year.astype(str).to_numpy()
    months = frame.time.dt.strftime('%Y-%m').to_numpy()

    seeds = {c: series_stats(hourly_returns(frame[c]), year_of, months) for c in agent_columns}
    agent = mean_stats(list(seeds.values()))
    trading = report['agent_mean']
    agent.update({k: trading[k] for k in ('position_changes', 'long_share', 'short_share', 'flat_share', 'win_rate_pct')})
    bench = {}
    for name in BENCHMARKS:
        bench[name] = series_stats(hourly_returns(frame[name]), year_of, months)
        bench[name].update({k: report['benchmarks'][name][k]
                            for k in ('position_changes', 'long_share', 'short_share', 'flat_share', 'win_rate_pct')})
    table = load()
    configs = json.loads((RUN / 'candidates.json').read_text())
    choices = {y: {**c, 'settings': {k: configs[c['config']][k] for k in ('lr', 'hidden', 'entropy_coef', 'eta')}}
               for y, c in report['choices'].items()}
    out = {'agent': agent, 'benchmarks': bench, 'random': report['benchmarks']['Random (median)'],
           'seeds': seeds, 'choices': choices, 'diagnostics': report['diagnostics'],
           'descriptive_by_year': descriptive_by_year(table)}
    (TABLES / 'walkforward.json').write_text(json.dumps(out, indent=1, default=float))
    from paper_tables import generate
    generate(out, frame)

    figure_price(table)
    figure_walk_forward()
    figure_equity_by_year(frame, agent_columns)
    figure_training()

    print(f"PPO agent: annual {agent['annual_return_pct']:.1f}%  Sharpe {agent['sharpe']:.2f}  maxDD {agent['max_drawdown_pct']:.1f}%  "
          f"win {agent['win_rate_pct']:.1f}%  changes {agent['position_changes']:.0f}")
    print('  by year', {y: (round(v['return_pct'], 1), round(v['sharpe'], 2), round(v['equity'])) for y, v in agent['by_year'].items()})
    print('  monthly', {k: round(v, 1) for k, v in agent['monthly'].items()})
    for name, b in bench.items():
        print(f"{name:22} annual {b['annual_return_pct']:6.1f}%  Sharpe {b['sharpe']:.2f}  maxDD {b['max_drawdown_pct']:.1f}%")
        print('  by year', {y: (round(v['return_pct'], 1), round(v['sharpe'], 2)) for y, v in b['by_year'].items()})
        print('  monthly', {k: round(v, 1) for k, v in b['monthly'].items()})


if __name__ == '__main__':
    main()
