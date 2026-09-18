"""Interactive financial charts for a completed experiment.

Reads the per-hour validation paths saved at the selected checkpoint, computes the
constant-long and constant-short benchmarks through the same ledger, and writes a
self-contained page (Plotly bundled locally, no internet needed) with one view per
validation fold and per training seed:

  price with the decisions marked on it, equity in BTC and in USD against the
  benchmarks, drawdown, costs, per-trade distributions, the policy's conviction,
  and a summary table.

    PYTHONPATH=src .venv/bin/python scripts/financial_charts.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from agent.config import OUTPUT, YEARS, Config, fold_config
from agent.data import Scaler, load_market, training_decisions
from agent.evaluate import evaluate_quarters

SEEDS_DEFAULT = (11, 29, 47, 61, 83)
HOURS_PER_YEAR = 24 * 365


def chained_equity(frame, column='equity_btc'):
    """Each quarter is evaluated from a flat account; compound them into one curve."""
    out, carry = [], 1.0
    for _, block in frame.groupby('quarter', sort=True):
        values = block[column].to_numpy(dtype=float)
        out.append(values * carry)
        carry = out[-1][-1]
    return np.concatenate(out)


def trades_of(position, quarter, grid, price, equity):
    """Maximal runs of one side, held over whole decision hours.

    A run never crosses a quarter boundary: every quarter is evaluated from a flat
    account, so a position open at the end of one quarter is closed and re-entered in
    the next. ``grid``, ``price`` and ``equity`` are indexed at decision instants and
    are one longer than ``position``: a run starting at i opens at grid[i] and a run
    ending at j closes at grid[j + 1].
    """
    trades, start = [], None
    for index in range(len(position) + 1):
        current = position[index] if index < len(position) else 0
        broken = 0 < index < len(position) and quarter[index] != quarter[index - 1]
        if start is not None and (index == len(position) or broken or current != position[start]):
            end = index - 1
            trades.append({'side': 'long' if position[start] > 0 else 'short',
                           'entry': int(grid[start]), 'exit': int(grid[end + 1]),
                           'hours': int(end - start + 1),
                           'entry_price': round(float(price[start]), 2),
                           'exit_price': round(float(price[end + 1]), 2),
                           'log_return': round(float(np.log(equity[end + 1] / equity[start])), 5)})
            start = None
        if start is None and current != 0:
            start = index
    return trades


def drawdown(equity):
    return equity / np.maximum.accumulate(equity) - 1.0


def statistics(path, equity, price, trades, benchmarks, across):
    """Everything in the summary table. ``equity`` and ``price`` are on the decision grid."""
    returns = path['log_return'].to_numpy(dtype=float)
    spread = returns.std()
    wins = [t for t in trades if t['log_return'] > 0]
    position = np.sign(path['contracts'].to_numpy())
    return {
        'mean_log_return_btc_4seeds': round(float(np.log(across[:, -1]).mean()), 4),
        'total_return_btc_pct': round(float(equity[-1] - 1) * 100, 2),
        'total_log_return_btc': round(float(np.log(equity[-1])), 4),
        'total_return_usd_pct': round(float(equity[-1] * price[-1] / price[0] - 1) * 100, 2),
        'buy_and_hold_usd_pct': round(float(price[-1] / price[0] - 1) * 100, 2),
        'constant_long_log_btc': round(float(np.log(benchmarks['long'][-1])), 4),
        'constant_short_log_btc': round(float(np.log(benchmarks['short'][-1])), 4),
        'annualised_sharpe_btc': round(float(returns.mean() / spread * np.sqrt(HOURS_PER_YEAR)), 2) if spread else 0.0,
        'max_drawdown_pct': round(float(drawdown(equity).min()) * 100, 2),
        'trades': len(trades),
        'win_rate_pct': round(100 * len(wins) / len(trades), 1) if trades else 0.0,
        'median_hold_hours': int(np.median([t['hours'] for t in trades])) if trades else 0,
        'longest_hold_hours': int(max((t['hours'] for t in trades), default=0)),
        'time_long_pct': round(float(np.mean(position > 0)) * 100, 1),
        'time_short_pct': round(float(np.mean(position < 0)) * 100, 1),
        'time_flat_pct': round(float(np.mean(position == 0)) * 100, 1),
        'fees_btc_pct': round(float(path['fees_btc'].sum()) * 100, 3),
        'funding_btc_pct': round(float(path['funding_btc'].sum()) * 100, 3),
        'liquidations': int(path['liquidation'].sum()),
    }


def fold_payload(experiment, market, base, year, update, seeds):
    """One validation year, with every series on the decision-instant grid.

    The saved frames label each row with the decision time but record price, position
    and equity as of the end of that hour. Charting them against the decision times
    would place every marker an hour early, so price and equity are shifted onto a grid
    of the decision instants themselves (one point longer, ending at the final close).
    """
    scaler = Scaler.from_dict(json.loads((experiment / f'scalers/{year}.json').read_text()))
    config = fold_config(base, scaler.bar_return_sd, training_decisions(market, pd.Timestamp(f'{year}-01-01', tz='UTC'), base))
    first, _ = market.bounds(pd.Timestamp(f'{year}-01-01', tz='UTC'), pd.Timestamp(f'{year + 1}-01-01', tz='UTC'))
    start_price = float(market.prices[first - 1, 3])
    benchmarks = {}
    for name in ('long', 'short'):
        _, frame = evaluate_quarters([], market, None, config, year, baseline=name)
        benchmarks[name] = np.concatenate([[1.0], chained_equity(frame)])
    payload = {'benchmarks': {k: np.round(v, 6).tolist() for k, v in benchmarks.items()}, 'seeds': {}}
    curves = []
    for seed in seeds:
        frame = pd.read_csv(experiment / f'runs/{year}/seed_{seed}/validation_{update:05d}.csv.gz', parse_dates=['time'])
        across = np.stack([np.concatenate([[1.0], chained_equity(frame[frame.seed == s].sort_values(['quarter', 'time']))])
                           for s in sorted(frame.seed.unique())])
        path = frame[frame.seed == 0].sort_values(['quarter', 'time']).reset_index(drop=True)
        stamps = path['time'].to_numpy().astype('datetime64[ms]').astype('int64')  # Plotly date axes read milliseconds
        hour = int(config.decision_bars * config.decision_minutes) * 60_000
        grid = np.concatenate([stamps, [stamps[-1] + hour]])
        price = np.concatenate([[start_price], path['mark'].to_numpy(dtype=float)])
        equity = across[0]
        position = np.sign(path['contracts'].to_numpy()).astype(int)
        quarter = path['quarter'].to_numpy()
        if 'time' not in payload:
            payload['time'] = grid.tolist()
            payload['price'] = np.round(price, 2).tolist()
        trade_list = trades_of(position, quarter, grid, price, equity)
        agent_quarter, long_quarter, short_quarter = [], [], []
        labels = sorted(np.unique(quarter))
        for q in labels:
            mask = quarter == q
            begin, end = int(np.argmax(mask)), int(len(mask) - 1 - np.argmax(mask[::-1]))
            agent_quarter.append(round(float(np.log(equity[end + 1] / equity[begin])), 4))
            for name, store in (('long', long_quarter), ('short', short_quarter)):
                curve = benchmarks[name]
                store.append(round(float(np.log(curve[end + 1] / curve[begin])), 4))
        every = 4
        payload['seeds'][str(seed)] = {
            'position': np.concatenate([position, [0]]).tolist(),  # flat after the final close
            'equity': np.round(equity, 6).tolist(),
            'equity_low': np.round(across.min(axis=0), 6).tolist(),
            'equity_high': np.round(across.max(axis=0), 6).tolist(),
            'equity_usd': np.round(equity * price / price[0], 6).tolist(),
            'drawdown': np.round(drawdown(equity) * 100, 3).tolist(),
            'fees': np.round(np.concatenate([[0.0], path['fees_btc'].cumsum().to_numpy()]) * 100, 4).tolist(),
            'funding': np.round(np.concatenate([[0.0], path['funding_btc'].cumsum().to_numpy()]) * 100, 4).tolist(),
            'conviction_time': stamps[::every].tolist(),
            'p_long': np.round(path['p_long'].to_numpy()[::every], 3).tolist(),
            'p_short': np.round(path['p_short'].to_numpy()[::every], 3).tolist(),
            'p_flat': np.round(path['p_flat'].to_numpy()[::every], 3).tolist(),
            'trades': trade_list,
            'quarter_labels': [f'Q{int(q)}' for q in labels],
            'quarter_agent': agent_quarter,
            'quarter_long': long_quarter,
            'quarter_short': short_quarter,
            'stats': statistics(path, equity, price, trade_list, benchmarks, across),
        }
        curves.append(across.mean(axis=0))
    portfolio = np.mean(np.stack(curves), axis=0)
    payload['portfolio'] = {'equity': np.round(portfolio, 6).tolist(),
                            'total_log_return_btc': round(float(np.log(portfolio[-1])), 4),
                            'max_drawdown_pct': round(float(drawdown(portfolio).min()) * 100, 2)}
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=OUTPUT / 'sharpe_bars12')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS_DEFAULT))
    args = parser.parse_args()
    experiment = args.experiment
    charts = args.output or experiment / 'charts'
    charts.mkdir(parents=True, exist_ok=True)
    selection = json.loads((experiment / 'comparison/selection.json').read_text())
    base = Config()
    market = load_market()
    data = {'checkpoint': selection['selected']['checkpoint'], 'passes': selection['selected']['passes'],
            'iqm': selection['selected']['iqm_quarter_log_return'],
            'interval': selection['selected_interval'],
            'pbo': selection['probability_of_backtest_overfitting']['probability'],
            'break_even_slippage': selection['break_even_slippage'],
            'seeds': [str(s) for s in args.seeds], 'folds': {}}
    for year in YEARS:
        update = selection['selected']['updates'][f'{year}_{args.seeds[0]}']
        print(f'fold {year}: checkpoint update {update}', flush=True)
        data['folds'][str(year)] = fold_payload(experiment, market, base, year, update, args.seeds)
    (charts / 'data.json').write_text(json.dumps(data, separators=(',', ':')))
    (charts / 'plotly.min.js').write_text(get_plotlyjs())
    (charts / 'index.html').write_text(PAGE)
    size = sum(f.stat().st_size for f in charts.iterdir()) / 1e6
    print(f'wrote {charts}/index.html ({size:.1f} MB including the Plotly bundle)')


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent financial results</title>
<script src="plotly.min.js"></script>
<style>
 :root { --ink:#101418; --muted:#5b6672; --line:#dfe4ea; --bg:#f6f7f9; --card:#fff;
         --long:#1a9850; --short:#d73027; --flat:#8a93a0; --accent:#2b6cb0; }
 * { box-sizing:border-box; }
 body { margin:0; background:var(--bg); color:var(--ink);
        font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
 header { background:var(--card); border-bottom:1px solid var(--line); padding:18px 24px; }
 h1 { margin:0 0 4px; font-size:20px; } .sub { color:var(--muted); font-size:13px; }
 .controls { display:flex; flex-wrap:wrap; gap:18px; align-items:center; padding:14px 24px;
             background:var(--card); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }
 .group { display:flex; gap:6px; align-items:center; }
 .group b { font-weight:600; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
 button { border:1px solid var(--line); background:var(--card); color:var(--ink); padding:6px 12px;
          border-radius:6px; cursor:pointer; font-size:13px; }
 button:hover { border-color:var(--accent); }
 button.on { background:var(--accent); border-color:var(--accent); color:#fff; }
 main { padding:20px 24px 60px; max-width:1500px; margin:0 auto; }
 .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:18px; }
 .card h2 { margin:0 0 2px; font-size:15px; } .card p { margin:0 0 10px; color:var(--muted); font-size:12.5px; }
 .chart { width:100%; height:420px; } .chart.tall { height:520px; }
 table { border-collapse:collapse; width:100%; font-size:13px; }
 th,td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--line); }
 th:first-child, td:first-child { text-align:left; color:var(--muted); }
 .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; }
 .pos { color:var(--long); } .neg { color:var(--short); }
 @media (max-width:700px){ main{padding:14px;} .chart{height:340px;} }
</style></head>
<body>
<header>
  <h1>Agent financial results — hourly decisions on BTC</h1>
  <div class="sub" id="headline"></div>
</header>
<div class="controls">
  <div class="group"><b>Fold</b><span id="folds"></span></div>
  <div class="group"><b>Training seed</b><span id="seeds"></span></div>
</div>
<main>
  <div class="card"><h2>Price and decisions</h2>
    <p>BTC price coloured by the position the agent held over each hour, with entries and exits marked at the
       instant the decision took effect. Drag to zoom, double-click to reset, or use the range slider below.</p>
    <div id="decisions" class="chart tall"></div></div>
  <div class="card"><h2>Equity in BTC</h2>
    <p>The account's BTC equity, compounded across the year's four quarters, against holding a constant 1x long or short through the same costs and funding. The band spans the sampling seeds when the run was evaluated with sampled actions (every run up to 2026-09-13).</p>
    <div id="equity" class="chart"></div></div>
  <div class="grid">
    <div class="card"><h2>Value in USD</h2>
      <p>The same account measured in dollars, against buying and holding BTC.</p>
      <div id="usd" class="chart"></div></div>
    <div class="card"><h2>Drawdown</h2>
      <p>Distance below the running peak of BTC equity.</p>
      <div id="drawdown" class="chart"></div></div>
  </div>
  <div class="grid">
    <div class="card"><h2>Costs paid</h2>
      <p>Cumulative commission and slippage, and funding received or paid, as a percentage of starting equity.</p>
      <div id="costs" class="chart"></div></div>
    <div class="card"><h2>Policy conviction</h2>
      <p>The probabilities the policy assigned to each action, sampled every four hours.</p>
      <div id="conviction" class="chart"></div></div>
  </div>
  <div class="grid">
    <div class="card"><h2>Quarter by quarter</h2>
      <p>Log return in each evaluation quarter, against the two constant benchmarks over the same quarter.</p>
      <div id="quarters" class="chart"></div></div>
    <div class="card"><h2>Holding periods</h2><p>How long each position was held.</p>
      <div id="holds" class="chart"></div></div>
    <div class="card"><h2>Trade outcomes</h2><p>Log return of each completed position, by side.</p>
      <div id="outcomes" class="chart"></div></div>
  </div>
  <div class="card"><h2>Summary</h2><p id="summary-note"></p><div id="summary"></div></div>
  <div class="card"><h2>Every trade</h2><p>All completed positions for this fold and seed.</p><div id="trades" style="max-height:420px;overflow:auto"></div></div>
</main>
<script>
const LAYOUT = {margin:{l:56,r:18,t:10,b:40}, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
  font:{family:'-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif', size:12, color:'#101418'},
  xaxis:{gridcolor:'#eef1f4', zeroline:false}, yaxis:{gridcolor:'#eef1f4', zeroline:false},
  legend:{orientation:'h', y:1.12, x:0}, hovermode:'x unified'};
const CONFIG = {displayModeBar:true, responsive:true, displaylogo:false,
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d']};
const COLOUR = {long:'#1a9850', short:'#d73027', flat:'#8a93a0', agent:'#2b6cb0', hold:'#b08d2b', cash:'#8a93a0'};
let DATA=null, fold=null, seed=null;

const layout = extra => JSON.parse(JSON.stringify(Object.assign({}, LAYOUT, extra||{})));
const fmt = (v,d=2) => (v>0?'+':'') + v.toFixed(d);

function splitByPosition(time, price, position){
  const series = {};
  for (const key of [1,-1,0]){
    const k = Number(key);
    // a point belongs to the run it starts and to the run that ends there, so
    // consecutive runs share a vertex instead of leaving an hour-wide gap
    series[key] = price.map((p,i)=> (position[i]===k || (i>0 && position[i-1]===k)) ? p : null);
  }
  return series;
}

function drawDecisions(f, s){
  const parts = splitByPosition(f.time, f.price, s.position);
  const marks = {longEntry:{x:[],y:[]}, shortEntry:{x:[],y:[]}, exit:{x:[],y:[]}};
  for (const t of s.trades){
    const i = f.time.indexOf(t.entry), j = f.time.indexOf(t.exit);
    (t.side==='long'?marks.longEntry:marks.shortEntry).x.push(t.entry);
    (t.side==='long'?marks.longEntry:marks.shortEntry).y.push(f.price[i>=0?i:0]);
    marks.exit.x.push(t.exit); marks.exit.y.push(f.price[j>=0?j:0]);
  }
  const line = (y,name,colour)=>({x:f.time, y, name, type:'scatter', mode:'lines',
    line:{color:colour, width:1.6}, connectgaps:false, hovertemplate:'%{y:$,.0f}<extra>'+name+'</extra>'});
  Plotly.react('decisions', [
    line(parts['0'],'flat',COLOUR.flat), line(parts['1'],'long',COLOUR.long), line(parts['-1'],'short',COLOUR.short),
    {x:marks.longEntry.x, y:marks.longEntry.y, name:'enter long', mode:'markers', type:'scatter',
     marker:{symbol:'triangle-up', size:9, color:COLOUR.long, line:{width:1,color:'#fff'}}},
    {x:marks.shortEntry.x, y:marks.shortEntry.y, name:'enter short', mode:'markers', type:'scatter',
     marker:{symbol:'triangle-down', size:9, color:COLOUR.short, line:{width:1,color:'#fff'}}},
    {x:marks.exit.x, y:marks.exit.y, name:'close', mode:'markers', type:'scatter',
     marker:{symbol:'x', size:7, color:'#39424c'}}
  ], layout({xaxis:{type:'date', gridcolor:'#eef1f4', rangeslider:{visible:true, thickness:0.07}},
             yaxis:{title:'BTC price (USD)', gridcolor:'#eef1f4'}, hovermode:'closest'}), CONFIG);
}

function drawEquity(f, s){
  const band = {x:f.time.concat(f.time.slice().reverse()),
                y:s.equity_high.concat(s.equity_low.slice().reverse()),
                fill:'toself', fillcolor:'rgba(43,108,176,.13)', line:{width:0}, type:'scatter',
                name:'sampling-seed range', hoverinfo:'skip'};
  const line = (y,name,colour,dash)=>({x:f.time, y, name, type:'scattergl', mode:'lines',
    line:{color:colour, width:name==='agent'?2:1.4, dash:dash||'solid'}, hovertemplate:'%{y:.3f}<extra>'+name+'</extra>'});
  Plotly.react('equity', [band, line(s.equity,'agent',COLOUR.agent),
    line(f.benchmarks.long,'constant long',COLOUR.long,'dot'),
    line(f.benchmarks.short,'constant short',COLOUR.short,'dot'),
    line(f.time.map(()=>1),'flat (hold BTC)',COLOUR.flat,'dash'),
    line(DATA.folds[fold].portfolio.equity,'5-seed portfolio','#7d3c98','dashdot')],
    layout({xaxis:{type:'date', gridcolor:'#eef1f4'}, yaxis:{title:'equity (BTC, start = 1)', gridcolor:'#eef1f4'}}), CONFIG);
}

function drawUsd(f, s){
  const hold = f.price.map(p=>p/f.price[0]);
  const line = (y,name,colour,dash)=>({x:f.time, y, name, type:'scattergl', mode:'lines',
    line:{color:colour, width:name==='agent'?2:1.4, dash:dash||'solid'}, hovertemplate:'%{y:.3f}<extra>'+name+'</extra>'});
  Plotly.react('usd', [line(s.equity_usd,'agent',COLOUR.agent), line(hold,'buy and hold BTC',COLOUR.hold,'dot'),
    line(f.time.map(()=>1),'cash',COLOUR.cash,'dash')],
    layout({xaxis:{type:'date', gridcolor:'#eef1f4'}, yaxis:{title:'value (USD, start = 1)', gridcolor:'#eef1f4'}}), CONFIG);
}

function drawDrawdown(f, s){
  Plotly.react('drawdown', [{x:f.time, y:s.drawdown, type:'scattergl', mode:'lines', name:'drawdown',
    line:{color:COLOUR.short, width:1.4}, fill:'tozeroy', fillcolor:'rgba(215,48,39,.12)',
    hovertemplate:'%{y:.2f}%<extra></extra>'}],
    layout({xaxis:{type:'date', gridcolor:'#eef1f4'}, yaxis:{title:'% below peak', gridcolor:'#eef1f4'}}), CONFIG);
}

function drawCosts(f, s){
  Plotly.react('costs', [
    {x:f.time, y:s.fees, name:'commission + slippage', type:'scattergl', mode:'lines', line:{color:COLOUR.short, width:1.5}},
    {x:f.time, y:s.funding, name:'funding', type:'scattergl', mode:'lines', line:{color:COLOUR.agent, width:1.5}}],
    layout({xaxis:{type:'date', gridcolor:'#eef1f4'}, yaxis:{title:'% of starting equity', gridcolor:'#eef1f4'}}), CONFIG);
}

function drawConviction(f, s){
  const area = (y,name,colour)=>({x:s.conviction_time, y, name, type:'scattergl', mode:'lines', stackgroup:'one',
    line:{width:0.5, color:colour}, fillcolor:colour.replace('rgb','rgba').replace(')',',.55)')});
  Plotly.react('conviction', [area(s.p_short,'short','rgb(215,48,39)'), area(s.p_flat,'flat','rgb(138,147,160)'),
    area(s.p_long,'long','rgb(26,152,80)')],
    layout({xaxis:{type:'date', gridcolor:'#eef1f4'}, yaxis:{title:'probability', range:[0,1], gridcolor:'#eef1f4'}}), CONFIG);
}

function drawQuarters(s){
  const bar = (y,name,colour)=>({x:s.quarter_labels, y, name, type:'bar', marker:{color:colour},
    hovertemplate:'%{y:.4f}<extra>'+name+'</extra>'});
  Plotly.react('quarters', [bar(s.quarter_agent,'agent',COLOUR.agent), bar(s.quarter_long,'constant long',COLOUR.long),
    bar(s.quarter_short,'constant short',COLOUR.short)],
    layout({barmode:'group', xaxis:{gridcolor:'#eef1f4'}, yaxis:{title:'log return', gridcolor:'#eef1f4'}, hovermode:'closest'}), CONFIG);
}

function drawTradeCharts(s){
  const longs = s.trades.filter(t=>t.side==='long'), shorts = s.trades.filter(t=>t.side==='short');
  Plotly.react('holds', [
    {x:longs.map(t=>t.hours), name:'long', type:'histogram', marker:{color:COLOUR.long}, opacity:.75},
    {x:shorts.map(t=>t.hours), name:'short', type:'histogram', marker:{color:COLOUR.short}, opacity:.75}],
    layout({barmode:'overlay', xaxis:{title:'hours held', gridcolor:'#eef1f4'}, yaxis:{title:'positions', gridcolor:'#eef1f4'}, hovermode:'closest'}), CONFIG);
  Plotly.react('outcomes', [
    {x:longs.map(t=>t.log_return), name:'long', type:'histogram', marker:{color:COLOUR.long}, opacity:.75},
    {x:shorts.map(t=>t.log_return), name:'short', type:'histogram', marker:{color:COLOUR.short}, opacity:.75}],
    layout({barmode:'overlay', xaxis:{title:'log return per position', gridcolor:'#eef1f4'}, yaxis:{title:'positions', gridcolor:'#eef1f4'}, hovermode:'closest'}), CONFIG);
}

const ROWS = [
  ['total_return_btc_pct','Total return (BTC), shown path','%'],
  ['mean_log_return_btc_4seeds','Mean log return over the evaluated seeds',''], ['total_return_usd_pct','Total value change (USD)','%'],
  ['buy_and_hold_usd_pct','Buy and hold BTC (USD)','%'], ['total_log_return_btc','Log return (BTC)',''],
  ['constant_long_log_btc','Constant long (log, BTC)',''], ['constant_short_log_btc','Constant short (log, BTC)',''],
  ['annualised_sharpe_btc','Annualised Sharpe (hourly, BTC)',''], ['max_drawdown_pct','Max drawdown','%'],
  ['trades','Completed positions',''], ['win_rate_pct','Winning positions','%'],
  ['median_hold_hours','Median hold','h'], ['longest_hold_hours','Longest hold','h'],
  ['time_long_pct','Time long','%'], ['time_short_pct','Time short','%'], ['time_flat_pct','Time flat','%'],
  ['fees_btc_pct','Commission + slippage paid','%'], ['funding_btc_pct','Funding','%'],
  ['liquidations','Liquidations','']];

function drawSummary(f, s){
  const st = s.stats;
  let html = '<table><tbody>';
  for (const [key,label,unit] of ROWS){
    const v = st[key];
    const cls = (typeof v === 'number' && ['total_return_btc_pct','total_log_return_btc','total_return_usd_pct','annualised_sharpe_btc'].includes(key)) ? (v>0?'pos':(v<0?'neg':'')) : '';
    html += `<tr><td>${label}</td><td class="${cls}">${typeof v==='number' ? v.toLocaleString(undefined,{maximumFractionDigits:4}) : v}${unit}</td></tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('summary').innerHTML = html;
  const p = DATA.folds[fold].portfolio;
  document.getElementById('summary-note').textContent =
    `Fold ${fold}, training seed ${seed}, checkpoint ${DATA.checkpoint} of 20 (pass ${Math.round(DATA.passes)}). ` +
    `Charts show the first evaluated path. The saved runs were evaluated with sampled actions, ` +
    `so the equity band and the row above cover their sampling seeds; an argmax evaluation has one path. ` +
    `Five-seed equal-weight portfolio for this fold: ${fmt(p.total_log_return_btc,3)} log, max drawdown ${p.max_drawdown_pct}%.`;
}

function drawTrades(s){
  let html = '<table><thead><tr><th>#</th><th>Side</th><th>Entered</th><th>Exited</th><th>Hours</th><th>Entry</th><th>Exit</th><th>Log return</th></tr></thead><tbody>';
  s.trades.forEach((t,i)=>{
    const d = ms => new Date(ms).toISOString().slice(0,16).replace('T',' ');
    html += `<tr><td>${i+1}</td><td class="${t.side==='long'?'pos':'neg'}">${t.side}</td><td>${d(t.entry)}</td><td>${d(t.exit)}</td>`+
            `<td>${t.hours}</td><td>${t.entry_price.toLocaleString()}</td><td>${t.exit_price.toLocaleString()}</td>`+
            `<td class="${t.log_return>0?'pos':'neg'}">${fmt(t.log_return,4)}</td></tr>`;
  });
  document.getElementById('trades').innerHTML = html + '</tbody></table>';
}

function render(){
  const f = DATA.folds[fold], s = f.seeds[seed];
  drawDecisions(f,s); drawEquity(f,s); drawUsd(f,s); drawDrawdown(f,s); drawCosts(f,s);
  drawConviction(f,s); drawQuarters(s); drawTradeCharts(s); drawSummary(f,s); drawTrades(s);
  document.querySelectorAll('#folds button').forEach(b=>b.classList.toggle('on', b.textContent===fold));
  document.querySelectorAll('#seeds button').forEach(b=>b.classList.toggle('on', b.textContent===seed));
}

fetch('data.json').then(r=>r.json()).then(d=>{
  DATA = d; fold = Object.keys(d.folds)[0]; seed = d.seeds[0];
  document.getElementById('headline').textContent =
    `Selected checkpoint ${d.checkpoint} of 20 (pass ${Math.round(d.passes)}) · typical quarterly log return ${fmt(d.iqm,4)} ` +
    `(95% interval ${fmt(d.interval.lower,4)} to ${fmt(d.interval.upper,4)}) · probability of backtest overfitting ${d.pbo.toFixed(2)} ` +
    `· break-even extra slippage ${d.break_even_slippage.median_bp.toFixed(1)} bp per leg · validation folds, 2026 test data untouched`;
  const folds = document.getElementById('folds');
  Object.keys(d.folds).forEach(name=>{
    const b = document.createElement('button'); b.textContent = name;
    b.onclick = ()=>{ fold = name; render(); }; folds.appendChild(b);
  });
  const seeds = document.getElementById('seeds');
  d.seeds.forEach(name=>{
    const b = document.createElement('button'); b.textContent = name;
    b.onclick = ()=>{ seed = name; render(); }; seeds.appendChild(b);
  });
  render();
});
</script></body></html>
"""


if __name__ == '__main__':
    main()
