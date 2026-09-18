"""Seed 11 attribution and controlled development-only diagnostics.

No training, no test access, no changes to the saved v4 experiment.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from perpetual.config import Config, TEST_START, source_hash
from perpetual.data import load_market, Scaler
from perpetual.network import ActorCritic
from perpetual.evaluate import evaluate

HERE=Path(__file__).resolve().parent
MODEL=HERE.parents[1]
ROOT=MODEL/'results/perpetual_v4'
RUN=ROOT/'runs/2024/seed_11'


def save(name,item):
    (HERE/name).write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')


def attribution(market,c):
    frame=pd.read_csv(RUN/'validation_0300.csv',parse_dates=['time'])
    assert frame.time.max()<TEST_START
    changes=np.flatnonzero(np.diff(np.r_[0,np.sign(frame.contracts)]))
    trades=[]
    for start,end in changes.reshape(-1,2):
        t0,t1=market.index(frame.time.iloc[start]),market.index(frame.time.iloc[end])
        fill0=t0+int(np.flatnonzero(market.present[t0:t0+5])[0])
        fill1=t1+int(np.flatnonzero(market.present[t1:t1+5])[0])
        p0,p1=market.prices[[fill0,fill1],0]
        quantity=frame.contracts.iloc[start]
        assert quantity>0
        initial=1. if start==0 else frame.equity_btc.iloc[start-1]
        final=frame.equity_btc.iloc[end]
        segment=frame.iloc[start:end+1]
        fees=float(segment.fees_btc.sum());funding=float(segment.funding_btc.sum())
        pnl_fill=quantity*(1/(p0*(1+c.slippage))-1/(p1*(1-c.slippage)))
        pnl_reference=quantity*(1/p0-1/p1)
        assert np.isclose(final-initial,pnl_fill-fees+funding,rtol=0,atol=1e-11)
        # Equity at each actual minute close while the position is open.
        mark=market.prices[fill0:fill1,3]
        settlement=-quantity/market.prices[fill0-1:fill1-1,3]*market.funding[fill0:fill1]
        settlement[0]=0 # Funding at entry boundary precedes the new entry.
        entry_fee=quantity/(p0*(1+c.slippage))*c.fee
        marked=initial-entry_fee+np.cumsum(settlement)+quantity*(1/(p0*(1+c.slippage))-1/mark)
        net=final/initial-1
        trades.append({'entry':str(frame.time.iloc[start]),'exit':str(frame.time.iloc[end]),
                       'holding_minutes':int((end-start)*5),'contracts':float(quantity),
                       'entry_price':float(p0),'exit_price':float(p1),
                       'net_btc':float(final-initial),'net_return':float(net),'log_return':float(np.log1p(net)),
                       'price_pnl_before_costs_btc':float(pnl_reference),
                       'slippage_cost_btc':float(pnl_reference-pnl_fill),'commission_btc':fees,'funding_btc':funding,
                       'mae_minute_close':float(np.min(marked/initial-1)),
                       'mfe_minute_close':float(np.max(marked/initial-1)),
                       'momentum_15m':float(market.slow[t0,0]),'momentum_1h':float(market.slow[t0,1]),
                       'momentum_1d':float(market.slow[t0,4]),'volatility_1h':float(market.slow[t0,7])})
    t=pd.DataFrame(trades);t.to_csv(HERE/'trades.csv',index=False)
    group=t.groupby(t.entry.str[:10]).log_return.sum()
    total=float(t.log_return.sum());big=float(t.log_return.max())
    summary={'net_return_btc':float(np.expm1(total)),'gross_price_pnl_btc':float(t.price_pnl_before_costs_btc.sum()),
             'slippage_cost_btc':float(t.slippage_cost_btc.sum()),'commission_btc':float(t.commission_btc.sum()),
             'funding_btc':float(t.funding_btc.sum()),'trades':len(t),'entry_days':len(group),
             'winning_trades':int(t.net_return.gt(0).sum()),'profit_factor_btc':float(t.loc[t.net_btc>0,'net_btc'].sum()/-t.loc[t.net_btc<0,'net_btc'].sum()),
             'mean_winning_trade_return':float(t.loc[t.net_return>0,'net_return'].mean()),
             'mean_losing_trade_return':float(t.loc[t.net_return<0,'net_return'].mean()),
             'holding_minutes':{k:float(v) for k,v in t.holding_minutes.describe().items()},
             'median_mae':float(t.mae_minute_close.median()),'worst_mae':float(t.mae_minute_close.min()),
             'median_mfe':float(t.mfe_minute_close.median()),
             'return_excluding_best_trade':float(np.expm1(total-big)),
             'return_excluding_best_entry_day':float(np.expm1(total-group.max())),
             'entry_day_log_returns':group.to_dict(),'net_return_by_quarter':{},
             'sum_holding_hours':float(t.holding_minutes.sum()/60),
             'funding_is_net_cost':bool(t.funding_btc.sum()<0),
             'test_locked':True,'test_performance_accessed':False}
    for quarter,sub in t.groupby(pd.to_datetime(t.entry).dt.quarter):
        summary['net_return_by_quarter'][str(quarter)]=float(np.expm1(sub.log_return.sum()))
    save('attribution.json',summary)
    print('ATTRIBUTION',json.dumps(summary),flush=True)
    return frame,t


def counterfactual_exits(market,c,trades):
    # Independent entry experiments, not a feasible joint strategy: horizons
    # can overlap. Only compare per-trade outcomes at the original entries.
    summaries=[]
    for minutes in [15,30,60,120,240]:
        net=[]
        for row in trades.to_dict('records'):
            start=market.index(row['entry']);end=start+minutes
            assert market.timestamp(end+5)<TEST_START
            a=start+int(np.flatnonzero(market.present[start:start+5])[0])
            b=end+int(np.flatnonzero(market.present[end:end+5])[0])
            p0,p1=market.prices[[a,b],0]
            # Unit entry wealth, original 1x quantity convention, fixed quantity.
            quantity=np.floor(p0)
            funding=np.sum(-quantity/market.prices[a:b,3]*market.funding[a+1:b+1])
            pnl=quantity*(1/(p0*(1+c.slippage))-1/(p1*(1-c.slippage)))
            fee=quantity*(1/(p0*(1+c.slippage))+1/(p1*(1-c.slippage)))*c.fee
            # Check adverse prices for liquidation during alternative holding.
            wallet=1-quantity/(p0*(1+c.slippage))*c.fee
            marks=market.prices[a:b,2]
            marks=np.where(market.present[a:b],marks,market.prices[a:b,3])
            accrual=-quantity/market.prices[a:b,3]*market.funding[a+1:b+1]
            worst=wallet+np.r_[0,np.cumsum(accrual)[:-1]]+quantity*(1/(p0*(1+c.slippage))-1/marks)
            assert np.all(worst>quantity/marks*(c.maintenance+c.fee)), 'alternative liquidates; use full ledger'
            net.append(float(pnl-fee+funding))
        original=trades.net_return.to_numpy()
        summaries.append({'fixed_exit_minutes':minutes,'mean_net_return_per_entry':float(np.mean(net)),
                          'median_net_return_per_entry':float(np.median(net)),
                          'wins':int(np.sum(np.array(net)>0)),
                          'learned_exit_better_count':int(np.sum(original>net)),
                          'mean_learned_minus_fixed_return':float(np.mean(original-net)),
                          'note':'Entry-matched attribution; overlapping independent trades, not an aggregate strategy backtest.'})
    save('exit_diagnostics.json',{'learned_mean_return_per_trade':float(trades.net_return.mean()),
                                  'alternatives':summaries,'test_locked':True})
    print('EXITS',json.dumps(summaries),flush=True)


def sensitivity(market,c):
    state=torch.load(RUN/'policy_0300.pt',map_location='cpu',weights_only=False)
    model=ActorCritic(c);model.load_state_dict(state['model']);model.eval()
    scaler=Scaler.from_dict(state['scaler'])
    results={}
    for slip in [.0003,.0005]:
        label=f'{round((c.fee+slip)*1e4)}bp'
        result,_=evaluate([model],market,scaler,c,'2024-01-01T00:00:00Z','2025-01-01T00:00:00Z',slippage=slip)
        results[label]=result
        save('cost_sensitivity.json',results)
        print('COST',label,json.dumps(result),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);c=Config()
    assert source_hash()==json.loads((ROOT/'protocol.json').read_text())['source_sha256']
    market=load_market()
    _,trades=attribution(market,c)
    counterfactual_exits(market,c,trades)
    sensitivity(market,c)
