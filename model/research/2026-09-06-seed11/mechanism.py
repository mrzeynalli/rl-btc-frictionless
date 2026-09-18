"""Causal replay and input interventions at seed 11's actual trade decisions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from perpetual.config import Config
from perpetual.data import Scaler,load_market
from perpetual.env import Environment,Ledger,FLAT,LONG
from perpetual.network import ActorCritic

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'results/perpetual_v4'
RUN=ROOT/'runs/2024/seed_11'


@torch.no_grad()
def main():
    torch.set_num_threads(2);c=Config();market=load_market()
    frame=pd.read_csv(RUN/'validation_0300.csv',parse_dates=['time'])
    bounds=market.bounds('2024-01-01T00:00:00Z','2025-01-01T00:00:00Z')
    env=Environment(market,bounds,c,np.random.default_rng(0),training=False)
    first,last=bounds;n=len(frame)
    history=np.zeros((n,16),np.float32)
    actions=frame.action.to_numpy()
    active=np.flatnonzero(actions!=FLAT)
    i=0
    while i<n:
        minute,p=env.observation()
        if env.ledger.contracts[0]==0 and actions[i]==FLAT and env.ledger.last_reward[0]==0:
            later=active[active>i];end=int(later[0]) if len(later) else n
            history[i:end]=p[0]
            env.cursor+=(end-i)*5;env.steps+=(end-i);i=end
            continue
        history[i]=p[0]
        _,_,info=env.step(actions[i:i+1])
        assert abs(info['equity_btc'][0]-frame.equity_btc.iloc[i])<1e-11
        i+=1
    state=torch.load(RUN/'policy_0300.pt',map_location='cpu',weights_only=False)
    model=ActorCritic(c);model.load_state_dict(state['model']);model.eval()
    scaler=Scaler.from_dict(state['scaler'])
    changes=np.flatnonzero(np.diff(np.r_[0,np.sign(frame.contracts)]))
    groups={'entry':changes[::2],'exit':changes[1::2],
            'mid_trade':(changes[::2]+changes[1::2])//2}
    selected=np.r_[groups['entry'],groups['exit'],groups['mid_trade']]
    blank=Ledger(1,c).portfolio(np.array([100.]))[0]
    prefix=np.broadcast_to(blank,(c.burn_in,16))
    positions=np.concatenate([prefix,history])
    starts=(selected//c.rollout)*c.rollout-c.burn_in
    offsets=c.burn_in+selected%c.rollout
    indices=starts[:,None]+np.arange(c.burn_in+c.rollout)[None,:]
    # No observation after the selected decision is materialized for a probe.
    indices=np.minimum(indices,selected[:,None])
    minutes=first+indices*5
    fast,slow=scaler.observations(market,minutes,c.fast_minutes)
    p=positions[indices+c.burn_in].copy()
    batch,steps,_,channels=fast.shape
    local=model.actor.local(torch.from_numpy(fast.reshape(-1,c.fast_minutes,channels)).transpose(1,2)).squeeze(-1).reshape(batch,steps,-1)
    market_input=torch.cat([local,torch.from_numpy(slow)],-1)
    zero_local=model.actor.local(torch.zeros(1,channels,c.fast_minutes)).squeeze(-1)[0]
    variants=['original','neutral_current_account','neutral_account_history','remove_fast_information',
              'remove_slow_momentum','remove_slow_volatility','remove_slow_volume','remove_funding']
    result={'probe':'Interventions at original entry, exit and mid-trade states; no strategy return is estimated.',
            'sample_counts':{k:len(v) for k,v in groups.items()},'variants':{},'test_locked':True,
            'test_performance_accessed':False}
    per_state=[]
    for name in variants:
        z=market_input.clone();account=p.copy()
        if name=='neutral_current_account':account[np.arange(batch),offsets]=blank
        elif name=='neutral_account_history':account[:]=blank
        elif name=='remove_fast_information':z[:,:,:32]=zero_local
        elif name=='remove_slow_momentum':z[:,:,32:39]=0
        elif name=='remove_slow_volatility':z[:,:,39:41]=0
        elif name=='remove_slow_volume':z[:,:,41]=0
        elif name=='remove_funding':z[:,:,42:45]=0
        encoded=model.actor.input(torch.cat([z,torch.from_numpy(account)],-1))
        recurrent,_=model.actor.gru(encoded)
        probability=model.actor.head(recurrent[torch.arange(batch),torch.from_numpy(offsets)]).softmax(-1).numpy()
        choice=probability.argmax(-1)
        if name=='original':assert np.array_equal(choice,actions[selected])
        result['variants'][name]={}
        for g,indices_g in zip(groups,np.split(np.arange(batch),3)):
            result['variants'][name][g]={'argmax_counts':np.bincount(choice[indices_g],minlength=3).tolist(),
                                         'mean_probabilities':probability[indices_g].mean(0,dtype=np.float64).tolist()}
        for j,s in enumerate(selected):
            per_state.append({'variant':name,'kind':['entry','exit','mid_trade'][j//25],
                              'time':str(frame.time.iloc[s]),'p_short':float(probability[j,0]),
                              'p_flat':float(probability[j,1]),'p_long':float(probability[j,2])})
    (HERE/'mechanism.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(per_state).to_csv(HERE/'mechanism_states.csv',index=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
