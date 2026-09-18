"""Read-only checkpoint probes on the unlocked development cache.

No model updates, test-period evaluation or modifications of original results.
"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import torch
from perpetual.config import Config, TRAIN_START, source_hash
from perpetual.data import Scaler, FAST_NAMES, SLOW_NAMES, load_market
from perpetual.env import Environment, Ledger, SHORT, FLAT, LONG
from perpetual.network import ActorCritic, tensors
from perpetual.ppo import Trainer, advantages

HERE=Path(__file__).resolve().parent
MODEL=HERE.parents[1]
RESULTS=MODEL/'results/perpetual_v4'


def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def flat_probabilities(model, market, scaler, config, first, last):
    """Exactly the deployed recurrent refresh, with one always-flat account."""
    count=(last-first)//5
    minutes=first+np.arange(-config.burn_in,count)*5
    cached=np.empty((len(minutes),config.local_width+len(SLOW_NAMES)),np.float32)
    for begin in range(0,len(minutes),2048):
        fast,slow=scaler.observations(market,minutes[begin:begin+2048],config.fast_minutes)
        local=model.actor.local(torch.from_numpy(fast).transpose(1,2)).squeeze(-1).numpy()
        cached[begin:begin+len(fast)]=np.concatenate([local,slow],1)
    blank=Ledger(1,config).portfolio(np.array([100.]))[0]
    starts=np.arange(0,count,config.rollout)
    output=np.zeros((count,3),np.float32)
    # Batching independent 512-step deployment blocks preserves their histories.
    for begin in range(0,len(starts),16):
        start=starts[begin:begin+16]
        indices=start[:,None]+np.arange(config.burn_in+config.rollout)[None,:]
        clipped=np.minimum(indices,len(cached)-1)
        p=np.broadcast_to(blank,clipped.shape+(len(blank),))
        z=model.actor.input(torch.from_numpy(np.concatenate([cached[clipped],p],axis=-1)))
        sequence,_=model.actor.gru(z)
        probs=model.actor.head(sequence[:,config.burn_in:]).softmax(-1).numpy()
        for i,s in enumerate(start):
            n=min(config.rollout,count-s)
            output[s:s+n]=probs[i,:n]
    return output


@torch.no_grad()
def portfolio_probe(model, market, scaler, config, minute):
    # Causal 128-step flat histories, changing only the current account state.
    sequence=minute[:,None]-np.arange(config.burn_in,-1,-1)[None,:]*5
    flat=Ledger(len(minute),config).portfolio(market.prices[minute-1,3])
    portfolio=np.broadcast_to(flat[:,None,:],(len(minute),config.burn_in+1,flat.shape[-1])).copy()
    fast,slow,p=tensors(scaler,market,sequence,portfolio,config)
    hidden=model.burn(fast[:,:-1],slow[:,:-1],p[:,:-1])
    output={}
    for label,action in [('flat',FLAT),('new_long',LONG),('new_short',SHORT)]:
        ledger=Ledger(len(minute),config)
        prices=market.prices[minute-1,3]
        ledger.execute(np.full(len(minute),action),prices,np.ones(len(minute),bool))
        ledger.mark(prices)
        ledger.previous_action[:]=action
        portfolio=ledger.portfolio(prices)
        logits,value,_=model(fast[:,-1:],slow[:,-1:],torch.from_numpy(portfolio[:,None,:]),hidden)
        probabilities=logits[:,0].softmax(-1).numpy()
        output[label]={'mean_probabilities':probabilities.mean(0).tolist(),
                       'argmax_share':(np.bincount(probabilities.argmax(-1),minlength=3)/len(minute)).tolist(),
                       'critic_value_mean':float(value.mean())}
    return output


def encoder_proof(config):
    torch.manual_seed(11)
    model=ActorCritic(config)
    x=torch.randn(1,len(FAST_NAMES),config.fast_minutes,requires_grad=True)
    model.actor.local(x).sum().backward()
    dead=torch.nonzero(x.grad.abs().sum((0,1))==0).flatten().tolist()
    a=x.detach().clone();a[:,:,-3:]+=100
    with torch.no_grad():
        difference=float((model.actor.local(a)-model.actor.local(x)).abs().max())
        z=torch.zeros_like(x);w=z.clone();z[:,:,20]=1;w[:,:,36]=1
        shift=float((model.actor.local(z)-model.actor.local(w)).abs().max())
    return {'unseen_minute_indices_zero_based':dead,'last_three_minute_perturbation_output_change':difference,
            'interior_pattern_shift_16_minutes_output_change':shift,'last_layer_receptive_field_minutes':9,
            'pooling_locations':13}


def cost_diagnostic(market,config):
    torch.manual_seed(11)
    end=pd.Timestamp('2023-01-01T00:00:00Z')
    scaler=Scaler.fit(market,end)
    env=Environment(market,market.bounds(TRAIN_START,end),config,np.random.default_rng(11))
    trainer=Trainer(ActorCritic(config),env,scaler,config,11)
    rows,bootstrap=trainer.collect()
    adv,target=advantages(rows['rewards'],rows['values'],bootstrap,rows['terminals'],rows['valid'],config.gamma,config.gae_lambda)
    free_config=replace(config,fee=0,slippage=0)
    free=Environment(market,market.bounds(TRAIN_START,end),free_config,np.random.default_rng(11))
    cost_free=[]
    for i in range(config.rollout):
        cost_free.append(free.step(rows['actions'][:,i])[0])
    original=json.loads((RESULTS/'runs/2023/seed_11/updates.jsonl').read_text().splitlines()[0])
    net=float(rows['rewards'].mean()); zero=float(np.mean(cost_free))
    result={'seed':11,'training_end_exclusive':str(end),'steps':int(rows['rewards'].size),
            'original_logged_mean_reward':original['mean_reward'],'reconstructed_mean_reward':net,
            'identical_actions_cost_free_mean_reward':zero,'paired_difference':net-zero,
            'uniform_target_expected_turnover':8/9,'uniform_cost_per_decision_reward_units':config.reward_scale*(config.fee+config.slippage)*8/9,
            'uniform_conditional_hold_at_least_one_hour_probability':(1/3)**11,
            'sampled_advantage_by_action':{label:float(adv[rows['actions']==a].mean()) for a,label in enumerate(['short','flat','long'])},
            'observed_mean_reward_by_action':{label:float(rows['rewards'][rows['actions']==a].mean()) for a,label in enumerate(['short','flat','long'])},
            'note':'Fixed action-sequence cost comparison; not a no-cost policy training ablation.'}
    assert np.isclose(net,original['mean_reward'],rtol=0,atol=1e-8)
    return result


def main():
    torch.set_num_threads(2)
    config=Config()
    assert source_hash()==json.loads((RESULTS/'protocol.json').read_text())['source_sha256']
    market=load_market()
    assert market.timestamp(len(market)-1)<pd.Timestamp('2026-01-01T00:00:00Z')
    report={'encoder':encoder_proof(config),'cost':cost_diagnostic(market,config),'folds':[],
            'test_locked':True,'test_performance_accessed':False}
    print('ENCODER_COST',json.dumps({k:v for k,v in report.items() if k in ['encoder','cost']}),flush=True)
    for year in (2023,2024,2025):
        start,end=pd.Timestamp(f'{year}-01-01T00:00:00Z'),pd.Timestamp(f'{year+1}-01-01T00:00:00Z')
        first,last=market.bounds(start,end)
        cohort=[];summaries=[]
        minute=first+np.linspace(0,(last-first)//5-1,128,dtype=int)*5
        for seed in (11,29,47,71,101):
            state=torch.load(RESULTS/'runs'/str(year)/f'seed_{seed}'/'policy_0300.pt',map_location='cpu',weights_only=False)
            model=ActorCritic(config);model.load_state_dict(state['model']);model.eval()
            scaler=Scaler.from_dict(state['scaler'])
            prob=flat_probabilities(model,market,scaler,config,first,last)
            cohort.append(prob)
            summaries.append({'seed':seed,'flat_path_mean_probabilities':prob.mean(0,dtype=np.float64).tolist(),
                              'flat_probability_min':float(prob[:,1].min()),
                              'flat_path_argmax_counts':np.bincount(prob.argmax(-1),minlength=3).tolist(),
                              'portfolio_probe':portfolio_probe(model,market,scaler,config,minute)})
        cohort=np.asarray(cohort)
        ensemble=cohort.mean(0)
        assert (ensemble.argmax(-1)==FLAT).all()
        long_votes=(cohort.argmax(-1)==LONG).sum(0)
        event=np.flatnonzero(cohort[0].argmax(-1)==LONG)
        entry_example=None
        if len(event):
            i=int(event[0]);entry_example={'time':str(market.timestamp(first+i*5)),
                    'seed_probabilities':cohort[:,i].tolist(),'ensemble_probabilities':ensemble[i].tolist()}
        fold={'year':year,'members':summaries,'ensemble_mean_probabilities':ensemble.mean(0,dtype=np.float64).tolist(),
              'ensemble_flat_probability_min':float(ensemble[:,1].min()),
              'max_simultaneous_long_votes':int(long_votes.max()),
              'long_vote_count_distribution':np.bincount(long_votes,minlength=6).tolist(),
              'seed_11_long_example_on_common_flat_path':entry_example}
        report['folds'].append(fold)
        np.savez_compressed(HERE/f'flat_path_probabilities_{year}.npz',probabilities=cohort,first=first)
        print('FOLD_PROBE',json.dumps(fold),flush=True)
        save('probes.json',report)


if __name__=='__main__':
    main()
