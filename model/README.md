# Recurrent PPO agent for Bitcoin perpetual futures

A deep reinforcement learning agent that trades Bitcoin perpetual futures once per hour. It holds a short, flat or
long position of one times the account value in a USD-margined (linear) account, so a flat position leaves the equity
untouched by Bitcoin price moves. The aim is an agent that adapts to changing market regimes and survives them,
rather than one that forecasts prices.

| | |
|---|---|
| Market | BitMEX XBTUSD hourly prices, volume and funding, rebuilt from the public trade archive |
| Account | frictionless (no fees, slippage or funding), $10,000 start, position in {-1, 0, +1} |
| State | 120 market variables, each relative to its own history (volatility-scaled returns, momentum, volatility and volume percentiles, trend persistence, funding), plus a one-hot of the current position |
| Agent | separate LSTM(64) actor and critic, each followed by a 64-unit tanh layer; recurrent PPO in the CleanRL style |
| Reward | differential Sharpe ratio (Moody & Saffell), adaptation rate 1/336 |
| Training | one continuous path, 8 parallel walkers from evenly spaced starts, 10 passes (1,270 updates), seeds 11, 29, 47, 61, 83; the final model of each seed is used |
| Hyperparameters | successive-halving random search (51 → 12 → 5 configurations) scored on four walk-forward years, 2021 to 2024 |
| Test | trained to 31 Dec 2024, evaluated on 1 Jan 2025 → 1 Aug 2026 |

Result on the test period: all five seeds finish above $10,000; $16,402 on average (+64.0%) against $6,713 (−32.9%)
for buy-and-hold.

## Layout

```
src/collector/          BitMEX public-archive import into one-minute OHLCV bars and funding settlements
data_pipeline/          hourly bars and the 120 state variables  -> data/processed/xbtusd_hourly_features.parquet
src/simple_agent/       config, data scaling, environment, actor-critic model, PPO, training and evaluation
scripts/hpo_multiyear.py   the multi-year hyperparameter search and the final training runs
scripts/final_report.py    evaluates the five final models on the test period
scripts/paper_figures.py   benchmarks, regime analysis, shuffled-timing test and figures
tests/                  unit tests
```

Market data, results and model weights are not part of the repository.

## Running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pytest -q                       # unit tests
PYTHONPATH=src .venv/bin/python -m collector.bitmex                # download and build one-minute bars
PYTHONPATH=src .venv/bin/python data_pipeline/hourly_features.py   # hourly features
PYTHONPATH=src .venv/bin/python scripts/hpo_multiyear.py --workers 6
PYTHONPATH=src .venv/bin/python scripts/final_report.py
PYTHONPATH=src .venv/bin/python scripts/paper_figures.py
```
