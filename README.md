# Reinforcement learning for Bitcoin perpetual futures

A recurrent deep reinforcement learning agent that trades Bitcoin perpetual futures once per hour. It holds a short,
flat or long position of one times the account value in a USD-margined (linear) account, so a flat position leaves the
equity untouched by Bitcoin price moves. The aim is an agent that adapts to changing market conditions and survives
them, rather than one that forecasts prices.

| | |
|---|---|
| Market | BitMEX XBTUSD hourly prices, volume and funding, rebuilt from the public trade archive |
| Account | frictionless (no fees, slippage or funding), position in {-1, 0, +1} |
| State | 120 market variables, each relative to its own history (volatility-scaled returns, momentum, volatility and volume percentiles, trend persistence, funding), plus a one-hot of the current position |
| Network | separate LSTM actor and critic (64 or 128 units), each followed by a tanh layer of the same width |
| Reward | differential Sharpe ratio (Moody & Saffell) |
| Algorithms | `agents/ppo`: recurrent PPO (the reported agent); `agents/reinforce`: critic-free One-Batch REINFORCE |
| Evaluation | nested expanding walk-forward, 1 Jan 2021 → 1 Aug 2026: before every test year the configuration and number of passes are chosen from the preceding years only, ten agents are retrained on all earlier data, and their parameters stay fixed while they trade the year |

## Result (PPO, mean of ten random initializations, frictionless)

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (to 1 Aug) | Sharpe 2021–26 |
|---|---|---|---|---|---|---|---|
| PPO agent | +108% | +207% | +63% | +182% | +34% | +21% | 1.39 |
| Buy and hold | +60% | −64% | +156% | +121% | −6% | −28% | 0.53 |
| Time series momentum | +60% | +75% | −33% | +121% | −2% | +24% | 0.78 |

Probability of backtest overfitting of the selection 0.20; deflated Sharpe ratio 0.999.

## Layout

```
core/                   shared framework: settings, data scaling, environment and reward, LSTM actor-critic,
                        policy evaluation, rule benchmarks, training entry point (python -m core.train)
agents/ppo/             PPO update (ppo.py); results/ holds its runs (not in the repository)
agents/reinforce/       critic-free One-Batch REINFORCE (reinforce.py); results/ likewise
data_pipeline/          BitMEX archive collector (collector/) and the hourly feature table (hourly_features.py)
scripts/                walkforward_nested.py (selection + final training), walkforward_report.py (test results
                        and overfitting diagnostics), paper_figures.py (figures and tables)
tests/                  unit tests
```

Market data, results and model weights are not part of the repository.

## Running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest -q                                     # unit tests
PYTHONPATH=. .venv/bin/python -m data_pipeline.collector.bitmex                # one-minute bars and funding
PYTHONPATH=. .venv/bin/python data_pipeline/hourly_features.py                 # hourly features
PYTHONPATH=. .venv/bin/python scripts/walkforward_nested.py --algo ppo --workers 6
PYTHONPATH=.:scripts .venv/bin/python scripts/walkforward_report.py --algo ppo
PYTHONPATH=.:scripts .venv/bin/python scripts/paper_figures.py
```
