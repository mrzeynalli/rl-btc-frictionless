# Position-independent PPO study

Selected `frac_persistent_g090_l090` with `mean` seed aggregation and execution kappa 50.

## Walk-forward result

- Pooled 2022-2025 net return at 4 bp: 18.53%
- Pooled Sharpe: 0.169
- Continuous maximum drawdown: -32.51%
- Positive-fold share: 75.0%
- Mean starting-position return spread: 0.05%
- Median permanent convergence: 0.0 hours

## Cost sensitivity

| One-way cost | Net return | Sharpe | Max drawdown |
|---:|---:|---:|---:|
| 4 bp | 18.53% | 0.169 | -32.51% |
| 6 bp | 8.54% | 0.081 | -35.53% |
| 8 bp | -2.24% | -0.023 | -40.29% |
| 10 bp | -10.79% | -0.113 | -43.84% |

## Absorbing-state checks

- all_paths_converge_within_two_weeks: PASS
- mean_start_return_spread_below_five_percent: PASS
- transition_diagonal_share_below_98_percent: PASS
- directional_logit_counterfactual_position_sensitivity_below_tolerance: PASS
- Overall: ACCEPTED

The actor is structurally position-independent and the numerical invariance test passed. January-July 2026 is reported only as an already-open development diagnostic. A genuinely untouched confirmation is deferred.
