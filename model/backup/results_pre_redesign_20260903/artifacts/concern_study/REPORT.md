# Recurrent PPO concern study

All choices use economic costs and walk-forward validation only. The existing 2025-2026 test year and the untouched July 2026 holdout were not read.

Fixed budget: 225 PPO updates; 4 folds × 5 seeds.

### Burn-in and discount

| rank | configuration | median log return | positive | turnover/h | holding h | max probability | policy entropy | top-2 margin | critic EV | grad cosine |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `control_g050_b000` | 0.1841 | 65% | 0.0567 | 23.9 | 0.6403 | 0.8321 | 0.3860 | 0.0055 | -0.0511 |
| 2 | `burn_g090_b240` | 0.0814 | 60% | 0.0698 | 19.2 | 0.6390 | 0.8211 | 0.3879 | 0.0286 | 0.0808 |
| 3 | `burn_g050_b240` | 0.0687 | 65% | 0.0634 | 22.4 | 0.6492 | 0.8161 | 0.3991 | 0.0061 | 0.0231 |
| 4 | `burn_g095_b240` | 0.0683 | 60% | 0.0695 | 19.2 | 0.6385 | 0.8237 | 0.3896 | 0.0565 | 0.1066 |
| 5 | `burn_g099_b240` | 0.0220 | 55% | 0.0472 | 27.1 | 0.6231 | 0.8619 | 0.3556 | 0.1211 | 0.0310 |

### Entropy schedule

| rank | configuration | median log return | positive | turnover/h | holding h | max probability | policy entropy | top-2 margin | critic EV | grad cosine |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `burn_g090_b240` | 0.0814 | 60% | 0.0698 | 19.2 | 0.6390 | 0.8211 | 0.3879 | 0.0286 | 0.0808 |
| 2 | `entropy_g0.90_010to000` | 0.0136 | 50% | 0.0470 | 41.7 | 0.9204 | 0.2643 | 0.9410 | 0.0459 | 0.0366 |
| 3 | `entropy_g0.90_const003` | -0.0038 | 45% | 0.0421 | 49.5 | 0.9449 | 0.2002 | 0.9434 | 0.0625 | 0.0976 |
| 4 | `entropy_g0.90_const001` | -0.0126 | 45% | 0.0221 | 75.9 | 0.9930 | 0.0293 | 0.9968 | 0.0693 | 0.0007 |
| 5 | `entropy_g0.90_003to000` | -0.0309 | 40% | 0.0211 | 74.8 | 0.9923 | 0.0307 | 0.9963 | 0.0579 | -0.0153 |
| 6 | `entropy_g0.90_010to001` | -0.0409 | 40% | 0.0437 | 44.8 | 0.8970 | 0.3219 | 0.9040 | 0.0403 | 0.0489 |

### Actor/critic encoder

| rank | configuration | median log return | positive | turnover/h | holding h | max probability | policy entropy | top-2 margin | critic EV | grad cosine |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `encoder_detached_burn_g090_b240` | 0.0886 | 70% | 0.1012 | 13.9 | 0.6340 | 0.8430 | 0.3725 | 0.0341 | n/a |
| 2 | `burn_g090_b240` | 0.0814 | 60% | 0.0698 | 19.2 | 0.6390 | 0.8211 | 0.3879 | 0.0286 | 0.0808 |
| 3 | `encoder_separate_burn_g090_b240` | 0.0390 | 60% | 0.1006 | 13.4 | 0.6225 | 0.8418 | 0.3397 | 0.0449 | n/a |

### Deployment-policy diagnostics

Selected training configuration: `encoder_detached_burn_g090_b240`.

Median top-two action-probability margin: 0.3739; 5.3% of decisions are below 0.05.

| policy | median log return | turnover/h | holding h | positive |
|---|---:|---:|---:|---:|
| deterministic argmax | +0.0886 | 0.1012 | 13.9 | 70% |
| stochastic sampling | -0.8841 | 0.5364 | 2.2 | 1% |
| best hysteresis (margin 0.10) | +0.1517 | 0.0607 | 26.2 | 65% |
| best five-seed ensemble (margin 0.20) | +0.3157 | 0.0209 | 111.7 | 75% |

### Warm-up sensitivity

| hours | agreement with 512 h | first-24 h agreement | median log return |
|---:|---:|---:|---:|
| 0 | 100.0% | 100.0% | +0.1006 |
| 24 | 100.0% | 100.0% | +0.0886 |
| 72 | 100.0% | 100.0% | +0.0886 |
| 240 | 100.0% | 100.0% | +0.0886 |
| 512 | 100.0% | 100.0% | +0.0886 |

Completed at 2026-09-02T18:33:26.762258+00:00.
