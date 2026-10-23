# Complete attempt accounting

The runner completed all 30 outer units. This file reports execution and prediction coverage. It does not make a scientific decision.

The frozen schedule requested 1470 fits. The journal records 1464 fit starts: 1464 succeeded and 0 failed. Six repeated exact requests used saved predictions. The run stayed within the 1500 attempt cap and the 14400 second time limit.

The fitting window was 135.964 seconds. The launcher process took 141.717 seconds. The difference is launcher and process overhead.

A cache key includes method, ordered train and predict IDs, model seed, parameters, feature hash, runner hash, and transform hash. Each reused prediction retains its source fit attempt.

| Reuse dataset | Seed | Stage | Method | Outer fold | Origin dataset | Origin attempt | Cache key |
|---|---:|---|---|---:|---|---:|---|
| hill_sensitivity | 2718 | inner_stack_crossfit | boltz | 4 | primary | 472 | `71b17c65550737a5090434a09ac0ae218722c1b2d329e30414c41486b4b766ec` |
| hill_sensitivity | 2718 | inner_stack_crossfit | aim | 4 | primary | 473 | `c8691d6c02b17e95456d7fae363819a69ad0aa0e91487ec6ca622472c7717b3a` |
| hill_sensitivity | 2718 | inner_stack_crossfit | chem | 4 | primary | 474 | `32e13b554dc42be088889ff227b19b93e4ebbea64f8a380148985187a4ac58e2` |
| hill_sensitivity | 2718 | inner_stack_crossfit | boltz | 4 | primary | 475 | `f3eedcd98a87da459c2d70c74659a97dd52026f1f259857550d778e7a2365b0b` |
| hill_sensitivity | 2718 | inner_stack_crossfit | aim | 4 | primary | 476 | `49109fbe704af82d831137b05474a5020128c1c833e0dfc5f35c0eb40ed4202b` |
| hill_sensitivity | 2718 | inner_stack_crossfit | chem | 4 | primary | 477 | `f32572850d5326aeaced2b57a56e5f71f633e867a1056ab2b87827d2ea9a5105` |

Each row below counts a frozen request, a fit start, or an exact cache reuse. A cache reuse remains part of the 1,470 planned requests.

| Dataset | Stage | Method | Requests | Fit starts | Cache reuses | Successes | Failures |
|---|---|---|---:|---:|---:|---:|---:|
| hill_sensitivity | inner | additive_chem | 30 | 30 | 0 | 30 | 0 |
| hill_sensitivity | inner | additive_q | 30 | 30 | 0 | 30 | 0 |
| hill_sensitivity | inner | aim | 30 | 30 | 0 | 30 | 0 |
| hill_sensitivity | inner | boltz | 30 | 30 | 0 | 30 | 0 |
| hill_sensitivity | inner | chem | 30 | 30 | 0 | 30 | 0 |
| hill_sensitivity | inner | kernel_chem | 60 | 60 | 0 | 60 | 0 |
| hill_sensitivity | inner | kernel_q | 60 | 60 | 0 | 60 | 0 |
| hill_sensitivity | inner | lowrank_chem | 30 | 30 | 0 | 30 | 0 |
| hill_sensitivity | inner | lowrank_q | 30 | 30 | 0 | 30 | 0 |
| hill_sensitivity | inner | stack_chem | 30 | 30 | 0 | 30 | 0 |
| hill_sensitivity | inner | stack_q | 30 | 30 | 0 | 30 | 0 |
| hill_sensitivity | inner_stack_crossfit | aim | 60 | 58 | 2 | 58 | 0 |
| hill_sensitivity | inner_stack_crossfit | boltz | 60 | 58 | 2 | 58 | 0 |
| hill_sensitivity | inner_stack_crossfit | chem | 60 | 58 | 2 | 58 | 0 |
| hill_sensitivity | outer | additive_chem | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | additive_q | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | aim | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | boltz | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | chem | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | kernel_chem | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | kernel_q | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | lowrank_chem | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | lowrank_q | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | stack_chem | 15 | 15 | 0 | 15 | 0 |
| hill_sensitivity | outer | stack_q | 15 | 15 | 0 | 15 | 0 |
| primary | inner | additive_chem | 30 | 30 | 0 | 30 | 0 |
| primary | inner | additive_q | 30 | 30 | 0 | 30 | 0 |
| primary | inner | aim | 30 | 30 | 0 | 30 | 0 |
| primary | inner | boltz | 30 | 30 | 0 | 30 | 0 |
| primary | inner | chem | 30 | 30 | 0 | 30 | 0 |
| primary | inner | kernel_chem | 60 | 60 | 0 | 60 | 0 |
| primary | inner | kernel_q | 60 | 60 | 0 | 60 | 0 |
| primary | inner | lowrank_chem | 30 | 30 | 0 | 30 | 0 |
| primary | inner | lowrank_q | 30 | 30 | 0 | 30 | 0 |
| primary | inner | stack_chem | 30 | 30 | 0 | 30 | 0 |
| primary | inner | stack_q | 30 | 30 | 0 | 30 | 0 |
| primary | inner_stack_crossfit | aim | 60 | 60 | 0 | 60 | 0 |
| primary | inner_stack_crossfit | boltz | 60 | 60 | 0 | 60 | 0 |
| primary | inner_stack_crossfit | chem | 60 | 60 | 0 | 60 | 0 |
| primary | outer | additive_chem | 15 | 15 | 0 | 15 | 0 |
| primary | outer | additive_q | 15 | 15 | 0 | 15 | 0 |
| primary | outer | aim | 15 | 15 | 0 | 15 | 0 |
| primary | outer | boltz | 15 | 15 | 0 | 15 | 0 |
| primary | outer | chem | 15 | 15 | 0 | 15 | 0 |
| primary | outer | kernel_chem | 15 | 15 | 0 | 15 | 0 |
| primary | outer | kernel_q | 15 | 15 | 0 | 15 | 0 |
| primary | outer | lowrank_chem | 15 | 15 | 0 | 15 | 0 |
| primary | outer | lowrank_q | 15 | 15 | 0 | 15 | 0 |
| primary | outer | stack_chem | 15 | 15 | 0 | 15 | 0 |
| primary | outer | stack_q | 15 | 15 | 0 | 15 | 0 |

Predictions contain 6240 rows. Each of 96 dataset, seed, and method groups has 65 rows. The corrected sidecar is `coverage_annotation.csv` (6ca6486790a2ae4edce29a6756829423fdc626b669aef2aa80fdd1c6d40750ff). It keeps the raw reason in `raw_reason` and adds `corrected_reason` and `coverage_class`.

| Dataset | Method | Corrected reason | Coverage class | Missing rows | Excluded CIDs | Raw reasons |
|---|---|---|---|---:|---|---|
| primary | additive_chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | additive_q | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | aim | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | boltz | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | kernel_chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | kernel_q | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | lowrank_chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | lowrank_q | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | prior_boltz_ridge | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | prior_feature_missing_or_fixed_sensitivity_exclusion |
| primary | prior_ordinary_readout | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | prior_feature_missing_or_fixed_sensitivity_exclusion |
| primary | prior_quantum_only | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | prior_feature_missing_or_fixed_sensitivity_exclusion |
| primary | selected_fusion | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | stack_chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| primary | stack_q | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | additive_chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | additive_chem | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | additive_q | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | additive_q | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | aim | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | aim | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | boltz | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | boltz | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | chem | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | kernel_chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | kernel_chem | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | kernel_q | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | kernel_q | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | lowrank_chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | lowrank_chem | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | lowrank_q | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | lowrank_q | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | native | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | fixed_hill_sensitivity_exclusion |
| sensitivity | native | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | prior_boltz_ridge | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | prior_feature_missing_or_fixed_sensitivity_exclusion |
| sensitivity | prior_boltz_ridge | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | prior_feature_missing_or_fixed_sensitivity_exclusion |
| sensitivity | prior_ordinary_readout | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | prior_feature_missing_or_fixed_sensitivity_exclusion |
| sensitivity | prior_ordinary_readout | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | prior_feature_missing_or_fixed_sensitivity_exclusion |
| sensitivity | prior_quantum_only | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | prior_feature_missing_or_fixed_sensitivity_exclusion |
| sensitivity | prior_quantum_only | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | prior_feature_missing_or_fixed_sensitivity_exclusion |
| sensitivity | selected_fusion | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | selected_fusion | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | stack_chem | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | stack_chem | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |
| sensitivity | stack_q | feature_invalid | common_cohort_hold | 9 | 660560, 3243170, 6375556 | feature_invalid |
| sensitivity | stack_q | fixed_hill_sensitivity_exclusion | fixed_hill_sensitivity_exclusion | 6 | 899202, 976329 | fixed_hill_sensitivity_exclusion |

`metadata_amendment_receipt.json` records this coverage-only correction. It states that predictions, fit receipts, model rules, and metrics did not change. The correction separates three feature-invalid common-cohort holds from two fixed Hill sensitivity exclusions.

The post-run decision report assesses model performance. This accounting file makes no performance claim.
