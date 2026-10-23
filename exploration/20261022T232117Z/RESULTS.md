# Feature-fusion result

No candidate passed the frozen fusion acceptance rule.
The experiment completed successfully.
The results do not support added predictive value from AIM features beyond the tested chemistry controls.
Do not expand atom-level fusion on this evidence.
Keep JNK1 and p38 reserved.

## Execution

The run used the local CPU.
All three sub-agents used gpt-6-luna with medium reasoning.
The run used no VM and made no encoder calls.

The frozen schedule contained 1470 estimator requests.
The run completed 1464 estimator fits.
Six exact cache reuses supplied the other requests.
All 1464 fits succeeded.
All 30 dataset, model-seed, and outer-fold units completed.
The fitting window was 135.964 seconds.
The process wall time was 141.717 seconds.
The limits were 1500 attempted fits and 14400 seconds.
The cumulative journal includes all component fits and stacking cross-fit fits.

The cached feature bundle kept SHA-256 `ee7f4c5e74f9bb3b1950c98395249948f0910348858fea0df3f4384e195fbd34`.
The final prediction SHA-256 is `ed3cc985306c0526d84c5648e8a1b96934939ee9fe74f2cd017e214d29f2083c`.
The prediction report contains 6240 rows.
Each of its 96 dataset, model-seed, and method units contains all 65 source records.

The primary metric cohort contains 62 records and 136 pairs.
The fixed-Hill cohort contains 60 records and 129 pairs.
Both cohorts contain eight groups with eligible pairs.
Both coverage reports keep the denominator of 65.
CIDs 660560, 3243170, and 6375556 remain stereochemistry holds.
The fixed-Hill cohort also excludes CIDs 899202 and 976329.

The report script needed one path repair after fitting.
The repair changed only the default feature-bundle path.
It changed no predictions, model fits, metric definitions, or acceptance rules.
The failed attempt and source snapshot remain in the repair directory.
The report retry completed with exit code zero.
The raw native sensitivity report gives one exclusion label to all five omitted records.
A separate coverage annotation corrects the three stereochemistry-hold labels.
The raw prediction file remains unchanged.

## All candidate results

These values are means over model seeds 1729, 2718, and 31415.
The metric is group-balanced within-group pair MAE.
The unit is log10 potency difference.
Lower values are better.
The outer table describes each family.
Only the selected-fusion procedure faces the frozen acceptance rule.

| Method | Primary MAE | Fixed-Hill MAE |
|---|---:|---:|
| Native Boltz | 0.776188 | 0.779113 |
| Prior Boltz ridge | 0.747985 | 0.748360 |
| Prior final-readout adaptation | 0.691603 | 0.695594 |
| Prior quantum-only control | 0.608974 | 0.614530 |
| B component: compressed Boltz | 0.657430 | 0.659080 |
| Q component: compressed AIM plus simple scalars | 0.627185 | 0.624983 |
| C component: simple chemistry and MMFF94s gap | 0.599402 | 0.610852 |
| Additive quantum fusion | 0.646323 | 0.644348 |
| Additive chemistry fusion | 0.616232 | 0.626752 |
| Multiple-kernel quantum fusion | 0.637634 | 0.633451 |
| Multiple-kernel chemistry fusion | 0.654416 | 0.650758 |
| Quantum score stacking | 0.622210 | 0.608757 |
| Chemistry score stacking | 0.620701 | 0.607094 |
| Low-rank quantum interaction | 0.639961 | 0.645047 |
| Low-rank chemistry interaction | 0.667471 | 0.661061 |
| Inner-selected fusion procedure | 0.655671 | 0.648069 |
| Fold-matched selected chemistry control | 0.682887 | 0.648284 |

All new component predictions and individual fusion-family predictions are identical across the three model seeds.
The seeds change the inner stacking cross-fit assignments.
Those assignments can change the selected family.
The selected-fusion primary MAE ranges from 0.641220 to 0.662896.
Its fixed-Hill MAE ranges from 0.624900 to 0.675128.
These repetitions do not represent three independent datasets.

## Frozen decision

The selected-fusion primary MAE improves on prior final-readout adaptation by 0.035932.
This is a 5.195 percent reduction.
The rule requires at least 0.10 and 10 percent.
The paired group-bootstrap interval includes zero in every model seed.

The selected-fusion primary MAE is worse than the prior quantum-only control by 0.046697.
It is worse than the Q component by 0.028486.
It is worse than the C component by 0.056269.
It therefore fails the component comparisons.

The selected-fusion mean Spearman correlation is -0.164765.
The prior final-readout value is 0.101659.
The seed-specific drops are 0.327063, 0.236106, and 0.236106.
Each drop exceeds the allowed 0.05.

The selected-fusion gain over its fold-matched chemistry control is 0.027216 on the primary cohort.
Its bootstrap lower bounds are -0.038456, -0.020145, and -0.020145.
The fixed-Hill mean gain is 0.000215.
The selected procedure also chooses chemistry families in 7 of 15 primary outer evaluations.
It chooses chemistry families in 11 of 15 fixed-Hill outer evaluations.
The added-quantum gate fails.

The original experiment retains its NO_PROMOTION result.
This new experiment also returns NO_PROMOTION.
No model receives promotion from a descriptive outer-fold result.

## What the family comparisons show

The intervals below use model seed 1729.
The other seeds give the same family predictions.
A positive gain favors the quantum family.
The interval uses 5000 paired chemical-group resamples.

| Quantum family versus matched chemistry family | Primary MAE gain | 95 percent interval |
|---|---:|---:|
| Additive fusion | -0.030091 | [-0.072066, 0.013550] |
| Multiple-kernel fusion | 0.016782 | [-0.006862, 0.042165] |
| Score stacking | -0.001509 | [-0.031955, 0.023946] |
| Low-rank interaction | 0.027511 | [-0.047868, 0.105020] |

Every interval includes zero.
The primary B versus Q pair-residual correlation is 0.979176.
The B versus C value is 0.954521.
The Q versus C value is 0.978876.
These values show similar pair errors across the component predictors.
They do not identify a physical cause.
The chemistry-only component has the lowest descriptive primary MAE in this comparison.
This result weakens an added-quantum interpretation of the previous quantum-only result.
It does not prove that AIM features contain no useful information.
The chemistry control contains 18 cached simple scalars. These scalars include the MMFF94s gap.
It does not cover every ordinary chemistry representation.

The new low-rank family uses pooled features and four paired PCA products.
It does not test learned atom alignment.
Its result does not justify another atom-attention expansion.
The earlier atom-attention and shuffled-atom comparisons also lacked the required gain.

## Limits and next decision

KLK5 remains development-only.
The previous results informed this search.
Nested evaluation cannot restore untouched status.
Training overlap remains unresolved.
The 136 pairs share ligands and chemical groups.
The bootstrap uses eight groups and fixed predictions.
It omits model-refit uncertainty.
The results establish no target transfer or independent improvement.
The learned ligand features do not compute protein-ligand quantum interaction energies.

Reject promotion and stop this fusion expansion.
Do not score reserved targets for model selection.
Any later experiment must define stronger chemistry controls before testing an added-quantum claim.
Select and freeze a candidate before any untouched-data confirmation.

## Independent verification

The independent evaluator matched all 102 saved metric rows within 1e-8.
It matched all 210 bootstrap mean and interval rows within 1e-8.
It matched the frozen acceptance fields.
It found no selected-family argmin or copied-prediction error.
It found no cross-fit group leakage.
It verified the six exact cache reuses and cumulative fit limits.
It verified the immutable feature hash and frozen source hashes.
The numerical result is NO_PROMOTION.

## Evidence

- Frozen design: `FROZEN_PLAN.md` and `plan_freeze_receipt.json`.
- Frozen report comparisons: `REPORT_CONTRACT.md`.
- Fit accounting and exclusion ledger: `classical/complete_attempt_accounting.json` and `.md`.
- Corrected coverage labels: see the coverage annotation and metadata amendment in `classical/`.
- Actual fit attempts: `classical/fit_journal.jsonl`.
- Saved predictions: `classical/predictions.csv`.
- Complete metric and decision tables: `interaction/postrun/decision_report.json` and `.md`.
- All 210 bootstrap comparisons: `interaction/postrun/paired_bootstrap.csv`.
- Successful report receipt: `interaction/postrun/final_postrun_receipt.json`.
- Report repair: `interaction/postrun/postrun_repair01/repair_receipt.json`.
- Independent computation: `audit/independent_audit.json` and `.md`.
- Final numerical reconciliation: `audit/numeric_reconciliation.json`.
- Direct component error comparisons: `audit/base_error_complementarity_18.csv`.

