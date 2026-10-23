# Full KLK5 dataset result

The run completed on the local CPU.
Luna Medium agents executed and audited the run.
No model passed the frozen promotion rules.

## Data and execution

The source dataset contains 65 records.
All 62 eligible ligands passed bound and free feature extraction.
The run completed 124 AIMNet2 evaluations in 61.984 seconds.
The feature bundle retains all 65 rows.
Coverage is 62/65, or 95.3846 percent.
CIDs 660560, 3243170, and 6375556 remain stereochemistry holds.

The original 200-step MMFF94s preparation failed for 6 ligands.
The run used a recorded global 1,000-step amendment for all 62 eligible ligands.
The amendment kept chemistry, seed, conformer count, and fixed bound heavy coordinates unchanged.
The run preserved the failed 200-step attempt.

The development comparison completed all 1,500 fitting attempts in 910.891 seconds.
It used 5 outer folds, 2 inner folds, and 3 seeds.
The primary cohort has 62 records and 136 pairs.
The fixed-Hill sensitivity has 60 records and 129 pairs.

## Main result

The values below are seed means of group-balanced pair MAE.
The unit is log10 potency difference.
Lower values are better.

| Method | Primary MAE | Fixed-Hill sensitivity MAE |
|---|---:|---:|
| Native Boltz | 0.7762 | 0.7791 |
| Boltz ridge | 0.7480 | 0.7484 |
| Final-readout adaptation | 0.6916 | 0.6956 |
| Pooled feature fusion | 0.7611 | 0.7585 |
| Atom-aligned attention | 0.7993 | 0.7822 |
| Quantum-only control | 0.6090 | 0.6145 |

Atom-aligned attention did not improve the required comparisons.
The shuffled-atom control gave a primary MAE of 0.7990.
The free-conformer control gave a primary MAE of 0.7951.
This experiment does not support added value from atom alignment.

The quantum-only control had the lowest pair MAE.
Its mean gain over final-readout adaptation was 0.0826.
The frozen rule requires at least 0.10.
Its ranking correlation also failed the frozen regression limit.
No candidate qualified for promotion.
Do not expand this adapter to reserved targets on this evidence.

KLK5 supplies development evidence only.
Training overlap remains unresolved.
Eight nonsingleton groups limit precision.
This result does not establish independent confirmation or target transfer.

## Verification and artifacts

The independent evaluator checked all 66 metric rows.
It recomputed all 129 bootstrap intervals.
It found no numerical mismatch.
The bootstrap CSV writer had overwritten the fit-seed labels.
The audit restored those labels from the frozen execution order.
The audit preserved the raw file and all row mappings.

- [Detailed results](preparation/20261022T182444Z/execution_full/max1000_20261022T203000Z/RESULTS.md)
- [Promotion evaluation](preparation/20261022T182444Z/execution_full/max1000_20261022T203000Z/promotion_evaluation.json)
- [Feature bundle](preparation/20261022T182444Z/execution_full/max1000_20261022T203000Z/aim_features.npz)
- [Feature schema](preparation/20261022T182444Z/execution_full/max1000_20261022T203000Z/aim_features.schema.json)
- [All-record failure ledger](preparation/20261022T182444Z/execution_full/max1000_20261022T203000Z/bundle_failure_ledger.csv)
- [Bootstrap label repair](preparation/20261022T182444Z/execution_full/max1000_20261022T203000Z/fit_retry01/bootstrap_seed_repair_receipt.json)
- [Execution plan](FULL_DATASET_RUN_PLAN_20261022.md)
