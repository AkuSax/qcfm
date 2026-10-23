# QCFM: collaborator edition

This repository contains the scripts and compact results for the KLK5 feature-fusion experiments.
It also contains the frozen labels, splits, Boltz inputs, and cached AIMNet2 feature bundle.
It omits trained checkpoints, per-fit arrays, raw fit journals, prediction caches, and extraction intermediates.
The tracked files stay below 100 MB.

**Current result: NO_PROMOTION.**
The tests did not establish added predictive value from AIMNet2 features beyond the tested chemistry controls.
KLK5 remains development-only.
Keep JNK1 and p38 reserved.

## Start here

- [Latest results](exploration/20261022T232117Z/RESULTS.md)
- [Frozen experiment plan](exploration/20261022T232117Z/FROZEN_PLAN.md)
- [Fusion runner](exploration/20261022T232117Z/classical/runner.py)
- [Interaction feature map](exploration/20261022T232117Z/interaction/interaction.py)
- [Saved-prediction evaluator](exploration/20261022T232117Z/interaction/postrun_decision.py)
- [Independent audit summary](exploration/20261022T232117Z/audit/independent_audit.md)
- [Numeric reconciliation](exploration/20261022T232117Z/audit/numeric_reconciliation.json)

## Verify the files

Use Python 3.11 or later.
The file check uses the Python standard library.

```bash
python scripts/verify_archive.py
```

The command checks the published file hashes and the cached feature-bundle hash.

## Reproduce the report

```bash
python -m pip install -r requirements-reports.txt
python scripts/recompute_report.py --output results/recomputed
```

This command uses saved predictions.
It runs no estimator fit and makes no encoder call.
It compares all 102 metric rows and 210 bootstrap comparisons with the archived report.
It writes new outputs outside the frozen experiment directory.

## Results

The completed experiment used 1,464 successful fits and six exact cache reuses.
The local CPU process took 141.717 seconds.
The independent audit matched all reported metrics and bootstrap comparisons within 1e-8.

| Method | Primary pair MAE | Fixed-Hill pair MAE |
|---|---:|---:|
| Chemistry-only component | 0.599402 | 0.610852 |
| Prior quantum-only control | 0.608974 | 0.614530 |
| Inner-selected fusion | 0.655671 | 0.648069 |
| Prior final-readout adaptation | 0.691603 | 0.695594 |

Lower pair MAE is better.
The selected fusion failed the frozen component, ranking, and uncertainty rules.
The complete candidate table is in the latest results.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | File verification and report reproduction |
| `data/frozen_inputs/` | KLK5 labels, splits, exact Boltz inputs, and source provenance |
| `preparation/20261022T182444Z/arm3_environment/` | Dependency pins and AIMNet2 extraction scripts |
| `preparation/20261022T182444Z/arm4_protocol/` | Original protocol, pair manifests, and model code |
| `preparation/20261022T182444Z/orchestrator/` | Chemistry preparation and verification code |
| `preparation/20261022T182444Z/execution_full/max1000_20261022T203000Z/` | Cached feature bundle and compact prior results |
| `exploration/20261022T232117Z/` | Fusion scripts, frozen plan, predictions, results, and audit summaries |
| `publication/` | Published file hashes and compact archive scope |

The primary metric cohort has 62 of 65 source records and 136 within-group pairs.
The fixed-Hill cohort has 60 records and 129 pairs.
Both cohorts have eight groups with eligible pairs.
The feature bundle keeps all 65 rows and the three stereochemistry holds.
The fixed-Hill cohort also excludes CIDs 899202 and 976329.

Historical fitting scripts retain their original Windows paths and run guards.
A collaborator must set paths in a new run before fitting.
The report reproduction command uses repository-relative paths.
The dependency setup scripts can retrieve the pinned AIMNet2 source and weights if needed.
No downloaded dependency or pretrained weight is included.

Historical audit records can refer to files omitted from this compact edition.
The publication manifest lists the files provided here.
The full local research archive remains unchanged.

## Interpretation limits

Previous KLK5 results informed this search.
Nested evaluation cannot restore untouched status.
Training overlap remains unresolved.
The bootstrap uses fixed predictions and omits model-refit uncertainty.
The results establish no target transfer or independent improvement.
The learned ligand features do not compute protein-ligand quantum interaction energies.
