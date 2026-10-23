# Post-run report contract

Run `postrun_decision.py` only after the classical runner saves
`classical/predictions.csv`. The script reads saved predictions and frozen
manifests. It fits no estimator.

The prediction file has one row for each of 65 source IDs for every dataset,
model seed, and new method. Required fields include dataset, model seed or
seed, method, CID, outer fold, true label, prediction, status, feature-valid
flag, cohort-valid flag, failure reason, selected family, and matched
chemistry family. Excluded rows must have status `missing` and the frozen
exclusion reason.

The script checks source hashes, row IDs, labels, folds, cohorts, exclusions,
selected families, copied selected-family predictions, and prior comparator
coverage. It uses the frozen feature-valid mask. It evaluates the primary
cohort on 62 records and 136 pairs. It evaluates fixed-Hill sensitivity on 60
records and 129 pairs. It retains the 65-record coverage denominator.

The report includes all candidate metrics, per-group pair MAE, and seed
ranges. It compares selected fusion with the frozen mandatory comparators.
It also compares each quantum family with all mandatory comparators and its
matched chemistry family. Only selected fusion can pass the frozen gate.

The script writes `decision_report.json`, `decision_report.md`, and
`paired_bootstrap.csv` under `interaction/postrun/`. It uses 5,000 paired
chemical-group bootstrap draws with seed 1729. The CSV keeps `model_seed` and
`bootstrap_seed` as separate columns. These intervals use fixed predictions
and omit model-refit uncertainty. Undefined Spearman or Kendall values become
JSON null values and appear in an explicit undefined-statistics list.

## Process launcher

`launch_classical.py` launches `classical/runner.py` with the staged Python.
It requires the lead's gate receipt. It checks the receipt, the frozen plan
hash, the 1,500 fit cap, and the 14,400-second wall cap. It sets BLAS and
OpenMP thread counts to one. It writes per-attempt stdout, stderr, and process
receipts under `interaction/launcher_runs/`. It keeps one start time and one
deadline in `interaction/launcher_state.json` across retries.
