# Arm 4: Leakage-safe development protocol

The source audit passes for 65 KLK5 records and 27 chemical groups.
The audit finds 136 eligible within-group pairs.
The fixed Hill-sign sensitivity contains 63 records and 129 pairs.
Every outer fold has 13 test records.
The saved test-pair counts are 36, 29, 28, 22, and 21.
The 5 outer folds keep each group intact.
The 2 inner folds keep each training group intact.

The 65 qualifier fields are blank.
Admission uses original curve evidence.
It requires no inequality, bracketing of 50% activity, in-range fitted values, and agreement within 0.02 log units.
The protocol preserves qualifier fields as metadata.

`primary_pairs.csv` and `sensitivity_pairs.csv` contain exact held-out pairs.
`record_manifest.csv` maps each record to its held-out fold.
`source_verification.json` records hashes and checks.
`comparison_config.json` freezes comparisons, metrics, thresholds, and feature schema.
`feature_failure_ledger_template.csv` retains all 65 records.
Missing features must not receive zero values.
The coverage denominator stays at 65 records.

`protocol_support.py` contains verified pooled-input loaders, feature validation, metrics, bootstrap, and model modules.
The atom-context input uses one deduplicated 128-channel Boltz context.
The quantum atom input uses 256 channels.
The 4 declared readouts contain 55,409 to 58,145 parameters.
They remain below the 60,000 parameter cap.
Synthetic tests cover group loss, metrics, padding, batch consistency, capacity, and coverage.

Run manifest generation with `python generate_manifests.py`.
Run synthetic tests with `python -m pytest -q test_protocol_support.py test_fit_helpers.py`.
`fit_entrypoint.py` implements the future nested fitting workflow.
It requires `--authorize-fitting`, a gate receipt, a frozen feature bundle, and a new output path.
The gate receipt must set `execution_authorized`, `joins_verified`, `aim_runtime_verified`, and `feature_manifest_frozen` to true.
The runner plans 735 primary fits, 735 fixed Hill sensitivity fits, and 30 primary control fits.
The total is 1,500 fits.
It writes out-of-fold predictions, common identities, coverage, primary and secondary metrics, paired group bootstrap results, and a fit manifest.
It writes a failure receipt if a future run fails after it creates the output directory.
It adapts only saved final readout layers for the ordinary readout baseline.
It does not implement published full affinity-head adaptation.

The arm 2 audit passed 62 atom joins.
Three stereochemistry records remain held.
Full-cohort hydrogen and free-conformer preparation remain unrun.
AIM atom and scalar features do not exist in this arm.
The feature bundle must also include free-conformer AIM atoms and scalars for the free control.
No model fit or pretrained forward has run.
KLK5 remains development-only because training overlap is unresolved.
