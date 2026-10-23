# Report comparison contract

The lead fixed this comparison list before new fitting.
This list implements the frozen plan's descriptive family comparisons.
It does not change the candidate grid or acceptance gate.

Report the four quantum fusion families separately.
Use additive_q, kernel_q, stack_q, and lowrank_q.
Compare each family with prior Boltz ridge, prior ordinary_readout, prior quantum_only, B, Q, and C.
Also compare each family with its corresponding chemistry family.
Report selected_fusion against those six component and baseline comparators.
Also compare selected_fusion with its fold-matched chemistry control.
Use both cohorts and all three model seeds.
Use 5000 paired chemical-group resamples with bootstrap_seed 1729.
This list has 5 candidates times 7 comparators times 2 cohorts times 3 model seeds.
The expected bootstrap table has 210 rows.
Only selected_fusion faces the frozen acceptance gate.
The other comparisons remain descriptive.

Retain all 65 source rows in new prediction reports.
The immutable prior source has 65 primary rows per method and seed.
The immutable prior source has 65 fixed-Hill rows per method and seed. The prior native predictor has 63 available fixed-Hill predictions and two missing rows.
The prior native predictor has values beyond the common feature cohort.
Validate the prior source coverage before restricting it to the common cohort.
Use exactly 62 primary records and 60 fixed-Hill records for metrics.
Keep missing-record reasons separate from numerical predictions.

Use distinct model_seed and bootstrap_seed fields.
Write an undefined mathematical statistic as null with an explicit reason.
Do not write NaN or Infinity as JSON numbers.
Do not treat an undefined rank statistic as a failed estimator fit.
An undefined required ranking gate cannot pass.

The lead checked source row counts directly before fitting.
The first report-contract draft confused 63 available native predictions with 65 source rows.
This correction changes no metric cohort or experiment rule.

