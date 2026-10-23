# Frozen feature-fusion experiment

The lead froze this plan before new model fitting.
The run uses the local CPU.
Every sub-agent uses gpt-6-luna with medium reasoning.
The lead controls scientific design and interpretation.

## Inputs and scope

Reuse `preparation/20261022T182444Z/execution_full/max1000_20261022T203000Z/aim_features.npz`.
Require SHA-256 `ee7f4c5e74f9bb3b1950c98395249948f0910348858fea0df3f4384e195fbd34`.
Require schema SHA-256 `7ca819d8679e41273b112682ff97be79dac49c0fea09be1eadce2a65224defe5`.
Require the existing record, pair, assay, identity, and split boundaries.
Keep all 65 records in each prediction coverage report.
Keep stereochemistry holds 660560, 3243170, and 6375556.
Evaluate the primary cohort on 62 records and 136 pairs.
Refit the fixed-Hill sensitivity without 899202 and 976329.
Evaluate that sensitivity on 60 records and 129 pairs.
Keep the coverage denominator at 65.
Require eight groups with eligible pairs.
Require at least one eligible pair in each outer fold.
Do not use reserved targets.
Do not run the encoder.

## Features

Use the guarded loader with active-atom finite checks and NaN padding checks.
Record the imported loader path and its hash.
Form masked atom means before any fitted transform.
Block B contains global Boltz features and mean Boltz atom features.
Block Q contains mean bound AIM atoms, mean free AIM atoms, both AIM scalar arrays, and simple scalars.
Block C contains the 18 cached simple scalars.
Block C includes the MMFF94s conformer energy gap.
It is a simple chemistry and conformer control.
It is not a complete test of all ordinary chemistry descriptors.

Fit each block transform on the applicable training IDs only.
Standardize each input feature with its training mean and standard deviation.
Replace a training standard deviation at or below 1e-12 with one. Apply this rule to raw columns, PCA scores, product columns, and meta score columns.
Use deterministic SVD PCA with rank min(4, n_train-1, n_features).
Normalize each projected column with its training standard deviation.
Divide each block by the square root of its retained rank.
Use a deterministic sign convention or record the fitted SVD state.
Do not use labels to fit compression.

## Candidates and objective

Use ridge alpha 0.1 for all new estimators.
Use an unpenalized intercept.
Fit direct log10 IC50 labels in micromolar.
Use group-balanced within-group pair MSE plus 0.1 ligand MSE.
Use only eligible training pairs from the frozen manifests.
Use the same objective for meta models and kernel models.

Three component predictors use B, Q, and C separately.
Eight fusion families use the following fixed order:

1. additive_q: concatenate B and Q.
2. additive_chem: concatenate B and C.
3. kernel_q: combine Gaussian kernels from B and Q.
4. kernel_chem: combine Gaussian kernels from B and C.
5. stack_q: combine cross-fitted B, Q, and C predictions.
6. stack_chem: combine cross-fitted B and C predictions.
7. lowrank_q: concatenate additive_q with paired coordinate products B_j Q_j.
8. lowrank_chem: concatenate additive_chem with paired coordinate products B_j C_j.

Each kernel is exp(-squared_distance/2).
Select the Boltz kernel weight from [0.25, 0.75].
The other kernel receives the remaining weight.
No other kernel parameter changes. Minimize group-balanced pair MSE plus 0.1 ligand MSE plus alpha times c-transpose K c for kernel coefficients c. Leave the intercept unpenalized. Use train-only kernel eigenfeatures or an equivalent stable solver. Treat eigenvalues at or below 1e-12 times max(1, largest eigenvalue) as a numerical null space.
Fit product centering and scaling on training rows only.
Divide the product block by the square root of its column count.
The paired coordinate map gives a bilinear interaction of rank at most four.
It does not test learned atom alignment.
Fit meta score scaling on cross-fitted training predictions only.

Include native Boltz as a fixed comparator.
Include prior Boltz ridge, prior ordinary_readout, and prior quantum_only outer predictions as fixed comparators.
Verify their identities, labels, folds, datasets, seeds, and source hashes.
Never use those prior outer predictions for new training or selection.
The prior ordinary_readout is the strongest prior Boltz baseline by the primary metric.
This baseline adapts final predictors only.

## Nested selection and stacking

Use five frozen outer folds and two frozen inner folds.
Use model seeds 1729, 2718, and 31415.
Report deterministic repetitions honestly when a seed has no effect.
Select each kernel weight with inner predictions only.
Aggregate validation pair errors equally across eligible groups from both inner folds.
Select the fusion family with that same inner metric.
Use the listed candidate order to resolve exact ties.
Refit each family on the full outer training partition.
Report each family's outer predictions as exploratory diagnostics.
Report selected_fusion as the outer evaluation of the full inner selection procedure.
Do not select a family from its outer result.
Copy the selected family's outer predictions for selected_fusion.
That copy needs no extra fit.

Create meta-training predictions through group-respecting cross-fitting.
For an outer training partition, use its two saved inner assignments.
Reuse exact component inner fits for these cross-fitted predictions.
Use the component full outer-training fits to predict the outer test partition.
For each inner training partition, make a new deterministic two-way group split.
Use seed-specific stable group shuffling and row-count balancing.
Require at least one eligible training pair in each cross-fit training side.
Freeze these splits before fitting.
Do not use labels to balance them.
Fit each base transform on that base fit's training rows only.
Never train a base predictor on its predicted ligand or chemical group.
Share the three component cross-fit caches between the two stacks.
Bind caches to exact train IDs, predict IDs, feature hash, parameters, code hash, and model seed.

## Budget and failure rules

Use one cumulative append-only fit journal.
Reserve an attempt before each estimator fit.
Count failed fits and base cross-fitting fits.
Keep the counter across every process retry.
Use one fitting deadline of 14400 seconds.
Use at most 1500 attempted fits.
Set Torch and BLAS threads to one.

The schedule has 30 dataset/seed/outer-fold units.
Each unit has 26 inner fits, 11 outer fits, and 12 additional inner-stack cross-fit fits.
The schedule therefore has 1470 fits.
Reserve 30 attempts for concrete engineering repairs.
Do not fit both kernel weights on outer test data.
Do not expand the candidate grid.
Stop at a failed scientific check.
Retain incomplete outputs if a limit stops the run.
Check finite active features, losses, coefficients, and predictions.
Report undefined rank statistics explicitly.

## Metrics and frozen decision rules

Keep the original primary metric: macro-average within-group pair MAE.
Keep the label difference orientation y_a-y_b and score_a-score_b.
Report pair RMSE, macro group pair RMSE, ligand MAE, ligand RMSE, Spearman, Kendall tau-b, pair ordering, exact sign accuracy, and sign accuracy for absolute label difference at least 0.3.
Report each group's error and each seed's results.
Report coverage and every exclusion.
Use paired chemical-group bootstrap with 5000 resamples and bootstrap_seed 1729.
Use distinct model_seed and bootstrap_seed fields.
Report 95 percent intervals for fixed predictions.
These intervals omit model-refit uncertainty.

The original experiment keeps its NO_PROMOTION result.
The new fusion acceptance gate applies to selected_fusion only.
Require equal coverage for every mandatory comparison.
Require at least 0.10 absolute and 10 percent relative mean primary MAE reduction against each of prior Boltz ridge, prior ordinary_readout, the prior quantum_only, the new B component, the new Q component, and the new C component.
Require a positive reduction against each comparator in every model seed.
Require a paired bootstrap lower bound above zero for every seed and mandatory comparator.
Require Spearman regression no greater than 0.05 against any mandatory comparator.
Require positive mean sensitivity gain against every mandatory comparator.
Require a positive primary gain over the matched chemistry fusion in every seed.
Require its paired bootstrap lower bound above zero in every seed.
For a selected quantum family, use the corresponding chemistry family as the matched control.
If selection chooses a chemistry family in any outer fold, do not claim added quantum value from the selected procedure.
Report per-family comparisons descriptively without post-hoc promotion.
Do not change these gates after reading new outer results.

## Interpretation and audit

KLK5 remains development-only.
The previous results informed this search.
Nested evaluation cannot restore untouched status.
Training overlap remains unresolved.
The 136 pairs share ligands and groups.
Eight groups limit uncertainty estimates.
Do not claim target transfer or independent improvement.
Do not interpret learned ligand features as protein-ligand quantum interaction energies.
The audit agent must reconcile metrics and intervals from saved predictions independently.
The audit agent must check leakage and cumulative fit accounting.
Deliver all candidate results and the frozen decision evaluation.
Plan untouched-data confirmation only after a candidate passes and its recipe is frozen.

## Pre-fit design correction

The first draft used alpha 10.
The lead changed alpha to 0.1 before any fit.
The normalized feature blocks have unit total training variance.
Alpha 10 would suppress nearly all fitted variation at this scale.
This correction used no outer results.
The lead also fixed the near-zero scale rule and kernel penalty definition before fitting.


The lead added prior quantum_only as a mandatory comparator before fitting.
This comparator prevents a weak compressed Q component from lowering the acceptance standard.
It uses only the previously reported result.
It adds no new fit.

