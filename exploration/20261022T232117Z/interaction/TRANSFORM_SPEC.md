# Interaction transform

The module uses bundle keys `global_boltz`, `boltz_atoms`, `atom_mask`,
`aim_atoms`, `aim_atoms_free`, `aim_scalars`, `aim_scalars_free`, and
`simple_scalars` inside `bundle["X"]`. It also uses `bundle["valid"]`.

The shared runner calls `build_designs(bundle, train_idx, query_idx, method,
params)`. The method names are `additive_q`, `lowrank_q`, `additive_chem`,
and `lowrank_chem`. The frozen parameters are rank cap 4 and ridge alpha 0.1.
The function returns train and query design matrices. It does not fit an
estimator.

Block B joins 768 global Boltz values and the masked mean of 128 Boltz atom
values. Block Q joins masked means of bound and free AIM atom values, bound
and free AIM scalar values, and 18 simple chemistry values. Block C contains
the same 18 simple chemistry values.

Each block uses feature means and standard deviations from train rows.
Standard deviations at or below 1e-12 become 1. Deterministic SVD produces
at most four PCs. The largest absolute loading in each PC is positive.
Each projected PC uses its train standard deviation. The block divides by
the square root of its retained rank. The additive designs contain B plus
Q or C. The low-rank designs add matching coordinate products. Each product
uses its train mean and standard deviation. Product standard deviations at
or below 1e-12 become 1. Each product block divides by the square root of its
retained rank. Query rows use only the fitted train statistics and
axes.

The runner owns every ridge fit, seed, fold, metric, journal, and fit count.
No real-data fit is part of this module's checks.
