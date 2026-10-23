# Research decision: 22 October 2026

## Revised decision after repository and literature review

**This revision supersedes the initial design below.**
Keep the initial design as a record of the earlier proposal.
Use this revision for the next experiment specification.

Select **Atom-Aligned Quantum Feature Fusion for Congeneric Affinity Prediction** as the method to test.
This is a proposed affinity adapter.
It does not retrain the co-folding model.
Do not claim priority for combining co-folding and molecular representations.
The bounded literature search does not establish a first-ever method.

Ask whether frozen quantum-model features add predictive information beyond frozen Boltz affinity features.
Then ask whether atom-aligned fusion adds value beyond pooled fusion and scalar physics descriptors.
Treat activity cliffs as a prespecified subgroup when enough verified pairs exist.
Do not assume that every congeneric series shows a cliff failure.

### Existing evidence changes the starting point

Inspect existing `ppi_IG` captures before preparing new inference.
Keep that repository read-only during this review.
The 16-ligand TYK2 archive exists locally.
Its SHA-256 matches the saved recovery receipt.
The historical inventory reports 16 verified captures.
Each saved capture contains affinity-module inputs.
One archive-member inspection confirms `s_inputs`, `z`, `x_pred`, and atom-mapping fields.
That inspection reads serialization metadata only.
It does not execute a model or establish native replay parity.

The TYK2 capture table records 1 seed and 1 diffusion sample per ligand.
It does not provide a complete 3-seed benchmark.
The experimental records contain 12 EJM measurements and 4 JMC measurements.
One EJM measurement is censored.
The historical larger-group analysis reports Spearman rho 0.866667 for the conditional score, with n=10.
That analysis excludes the reference ligand and the censored measurement.
All 16 rows retain `official_parity_status=informational_mismatch`.
Do not substitute these conditional scores for a verified native benchmark.

The KLK5 study records 65 compounds and 27 chemical groups.
It saves 2 pooled feature vectors of dimension 384 per compound.
Its gated 2-expert model fails the frozen comparison rules.
That result does not test quantum features.
It does argue against repeating a generic gate without new evidence.
The saved handoff reports that A100 replay passes and CPU replay fails.
Use the runtime required by the applicable parity checks.

The Scp-1 study records 402 scores from 414 compounds.
Its saved Spearman rho is 0.1791098923.
It has unresolved training overlap.
Its frozen cliff definition yields 0 cliff pairs.
Use it only for its stated development task.

Preserve JNK1 and p38 as reserved replication targets.
Do not inspect their model errors during adapter selection.
The prior OpenFF audit already recovered release 0.2.1.
Reuse its source manifest before downloading another copy.
Keep TYK2's publication groups separate.

### Proposed adapter and required comparisons

Freeze Boltz-2 and the selected quantum model.
Use ligand-only AIMNet2 features for the first experiment.
Use the exact ligand geometry supplied to the affinity module.
Verify element, charge, stereochemistry, and heavy-atom correspondence.
Do not infer atom correspondence from array order.
Treat quantum features as learned descriptors.
They do not establish a quantum treatment of protein-ligand interactions.

For each ligand atom, pool the Boltz pair features over receptor tokens.
Keep atom identity through pooling.
Project the resulting atom vectors into a small feature space.
Project frozen AIMNet2 invariant atomic vectors into the same feature dimension.
Train 1 small cross-attention layer and a residual readout.
Use Boltz context as queries.
Use quantum atomic vectors as keys and values.
Mask attention to the matching ligand atom and its bonded neighbors.
Build this mask from the verified ligand graph.
This mask makes atom correspondence part of the adapter.
Retain explicit strain and force descriptors as separate inputs or controlled variants.

Use 1 corrected score per ligand.
Train and evaluate differences between these scores for eligible ligand pairs.
This preserves antisymmetry and cycle consistency.
These properties do not prove accurate predictions.

Require these comparisons under identical splits and labels:

1. Unchanged native Boltz-2.
2. A regularized predictor using Boltz latent features only.
3. Ordinary affinity-readout fine-tuning with the same labels.
4. Boltz latent features plus simple molecular descriptors and MMFF strain.
5. Boltz latent features plus AIMNet2 scalar descriptors.
6. Pooled concatenation of Boltz and AIMNet2 latent features.
7. Atom-aligned cross-attention between the frozen latent features.

Add a free-conformer feature control.
This tests whether a gain depends on bound geometry.
Add a shuffled atom-correspondence control where its interpretation is valid.
Shuffle the atom-specific quantum assignment before fusion.
Keep all outputs fixed to the same chemical identity and label splits.
Keep the graph mask fixed when testing shuffled feature assignments.
Do not treat a joint reorder of features and masks as a changed correspondence.
Also compare direct per-atom concatenation before pooling.
Inspect finite prediction changes for matched chemical edits.
Keep Integrated Gradients optional.

Fit scaling, compression, projections, and hyperparameters within training folds.
Do not use representations from reserved targets to fit preprocessing.
Use target-held-out evaluation for target-transfer claims.
Use chemical-group splits for within-target development.
Report both the selected failure metric and general ranking performance.
Record coverage, missing predictions, and regressions.

### Literature changes the novelty claim

Boltz affinity fine-tuning already uses frozen structure weights and cached embeddings.
Boltz representation research already combines Boltz features with other molecular encoders.
PROBE already predicts neural-potential energy reliability from frozen atomic features with attention.
AIMNet2(Score) already combines interaction energy, desolvation, and strain.
taqo already tests single-point neural-potential scoring on congeneric series.
Rowan already reports the same 8-target, 199-ligand panel.
The evidence file records publication status and access limits.

The remaining question concerns additional affinity information under controlled evaluation.
The experiment must establish this information before assigning a physical cause.
Cross-attention alone does not establish methodological novelty.
A positive benchmark result alone does not guarantee sufficient novelty for RECOMB.

### Revised execution sequence

| Date in New York | Required result |
|---|---|
| 22-23 October | Audit retained captures, chemistry, endpoints, and native score compatibility. Freeze the feature boundary. |
| 24-25 October | Extract quantum features after execution authorization. Test pooled and atom-aligned fusion on development data. |
| 26-27 October | Apply the frozen variants to independent evaluation. Preserve reserved targets until protocol freeze. |
| 28 October | Register a substantive abstract based on verified results. |
| 29 October-4 November | Complete controls, uncertainty estimates, figures, code, and manuscript. |

Allocate 1 working day to the first feature-extraction attempt.
Allocate 1 further working day to the first fusion comparisons.
These are planning limits.
They are not measured runtimes.
If retained captures fail compatibility checks, specify a separate bounded recapture job.
Do not assume that every historical run must repeat.
Do not assume that every historical capture can support the new experiment.

Promote the quantum adapter only if it improves held-out predictions beyond latent-only adaptation and ordinary fine-tuning.
Promote cross-attention only if it improves on pooled fusion under the same evaluation.
Keep the practical thresholds in the frozen experiment specification.
Do not choose them after inspecting reserved-target results.
The initial design's generic gate and automatic 3-seed rerun are no longer default requirements.
Its proposed 6-target confirmatory summary is not an established independent cohort.
Reconcile the new assignments with existing reserved-target rules before execution.

No model inference, training, GCP authentication, or instance operation occurs during this review.
Detailed local evidence and literature records are in `RESEARCH_REVIEW.json`.

## Initial proposal retained for reference

Select **Reliability-Gated Physics Residuals for Activity-Cliff Prediction with Boltz-2**.

This is the selected research direction for RECOMB 2027.
The method and its benefit remain hypotheses.
This document records a plan, not experimental results.
The present task covers idea selection and planning.
GPU execution requires a later user instruction.

## Research question

Can a small physics correction improve Boltz-2 affinity differences between similar ligands on targets excluded from adapter training?
Can pose reliability prevent that correction from increasing errors when poses change between runs?

The proposed contribution is a reliability gate for a physics residual.
Ligand strain, neural network potentials, Boltz-2 refinement, and affinity adapters already have prior work.
Their combination alone does not establish methodological novelty.
The literature search on 22 October found related work but did not establish priority for the exact proposed method.
Complete the focused novelty audit before large GPU runs.

## Scope

Use frozen Boltz-2 weights.
Use AIMNet2 through the maintained `isayevlab/aimnetcentral` package.
Pin the package version, model revision, and model member.
Start with the public `aimnet2` model.
Treat its ligand strain as a gas-phase descriptor.
Do not interpret it as a binding free energy.

The first version uses ligand strain, ligand force summaries, and simple geometry diagnostics.
It does not require a neural potential for a cropped protein pocket.
Pocket preparation would add capping, protonation, charge, and boundary assumptions.
Those assumptions need separate validation.

Save the coordinates that the affinity module actually scores.
Also save the structure output from the structure module.
Verify the coordinate route in the pinned Boltz-2 version.
Do not assume that these two coordinate sets are identical.

Keep latent-feature fusion and Integrated Gradients as optional analyses.
They must not delay the main evaluation.
Do not train the Boltz-2 trunk.

## Proposed method

Use three fixed Boltz-2 seeds per ligand.
Use the same seed policy for every method.
Record the affinity score, coordinates, confidence values, atom mapping, and failure status.
Cache the MSA for each target.

For each pose, calculate these descriptors:

1. Add hydrogens with the specified protonation state and stereochemistry.
2. Relax hydrogen positions while fixing ligand heavy atoms.
3. Calculate heavy-atom force summaries on this geometry.
4. Relax bond lengths and angles while fixing rotatable-bond dihedrals.
5. Calculate the energy of this constrained conformation.
6. Generate 32 free conformers with fixed random seeds.
7. Optimize the free conformers with the same potential and charge.
8. Subtract the lowest sampled free-conformer energy from the constrained energy.

Call the result a **strain proxy**.
A finite conformer search does not establish a global minimum.
Check 128 conformers on a predefined subset.
Report negative strain estimates and optimization failures.
Do not silently set negative estimates to zero.
Use a force tolerance and a step limit for optimization.
Report convergence rates.
Do not assume that 25 steps establish convergence.

Use the median descriptors across the three poses as `phi_i`.
Use the median absolute deviation of the strain proxy as `u_i`.
Define the proposed score as:

```text
b_i       = median Boltz-2 score in a higher-is-stronger convention
r_i       = 1 / (1 + u_i / tau)
s_i       = b_i + r_i * (beta^T phi_i)
delta_ij  = s_i - s_j
```

Here, `tau` is positive and has the same units as `u_i`.
The reliability gate is a proposed heuristic.
It is not a calibrated probability of pose correctness.
Compare it with the same correction without the gate.

Fit `beta` with a target-balanced, regularized pairwise loss.
Use all eligible training pairs.
Do not train only on activity cliffs.
Fit descriptor scaling and select regularization within training targets.
Select `tau` within training targets.
Do not use a test target to select these values.
Set the correction to zero when the required physics descriptors fail.
Report these fallback cases.

The score difference changes sign when the ligand order changes.
Score differences also sum to zero around a closed cycle.
These properties follow from assigning one score to each ligand.
They do not prove accurate affinity prediction.

## Data and evaluation

Use OpenFF `protein-ligand-benchmark` release `0.2.1`.
Record the resolved commit and file hashes.
Fetch actual Git LFS objects.
Reject LFS pointer files during validation.

Use these eight target series:

| Target | Ligands in the current repository summary |
|---|---:|
| TYK2 | 16 |
| CDK2 | 16 |
| BACE | 36 |
| MCL1 | 42 |
| P38 | 34 |
| Thrombin | 11 |
| PTP1B | 23 |
| JNK1 | 21 |
| Total | 199 |

The table gives planning counts from the current README.
Verify the counts against the pinned release before reporting results.
These series come from multiple source studies.
Do not describe all eight as the Schindler dataset.

Use TYK2 and CDK2 for development.
Mark their results as exploratory.
Freeze the protocol before inspecting adapter performance on the other six targets.
For each confirmatory target, train on the other seven targets.
Use inner target splits for model selection.
Report exploratory targets separately from the six-target confirmatory summary.

Audit protein similarity and ligand duplicates across targets.
LOTO excludes a target from adapter training.
It does not establish independence between related protein families.
Add a kinase-family holdout sensitivity analysis if the data support it.
Document possible Boltz-2 pretraining exposure.
Do not claim that LOTO removes pretraining overlap.

Use matched molecular pairs with one transformation as the primary pair set.
Use RDKit matched-pair fragmentation.
Limit each variable fragment to eight heavy atoms.
Require the common core to contain at least 50% of the heavy atoms of each ligand.
Freeze atom mapping and pair generation before inspecting model errors.
Report the benchmark perturbation edges as a secondary pair set.
Count pairs per target before GPU work.
Do not change the pair definition to increase a model's reported gain.

Compare measurements within the same assay and endpoint type.
Keep Ki and IC50 labels distinct.
Exclude censored measurements from the primary error calculation.
Record their exclusion.
Check units and score direction with explicit examples.
Boltz-2 reports `log10(IC50 in micromolar)`.
For an IC50 interpretation, the corresponding pIC50 score is `6 - affinity_pred_value`.
Do not interpret that conversion as a conversion from IC50 to Ki.

Define a cliff as an eligible pair with an experimental pActivity difference of at least 1.5.
Use the absolute difference for this definition.
Define pActivity separately for each assay endpoint.

Primary metric: mean absolute error of the predicted pActivity difference on cliff pairs.
Calculate each target's metric first.
Then average the target metrics with equal weight.
Report pair counts and target counts with every summary.
Report pairwise sign accuracy, all-pair error, Spearman correlation, and Kendall correlation as secondary metrics.
Use a tie-aware definition for sign accuracy.
Report results for charged and neutral ligands separately.
Use target-level resampling for uncertainty estimates.
Do not treat pairs that share ligands as independent observations.
State that six confirmatory targets provide limited precision.

## Required comparisons

Use the same successful-input set for paired metric comparisons.
Also report full-set performance with the predefined baseline fallback.
Report coverage for each method.

| Comparison | Purpose |
|---|---|
| Frozen Boltz-2 | Establish the reference error. |
| Boltz-2 plus simple molecular descriptors | Test whether charge or size explains the gain. |
| MMFF strain residual | Test whether the neural potential adds useful information. |
| AIMNet2 descriptor-only model | Test whether Boltz-2 adds useful information. |
| Boltz-2 plus ungated AIMNet2 residual | Test the physics correction. |
| Boltz-2 plus reliability-gated residual | Test the proposed method. |
| Shuffled reliability values | Test whether reliability carries useful information. |

Simple molecular descriptors include formal charge, heavy-atom count, and rotatable-bond count.
Apply the same training split rules to all learned comparisons.
Fit shuffled-gate comparisons within training folds.
Record MMFF coverage failures.

## Decision criteria

These are project criteria.
They are not conference requirements or expected results.

Before the first GPU run:

- Confirm model and data access.
- Complete the focused novelty audit.
- Verify units, assay groups, atom mappings, and pair counts.
- Require at least 20 confirmatory cliff pairs across at least four targets.
- Require at least three cliff pairs in each target included in the cliff summary.
- Preserve the selected research question if a target fails data validation.

By 25 October:

- Complete TYK2 and CDK2 runs.
- Verify the scored-coordinate capture.
- Measure runtime, memory use, and optimization coverage.
- Require at least 90% usable physics coverage on the development ligands.
- Check whether force and strain descriptors change strongly with pose seed.

By 27 October:

- Produce predictions from the frozen confirmatory evaluation.
- Compare the proposed method with Boltz-2 and the ungated residual.
- Check the direction of the gain across targets.
- Use a 10% reduction in macro cliff error as a practical project target.
- Require improvement on a majority of eligible confirmatory targets.
- Report uncertainty instead of treating the practical target as statistical proof.
- Investigate any loss above 0.1 pActivity units in macro all-pair error.

If the gate adds no benefit, do not claim it as the contribution.
If the neural potential adds no benefit over MMFF, report that result.
If the evidence does not support the method, do not promise a positive RECOMB paper.
Do not replace the method with a different central idea after abstract registration.

## Schedule

| Date in New York | Work and output |
|---|---|
| 22 October | Select the method. Record the protocol. Complete the focused novelty audit. |
| 23 October | Validate data. Generate pairs. Prepare isolated environments and GPU job specifications. |
| 24-25 October | Run the development targets after GCP approval. Measure runtime and coverage. Freeze the protocol. |
| 26-27 October | Run the remaining targets. Fit adapters with nested target splits. Produce the comparison table. |
| 28 October | Register a substantive abstract. Use only verified results. Keep the central method stable. |
| 29-31 October | Complete sensitivity analyses. Prepare case studies and figures. Write the main paper. |
| 1-3 November | Complete the paper and documented code. Verify reproduction from cached inputs. |
| 4 November | Check the PDF, public repository link, and submission files. Submit the paper. |

Official abstract deadline: **28 October 2026, 23:59 AoE**.
This equals **29 October, 07:59 EDT** in New York.
Use **28 October, 18:00 EDT** as the internal submission target.

Official paper deadline: **4 November 2026, 23:59 AoE**.
This equals **5 November, 06:59 EST** in New York.
Use **4 November, 18:00 EST** as the internal submission target.
New York changes from EDT to EST on 1 November.

The official rules require meaningful abstract content.
They do not specify a required number of pilot experiments.
Substantial title or abstract changes can cause removal from consideration.
The main paper has a 10-page limit.
Use Letter pages, a font of at least 10 pt, and 1-inch margins.
An additional title and abstract page is permitted.
References do not count toward the limit.
Review is single-blind.
This methods project requires public source code with a documented README by submission.
Include the repository link in the paper.
The main paper must remain understandable without the optional appendix.

## Attribution and scientific claims

Use descriptor ablation and matched-pair examples as the main interpretation tools.
Include both successful corrections and failed corrections.
Select cases with a recorded rule.
Do not present one selected pair as proof of a general mechanism.

The Boltz-2 affinity code converts representative-atom distances into discrete bins.
Double precision does not make that operation differentiable.
Integrated Gradients on embedded features can describe model sensitivity.
It does not prove a physical cause.
A smooth replacement for the distance bins changes the model.
Validate that replacement before using its gradients.

Do not claim that evolutionary smoothing causes affinity errors without evidence.
Do not call neural-potential energies exact quantum energies.
Do not call a high ligand force a specific protein-ligand clash diagnosis.
Do not equate gas-phase strain with a binding free-energy penalty.
Charged ligands need special scrutiny because gas-phase minima can distort the strain reference.

## GCP plan

No GCP authentication or instance operation occurred during planning.
Use the existing A100 80 GB VM only after the user confirms the GPU run.
Verify current instance status and quota after login.
Use an isolated project directory and an isolated environment.
The current AIMNet2 package requires Python 3.11+ and PyTorch 2.10+.
The VM's documented Python 3.10 environment therefore needs a separate environment.
Do not alter previous project environments.

Start with a bounded timing run.
Use the measured runtime to estimate the full job cost.
Set a run limit and arrange VM shutdown after completion or failure.
Retain logs and partial outputs.
Only one A2 VM can run under the documented CPU quota.
Do not use the capture VM for this work.
Its documented watchdog can stop jobs.

## Evidence checked on 22 October 2026

- [Official RECOMB call for papers](https://recomb.org/recomb2027/call_for_papers): deadlines, scope, format, code, and abstract rules.
- [Official RECOMB dates](https://recomb.org/recomb2027/keydates_and_deadlines.html): registration cutoff and update window.
- [Boltz repository](https://github.com/jwohlwend/boltz): score meaning and public model access.
- [Boltz affinity source](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/affinity.py): discrete distance bins and affinity inputs.
- [Boltz-2 technical report](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full): existing affinity training, pairwise supervision, activity-cliff sampling, and FEP benchmark evaluation.
- [OpenFF benchmark](https://github.com/openforcefield/protein-ligand-benchmark): current target counts and provenance.
- [OpenFF release 0.2.1 README](https://github.com/openforcefield/protein-ligand-benchmark/blob/0.2.1/README.md): metadata, units, perturbation edges, Git LFS, and licenses.
- [Maintained AIMNet2 repository](https://github.com/isayevlab/aimnetcentral): installation, model choices, supported elements, and energy units.
- [AIMNet2 strain project](https://github.com/isayevlab/ligand-strain-aimnet2): existing strain methods and charged-ligand reference problems. Its README states that CPCM model weights will be released with the paper. Do not depend on those weights until access works.
- [Boltz-ABFE](https://arxiv.org/abs/2508.19385): prior physics refinement from Boltz structures.
- [Boltz-2 affinity fine-tuning](https://www.biorxiv.org/content/10.64898/2026.05.26.727958v1): prior adaptation of the affinity model.
- [Pose-to-affinity analysis and code](https://github.com/xeureka-research/boltz2-affinity-pose-sensitivity): prior analysis of scored coordinates and pose sensitivity. Verify its version-specific findings against the selected version.

The plan rejects the Gemini runtime promises of 200 complex relaxations in under four minutes and a two-minute installation.
No local measurements support those promises.
The plan also rejects a guaranteed positive affinity result.
