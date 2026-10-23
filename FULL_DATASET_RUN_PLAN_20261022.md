# Full KLK5 dataset run

The user authorized this run on 22 October 2026.
The user requested Luna Medium agents for execution.
The main agent controls scientific scope and reviews results.

## Scope

Use the frozen KLK5 development dataset.
Retain all 65 source records in the coverage ledger.
Prepare bound and free inputs for all 62 compatible records.
Keep stereochemistry holds for CIDs 660560, 3243170, and 6375556.
Do not change labels, atom maps, coordinates, splits, or tolerances.
Use the verified local source files and pinned model weight.
Keep the source project ppi_IG read-only.
Keep TYK2, CDK2, Scp1, JNK1, and p38 outside this run.

## Existing evidence

The latest CPU smoke passed for CIDs 57341, 68261, and 570059.
The successful process took 35.985 seconds.
It used 14 calculator calls and 18 molecular evaluations.
All smoke attempts used 41 calls and 55 evaluations.
The successful smoke disabled compilation.
Its adapter removed zero padding before model calls.
Its adapter restored zero output rows after model calls.
Audit this adapter before full extraction.
Use individual molecule calls for full extraction.

## Execution

1. Audit the successful smoke and source hashes.
2. Prepare fixed-heavy-atom bound inputs and free conformers.
3. Verify each manifest against its source atom map and geometry.
4. Extract bound and free AIMNet2 features on the local CPU.
5. Preserve a journal before each model call.
6. Validate and cache each successful result.
7. Assemble a 65-row feature bundle and failure ledger.
8. Check both saved final affinity readouts in the current runtime.
9. Build the strict fitting provenance gate.
10. Run the frozen nested development comparisons and controls.
11. Report paired metrics, coverage, uncertainty, and promotion gates.

Use at most 130 cumulative extraction evaluations.
Use a 3,600 second limit for each extraction process.
Stop extraction at the first failed check.
Retry only after a concrete remedy.
Count failed calls in the cumulative journal.
Require at least 59 valid rows out of 65 before fitting.
Require eligible pairs in every outer fold.
Use the same eligible identities for principal comparisons.
Use the frozen 5 outer folds and 2 inner folds.
Use seeds 1729, 2718, and 31415.
Use at most 1,500 fitting attempts and 14,400 seconds.
Keep incomplete comparisons if the time limit stops fitting.
Do not promote a model from unequal comparison coverage.

## Interpretation

KLK5 supplies development evidence only.
Training overlap remains unresolved.
Eight nonsingleton chemical groups limit precision.
Report the gas-phase conformer energy gap in its stated units.
Do not infer a protein-ligand interaction energy from this gap.
Apply the frozen promotion thresholds without changes.
A negative result completes this experiment.

## Workspace change

The original qgcf workspace path is missing after resume.
The project files now reside at C:/Users/Saxen/OneDrive/Documents/qcfm.
Record path repairs and code hashes in each run.
Keep frozen preparation artifacts unchanged.
Place final evidence under the preparation package for the provenance gate.
No cloud VM start is required for this plan.

## Numerical preparation amendment

The original 200-step recipe prepared 56 of 65 records.
It failed the unchanged minimum of 59 valid records.
All 8 free conformers failed to converge for 6 ligands.
These ligands are 646525, 651936, 653862, 655490, 3236830, and 3240114.
Preserve this failed attempt.

Use a new numerical-preparation contract for the next attempt.
Raise the MMFF94s step limit from 200 to 1,000 for all 62 eligible ligands.
Apply this limit to bound hydrogen minimization and free conformer minimization.
Recompute every eligible record from its original deterministic inputs.
Keep 8 ETKDGv3 conformers and seed 1729.
Keep RDKit 2026.03.6 and all atom identities.
Keep bound heavy-atom coordinates fixed.
Require MMFF94s convergence before admitting a conformer.
Select the lowest-energy converged conformer.
The larger limit can change the selected free conformer for any ligand.
Record these geometry changes in the next attempt.
Keep the coverage, splitting, fitting, and promotion gates unchanged.
Record the amendment hash in the feature schema and run manifest.
No AIM output or affinity fit informed this amendment.

## Completed execution

The full eligible KLK5 run completed on the local CPU.
Luna Medium agents executed and audited the run.
All 124 bound/free AIMNet2 evaluations passed in 61.984 seconds.
The bundle retains all 65 records.
It contains 62 valid feature rows.
The three stereochemistry holds remain.
The fitting process completed all 1,500 attempts in 910.891 seconds.
It completed both cohorts, three seeds, five outer folds, and two inner folds.
The independent evaluator checked all 66 metric rows.
It recomputed all 129 bootstrap rows.
No model passed the frozen promotion rules.

The fit loader needed a masked-padding validation fix.
The first fit attempt used zero fits.
The bootstrap writer overwrote the fit-seed labels.
The audit repaired those labels from the frozen row order.
The audit verified every interval against the saved predictions.
The raw outputs remain unchanged.

Use FULL_DATASET_RESULT.md for the completed result.
