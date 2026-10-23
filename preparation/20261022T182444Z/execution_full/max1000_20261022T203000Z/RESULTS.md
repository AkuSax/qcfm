# Frozen fit results

The guarded CPU run completed with 1,500 fits. It used all 1,500 allowed fits.

The run produced 4,290 prediction rows, 300 outer-fit rows, 66 metric rows, and 129 bootstrap rows. The independent evaluator checked all 66 metric rows. It recomputed all 129 bootstrap rows from the frozen predictions. It found no numeric mismatches after seed repair.

The primary cohort has 62 of 65 records. It has 136 common pairs. The sensitivity cohort has 60 of 65 records. It has 129 common pairs. Every seed retains at least one pair in each of the five folds.

## Primary pair macro MAE

Lower values are better. Each seed uses the same labels and pairs.

| Method | Seed 1729 | Seed 2718 | Seed 31415 | Seed mean |
|---|---:|---:|---:|---:|
| Native | 0.7762 | 0.7762 | 0.7762 | 0.7762 |
| Boltz ridge | 0.7480 | 0.7480 | 0.7480 | 0.7480 |
| Ordinary readout | 0.6916 | 0.6916 | 0.6916 | 0.6916 |
| Boltz atom capacity | 0.6754 | 0.7022 | 0.8226 | 0.7334 |
| Simple scalars | 0.9319 | 0.7805 | 0.9032 | 0.8719 |
| Quantum scalars | 0.8956 | 0.9325 | 0.8173 | 0.8818 |
| Pooled fusion | 0.7336 | 0.8029 | 0.7468 | 0.7611 |
| Atom concat | 0.7675 | 0.8794 | 0.7482 | 0.7984 |
| Atom attention | 0.8599 | 0.7347 | 0.8032 | 0.7993 |
| Quantum only | 0.6135 | 0.5886 | 0.6248 | 0.6090 |
| Attention-free control | 0.8578 | 0.7262 | 0.8013 | 0.7951 |
| Attention-shuffle control | 0.8592 | 0.7351 | 0.8028 | 0.7990 |

## Sensitivity pair macro MAE

| Method | Seed 1729 | Seed 2718 | Seed 31415 | Seed mean |
|---|---:|---:|---:|---:|
| Native | 0.7791 | 0.7791 | 0.7791 | 0.7791 |
| Boltz ridge | 0.7484 | 0.7484 | 0.7484 | 0.7484 |
| Ordinary readout | 0.6956 | 0.6956 | 0.6956 | 0.6956 |
| Boltz atom capacity | 0.6822 | 0.7300 | 0.8293 | 0.7472 |
| Simple scalars | 0.9800 | 0.7750 | 0.9650 | 0.9066 |
| Quantum scalars | 0.8993 | 0.9285 | 0.8131 | 0.8803 |
| Pooled fusion | 0.7263 | 0.8028 | 0.7463 | 0.7585 |
| Atom concat | 0.7670 | 0.8577 | 0.7131 | 0.7793 |
| Atom attention | 0.8364 | 0.7076 | 0.8026 | 0.7822 |
| Quantum only | 0.6276 | 0.5905 | 0.6255 | 0.6145 |

## Promotion checks

No model passes the frozen promotion gate.

Quantum-only reduces primary MAE versus Boltz ridge by 0.1345, 0.1594, and 0.1232 across the three seeds. The mean reduction is 0.1390, or 18.6%. Its paired bootstrap lower bound is above zero for each seed. It fails the ordinary-readout absolute threshold. Its reductions versus ordinary readout are 0.0781, 0.1030, and 0.0668. The mean is 0.0826, below the required 0.10. Its Spearman drops versus ordinary readout are -0.116, -0.177, and -0.179. Each drop exceeds the allowed 0.05 loss.

Atom attention does not improve over pooled fusion or atom concat in all seeds. Its mean reductions are -0.0382 versus pooled fusion and -0.0009 versus atom concat. It also fails the gain checks against the matched Boltz and scalar controls.

The seed mean for quantum-only is lower than the seed means for every other method. This result remains exploratory. The frozen sample has eight nonsingleton chemical groups. It does not support an independent confirmation or target-transfer claim.

## Bootstrap seed repair

The raw `bootstrap.csv` has two columns named `seed`. The writer overwrote the model seed with the fixed bootstrap RNG seed before it wrote each row. The raw file remains unchanged. The repair maps each row to the model seed from the frozen writer order. The order has 27 primary rows per seed and 16 sensitivity rows per seed. The candidate/comparator sequence matches in each repeated block.

The repaired file is `fit_retry01/bootstrap_corrected.csv`. Its repair receipt records all 129 source row mappings and both file hashes. The evaluator recomputed every confidence interval from predictions. All saved bootstrap values matched within 1e-8.

The evaluation files are `promotion_evaluation.json`, `bootstrap_corrected.csv`, `bootstrap_seed_repair_receipt.json`, and `evaluator_review.json`.
