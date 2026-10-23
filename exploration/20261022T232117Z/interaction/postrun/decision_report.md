# Frozen fusion decision report

This report uses saved predictions. It fits no estimator.

Bootstrap intervals use 5,000 paired chemical-group draws with bootstrap seed 1729.
They use fixed predictions. They omit model-refit uncertainty.

## Candidate metrics

| Dataset | Method | Pair MAE mean | Seed range | Pair RMSE mean | Ligand MAE mean | Spearman mean |
|---|---|---:|---:|---:|---:|---:|
| primary | additive_chem | 0.616232 | 0.616232–0.616232 | 0.787302 | 0.558062 | -0.051598 |
| primary | additive_q | 0.646323 | 0.646323–0.646323 | 0.801116 | 0.513223 | -0.214172 |
| primary | aim | 0.627185 | 0.627185–0.627185 | 0.770471 | 0.517224 | -0.280099 |
| primary | boltz | 0.657430 | 0.657430–0.657430 | 0.820095 | 0.498218 | -0.152326 |
| primary | chem | 0.599402 | 0.599402–0.599402 | 0.755268 | 0.553906 | 0.020674 |
| primary | kernel_chem | 0.654416 | 0.654416–0.654416 | 0.795067 | 0.491548 | -0.154693 |
| primary | kernel_q | 0.637634 | 0.637634–0.637634 | 0.777391 | 0.500063 | -0.250233 |
| primary | lowrank_chem | 0.667471 | 0.667471–0.667471 | 0.848514 | 0.588414 | -0.214978 |
| primary | lowrank_q | 0.639961 | 0.639961–0.639961 | 0.782388 | 0.524179 | -0.066455 |
| primary | matched_selected_chem | 0.682887 | 0.648902–0.699880 | 0.839695 | 0.551918 | -0.177776 |
| primary | native | 0.776188 | 0.776188–0.776188 | 0.911855 | 0.573631 | 0.311324 |
| primary | prior_boltz_ridge | 0.747985 | 0.747985–0.747985 | 0.813225 | 0.459447 | 0.296769 |
| primary | prior_ordinary_readout | 0.691603 | 0.691603–0.691603 | 0.799302 | 0.494945 | 0.101659 |
| primary | prior_quantum_only | 0.608974 | 0.588613–0.624826 | 0.754753 | 0.508678 | -0.055896 |
| primary | selected_fusion | 0.655671 | 0.641220–0.662896 | 0.799720 | 0.547685 | -0.164765 |
| primary | stack_chem | 0.620701 | 0.620701–0.620701 | 0.750069 | 0.500580 | -0.070938 |
| primary | stack_q | 0.622210 | 0.622210–0.622210 | 0.743940 | 0.482913 | 0.084687 |
| sensitivity | additive_chem | 0.626752 | 0.626752–0.626752 | 0.792346 | 0.578424 | -0.112698 |
| sensitivity | additive_q | 0.644348 | 0.644348–0.644348 | 0.797391 | 0.513318 | -0.212170 |
| sensitivity | aim | 0.624983 | 0.624983–0.624983 | 0.760501 | 0.518312 | -0.299194 |
| sensitivity | boltz | 0.659080 | 0.659080–0.659080 | 0.819726 | 0.504720 | -0.228174 |
| sensitivity | chem | 0.610852 | 0.610852–0.610852 | 0.757564 | 0.576126 | -0.041734 |
| sensitivity | kernel_chem | 0.650758 | 0.650758–0.650758 | 0.786237 | 0.498867 | -0.191109 |
| sensitivity | kernel_q | 0.633451 | 0.633451–0.633451 | 0.766172 | 0.502281 | -0.206946 |
| sensitivity | lowrank_chem | 0.661061 | 0.661061–0.661061 | 0.848845 | 0.624218 | -0.314976 |
| sensitivity | lowrank_q | 0.645047 | 0.645047–0.645047 | 0.780203 | 0.513467 | 0.034787 |
| sensitivity | matched_selected_chem | 0.648284 | 0.628133–0.678361 | 0.793399 | 0.549988 | -0.182847 |
| sensitivity | native | 0.779113 | 0.779113–0.779113 | 0.913450 | 0.580078 | 0.301473 |
| sensitivity | prior_boltz_ridge | 0.748360 | 0.748360–0.748360 | 0.812809 | 0.471800 | 0.264796 |
| sensitivity | prior_ordinary_readout | 0.695594 | 0.695594–0.695594 | 0.801178 | 0.494470 | 0.087413 |
| sensitivity | prior_quantum_only | 0.614530 | 0.590452–0.627640 | 0.749656 | 0.513819 | -0.107382 |
| sensitivity | selected_fusion | 0.648069 | 0.624900–0.675128 | 0.793680 | 0.551313 | -0.182717 |
| sensitivity | stack_chem | 0.607094 | 0.607094–0.607094 | 0.725182 | 0.496747 | -0.011114 |
| sensitivity | stack_q | 0.608757 | 0.608757–0.608757 | 0.723189 | 0.486114 | 0.082801 |

## Frozen acceptance

primary: acceptance pass = False; added quantum claim allowed = False.
prior_boltz_ridge: absolute gain gate = False; relative gain gate = True; positive every seed = True; bootstrap lower bound positive = False; Spearman gate = False.
prior_ordinary_readout: absolute gain gate = False; relative gain gate = False; positive every seed = True; bootstrap lower bound positive = False; Spearman gate = False.
prior_quantum_only: absolute gain gate = False; relative gain gate = False; positive every seed = False; bootstrap lower bound positive = False; Spearman gate = False.
boltz: absolute gain gate = False; relative gain gate = False; positive every seed = False; bootstrap lower bound positive = False; Spearman gate = False.
aim: absolute gain gate = False; relative gain gate = False; positive every seed = False; bootstrap lower bound positive = False; Spearman gate = True.
chem: absolute gain gate = False; relative gain gate = False; positive every seed = False; bootstrap lower bound positive = False; Spearman gate = False.
Matched chemistry: mean gain 0.027216; all seed gains positive = True; all bootstrap lower bounds positive = False.
sensitivity: acceptance pass = False; added quantum claim allowed = False.
prior_boltz_ridge: absolute gain gate = True; relative gain gate = True; positive every seed = True; bootstrap lower bound positive = False; Spearman gate = False.
prior_ordinary_readout: absolute gain gate = False; relative gain gate = False; positive every seed = True; bootstrap lower bound positive = False; Spearman gate = False.
prior_quantum_only: absolute gain gate = False; relative gain gate = False; positive every seed = False; bootstrap lower bound positive = False; Spearman gate = False.
boltz: absolute gain gate = False; relative gain gate = False; positive every seed = False; bootstrap lower bound positive = False; Spearman gate = True.
aim: absolute gain gate = False; relative gain gate = False; positive every seed = False; bootstrap lower bound positive = False; Spearman gate = True.
chem: absolute gain gate = False; relative gain gate = False; positive every seed = False; bootstrap lower bound positive = False; Spearman gate = False.
Matched chemistry: mean gain 0.000215; all seed gains positive = False; all bootstrap lower bounds positive = False.

## Descriptive family comparisons

The JSON report and paired_bootstrap.csv include each quantum family against all mandatory comparators and its matched chemistry family.
These comparisons do not change the selected_fusion acceptance rule.

Paired bootstrap rows: 210. Each row stores model_seed and bootstrap_seed.
