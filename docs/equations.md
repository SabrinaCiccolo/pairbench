# Equation-to-code map

Each equation mapped to the function that implements it. `file::name` is a
module-level function in that file.

| # | What it states | Implemented by | Notes |
|---|---|---|---|
| E1 | Pool selection set objective `J(S)` | `src/pairbench/detect/pool.py::select_quality_aware` | the default selector: `select_greedy` plus a quality-dominance correction pass |
| E2 | Marginal-gain surrogate `g(k \| E)` actually optimized | `src/pairbench/detect/pool.py::select_greedy` | gain recomputed against the current global explained set each step |
| E3 | Symmetry period from mask IoU, largest passing k | `src/pairbench/symmetry.py::measure_symmetry_period_deg` | |
| E4 | Folded angle difference, alias branch | `src/pairbench/pairing/common.py::angle_difference`, `src/pairbench/pairing/common.py::fold_angle` | `fold_angle` applies the alias, `angle_difference` the signed residual |
| E5 | Pair cost: twist + saturating YZ distance + level + height-outlier terms | `src/pairbench/pairing/hungarian.py::pair_cost_matrix` | one term per summand |
| E6 | Padded assignment with constant unmatch cost `u` | `src/pairbench/pairing/hungarian.py::pair_hungarian` | pads the E5 matrix to square, then solves |
| E7 | Crossing gate: cross-view corroboration AND rigid-motion break; or, for a bar elevated in one view only, a rigid-motion break of that bar alone | `src/pairbench/pairing/hybrid.py::pair_hybrid_aligned` | the two-view gate is the conjunction of `elevated_outliers` (corroboration) and `moves_with_the_bundle` (rigid-motion break); the one-view gate is `breaks_rigid_motion`, and its peeled bars are matched by `_match_one_view` on angle and translated distance only |
| E8 | Hypothesis cost: assignment cost at the hypothesis' own median translation plus half an unmatch cost per leftover detection | `src/pairbench/pairing/hypotheses.py::hypothesis_cost` | scored at `translation_of(pairs)`, so no hypothesis is charged for the anchor that proposed it |
| E9 | Re-projection model with residual `(0, t_y, t_z)` | `src/pairbench/selfcalib.py::length_residuals` | |
| E10 | Length residual, median-absolute objective | `src/pairbench/selfcalib.py::fit_steps` | minimises the median absolute `length_residuals` |
| E11 | Identifiability derivation, rank-one Jacobian | `src/pairbench/selfcalib.py::observability` | returns the rank/conditioning the derivation predicts |
| E12 | Length change under a `k`-bar lateral error, `dl(k) = sqrt(L^2 + (k p)^2) - L` | `src/pairbench/experiments/length_discriminant.py::delta_length_mm` | exact, not the `(kp)^2/2L` approximation; `offset_for_delta_mm` is its inverse |
| E13 | Hypothesis cost split into its geometric and bookkeeping summands | `src/pairbench/experiments/ambiguity_sweep.py::_cost_terms` | equals `pairing.hypotheses.hypothesis_cost`, returned as two terms |

