# Synthetic twin — controlled experiments

Regenerate: `python -m pairbench.experiments.build_synthetic_twin --suite all` then `python -m pairbench.experiments.synth_ablation`. Realism check: `results/pairbench/synthetic_realism.md`.

A pair is correct iff its two detections landed on the same physical bar (Hungarian match to the analytic face centres). Recall is over `n_pairable`, the bars detected on both sides.

## A. Pitch regularity x cross-view transport

One 6-bar `l-profile` bundle against the left edge of the scan window, so transport can carry its leftmost bar in or out between captures. `pitch_jitter` is the per-bar Gaussian sigma added to a regular 69.9 mm spacing.

| pitch jitter (mm) | scenes | count-mismatch sides | greedy_index_nolen | hungarian | hungarian_aligned | monotone_aligned | hybrid_aligned |
|---|---|---|---|---|---|---|---|
| 0 | 21 | 24/42 | 0.971 | 0.362 | 0.619 | 0.619 | 0.619 |
| 1 | 21 | 24/42 | 0.990 | 0.333 | 0.714 | 0.714 | 0.714 |
| 2 | 21 | 24/42 | 0.990 | 0.333 | 0.810 | 0.810 | 0.810 |
| 5 | 21 | 24/42 | 0.981 | 0.362 | 0.952 | 0.952 | 0.952 |
| 10 | 21 | 25/42 | 0.942 | 0.365 | 1.000 | 1.000 | 1.000 |
| 20 | 21 | 26/42 | 0.922 | 0.417 | 1.000 | 1.000 | 1.000 |

From 0 mm to 20 mm of jitter: helped `hungarian` (+0.056), `hungarian_aligned` (+0.381), `monotone_aligned` (+0.381), `hybrid_aligned` (+0.381); hurt `greedy_index_nolen` (-0.049); within +-0.02 none.

Arms that estimate a cross-view translation need an irregular bundle for it to be identifiable: under regular spacing a whole-pitch shift fits as well as the truth. Instance-order arms estimate no translation.

21 scenes per jitter level; the significance section tests only the two extreme rows, and the bundle sits at the window edge, so jitter and truncation are not separated here (see section D).

Recall differences between the aligned arms, per jitter level:

| pitch jitter (mm) | hybrid - monotone | monotone - hungarian_aligned |
|---|---|---|
| 0 | +0.000 | +0.000 |
| 1 | +0.000 | +0.000 |
| 2 | +0.000 | +0.000 |
| 5 | +0.000 | +0.000 |
| 10 | +0.000 | +0.000 |
| 20 | +0.000 | +0.000 |

`hybrid_aligned` recall per (jitter, transport/pitch) cell. Periodicity in the pitch is tested in `ambiguity_sweep.md`.

| jitter \ transport/pitch | 0 | 0.25 | 0.501 | 0.751 | 1.001 | 1.252 | 1.502 |
|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.00 | 0.00 |
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.33 | 0.00 |
| 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.00 |
| 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 |
| 10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## B. Elevation x yaw: the crossing gate, dose-response

Three bed bars plus one probe bar. `elevated` puts the probe one section height above the bed between two bed bars; otherwise it lies flat one extra pitch beyond the row's end. `yaw` separates the probe's two end faces in transport-Y.

| elevated | yaw (deg) | face offset from yaw (mm) | scenes | elevated flagged (1/2) | corroborated | rigid-motion break | crossing applied | greedy_index_nolen | hungarian | hungarian_aligned | monotone_aligned | hybrid_aligned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yes | 0 | 0 | 3 | 1.0/1.0 | 3/3 | 0/3 | 0/3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| yes | 0.25 | 22 | 3 | 1.0/1.0 | 3/3 | 0/3 | 0/3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| yes | 0.5 | 43 | 3 | 1.0/1.0 | 3/3 | 3/3 | 3/3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| yes | 1 | 86 | 3 | 1.0/1.0 | 3/3 | 3/3 | 3/3 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 |
| yes | 2 | 172 | 3 | 1.0/1.0 | 3/3 | 3/3 | 3/3 | 0.750 | 1.000 | 1.000 | 0.750 | 1.000 |
| yes | 3 | 258 | 3 | 1.0/1.0 | 3/3 | 3/3 | 3/3 | 0.750 | 1.000 | 1.000 | 0.750 | 1.000 |
| yes | 5 | 430 | 3 | 1.0/0.0 | 0/3 | 0/3 | 3/3 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no | 0 | 0 | 3 | 0.0/0.0 | 0/3 | 0/3 | 0/3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no | 0.25 | 22 | 3 | 0.0/0.0 | 0/3 | 0/3 | 0/3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no | 0.5 | 43 | 3 | 0.0/0.0 | 0/3 | 0/3 | 0/3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no | 1 | 86 | 3 | 0.0/0.0 | 0/3 | 0/3 | 0/3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no | 2 | 172 | 3 | 0.0/0.0 | 0/3 | 0/3 | 0/3 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no | 3 | 258 | 3 | 0.0/0.0 | 0/3 | 0/3 | 0/3 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| no | 5 | 430 | 3 | 0.0/0.0 | 0/3 | 0/3 | 0/3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

`rigid-motion break`: elevated bars found not to travel with the bundle. `crossing applied`: the gate's final output.


## B2. A bar raised at one end only

Three bed bars plus one probe whose raised end rests between the last two bed bars and whose other end lies on the bed `lateral` mm further along Y, so it is elevated in one view only. The hybrid arm's one-view path fires when the raised end has no partner near it under the scene translation.

| lateral (mm) | raised in view | scenes | pairable | one-view crossing applied | greedy_index_nolen | hungarian | hungarian_aligned | monotone_aligned | hybrid_aligned |
|---|---|---|---|---|---|---|---|---|---|
| 120 | 1 | 3 | 12 | 3/3 | 0.000 | 0.750 | 1.000 | 0.583 | 1.000 |
| 120 | 2 | 3 | 12 | 3/3 | 0.000 | 0.750 | 1.000 | 0.583 | 1.000 |
| 170 | 1 | 3 | 12 | 3/3 | 0.000 | 0.750 | 0.750 | 0.750 | 1.000 |
| 170 | 2 | 3 | 12 | 3/3 | 0.000 | 0.750 | 0.750 | 0.750 | 1.000 |
| 220 | 1 | 3 | 9 | 3/3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 220 | 2 | 3 | 12 | 3/3 | 0.000 | 0.750 | 0.750 | 0.750 | 1.000 |


## C. Localization error, in millimetres

Detected centroid against the analytic face centre.

| profile | section height (mm) | faces | median |err| | p95 |err| | median dY | median dZ |
|---|---|---|---|---|---|---|
| l-profile | 23.8 | 3763 | 0.40 | 2.12 | -0.09 | +0.25 |
| square-profile | 40.0 | 305 | 1.00 | 5.99 | -0.35 | +0.46 |
| heavy-profile | 95.0 | 54 | 1.25 | 1.61 | -0.08 | +1.25 |
| complex-profile | 122.3 | 13 | 2.92 | 3.43 | -0.23 | +2.91 |

A dZ bias that grows with section height comes from the centroid estimator: the sensor samples the upper part of a tall face more densely. It is shared by both views and cancels in cross-view comparisons.

## D. Transport with and without truncation

Same 4-bar bundle at the window edge and centred (no bar leaves the window at any swept transport). `hybrid_aligned` recall pooled over the 6 jitter levels; `mismatch` = scene-sides whose detected count differs from the faces present.

| transport/pitch | edge recall | edge mismatch | centre recall | centre mismatch |
|---|---|---|---|---|
| 0 | 1.000 | 35/36 | 1.000 | 0/36 |
| 0.25 | 1.000 | 20/36 | 1.000 | 0/36 |
| 0.501 | 1.000 | 18/36 | 1.000 | 0/36 |
| 0.751 | 0.889 | 18/36 | 1.000 | 0/36 |
| 1.001 | 0.944 | 18/36 | 1.000 | 0/36 |
| 1.252 | 0.944 | 18/36 | 1.000 | 0/36 |
| 1.502 | 0.944 | 18/36 | 1.000 | 0/36 |

At zero transport both views truncate the same bar, so their visible sets agree; non-zero transport makes the truncation differ between views.

Zero pitch jitter only:

| transport/pitch | edge recall | centre recall |
|---|---|---|
| 0 | 1.000 | 1.000 |
| 0.25 | 1.000 | 1.000 |
| 0.501 | 1.000 | 1.000 |
| 0.751 | 0.333 | 1.000 |
| 1.001 | 0.667 | 1.000 |
| 1.252 | 0.667 | 1.000 |
| 1.502 | 1.000 | 1.000 |


## Significance at the ends of the jitter sweep

Paired scene bootstrap against `hybrid_aligned`, within each of the two extreme jitter levels. The jitter x arm interaction is not tested.

| jitter | arm | recall | delta | 95% CI | p |
|---|---|---|---|---|---|
| 0 | greedy_index_nolen | 0.971 | +0.352 | [+0.133, +0.571] | 0.000 |
| 0 | hungarian | 0.362 | -0.257 | [-0.429, -0.105] | 0.000 |
| 0 | hungarian_aligned | 0.619 | +0.000 | [+0.000, +0.000] | 1.000 |
| 0 | monotone_aligned | 0.619 | +0.000 | [+0.000, +0.000] | 1.000 |
| 20 | greedy_index_nolen | 0.922 | -0.078 | [-0.200, +0.000] | 0.249 |
| 20 | hungarian | 0.417 | -0.583 | [-0.760, -0.392] | 0.000 |
| 20 | hungarian_aligned | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| 20 | monotone_aligned | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |

