# Joint self-calibration + assignment

Arm driving the alternation: `hybrid_aligned`. 108 scenes with pairs.

Joint estimation of the cross-view transform and the assignment. Per-scan carriage position is not in the data, so metric world-X calibration comes from scene content plus the 6005 mm bar-length prior.

## Result

**Separation s1 - s2 = 5403.3 mm**, from s1 = 2064.3, s2 = -3339.0; only the difference is identified (see observability below). Per-scene fits have IQR [5401.9, 5405.3] and range [5378.2, 5421.2] mm. A bundle-cluster bootstrap of the pooled fit (s2 held at its clamp) gives 95% [5402.8, 5403.8]. Global cross-view Z residual +1.66 mm; no global Y term is fitted (see the transport section). Converged in 2 iteration(s) (assignment identical between two rounds).

| bar-length residual | median | IQR | within +-10 mm |
|---|---|---|---|
| offline single-knot calibration | -1077.47 mm | [-1080.36, -1076.02] | 0/425 |
| one-shot fit (experiments.fit_step_positions) | -1.48 mm | [-5.11, +0.15] | — / 415 |
| joint self-calibration | -0.15 mm | [-2.99, +1.38] | 403/425 |

The one-shot row is fitted once against a fixed `hungarian` assignment. IQR 4.4 mm (joint) vs 5.3 mm (one-shot); the joint fit does not fit the calibration to pairs the calibration itself would reject.

## Pairing does not regress

Pair-count P/R/F1 vs gt, same arms, before and after recalibration. Bar-level correspondence is checked separately: `python -m pairbench.experiments.correspondence --detections results/pairbench/a2_recalibrated.json`.

| arm | F1 before | F1 after |
|---|---|---|
| greedy_index | 0.000 | 0.917 |
| hungarian | 0.973 | 0.973 |
| hungarian_aligned | 0.973 | 0.973 |
| monotone_aligned | 0.961 | 0.963 |
| hybrid_aligned | 0.975 | 0.975 |

## Observability of (s1, s2) under the bar-length prior

Gauss-Newton conditioning of the length constraint in (s1, s2), per pair. The eigenvalues of J^T J are the curvature along the best- and worst-observed directions.

Pooled over 425 pairs: eigenvalues 2.008e+00 / 2.913e-08, ratio **6.9e+07**, best-observed direction (+0.708, -0.707).

s2 comes to rest at -3339.0 mm (lowest calibration knot: -3339.0 mm): the optimizer slides along the null direction until the knot range stops it, so only s1 - s2 is a measurement.

The observed direction is the separation s1 - s2. The sum direction is NOT identified by this prior: moving both carriages together leaves every pair length unchanged, so absolute world-X is not recovered.

### Which scenes identify the separation

Per-pair curvature along the observed direction is 2.007–2.009 across all 108 scenes: d|w1 - w2| / ds is close to 1 for any bar, so precision comes from pair count and pair correctness, not scene geometry.

Separation fitted per scene: median 5403.0 mm, IQR [5401.9, 5405.3], full range [5378.2, 5421.2] mm, against the pooled 5403.3 mm. A scene far from the pooled value points to wrong pairs in that scene.

| scene | pairs | sep_alone (mm) | delta vs pooled |
|---|---|---|---|
| l-profile/tilted-01 | 2 | 5378.2 | -25.1 |
| l-profile-reshoot/overlap-08 | 4 | 5421.2 | +17.9 |
| square-profile/single-24 | 8 | 5413.5 | +10.2 |
| square-profile/single-27 | 9 | 5412.1 | +8.8 |
| square-profile/single-28 | 9 | 5412.0 | +8.8 |
| l-profile-reshoot/single-02 | 2 | 5410.6 | +7.3 |
| l-profile/overlap-14 | 3 | 5409.3 | +6.0 |
| square-profile/single-23 | 7 | 5409.3 | +6.0 |
| l-profile/standing-05 | 3 | 5408.8 | +5.5 |
| l-profile/single-10 | 2 | 5408.7 | +5.4 |

### Cross-view YZ residual: calibration in Z, transport in Y

Per-scene median (side 1 - side 2) YZ offset over that scene's pairs. A calibration residual is constant across scenes; the two views of one scan are captured at different times while the conveyor runs.

- **Z: median +1.64 mm, IQR [+1.12, +2.30], range [-9.9, +13.3]**, fitted as a calibration constant.
- **Y: median -5.36 mm, IQR [-47.56, +21.04], range [-150.1, +68.1]**, with 62 of 108 scenes past +-30 mm: bar motion along the transport axis, so no global Y term is fitted. The aligned pairing arms estimate a per-scene YZ translation by RANSAC instead.

## Alternation history

| iter | s1 | s2 | separation | t_yz | pairs | median\|resid\| |
|---|---|---|---|---|---|---|
| 1 | 2064.3 | -3339.0 | 5403.3 | (+0.0, +1.7) | 425 | 1.62 |
| 2 | 2064.3 | -3339.0 | 5403.3 | (+0.0, +1.7) | 425 | 1.62 |

## Timing (alternation)

Wall-clock, single machine, CPU only. Mean is per call, not per scene, where a stage runs once per scene-side.

| stage | calls | total s | mean ms/call |
|---|---|---|---|
| selfcalib_iter | 2 | 0.647 | 323.537 |
