# Pairing strategies vs gt (B1 detections)

Detections: `b1_detections_pool.json`. 108 scenes scored (`gt.FLAGGED_SCENES` excluded).

gt_pairs is a count: TP = min(pred, gt) does not verify which bars are paired (see overlay PNGs and correspondence_report.md).

Arms:

- `greedy_index`: index order per level, twist, tilt and length checks
- `greedy_index_debiased`: as above, length check centred on a global bias of -1077.4 mm (median residual over all index-matched candidates)
- `greedy_index_nolen`: as above, no length check
- `greedy_index_order`: index order only, no checks
- `hungarian`: cost-matrix assignment
- `hungarian_aligned`: `hungarian` after RANSAC YZ translation compensation
- `monotone_aligned`: same cost and translation, order-constrained per level
- `hybrid_aligned`: `monotone_aligned` per stacking level plus an evidence-gated crossing hypothesis

Angle costs fold at each profile's symmetry period (`pairbench.symmetry`): 360° for complex-profile; 180° for heavy-profile; 360° for l-profile, l-profile-reshoot; 90° for square-profile.

## Overall

`inv` = same-level Y-order inversions on non-crossed scenes, a swap proxy that count-F1 does not see. `F1 95% CI` resamples bundles with replacement (2000 reps, `pairbench.metrics.bootstrap_f1_ci`); repeat captures of one bundle (`data/repeat_captures.csv`) are resampled together.

| strategy | P | R | F1 | F1 95% CI | inv |
|---|---|---|---|---|---|
| greedy_index | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0 |
| greedy_index_debiased | 1.000 | 0.847 | 0.917 | [0.891, 0.941] | 0 |
| greedy_index_nolen | 0.997 | 0.876 | 0.933 | [0.909, 0.954] | 0 |
| greedy_index_order | 0.995 | 0.948 | 0.971 | [0.958, 0.984] | 0 |
| hungarian | 0.995 | 0.953 | 0.973 | [0.961, 0.985] | 59 |
| hungarian_aligned | 0.995 | 0.953 | 0.973 | [0.961, 0.985] | 40 |
| monotone_aligned | 0.998 | 0.928 | 0.961 | [0.940, 0.979] | 0 |
| hybrid_aligned | 0.995 | 0.955 | 0.975 | [0.961, 0.987] | 0 |

`hybrid_aligned`'s interval ([0.961, 0.987]) overlaps `greedy_index_order`, `hungarian`, `hungarian_aligned`, `monotone_aligned`: count-F1 does not separate them at this sample size (108 scenes). See `results/pairbench/correspondence_report.md` for correspondence-level scores.

## By category

| strategy | crossed | overlap | single | standing | tilted |
|---|---|---|---|---|---|
| greedy_index | 0.000 (n=32) | 0.000 (n=82) | 0.000 (n=238) | 0.000 (n=73) | 0.000 (n=18) |
| greedy_index_debiased | 0.667 (n=32) | 0.942 (n=82) | 0.921 (n=238) | 0.957 (n=73) | 0.941 (n=18) |
| greedy_index_nolen | 0.769 (n=32) | 0.942 (n=82) | 0.940 (n=238) | 0.957 (n=73) | 0.941 (n=18) |
| greedy_index_order | 0.857 (n=32) | 0.988 (n=82) | 0.970 (n=238) | 0.993 (n=73) | 1.000 (n=18) |
| hungarian | 0.857 (n=32) | 0.988 (n=82) | 0.974 (n=238) | 0.993 (n=73) | 1.000 (n=18) |
| hungarian_aligned | 0.857 (n=32) | 0.988 (n=82) | 0.974 (n=238) | 0.993 (n=73) | 1.000 (n=18) |
| monotone_aligned | 0.815 (n=32) | 0.988 (n=82) | 0.956 (n=238) | 0.993 (n=73) | 1.000 (n=18) |
| hybrid_aligned | 0.897 (n=32) | 0.988 (n=82) | 0.972 (n=238) | 0.993 (n=73) | 1.000 (n=18) |

## By profile family

`complex-profile` and `heavy-profile` carry few pairs each. `l-profile-reshoot` re-shoots `l-profile`'s scenarios with the same profile, so the two are not independent samples.

| strategy | complex-profile | heavy-profile | l-profile | l-profile-reshoot | square-profile |
|---|---|---|---|---|---|
| greedy_index | 0.000 (n=7) | 0.000 (n=33) | 0.000 (n=133) | 0.000 (n=101) | 0.000 (n=169) |
| greedy_index_debiased | 0.923 (n=7) | 0.969 (n=33) | 0.868 (n=133) | 0.964 (n=101) | 0.913 (n=169) |
| greedy_index_nolen | 0.923 (n=7) | 0.969 (n=33) | 0.892 (n=133) | 0.964 (n=101) | 0.938 (n=169) |
| greedy_index_order | 1.000 (n=7) | 1.000 (n=33) | 0.957 (n=133) | 0.990 (n=101) | 0.964 (n=169) |
| hungarian | 1.000 (n=7) | 1.000 (n=33) | 0.957 (n=133) | 0.990 (n=101) | 0.970 (n=169) |
| hungarian_aligned | 1.000 (n=7) | 1.000 (n=33) | 0.957 (n=133) | 0.990 (n=101) | 0.970 (n=169) |
| monotone_aligned | 1.000 (n=7) | 1.000 (n=33) | 0.949 (n=133) | 0.985 (n=101) | 0.947 (n=169) |
| hybrid_aligned | 1.000 (n=7) | 1.000 (n=33) | 0.965 (n=133) | 0.990 (n=101) | 0.967 (n=169) |

## Length-check residuals (greedy, |c1-c2| - 6005 mm)

n=427, median -1077.4 mm, IQR [-1080.2, -1076.0] mm; the gate is ±10 mm.

## Assignment confidence (hungarian_aligned)

Per accepted pair, how much cheaper the assigned partner was than the next-best alternative (own-index cost minus runner-up cost). Not computed for the order-constrained arms.

n=424, median 1.02, IQR [0.63, 1.95], 82 pair(s) below the 0.5 low-margin threshold (close calls, not necessarily wrong pairs).

| scene | side1 idx | side2 idx | margin |
|---|---|---|---|
| square-profile/single-16 | 0 | 2 | -4.301 |
| l-profile-reshoot/standing-05 | 0 | 10 | -3.876 |
| l-profile-reshoot/standing-06 | 0 | 10 | -3.864 |
| l-profile/overlap-14 | 1 | 0 | -3.664 |
| l-profile/single-11 | 1 | 0 | -3.551 |
| square-profile/single-02 | 3 | 0 | -3.538 |
| square-profile/single-15 | 0 | 2 | -3.489 |
| square-profile/single-12 | 2 | 0 | -3.171 |
| square-profile/single-14 | 0 | 2 | -3.048 |
| l-profile/crossed-02 | 0 | 3 | -2.990 |
| square-profile/single-18 | 15 | 20 | -2.779 |
| l-profile-reshoot/standing-07 | 6 | 7 | -2.612 |
| l-profile/single-05 | 0 | 2 | -2.159 |
| square-profile/single-05 | 1 | 2 | -2.019 |
| square-profile/single-18 | 3 | 0 | -1.995 |
| square-profile/single-17 | 5 | 7 | -1.783 |
| l-profile-reshoot/standing-07 | 4 | 5 | -1.425 |
| square-profile/single-01 | 9 | 6 | -1.124 |
| l-profile/crossed-01 | 0 | 0 | -1.046 |
| l-profile-reshoot/standing-01 | 3 | 5 | -0.888 |
| ... | | | (62 more) |


## Timing (per arm, per scene)

Wall-clock, single machine, CPU only. Mean is per call, not per scene, where a stage runs once per scene-side.

| stage | calls | total s | mean ms/call |
|---|---|---|---|
| pairing/greedy_index | 109 | 0.010 | 0.094 |
| pairing/greedy_index_debiased | 109 | 0.018 | 0.164 |
| pairing/greedy_index_nolen | 109 | 0.014 | 0.131 |
| pairing/greedy_index_order | 109 | 0.007 | 0.062 |
| pairing/hungarian | 109 | 0.034 | 0.315 |
| pairing/hungarian_aligned | 109 | 0.347 | 3.181 |
| pairing/hybrid_aligned | 109 | 0.383 | 3.518 |
| pairing/monotone_aligned | 109 | 0.352 | 3.226 |

Real-data scale only; see `results/pairbench/scaling_report.md` for the synthetic bar-count sweep.
