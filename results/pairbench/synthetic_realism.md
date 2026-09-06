# Synthetic twin — realism check

Regenerate: `python -m pairbench.experiments.build_synthetic_twin --suite realism` then `python -m pairbench.experiments.synth_realism`.
Real reference: 112 clouds from the 56 `single`-category scenes. Twin: 112 clouds from the 56 mirrored scenes (same family, same bar count).

The twin's constants are measured in `results/pairbench/synthetic_sensor_model.md`.

| statistic | real median [IQR] | twin median [IQR] | twin/real | KS D |
|---|---|---|---|---|
| n_kept | 3.16e+04 [1.61e+04, 5.96e+04] | 1.44e+04 [1.01e+04, 2.78e+04] | 0.46 | 0.33 |
| density_pts_mm2 | 16 [13.6, 19] | 11.3 [11.1, 11.5] | 0.70 | 0.92 |
| area_mm2 | 1.64e+03 [1.12e+03, 3.26e+03] | 1.29e+03 [897, 2.43e+03] | 0.79 | 0.23 |
| radial_mm | 656 [634, 674] | 654 [650, 660] | 1.00 | 0.29 |
| x_spread_mm | 127 [86.2, 211] | 24 [16.8, 32] | 0.19 | 0.84 |
| n_observed_px | 1.02e+04 [6.98e+03, 2.04e+04] | 7.96e+03 [5.61e+03, 1.53e+04] | 0.78 | 0.23 |
| n_det | 3 [3, 4] | 3 [3, 4.25] | 1.00 | 0.04 |
| px_density | 2 [1, 3] | 2 [1, 2] | 1.00 | 0.13 |
| px_x_spread_mm | 0.0249 [0, 0.0911] | 0.0391 [0, 0.149] | 1.57 | 0.13 |
| det ccorr_frac | 0.991 [0.981, 0.997] | 0.978 [0.967, 0.984] | 0.99 | 0.50 |
| det score | 0.916 [0.898, 0.93] | 0.949 [0.926, 0.965] | 1.04 | 0.54 |
| det n_points | 4.62e+03 [4.18e+03, 5.44e+03] | 3.48e+03 [3.25e+03, 3.72e+03] | 0.75 | 0.71 |

KS D: two-sample Kolmogorov-Smirnov statistic (0 = identical, 1 = disjoint).

## Detection precision/recall on real vs. synthetic

`pool` detector, count metric of `experiments.detect`.

| set | P | R | F1 | tp | fp | fn |
|---|---|---|---|---|---|---|
| real | 0.974 | 0.964 | 0.969 | 481 | 13 | 18 |
| synth | 1.000 | 0.918 | 0.957 | 461 | 0 | 41 |

Side-by-side projections, one per family: `synthetic_realism/`.

