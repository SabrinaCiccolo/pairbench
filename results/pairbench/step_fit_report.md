# Step-position fit (full calibration, bar-length prior)

Input: 415 hungarian centroid pairs, 108 scenes.

**Fitted: s1 = 2062.8 mm, s2 = -3339.0 mm** -> length residual median -1.48 mm, IQR [-5.11, +0.15] mm.

Length check (±10 mm around 6005): 359/415 pairs pass.

Greedy pairing with the length check, on the recalibrated centroids: P 1.000 R 0.767 F1 0.868.

## Degeneracy valley (top coarse-grid cells)

The bar length constrains the X separation, not s1/s2 individually;
the cells below are near-equivalent under the fitted objective.

| s1 | s2 | median\|resid\| mm |
|---|---|---|
| 2081 | -3339 | 17.06 |
| 2131 | -3289 | 17.14 |
| 2182 | -3238 | 17.21 |
| 2232 | -3188 | 17.27 |
| 2282 | -3138 | 17.33 |
| 2333 | -3087 | 17.40 |
| 2383 | -3037 | 17.48 |
| 2433 | -2987 | 17.56 |
