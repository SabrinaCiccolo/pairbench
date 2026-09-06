# Pairing-strategy scaling bench (synthetic bar counts)

Synthetic scenes (`synthetic_scene`, one flat stacking level, constant 5 mm cross-view Y offset). Wall-clock, one machine, CPU only, median of 3 reps per (strategy, n).

`hungarian_aligned`/`monotone_aligned`/`hybrid_aligned` call the RANSAC translation estimate (`pairing.common.estimate_translation_ransac`) once; the other arms do not, and are swept to larger n. Blank cells are outside an arm's swept range. The fitted exponent is a regression over the swept range, not an asymptotic bound.

| strategy | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 128 | 256 | 512 | fit exponent (n>=16) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy_index | 0.10 | 0.16 | 0.29 |  | 0.66 |  | 1.68 | 4.87 | 16.43 | 64.12 | 1.55 |
| greedy_index_nolen | 0.11 | 0.15 | 0.30 |  | 0.68 |  | 1.73 | 5.04 | 16.56 | 63.46 | 1.54 |
| greedy_index_order | 0.03 | 0.06 | 0.12 |  | 0.34 |  | 1.04 | 4.20 | 14.48 | 56.15 | 1.78 |
| hungarian | 0.17 | 0.24 | 0.73 |  | 3.21 |  | 8.80 | 34.54 | 135.57 | 518.78 | 1.87 |
| hungarian_aligned | 0.91 | 3.72 | 21.86 | 68.82 | 160.38 | 592.63 | 1550.36 |  |  |  | 3.08 |
| monotone_aligned | 1.05 | 4.15 | 22.88 | 70.21 | 168.36 | 603.50 | 1594.47 |  |  |  | 3.07 |
| hybrid_aligned | 1.05 | 4.10 | 22.63 | 71.21 | 167.09 | 601.90 | 1575.40 |  |  |  | 3.06 |

Reference exponents: O(n^2) fits ~2.0, O(n^3) ~3.0, O(n^4) ~4.0.

At the bar counts of the real dataset every arm costs well under the per-scene detection time; see docs/method.md.
