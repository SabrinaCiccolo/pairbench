# What the bar-length check can resolve

Regenerate: `python -m pairbench.experiments.length_discriminant`.

Pairing bar `i` with bar `i+k` changes the reconstructed length by `dl(k) = sqrt(L^2 + (k*p)^2) - L ~= (k*p)^2 / (2L)`, L = 6005 mm, `p` the per-family median pitch (`synthetic_sensor_model.json`). Residual: post-self-calibration length residual (`a2_calibration.json`), IQR 4.36 mm, sigma ~= 3.23 mm under a normal.

## The signal, per family

| family | pitch (mm) | n gaps | dl, 1 pitch | dl, 2 pitches | as a fraction of the ±10 mm gate | in residual sigmas |
|---|---|---|---|---|---|---|
| complex-profile | 118.4 | 3 | 1.17 mm | 4.67 mm | 0.117 | 0.36 |
| heavy-profile | 96.0 | 44 | 0.77 mm | 3.07 mm | 0.077 | 0.24 |
| l-profile | 69.2 | 167 | 0.40 mm | 1.60 mm | 0.040 | 0.12 |
| l-profile-reshoot | 38.6 | 161 | 0.12 mm | 0.50 mm | 0.012 | 0.04 |
| square-profile | 40.2 | 279 | 0.13 mm | 0.54 mm | 0.013 | 0.04 |

A one-bar error changes the length by 0.12 to 1.17 mm, against a ±10 mm gate and a residual sigma of about 3.2 mm.

## Lateral offset needed to be detectable

- to reach the ±10 mm gate: **347 mm** of lateral offset — 3 pitches for `complex-profile`, 4 pitches for `heavy-profile`, 5 pitches for `l-profile`, 9 pitches for `l-profile-reshoot`, 9 pitches for `square-profile`;
- to reach three residual sigmas (9.7 mm): **341 mm**.

Scan window after the radial gate (`synthetic_sensor_model.md`): median 258 mm, 90th percentile 337 mm, max 414 mm.

## Against the measured cost gap

`gap = cost(cheapest rival) - cost(true)` from `experiments.cost_gap`. Over the 84 scenes whose cheapest rival is a shift, Spearman rho between `dl` and `gap` is +0.251.

The objective (`pairing.hungarian.pair_cost_matrix`) has no length term; `dl` and `gap` both increase with the pitch.

On none of those scenes does the objective prefer the shift to the truth.

| scene | family | shift k | pitch (mm) | implied dl (mm) | cost gap |
|---|---|---|---|---|---|
| l-profile/tilted-01 | l-profile | -1 | 69.2 | 0.40 | +2.718 |
| square-profile/single-05 | square-profile | +1 | 40.2 | 0.13 | +3.713 |
| l-profile/overlap-13 | l-profile | +1 | 69.2 | 0.40 | +4.481 |
| square-profile/single-16 | square-profile | +1 | 40.2 | 0.13 | +4.818 |
| square-profile/single-17 | square-profile | +1 | 40.2 | 0.13 | +5.086 |
| square-profile/single-02 | square-profile | -1 | 40.2 | 0.13 | +5.312 |
| square-profile/single-14 | square-profile | +1 | 40.2 | 0.13 | +5.328 |
| l-profile/overlap-14 | l-profile | -1 | 69.2 | 0.40 | +5.457 |
| l-profile/single-11 | l-profile | -1 | 69.2 | 0.40 | +5.493 |
| square-profile/single-15 | square-profile | +1 | 40.2 | 0.13 | +5.642 |
| l-profile/single-06 | l-profile | +1 | 69.2 | 0.40 | +5.882 |
| square-profile/single-13 | square-profile | -1 | 40.2 | 0.13 | +5.948 |
| square-profile/single-23 | square-profile | -1 | 40.2 | 0.13 | +6.037 |
| heavy-profile/single-05 | heavy-profile | -1 | 96.0 | 0.77 | +6.112 |
| l-profile-reshoot/standing-05 | l-profile-reshoot | +1 | 38.6 | 0.12 | +6.149 |
| l-profile-reshoot/standing-06 | l-profile-reshoot | +1 | 38.6 | 0.12 | +6.163 |
| l-profile/overlap-03 | l-profile | -1 | 69.2 | 0.40 | +6.414 |
| square-profile/single-09 | square-profile | -1 | 40.2 | 0.13 | +6.465 |
| l-profile/overlap-04 | l-profile | -1 | 69.2 | 0.40 | +6.483 |
| l-profile/single-05 | l-profile | +1 | 69.2 | 0.40 | +6.494 |
| square-profile/single-12 | square-profile | -1 | 40.2 | 0.13 | +6.498 |
| l-profile/overlap-12 | l-profile | +1 | 69.2 | 0.40 | +6.504 |
| square-profile/single-08 | square-profile | +1 | 40.2 | 0.13 | +6.544 |
| l-profile/single-10 | l-profile | +1 | 69.2 | 0.40 | +6.558 |
| l-profile/overlap-02 | l-profile | -1 | 69.2 | 0.40 | +6.581 |
| square-profile/single-04 | square-profile | -1 | 40.2 | 0.13 | +6.606 |
| l-profile-reshoot/overlap-05 | l-profile-reshoot | +1 | 38.6 | 0.12 | +6.607 |
| square-profile/single-03 | square-profile | -1 | 40.2 | 0.13 | +6.620 |
| l-profile-reshoot/overlap-08 | l-profile-reshoot | +1 | 38.6 | 0.12 | +6.644 |
| l-profile-reshoot/overlap-09 | l-profile-reshoot | +1 | 38.6 | 0.12 | +6.653 |
| square-profile/single-10 | square-profile | +1 | 40.2 | 0.13 | +6.661 |
| l-profile/overlap-10 | l-profile | +1 | 69.2 | 0.40 | +6.711 |
| l-profile/single-01 | l-profile | +1 | 69.2 | 0.40 | +6.749 |
| l-profile/single-02 | l-profile | +1 | 69.2 | 0.40 | +6.753 |
| l-profile/single-03 | l-profile | +1 | 69.2 | 0.40 | +6.773 |
| complex-profile/single-03 | complex-profile | -1 | 118.4 | 1.17 | +6.833 |
| l-profile/crossed-04 | l-profile | -1 | 69.2 | 0.40 | +6.991 |
| l-profile/overlap-01 | l-profile | -1 | 69.2 | 0.40 | +7.004 |
| heavy-profile/single-01 | heavy-profile | +1 | 96.0 | 0.77 | +7.066 |
| heavy-profile/single-03 | heavy-profile | +1 | 96.0 | 0.77 | +7.082 |
| heavy-profile/single-02 | heavy-profile | +1 | 96.0 | 0.77 | +7.101 |
| square-profile/single-07 | square-profile | -1 | 40.2 | 0.13 | +7.222 |
| l-profile/single-04 | l-profile | -1 | 69.2 | 0.40 | +7.267 |
| l-profile-reshoot/overlap-03 | l-profile-reshoot | -1 | 38.6 | 0.12 | +7.296 |
| l-profile/crossed-05 | l-profile | -1 | 69.2 | 0.40 | +7.318 |
| l-profile/single-12 | l-profile | -1 | 69.2 | 0.40 | +7.346 |
| square-profile/single-06 | square-profile | -1 | 40.2 | 0.13 | +7.378 |
| heavy-profile/single-06 | heavy-profile | +1 | 96.0 | 0.77 | +7.379 |
| l-profile/crossed-06 | l-profile | -1 | 69.2 | 0.40 | +7.380 |
| heavy-profile/single-09 | heavy-profile | +1 | 96.0 | 0.77 | +7.387 |
| heavy-profile/single-07 | heavy-profile | +1 | 96.0 | 0.77 | +7.403 |
| heavy-profile/single-10 | heavy-profile | -1 | 96.0 | 0.77 | +7.416 |
| heavy-profile/single-08 | heavy-profile | +1 | 96.0 | 0.77 | +7.416 |
| square-profile/single-19 | square-profile | -1 | 40.2 | 0.13 | +7.426 |
| l-profile/overlap-11 | l-profile | +1 | 69.2 | 0.40 | +7.441 |
| square-profile/single-20 | square-profile | +1 | 40.2 | 0.13 | +7.442 |
| square-profile/single-21 | square-profile | +1 | 40.2 | 0.13 | +7.469 |
| square-profile/single-26 | square-profile | +1 | 40.2 | 0.13 | +7.476 |
| square-profile/single-25 | square-profile | +1 | 40.2 | 0.13 | +7.503 |
| square-profile/single-24 | square-profile | -1 | 40.2 | 0.13 | +7.749 |
| l-profile-reshoot/overlap-02 | l-profile-reshoot | -1 | 38.6 | 0.12 | +8.627 |
| l-profile-reshoot/overlap-06 | l-profile-reshoot | -1 | 38.6 | 0.12 | +8.962 |
| l-profile/single-09 | l-profile | -1 | 69.2 | 0.40 | +9.025 |
| l-profile-reshoot/overlap-04 | l-profile-reshoot | +1 | 38.6 | 0.12 | +9.032 |
| l-profile/overlap-08 | l-profile | -1 | 69.2 | 0.40 | +9.545 |
| l-profile/overlap-06 | l-profile | -1 | 69.2 | 0.40 | +9.556 |
| l-profile/overlap-09 | l-profile | -1 | 69.2 | 0.40 | +9.558 |
| l-profile/overlap-07 | l-profile | -1 | 69.2 | 0.40 | +9.561 |
| l-profile/overlap-05 | l-profile | -1 | 69.2 | 0.40 | +9.562 |
| l-profile-reshoot/tilted-04 | l-profile-reshoot | -2 | 38.6 | 0.50 | +14.621 |
| l-profile-reshoot/tilted-01 | l-profile-reshoot | -1 | 38.6 | 0.12 | +14.793 |
| l-profile-reshoot/tilted-02 | l-profile-reshoot | -1 | 38.6 | 0.12 | +14.802 |
| l-profile-reshoot/standing-04 | l-profile-reshoot | -1 | 38.6 | 0.12 | +21.480 |
| square-profile/standing-01 | square-profile | +2 | 40.2 | 0.54 | +24.287 |
| heavy-profile/single-04 | heavy-profile | -1 | 96.0 | 0.77 | +41.420 |
| heavy-profile/standing-01 | heavy-profile | -1 | 96.0 | 0.77 | +41.424 |
| l-profile/standing-03 | l-profile | -1 | 69.2 | 0.40 | +43.970 |
| l-profile/standing-04 | l-profile | -1 | 69.2 | 0.40 | +44.990 |
| l-profile/standing-01 | l-profile | +1 | 69.2 | 0.40 | +45.380 |
| l-profile/standing-05 | l-profile | -2 | 69.2 | 1.60 | +45.800 |
| l-profile/standing-02 | l-profile | -1 | 69.2 | 0.40 | +45.835 |
| l-profile-reshoot/standing-01 | l-profile-reshoot | -2 | 38.6 | 0.50 | +52.765 |
| l-profile-reshoot/standing-02 | l-profile-reshoot | -2 | 38.6 | 0.50 | +59.170 |
| l-profile-reshoot/standing-03 | l-profile-reshoot | -2 | 38.6 | 0.50 | +59.406 |

## Limits

- `dl` covers a pure lateral shift only (the length channel `greedy_index` gates on); a real mis-pair also changes tilt and twist.
- The pitch is a per-family median; per-scene spacing varies.
- The correlation is over 84 scenes in 5 families, descriptive only.

