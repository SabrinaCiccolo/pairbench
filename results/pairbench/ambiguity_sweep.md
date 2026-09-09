# Is the cross-view ambiguity periodic in the bar pitch?

Regenerate: `python -m pairbench.experiments.build_synthetic_twin --suite ambiguity` then `python -m pairbench.experiments.ambiguity_sweep`. 234 synthetic scenes: 5 `square-profile` bars, pitch 42.1 mm, cross-view transport from 0 to 3 pitches, both placements, 3 pitch-jitter levels, 1 seeds per cell.

`margin(k) = cost(hypothesis shifted by k bars) - cost(truth)` under the aligned arms' order-constrained objective, with analytic bar identity. Positive = the objective prefers the truth. Periodicity would show as dips at whole values of transport / pitch.

## margin(k=+1), centred (no bar leaves the window)

| jitter (mm) | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 1.75 | 2 | 2.25 | 2.5 | 2.75 | 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.52 | 7.51 | 7.53 | 7.51 | 7.52 | 7.51 | 7.53 | 7.52 | 7.52 | 7.51 | 7.52 | 7.53 | 7.52 |
| 2 | 7.69 | 7.68 | 7.69 | 7.68 | 7.69 | 7.69 | 7.69 | 7.68 | 7.69 | 7.69 | 7.68 | 7.70 | 7.48 |
| 10 | 8.60 | 8.59 | 8.60 | 8.59 | 8.60 | 8.58 | 8.60 | 8.60 | 8.59 | 8.59 | 8.59 | 8.46 | 8.17 |

## margin(k=+1), at the window edge (transport moves bars in and out)

| jitter (mm) | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 1.75 | 2 | 2.25 | 2.5 | 2.75 | 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.50 | 7.50 | 7.48 | 7.50 | 7.49 | 7.49 | 7.48 | 7.48 | 7.48 | 7.49 | 7.49 | 7.47 | 7.49 |
| 2 | 7.63 | 7.61 | 7.63 | 7.60 | 7.62 | 7.63 | 7.62 | 7.62 | 7.62 | 7.61 | 7.62 | 7.62 | 7.62 |
| 10 | 8.30 | 8.32 | 8.31 | 8.32 | 8.33 | 8.32 | 8.32 | 8.32 | 8.32 | 8.32 | 8.32 | 8.32 | 8.32 |

## Where the margin comes from

Medians per placement x jitter, pooled over transport. `assignment`: matched-pair cost at the hypothesis' own translation. `bookkeeping`: half an unmatch cost per leftover detection.

| placement | jitter (mm) | assignment (truth) | assignment (k=+1) | bookkeeping (truth) | bookkeeping (k=+1) | margin |
|---|---|---|---|---|---|---|
| centre | 0 | 0.04 | 0.06 | 0.00 | 7.50 | 7.52 |
| centre | 2 | 0.07 | 0.22 | 0.00 | 7.50 | 7.69 |
| centre | 10 | 0.05 | 1.12 | 0.00 | 7.50 | 8.59 |
| edge | 0 | 0.25 | 0.23 | 3.75 | 11.25 | 7.49 |
| edge | 2 | 0.27 | 0.27 | 3.75 | 11.25 | 7.62 |
| edge | 10 | 0.23 | 0.84 | 3.75 | 11.25 | 8.32 |

## margin(k=-1), at the window edge (shift towards the truncated side)

When the two views hold different bars, a shift towards the cut side pairs as many detections as the truth, so only the assignment term separates them.

| jitter (mm) | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 1.75 | 2 | 2.25 | 2.5 | 2.75 | 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.34 | 7.34 | 7.35 | -0.08 | 0.03 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.03 | 0.01 | 0.02 |
| 2 | 7.53 | 7.30 | 7.53 | 0.08 | 0.19 | 0.23 | 0.22 | 0.23 | 0.24 | 0.18 | 0.19 | 0.18 | 0.23 |
| 10 | 8.34 | 8.31 | 7.67 | 0.92 | 1.09 | 1.11 | 1.11 | 1.11 | 1.12 | 1.10 | 1.08 | 1.11 | 1.10 |

Decomposition of margin(k=-1) at the edge, split by whether the two views detect the same number of bars (medians):

| jitter (mm) | views agree on count | scenes | assignment (truth) | assignment (k=-1) | bookkeeping (truth) | bookkeeping (k=-1) | margin |
|---|---|---|---|---|---|---|---|
| 0 | yes | 9 | 0.24 | 0.04 | 0.00 | 7.50 | 7.34 |
| 0 | no | 30 | 0.25 | 0.27 | 3.75 | 3.75 | 0.02 |
| 2 | yes | 8 | 0.27 | 0.21 | 0.00 | 7.50 | 7.53 |
| 2 | no | 31 | 0.25 | 0.38 | 3.75 | 3.75 | 0.19 |
| 10 | yes | 8 | 0.21 | 0.84 | 0.00 | 7.50 | 8.30 |
| 10 | no | 31 | 0.23 | 1.13 | 3.75 | 3.75 | 1.11 |

margin(k=-1) is negative, i.e. the objective prefers the shift, on 5 of 117 edge scenes.

## Periodicity test

- centre, jitter 0 mm: margin at whole multiples 7.52, between them 7.52, difference -0.00, range over transport 0.02.
- centre, jitter 2 mm: margin at whole multiples 7.69, between them 7.69, difference +0.00, range over transport 0.21.
- centre, jitter 10 mm: margin at whole multiples 8.60, between them 8.59, difference +0.00, range over transport 0.43.
- edge, jitter 0 mm: margin at whole multiples 7.49, between them 7.49, difference +0.01, range over transport 0.03.
- edge, jitter 2 mm: margin at whole multiples 7.62, between them 7.62, difference +0.00, range over transport 0.02.
- edge, jitter 10 mm: margin at whole multiples 8.32, between them 8.32, difference +0.00, range over transport 0.02.

Periodicity would show as a negative `difference` repeated across cells and large relative to the range (1 seeds per cell).

## Limits

- Centred, sides whose detected count differs from the faces present: jitter 0 mm, 0/6 at 0 pitches and 0/6 at 3; jitter 2 mm, 0/6 at 0 pitches and 0/6 at 3; jitter 10 mm, 0/6 at 0 pitches and 0/6 at 3.
- Centred, median truth assignment cost at the smallest vs largest transport: jitter 0 mm, 0.03 at 0 pitches against 0.25 at 3; jitter 2 mm, 0.05 at 0 pitches against 0.24 at 3; jitter 10 mm, 0.03 at 0 pitches against 0.26 at 3.
- One family (`square-profile`), 5 bars, 1 seeds per cell.
- The margin is a cost difference in the aligned arms' units, not an error rate.
- Shifted hypotheses are constructed at a chosen offset; the CSV's `margin_best` is the searched counterpart (same enumeration as `experiments.cost_gap`).

