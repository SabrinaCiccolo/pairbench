# Detector transfer — which pairing arms survive a change of detector

Regenerate: `python -m pairbench.experiments.detector_transfer`.

Scorings `B1 confirmed` and `b1_detections_ccorr confirmed` (confirmed detections only), on the 108 scenes where both detectors resolved at least one claim (407 resolved claims under `pool`, 382 under `ccorr`). Recall is pooled over each detector's own resolved claims. `greedy_index` is excluded: the calibration's world-X bias puts every candidate outside its length gate under either detector.

## Transfer delta per arm

`delta` = recall(pool) - recall(ccorr). Paired bootstrap over 96 bundles (10000 replicates, one draw shared by both detectors and every arm); `moves?` = interval excludes zero.

| arm | recall (pool) | recall (ccorr) | delta | 95% paired CI | p | moves? |
|---|---|---|---|---|---|---|
| `monotone_aligned` | 0.865 | 0.861 | +0.004 | [-0.058, +0.065] | 0.8946 | no |
| `greedy_index_order` | 0.872 | 0.856 | +0.016 | [-0.021, +0.053] | 0.4010 | no |
| `hybrid_aligned` | 0.929 | 0.911 | +0.018 | [-0.049, +0.082] | 0.5928 | no |
| `hungarian_aligned` | 0.779 | 0.801 | -0.022 | [-0.108, +0.063] | 0.6496 | no |
| `hungarian` | 0.762 | 0.694 | +0.068 | [+0.017, +0.122] | 0.0042 | yes |
| `greedy_index_debiased` | 0.816 | 0.309 | +0.507 | [+0.431, +0.583] | 0.0000 | yes |
| `greedy_index_nolen` | 0.835 | 0.317 | +0.519 | [+0.439, +0.597] | 0.0000 | yes |

## Difference of differences vs the order anchor

|delta(greedy_index_nolen)| - |delta(arm)| on the same paired draw; positive = the arm is more detector-stable. Holm-Bonferroni over 6 comparisons, alpha 0.05.

| arm | |delta| | advantage over the order anchor | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `monotone_aligned` | 0.004 | +0.515 | [+0.406, +0.577] | 0.0000 | yes | yes |
| `greedy_index_order` | 0.016 | +0.502 | [+0.418, +0.579] | 0.0000 | yes | yes |
| `hybrid_aligned` | 0.018 | +0.501 | [+0.397, +0.576] | 0.0000 | yes | yes |
| `hungarian_aligned` | 0.022 | +0.496 | [+0.374, +0.574] | 0.0000 | yes | yes |
| `hungarian` | 0.068 | +0.451 | [+0.353, +0.553] | 0.0000 | yes | yes |
| `greedy_index_debiased` | 0.507 | +0.012 | [-0.011, +0.037] | 0.3746 | no | no |

## Reading

`greedy_index_nolen` (the order-anchored arm: i-th face to i-th face, then its twist and tilt checks) has delta +0.519. `greedy_index_order` (same ordering, no checks) has delta +0.016 (0.872 -> 0.856; interval [-0.021, +0.053]), so the difference between them is due to the checks.

Arms whose transfer interval includes zero: `monotone_aligned`, `hybrid_aligned`, `hungarian_aligned`. Per-detector rankings are in `correspondence_report.md` and `correspondence_report_b1_detections_ccorr.md`.

The correspondence anchors were annotated from `pool` output, so re-linking `ccorr` faces to them is less exact and resolves fewer claims; the comparison is within-column stability, not a cross-detector ranking.

