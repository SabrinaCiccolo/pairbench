# Sensitivity to the shared distance scale

Regenerate: `python -m pairbench.experiments.dist_scale`. Confirmed detections only, every scene with a claimed pair (108 scenes), recall pooled over resolved claims as in correspondence_report.md.

`DIST_SCALE_MM` is 45 mm, selected by leave-one-family-out + k-fold on these scenes. This report shows how the results move with it and re-runs a leave-one-profile-out selection.

## Invariants

| scale (mm) | `test_aligned_impostor_does_not_beat_noisy_true_pair` | `test_hybrid_keeps_refusing_the_impostor_crossing` |
|---|---|---|
| 10 | pass | pass |
| 20 | pass | pass |
| 30 | pass | pass |
| 40 | pass | pass |
| 45 | pass | pass |
| 50 | pass | pass |
| 60 | FAIL | pass |
| 90 | FAIL | pass |

## Correspondence recall by distance scale (`pool` detections)

| arm | 10 | 20 | 30 | 40 | 45 | 50 | 60 | 90 |
|---|---|---|---|---|---|---|---|---|
| `hungarian` | 0.705 | 0.708 | 0.725 | 0.749 | 0.762 | 0.771 | 0.789 | 0.855 |
| `hungarian_aligned` | 0.791 | 0.789 | 0.791 | 0.796 | 0.779 | 0.781 | 0.818 | 0.877 |
| `monotone_aligned` | 0.740 | 0.754 | 0.754 | 0.803 | 0.865 | 0.850 | 0.848 | 0.833 |
| `hybrid_aligned` | 0.803 | 0.818 | 0.818 | 0.867 | 0.929 | 0.921 | 0.921 | 0.919 |
| `greedy_index_order` (reads no distance) | 0.872 | | | | | | | |

IPAA-1.0 and wholly wrong scenes (at least two claims, none correct) of `hybrid_aligned`:

| | 10 | 20 | 30 | 40 | 45 | 50 | 60 | 90 |
|---|---|---|---|---|---|---|---|---|
| IPAA-1.0 | 0.778 | 0.796 | 0.796 | 0.861 | 0.889 | 0.880 | 0.880 | 0.880 |
| wholly wrong | 14 | 13 | 13 | 8 | 5 | 5 | 5 | 5 |

## Correspondence recall by distance scale (`ccorr` detections)

| arm | 10 | 20 | 30 | 40 | 45 | 50 | 60 | 90 |
|---|---|---|---|---|---|---|---|---|
| `hungarian` | 0.644 | 0.660 | 0.670 | 0.683 | 0.694 | 0.704 | 0.730 | 0.796 |
| `hungarian_aligned` | 0.822 | 0.817 | 0.806 | 0.793 | 0.801 | 0.783 | 0.793 | 0.804 |
| `monotone_aligned` | 0.806 | 0.812 | 0.817 | 0.861 | 0.861 | 0.848 | 0.806 | 0.796 |
| `hybrid_aligned` | 0.848 | 0.851 | 0.861 | 0.911 | 0.911 | 0.908 | 0.866 | 0.856 |
| `greedy_index_order` (reads no distance) | 0.856 | | | | | | | |

IPAA-1.0 and wholly wrong scenes (at least two claims, none correct) of `hybrid_aligned`:

| | 10 | 20 | 30 | 40 | 45 | 50 | 60 | 90 |
|---|---|---|---|---|---|---|---|---|
| IPAA-1.0 | 0.778 | 0.769 | 0.769 | 0.824 | 0.824 | 0.824 | 0.769 | 0.759 |
| wholly wrong | 9 | 9 | 9 | 3 | 3 | 3 | 3 | 3 |

## Leave-one-profile-out selection (`pool` detections)

For each held-out profile, the grid value with the highest pooled `hybrid_aligned` recall on the other profiles, among values that pass both invariants (ties broken towards 45 mm), scored on the held-out profile. The L profile's two acquisition campaigns are one cross-section and are held out together.

| held-out profile | scenes | chosen scale (mm) | train recall | held-out recall | held-out recall at 45 mm |
|---|---|---|---|---|---|
| complex-profile | 4 | 45 | 0.928 | 1.000 | 1.000 |
| heavy-profile | 11 | 45 | 0.925 | 0.970 | 0.970 |
| l-profile | 64 | 45 | 0.948 | 0.911 | 0.911 |
| square-profile | 29 | 45 | 0.920 | 0.942 | 0.942 |

## Limits

- The selection uses one criterion (pooled recall of one arm) and no k-fold stage.
- Every value here is scored on the scenes that also informed the cost model's other constants, so no row is an out-of-sample estimate for the model as a whole.
