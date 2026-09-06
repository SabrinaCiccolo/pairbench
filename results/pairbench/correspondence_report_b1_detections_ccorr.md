# Correspondence-level GT — pairing strategy accuracy

Scored against `results/pairbench/gt_correspondence.csv` (108 scenes with claimed pairs).

Claims are re-linked to detections by stored YZ anchors, not by index, so the same GT scores both detection arms. `resolved` = claims whose bar the arm actually detected; recall is over those, and the unresolved remainder is a detection loss, not a pairing error.

**Denominators.** 108 scenes are scored (`gt.FLAGGED_SCENES` excluded); 108 claim at least one cross-view pair. The bootstrap runs over scenes with at least one resolved claim, grouped into bundles (repeat captures of one bundle move together).

## b1_detections_ccorr confirmed

Resolved 382/407 claimed pairs (0.939 coverage).

| strategy | correct | resolved | recall | precision | IPAA | IPAA-1.0 | IPAA-0.9 | IPAA-0.8 | IPAA-0.5 |
|---|---|---|---|---|---|---|---|---|---|
| greedy_index | 0 | 382 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| greedy_index_debiased | 118 | 382 | 0.309 | 0.915 | 0.347 | 0.139 | 0.139 | 0.139 | 0.380 |
| greedy_index_nolen | 121 | 382 | 0.317 | 0.883 | 0.358 | 0.139 | 0.139 | 0.139 | 0.389 |
| greedy_index_order | 327 | 382 | 0.856 | 0.872 | 0.864 | 0.824 | 0.824 | 0.833 | 0.889 |
| hungarian | 265 | 382 | 0.694 | 0.718 | 0.721 | 0.620 | 0.620 | 0.639 | 0.731 |
| hungarian_aligned | 306 | 382 | 0.801 | 0.827 | 0.798 | 0.704 | 0.704 | 0.731 | 0.824 |
| monotone_aligned | 329 | 382 | 0.861 | 0.909 | 0.868 | 0.759 | 0.769 | 0.796 | 0.907 |
| hybrid_aligned | 348 | 382 | 0.911 | 0.935 | 0.903 | 0.824 | 0.833 | 0.870 | 0.926 |

IPAA (Cai et al. 2020, MessyTable) is the unweighted mean of each scene's own recall — pooled `recall` above lets a big scene swamp small ones, IPAA does not. IPAA-X is the fraction of scenes reaching at least X per-scene accuracy; IPAA-1.0 is the fraction of scenes an arm gets completely right.

**Localization error** (annotated anchor to resolved detection centroid, YZ mm; bounded by the anchor resolution radius, so this reads as the noise floor of the anchor re-link, not full-pipeline accuracy): n=837, median 0.96 mm, IQR [0.14, 3.92] mm, max 18.94 mm.

### Paired-bootstrap significance vs `hybrid_aligned`

Every arm is re-scored on the same resampled datasets (10000 replicates, `pairbench.significance.paired_bootstrap`).

The resampling unit is the bundle: claims within a scene are correlated, and repeat captures of one bundle (`data/repeat_captures.csv`) are resampled together. The claim-level table is for comparison.

Resampling unit: **bundle** (n=96).

| strategy | recall | delta vs baseline | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `hybrid_aligned` (baseline) | 0.911 | — | — | — | — | — |
| `monotone_aligned` | 0.861 | -0.050 | [-0.094, -0.018] | 0.0000 | yes | yes |
| `greedy_index_order` | 0.856 | -0.055 | [-0.124, +0.003] | 0.0760 | no | no |
| `hungarian_aligned` | 0.801 | -0.110 | [-0.169, -0.057] | 0.0000 | yes | yes |
| `hungarian` | 0.694 | -0.217 | [-0.311, -0.124] | 0.0000 | yes | yes |
| `greedy_index_nolen` | 0.317 | -0.594 | [-0.667, -0.522] | 0.0000 | yes | yes |
| `greedy_index_debiased` | 0.309 | -0.602 | [-0.675, -0.528] | 0.0000 | yes | yes |
| `greedy_index` | 0.000 | -0.911 | [-0.954, -0.857] | 0.0000 | yes | yes |

Resampling unit: **claim** (n=382).

| strategy | recall | delta vs baseline | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `hybrid_aligned` (baseline) | 0.911 | — | — | — | — | — |
| `monotone_aligned` | 0.861 | -0.050 | [-0.073, -0.026] | 0.0000 | yes | yes |
| `greedy_index_order` | 0.856 | -0.055 | [-0.092, -0.018] | 0.0038 | yes | yes |
| `hungarian_aligned` | 0.801 | -0.110 | [-0.144, -0.079] | 0.0000 | yes | yes |
| `hungarian` | 0.694 | -0.217 | [-0.264, -0.170] | 0.0000 | yes | yes |
| `greedy_index_nolen` | 0.317 | -0.594 | [-0.644, -0.545] | 0.0000 | yes | yes |
| `greedy_index_debiased` | 0.309 | -0.602 | [-0.652, -0.552] | 0.0000 | yes | yes |
| `greedy_index` | 0.000 | -0.911 | [-0.937, -0.882] | 0.0000 | yes | yes |

`separates?` = the uncorrected 95% paired CI excludes zero. `survives Holm?` applies Holm-Bonferroni over the 7 comparisons in each table (family-wise alpha=0.05, `pairbench.significance.holm_reject`).

### IPAA-1.0 paired-bootstrap vs `hybrid_aligned`

Same shared-resample machinery as the recall test above, applied to the binary indicator "this scene is completely correct" instead of pooled correct claims.

| strategy | IPAA-1.0 | delta vs baseline | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `greedy_index_order` | 0.824 | +0.000 | [-0.078, +0.076] | 1.0000 | no | no |
| `hybrid_aligned` (baseline) | 0.824 | — | — | — | — | — |
| `monotone_aligned` | 0.759 | -0.065 | [-0.118, -0.020] | 0.0020 | yes | yes |
| `hungarian_aligned` | 0.704 | -0.120 | [-0.189, -0.063] | 0.0000 | yes | yes |
| `hungarian` | 0.620 | -0.204 | [-0.322, -0.090] | 0.0002 | yes | yes |
| `greedy_index_debiased` | 0.139 | -0.685 | [-0.784, -0.581] | 0.0000 | yes | yes |
| `greedy_index_nolen` | 0.139 | -0.685 | [-0.784, -0.581] | 0.0000 | yes | yes |
| `greedy_index` | 0.000 | -0.824 | [-0.895, -0.740] | 0.0000 | yes | yes |

## Robustness to the detector

The same arms scored on the other detector's detections (`correspondence_report.md`). The 3 aligned arms move by at most 0.022 recall between the two detection sets; `greedy_index_nolen` scores 0.317 here and 0.835 there (15 / 55 / 38 clean / partial / wrong scenes here, 80 / 12 / 16 there).

An order-anchored arm assumes both sides detected the same bars in the same order, so detector disagreement passes directly into its pairing accuracy. Rankings in this report hold at this detector's operating point.

The anchors were annotated from the pool detector's output: re-linking the other detector's detections has median error 0.96 mm against 0.00 mm and resolves 382 claims against 407.

## Error structure (b1_detections_ccorr confirmed)

A scene is `clean` when every resolved claim is right, `wrong` when none is, and `partial` otherwise. `shift k` is the modal offset `j_pred - j_true` over a scene's claims; `pure shift` counts the wrong scenes with at least two claims where every claim sits at the same non-zero offset.

| arm | clean | partial | wrong | wrong (>=2 claims) | pure shift | shift sizes |
|---|---|---|---|---|---|---|
| `greedy_index` | 0 | 0 | 108 | 101 | 0 | — |
| `greedy_index_debiased` | 15 | 52 | 41 | 35 | 5 | k=-1: 2, k=1: 3 |
| `greedy_index_nolen` | 15 | 55 | 38 | 32 | 8 | k=-1: 3, k=1: 5 |
| `greedy_index_order` | 89 | 8 | 11 | 11 | 8 | k=-1: 1, k=1: 7 |
| `hungarian` | 67 | 22 | 19 | 19 | 4 | k=-1: 1, k=1: 3 |
| `hungarian_aligned` | 76 | 18 | 14 | 11 | 3 | k=-1: 3 |
| `monotone_aligned` | 82 | 18 | 8 | 5 | 3 | k=-1: 3 |
| `hybrid_aligned` | 89 | 13 | 6 | 3 | 3 | k=-1: 3 |

## Gate firing on real scenes (b1_detections_ccorr confirmed)

- crossing hypothesis admitted: **12 of 108** scenes (`hybrid_aligned`)

## By scene (b1_detections_ccorr confirmed)

correct/resolved per arm, every arm. The three aligned arms differ only in what they are allowed to express: free assignment, order constraint, order constraint + evidence-gated crossings. `greedy_index_nolen` is the order-anchored arm with its nominal-length check disabled.

| scene | claimed | greedy_index | greedy_index_debiased | greedy_index_nolen | greedy_index_order | hungarian | hungarian_aligned | monotone_aligned | hybrid_aligned |
|---|---|---|---|---|---|---|---|---|---|
| complex-profile/single-01 | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| complex-profile/single-02 | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| complex-profile/single-03 | 2 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| complex-profile/single-04 | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| heavy-profile/single-01 | 3 | 0/2 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| heavy-profile/single-02 | 3 | 0/2 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| heavy-profile/single-03 | 3 | 0/2 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| heavy-profile/single-04 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-05 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-06 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-07 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-08 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-09 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-10 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/standing-01 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/crossed-01 | 2 | 0/2 | 1/2 | 1/2 | 1/2 | 2/2 | 1/2 | 1/2 | 1/2 |
| l-profile-reshoot/overlap-02 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/overlap-03 | 4 | 0/4 | 2/4 | 2/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/overlap-04 | 4 | 0/4 | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/overlap-05 | 3 | 0/2 | 0/2 | 0/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile-reshoot/overlap-06 | 3 | 0/2 | 1/2 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile-reshoot/overlap-07 | 4 | 0/4 | 1/4 | 1/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/overlap-08 | 4 | 0/4 | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/overlap-09 | 4 | 0/4 | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/single-01 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/single-02 | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| l-profile-reshoot/single-03 | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| l-profile-reshoot/standing-01 | 4 | 0/4 | 2/4 | 2/4 | 4/4 | 1/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/standing-02 | 4 | 0/4 | 1/4 | 1/4 | 1/4 | 1/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/standing-03 | 4 | 0/4 | 3/4 | 3/4 | 4/4 | 0/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/standing-04 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile-reshoot/standing-05 | 11 | 0/9 | 5/9 | 5/9 | 8/9 | 4/9 | 4/9 | 8/9 | 8/9 |
| l-profile-reshoot/standing-06 | 11 | 0/10 | 4/10 | 4/10 | 10/10 | 6/10 | 6/10 | 9/10 | 9/10 |
| l-profile-reshoot/standing-07 | 8 | 0/5 | 3/5 | 3/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| l-profile-reshoot/tilted-01 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/tilted-02 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/tilted-03 | 4 | 0/3 | 2/3 | 2/3 | 3/3 | 2/3 | 1/3 | 3/3 | 3/3 |
| l-profile-reshoot/tilted-04 | 3 | 0/3 | 1/3 | 1/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/tilted-05 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile/crossed-01 | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 4/4 |
| l-profile/crossed-02 | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 4/4 | 4/4 | 3/4 | 4/4 |
| l-profile/crossed-03 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 1/3 |
| l-profile/crossed-04 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/crossed-05 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/crossed-06 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/crossed-07 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 2/3 | 3/3 |
| l-profile/crossed-08 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 2/3 | 3/3 |
| l-profile/overlap-01 | 5 | 0/5 | 1/5 | 1/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| l-profile/overlap-02 | 2 | 0/2 | 1/2 | 1/2 | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 |
| l-profile/overlap-03 | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 2/4 | 2/4 | 2/4 | 2/4 |
| l-profile/overlap-04 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 2/3 | 2/3 | 2/3 |
| l-profile/overlap-05 | 3 | 0/3 | 2/3 | 2/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-06 | 3 | 0/3 | 2/3 | 2/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-07 | 3 | 0/3 | 2/3 | 2/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-08 | 3 | 0/3 | 2/3 | 2/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-09 | 3 | 0/3 | 2/3 | 2/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-10 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 2/3 | 2/3 | 2/3 |
| l-profile/overlap-11 | 4 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-12 | 3 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| l-profile/overlap-13 | 2 | 0/2 | 1/2 | 1/2 | 2/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| l-profile/overlap-14 | 3 | 0/3 | 2/3 | 2/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 |
| l-profile/single-01 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/single-02 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/single-03 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/single-04 | 2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| l-profile/single-05 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 3/3 | 3/3 |
| l-profile/single-06 | 3 | 0/3 | 2/3 | 2/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/single-07 | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| l-profile/single-08 | 1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| l-profile/single-09 | 2 | 0/2 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/single-10 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/single-11 | 3 | 0/3 | 2/3 | 2/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 |
| l-profile/single-12 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 |
| l-profile/standing-01 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile/standing-02 | 4 | 0/4 | 2/4 | 2/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile/standing-03 | 4 | 0/4 | 0/4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile/standing-04 | 4 | 0/4 | 1/4 | 1/4 | 4/4 | 2/4 | 2/4 | 4/4 | 4/4 |
| l-profile/standing-05 | 3 | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile/tilted-01 | 2 | 0/2 | 0/2 | 1/2 | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 |
| square-profile/single-01 | 8 | 0/8 | 0/8 | 0/8 | 8/8 | 2/8 | 8/8 | 2/8 | 8/8 |
| square-profile/single-02 | 4 | 0/4 | 0/4 | 0/4 | 4/4 | 4/4 | 0/4 | 4/4 | 4/4 |
| square-profile/single-03 | 4 | 0/4 | 0/4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| square-profile/single-04 | 4 | 0/4 | 0/4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| square-profile/single-05 | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-06 | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-07 | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-08 | 4 | 0/4 | 0/4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| square-profile/single-09 | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-10 | 2 | 0/2 | 0/2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| square-profile/single-11 | 5 | 0/5 | 0/5 | 0/5 | 5/5 | 1/5 | 5/5 | 5/5 | 5/5 |
| square-profile/single-12 | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| square-profile/single-13 | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-14 | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| square-profile/single-15 | 3 | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| square-profile/single-16 | 3 | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| square-profile/single-17 | 4 | 0/4 | 0/4 | 0/4 | 4/4 | 2/4 | 0/4 | 0/4 | 0/4 |
| square-profile/single-18 | 22 | 0/12 | 0/12 | 0/12 | 6/12 | 5/12 | 4/12 | 4/12 | 5/12 |
| square-profile/single-19 | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-20 | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| square-profile/single-21 | 7 | 0/7 | 1/7 | 1/7 | 7/7 | 0/7 | 7/7 | 7/7 | 7/7 |
| square-profile/single-22 | 7 | 0/7 | 0/7 | 0/7 | 0/7 | 7/7 | 7/7 | 6/7 | 7/7 |
| square-profile/single-23 | 7 | 0/7 | 1/7 | 1/7 | 7/7 | 3/7 | 7/7 | 7/7 | 7/7 |
| square-profile/single-24 | 8 | 0/8 | 1/8 | 1/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| square-profile/single-25 | 7 | 0/7 | 1/7 | 1/7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 |
| square-profile/single-26 | 7 | 0/7 | 1/7 | 1/7 | 7/7 | 5/7 | 6/7 | 6/7 | 6/7 |
| square-profile/single-27 | 9 | 0/9 | 1/9 | 1/9 | 9/9 | 8/9 | 8/9 | 7/9 | 8/9 |
| square-profile/single-28 | 9 | 0/9 | 1/9 | 1/9 | 9/9 | 8/9 | 8/9 | 7/9 | 8/9 |
| square-profile/standing-01 | 4 | 0/4 | 1/4 | 1/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
