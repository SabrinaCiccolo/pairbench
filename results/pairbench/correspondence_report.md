# Correspondence-level GT — pairing strategy accuracy

Scored against `results/pairbench/gt_correspondence.csv` (108 scenes with claimed pairs).

Claims are re-linked to detections by stored YZ anchors, not by index, so the same GT scores both detection arms. `resolved` = claims whose bar the arm actually detected; recall is over those, and the unresolved remainder is a detection loss, not a pairing error.

**Denominators.** 108 scenes are scored (`gt.FLAGGED_SCENES` excluded); 108 claim at least one cross-view pair. The bootstrap runs over scenes with at least one resolved claim, grouped into bundles (repeat captures of one bundle move together).

## B1 confirmed

Resolved 407/407 claimed pairs (1.000 coverage).

| strategy | correct | resolved | recall | precision | IPAA | IPAA-1.0 | IPAA-0.9 | IPAA-0.8 | IPAA-0.5 |
|---|---|---|---|---|---|---|---|---|---|
| greedy_index | 0 | 407 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| greedy_index_debiased | 332 | 407 | 0.816 | 0.910 | 0.820 | 0.731 | 0.750 | 0.769 | 0.852 |
| greedy_index_nolen | 340 | 407 | 0.835 | 0.899 | 0.828 | 0.741 | 0.759 | 0.796 | 0.852 |
| greedy_index_order | 355 | 407 | 0.872 | 0.896 | 0.861 | 0.843 | 0.843 | 0.852 | 0.870 |
| hungarian | 310 | 407 | 0.762 | 0.781 | 0.757 | 0.676 | 0.676 | 0.676 | 0.769 |
| hungarian_aligned | 317 | 407 | 0.779 | 0.792 | 0.808 | 0.741 | 0.741 | 0.750 | 0.833 |
| monotone_aligned | 352 | 407 | 0.865 | 0.905 | 0.892 | 0.796 | 0.796 | 0.824 | 0.926 |
| hybrid_aligned | 378 | 407 | 0.929 | 0.940 | 0.926 | 0.889 | 0.898 | 0.898 | 0.944 |

IPAA (Cai et al. 2020, MessyTable) is the unweighted mean of each scene's own recall — pooled `recall` above lets a big scene swamp small ones, IPAA does not. IPAA-X is the fraction of scenes reaching at least X per-scene accuracy; IPAA-1.0 is the fraction of scenes an arm gets completely right.

**Localization error** (annotated anchor to resolved detection centroid, YZ mm; bounded by the anchor resolution radius, so this reads as the noise floor of the anchor re-link, not full-pipeline accuracy): n=870, median 0.00 mm, IQR [0.00, 0.00] mm, max 0.04 mm.

### Paired-bootstrap significance vs `hybrid_aligned`

Every arm is re-scored on the same resampled datasets (10000 replicates, `pairbench.significance.paired_bootstrap`).

The resampling unit is the bundle: claims within a scene are correlated, and repeat captures of one bundle (`data/repeat_captures.csv`) are resampled together. The claim-level table is for comparison.

Resampling unit: **bundle** (n=96).

| strategy | recall | delta vs baseline | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `hybrid_aligned` (baseline) | 0.929 | — | — | — | — | — |
| `greedy_index_order` | 0.872 | -0.057 | [-0.133, +0.016] | 0.1348 | no | no |
| `monotone_aligned` | 0.865 | -0.064 | [-0.127, -0.018] | 0.0000 | yes | yes |
| `greedy_index_nolen` | 0.835 | -0.093 | [-0.172, -0.016] | 0.0220 | yes | yes |
| `greedy_index_debiased` | 0.816 | -0.113 | [-0.192, -0.034] | 0.0074 | yes | yes |
| `hungarian_aligned` | 0.779 | -0.150 | [-0.259, -0.065] | 0.0000 | yes | yes |
| `hungarian` | 0.762 | -0.167 | [-0.270, -0.067] | 0.0012 | yes | yes |
| `greedy_index` | 0.000 | -0.929 | [-0.971, -0.874] | 0.0000 | yes | yes |

Resampling unit: **claim** (n=407).

| strategy | recall | delta vs baseline | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `hybrid_aligned` (baseline) | 0.929 | — | — | — | — | — |
| `greedy_index_order` | 0.872 | -0.057 | [-0.093, -0.020] | 0.0038 | yes | yes |
| `monotone_aligned` | 0.865 | -0.064 | [-0.088, -0.039] | 0.0000 | yes | yes |
| `greedy_index_nolen` | 0.835 | -0.093 | [-0.135, -0.054] | 0.0000 | yes | yes |
| `greedy_index_debiased` | 0.816 | -0.113 | [-0.157, -0.071] | 0.0000 | yes | yes |
| `hungarian_aligned` | 0.779 | -0.150 | [-0.187, -0.115] | 0.0000 | yes | yes |
| `hungarian` | 0.762 | -0.167 | [-0.216, -0.118] | 0.0000 | yes | yes |
| `greedy_index` | 0.000 | -0.929 | [-0.953, -0.902] | 0.0000 | yes | yes |

`separates?` = the uncorrected 95% paired CI excludes zero. `survives Holm?` applies Holm-Bonferroni over the 7 comparisons in each table (family-wise alpha=0.05, `pairbench.significance.holm_reject`).

### IPAA-1.0 paired-bootstrap vs `hybrid_aligned`

Same shared-resample machinery as the recall test above, applied to the binary indicator "this scene is completely correct" instead of pooled correct claims.

| strategy | IPAA-1.0 | delta vs baseline | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `hybrid_aligned` (baseline) | 0.889 | — | — | — | — | — |
| `greedy_index_order` | 0.843 | -0.046 | [-0.123, +0.028] | 0.2592 | no | no |
| `monotone_aligned` | 0.796 | -0.093 | [-0.153, -0.043] | 0.0000 | yes | yes |
| `greedy_index_nolen` | 0.741 | -0.148 | [-0.248, -0.055] | 0.0018 | yes | yes |
| `hungarian_aligned` | 0.741 | -0.148 | [-0.225, -0.079] | 0.0000 | yes | yes |
| `greedy_index_debiased` | 0.731 | -0.157 | [-0.259, -0.063] | 0.0010 | yes | yes |
| `hungarian` | 0.676 | -0.213 | [-0.339, -0.093] | 0.0006 | yes | yes |
| `greedy_index` | 0.000 | -0.889 | [-0.945, -0.823] | 0.0000 | yes | yes |

## B1 + B2 rescued

Resolved 407/407 claimed pairs (1.000 coverage).

| strategy | correct | resolved | recall | precision | IPAA | IPAA-1.0 | IPAA-0.9 | IPAA-0.8 | IPAA-0.5 |
|---|---|---|---|---|---|---|---|---|---|
| greedy_index | 0 | 407 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| greedy_index_debiased | 332 | 407 | 0.816 | 0.910 | 0.820 | 0.731 | 0.750 | 0.769 | 0.852 |
| greedy_index_nolen | 340 | 407 | 0.835 | 0.899 | 0.828 | 0.741 | 0.759 | 0.796 | 0.852 |
| greedy_index_order | 355 | 407 | 0.872 | 0.896 | 0.861 | 0.843 | 0.843 | 0.852 | 0.870 |
| hungarian | 311 | 407 | 0.764 | 0.783 | 0.760 | 0.676 | 0.676 | 0.676 | 0.769 |
| hungarian_aligned | 315 | 407 | 0.774 | 0.786 | 0.802 | 0.741 | 0.741 | 0.750 | 0.824 |
| monotone_aligned | 346 | 407 | 0.850 | 0.885 | 0.884 | 0.796 | 0.796 | 0.824 | 0.917 |
| hybrid_aligned | 375 | 407 | 0.921 | 0.933 | 0.920 | 0.889 | 0.898 | 0.898 | 0.935 |

IPAA (Cai et al. 2020, MessyTable) is the unweighted mean of each scene's own recall — pooled `recall` above lets a big scene swamp small ones, IPAA does not. IPAA-X is the fraction of scenes reaching at least X per-scene accuracy; IPAA-1.0 is the fraction of scenes an arm gets completely right.

**Localization error** (annotated anchor to resolved detection centroid, YZ mm; bounded by the anchor resolution radius, so this reads as the noise floor of the anchor re-link, not full-pipeline accuracy): n=870, median 0.00 mm, IQR [0.00, 0.00] mm, max 0.04 mm.

### Paired-bootstrap significance vs `hybrid_aligned`

Every arm is re-scored on the same resampled datasets (10000 replicates, `pairbench.significance.paired_bootstrap`).

The resampling unit is the bundle: claims within a scene are correlated, and repeat captures of one bundle (`data/repeat_captures.csv`) are resampled together. The claim-level table is for comparison.

Resampling unit: **bundle** (n=96).

| strategy | recall | delta vs baseline | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `hybrid_aligned` (baseline) | 0.921 | — | — | — | — | — |
| `greedy_index_order` | 0.872 | -0.049 | [-0.123, +0.021] | 0.1788 | no | no |
| `monotone_aligned` | 0.850 | -0.071 | [-0.149, -0.018] | 0.0000 | yes | yes |
| `greedy_index_nolen` | 0.835 | -0.086 | [-0.160, -0.012] | 0.0270 | yes | no |
| `greedy_index_debiased` | 0.816 | -0.106 | [-0.182, -0.029] | 0.0100 | yes | yes |
| `hungarian_aligned` | 0.774 | -0.147 | [-0.257, -0.062] | 0.0000 | yes | yes |
| `hungarian` | 0.764 | -0.157 | [-0.260, -0.058] | 0.0018 | yes | yes |
| `greedy_index` | 0.000 | -0.921 | [-0.967, -0.864] | 0.0000 | yes | yes |

Resampling unit: **claim** (n=407).

| strategy | recall | delta vs baseline | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `hybrid_aligned` (baseline) | 0.921 | — | — | — | — | — |
| `greedy_index_order` | 0.872 | -0.049 | [-0.086, -0.015] | 0.0080 | yes | yes |
| `monotone_aligned` | 0.850 | -0.071 | [-0.098, -0.047] | 0.0000 | yes | yes |
| `greedy_index_nolen` | 0.835 | -0.086 | [-0.128, -0.047] | 0.0000 | yes | yes |
| `greedy_index_debiased` | 0.816 | -0.106 | [-0.147, -0.064] | 0.0000 | yes | yes |
| `hungarian_aligned` | 0.774 | -0.147 | [-0.184, -0.113] | 0.0000 | yes | yes |
| `hungarian` | 0.764 | -0.157 | [-0.204, -0.108] | 0.0000 | yes | yes |
| `greedy_index` | 0.000 | -0.921 | [-0.946, -0.894] | 0.0000 | yes | yes |

`separates?` = the uncorrected 95% paired CI excludes zero. `survives Holm?` applies Holm-Bonferroni over the 7 comparisons in each table (family-wise alpha=0.05, `pairbench.significance.holm_reject`).

### IPAA-1.0 paired-bootstrap vs `hybrid_aligned`

Same shared-resample machinery as the recall test above, applied to the binary indicator "this scene is completely correct" instead of pooled correct claims.

| strategy | IPAA-1.0 | delta vs baseline | 95% paired CI | p | separates? | survives Holm? |
|---|---|---|---|---|---|---|
| `hybrid_aligned` (baseline) | 0.889 | — | — | — | — | — |
| `greedy_index_order` | 0.843 | -0.046 | [-0.118, +0.019] | 0.2320 | no | no |
| `monotone_aligned` | 0.796 | -0.093 | [-0.153, -0.043] | 0.0000 | yes | yes |
| `greedy_index_nolen` | 0.741 | -0.148 | [-0.243, -0.057] | 0.0004 | yes | yes |
| `hungarian_aligned` | 0.741 | -0.148 | [-0.225, -0.079] | 0.0000 | yes | yes |
| `greedy_index_debiased` | 0.731 | -0.157 | [-0.255, -0.067] | 0.0002 | yes | yes |
| `hungarian` | 0.676 | -0.213 | [-0.339, -0.093] | 0.0000 | yes | yes |
| `greedy_index` | 0.000 | -0.889 | [-0.944, -0.824] | 0.0000 | yes | yes |

## Robustness to the detector

The same arms scored on the other detector's detections (`correspondence_report_b1_detections_ccorr.md`). The 3 aligned arms move by at most 0.022 recall between the two detection sets; `greedy_index_nolen` scores 0.835 here and 0.317 there (80 / 12 / 16 clean / partial / wrong scenes here, 15 / 55 / 38 there).

An order-anchored arm assumes both sides detected the same bars in the same order, so detector disagreement passes directly into its pairing accuracy. Rankings in this report hold at this detector's operating point.

The anchors were annotated from the pool detector's output: re-linking the other detector's detections has median error 0.96 mm against 0.00 mm and resolves 382 claims against 407.

## Error structure (B1 confirmed)

A scene is `clean` when every resolved claim is right, `wrong` when none is, and `partial` otherwise. `shift k` is the modal offset `j_pred - j_true` over a scene's claims; `pure shift` counts the wrong scenes with at least two claims where every claim sits at the same non-zero offset.

| arm | clean | partial | wrong | wrong (>=2 claims) | pure shift | shift sizes |
|---|---|---|---|---|---|---|
| `greedy_index` | 0 | 0 | 108 | 102 | 0 | — |
| `greedy_index_debiased` | 79 | 13 | 16 | 14 | 9 | k=-1: 4, k=1: 5 |
| `greedy_index_nolen` | 80 | 12 | 16 | 14 | 12 | k=-1: 5, k=1: 7 |
| `greedy_index_order` | 91 | 3 | 14 | 13 | 10 | k=-1: 3, k=1: 7 |
| `hungarian` | 73 | 19 | 16 | 16 | 3 | k=-1: 1, k=1: 2 |
| `hungarian_aligned` | 80 | 14 | 14 | 14 | 5 | k=-1: 3, k=1: 2 |
| `monotone_aligned` | 86 | 16 | 6 | 6 | 5 | k=-1: 3, k=1: 2 |
| `hybrid_aligned` | 96 | 7 | 5 | 5 | 5 | k=-1: 3, k=1: 2 |

## Gate firing on real scenes (B1 confirmed)

- crossing hypothesis admitted: **9 of 108** scenes (`hybrid_aligned`)

## Error structure (B1 + B2 rescued)

A scene is `clean` when every resolved claim is right, `wrong` when none is, and `partial` otherwise. `shift k` is the modal offset `j_pred - j_true` over a scene's claims; `pure shift` counts the wrong scenes with at least two claims where every claim sits at the same non-zero offset.

| arm | clean | partial | wrong | wrong (>=2 claims) | pure shift | shift sizes |
|---|---|---|---|---|---|---|
| `greedy_index` | 0 | 0 | 108 | 102 | 0 | — |
| `greedy_index_debiased` | 79 | 13 | 16 | 14 | 9 | k=-1: 4, k=1: 5 |
| `greedy_index_nolen` | 80 | 12 | 16 | 14 | 12 | k=-1: 5, k=1: 7 |
| `greedy_index_order` | 91 | 3 | 14 | 13 | 10 | k=-1: 3, k=1: 7 |
| `hungarian` | 73 | 20 | 15 | 15 | 4 | k=-1: 2, k=1: 2 |
| `hungarian_aligned` | 80 | 13 | 15 | 15 | 6 | k=-2: 1, k=-1: 3, k=1: 2 |
| `monotone_aligned` | 86 | 15 | 7 | 7 | 6 | k=-2: 1, k=-1: 3, k=1: 2 |
| `hybrid_aligned` | 96 | 6 | 6 | 6 | 6 | k=-2: 1, k=-1: 3, k=1: 2 |

## Gate firing on real scenes (B1 + B2 rescued)

- crossing hypothesis admitted: **9 of 108** scenes (`hybrid_aligned`)

## By scene (B1 + B2 rescued)

correct/resolved per arm, every arm. The three aligned arms differ only in what they are allowed to express: free assignment, order constraint, order constraint + evidence-gated crossings. `greedy_index_nolen` is the order-anchored arm with its nominal-length check disabled.

| scene | claimed | greedy_index | greedy_index_debiased | greedy_index_nolen | greedy_index_order | hungarian | hungarian_aligned | monotone_aligned | hybrid_aligned |
|---|---|---|---|---|---|---|---|---|---|
| complex-profile/single-01 | 1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| complex-profile/single-02 | 1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| complex-profile/single-03 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| complex-profile/single-04 | 1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| heavy-profile/single-01 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-02 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-03 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-04 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-05 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-06 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-07 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-08 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-09 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| heavy-profile/single-10 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 2/3 | 2/3 | 2/3 |
| heavy-profile/standing-01 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/crossed-01 | 2 | 0/2 | 1/2 | 1/2 | 1/2 | 2/2 | 1/2 | 1/2 | 1/2 |
| l-profile-reshoot/overlap-02 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/overlap-03 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/overlap-04 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/overlap-05 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/overlap-06 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/overlap-07 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/overlap-08 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | 0/4 |
| l-profile-reshoot/overlap-09 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 1/4 | 1/4 | 1/4 |
| l-profile-reshoot/single-01 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/single-02 | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| l-profile-reshoot/single-03 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| l-profile-reshoot/standing-01 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 1/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/standing-02 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 1/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/standing-03 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 1/4 | 4/4 | 4/4 | 4/4 |
| l-profile-reshoot/standing-04 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile-reshoot/standing-05 | 11 | 0/11 | 10/11 | 10/11 | 11/11 | 11/11 | 0/11 | 11/11 | 11/11 |
| l-profile-reshoot/standing-06 | 11 | 0/11 | 10/11 | 10/11 | 11/11 | 11/11 | 0/11 | 11/11 | 11/11 |
| l-profile-reshoot/standing-07 | 8 | 0/8 | 7/8 | 7/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| l-profile-reshoot/tilted-01 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/tilted-02 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/tilted-03 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 1/4 | 2/4 | 2/4 |
| l-profile-reshoot/tilted-04 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile-reshoot/tilted-05 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile/crossed-01 | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 4/4 |
| l-profile/crossed-02 | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 4/4 | 4/4 | 3/4 | 4/4 |
| l-profile/crossed-03 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 3/3 | 2/3 | 3/3 |
| l-profile/crossed-04 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/crossed-05 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/crossed-06 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/crossed-07 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 2/3 | 2/3 | 3/3 |
| l-profile/crossed-08 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 2/3 | 3/3 |
| l-profile/overlap-01 | 5 | 0/5 | 5/5 | 5/5 | 5/5 | 3/5 | 3/5 | 5/5 | 5/5 |
| l-profile/overlap-02 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 |
| l-profile/overlap-03 | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 2/4 | 2/4 | 4/4 | 4/4 |
| l-profile/overlap-04 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| l-profile/overlap-05 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-06 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-07 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-08 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-09 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-10 | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| l-profile/overlap-11 | 4 | 0/4 | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile/overlap-12 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile/overlap-13 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| l-profile/overlap-14 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 |
| l-profile/single-01 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/single-02 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/single-03 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/single-04 | 2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| l-profile/single-05 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 3/3 | 3/3 |
| l-profile/single-06 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 |
| l-profile/single-07 | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| l-profile/single-08 | 1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| l-profile/single-09 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/single-10 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| l-profile/single-11 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 |
| l-profile/single-12 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 |
| l-profile/standing-01 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile/standing-02 | 4 | 0/4 | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile/standing-03 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| l-profile/standing-04 | 4 | 0/4 | 3/4 | 3/4 | 4/4 | 2/4 | 2/4 | 4/4 | 4/4 |
| l-profile/standing-05 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| l-profile/tilted-01 | 2 | 0/2 | 0/2 | 0/2 | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 |
| square-profile/single-01 | 8 | 0/8 | 8/8 | 8/8 | 8/8 | 0/8 | 8/8 | 6/8 | 8/8 |
| square-profile/single-02 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 0/4 | 4/4 | 4/4 |
| square-profile/single-03 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| square-profile/single-04 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| square-profile/single-05 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-06 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-07 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-08 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| square-profile/single-09 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-10 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| square-profile/single-11 | 5 | 0/5 | 5/5 | 5/5 | 5/5 | 1/5 | 5/5 | 5/5 | 5/5 |
| square-profile/single-12 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| square-profile/single-13 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-14 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| square-profile/single-15 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| square-profile/single-16 | 3 | 0/3 | 2/3 | 2/3 | 2/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| square-profile/single-17 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 2/4 | 4/4 | 4/4 | 4/4 |
| square-profile/single-18 | 22 | 0/22 | 16/22 | 16/22 | 19/22 | 17/22 | 19/22 | 6/22 | 21/22 |
| square-profile/single-19 | 3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| square-profile/single-20 | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| square-profile/single-21 | 7 | 0/7 | 6/7 | 6/7 | 7/7 | 0/7 | 7/7 | 7/7 | 7/7 |
| square-profile/single-22 | 7 | 0/7 | 0/7 | 0/7 | 0/7 | 7/7 | 7/7 | 6/7 | 7/7 |
| square-profile/single-23 | 7 | 0/7 | 7/7 | 7/7 | 7/7 | 5/7 | 7/7 | 7/7 | 7/7 |
| square-profile/single-24 | 8 | 0/8 | 5/8 | 7/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| square-profile/single-25 | 7 | 0/7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 |
| square-profile/single-26 | 7 | 0/7 | 7/7 | 7/7 | 7/7 | 7/7 | 0/7 | 0/7 | 0/7 |
| square-profile/single-27 | 9 | 0/9 | 6/9 | 9/9 | 9/9 | 9/9 | 9/9 | 8/9 | 9/9 |
| square-profile/single-28 | 9 | 0/9 | 5/9 | 8/9 | 9/9 | 9/9 | 9/9 | 8/9 | 9/9 |
| square-profile/standing-01 | 4 | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
