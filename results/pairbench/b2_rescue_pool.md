# Cross-view guided re-detection (rescue)

Detections: b1_detections_pool.json. Count proxy metrics as in the detection/pairing reports.

An unmatched confirmed detection on one side plus hungarian_aligned's fitted translation predicts the partner position on the other side; a local template match on the confirmed-erased image (partner angle +/- one coarse step) is accepted only with a face-like 3D footprint (z-extent >= 15 mm, >= 500 points, centroid within 100 mm of the prediction). Faces with no partner in GT cannot be rescued.

## Detection counts (both sides)

| arm | P | R | F1 |
|---|---|---|---|
| B1 confirmed | 0.986 | 0.955 | 0.970 |
| B1 + rescued | 0.985 | 0.967 | 0.976 |

### By profile family (B1 + rescued)

l-profile and l-profile-reshoot are the same profile with overlapping scenarios, not independent samples. complex-profile and heavy-profile have few scenes.

| family | P | R | F1 | faces |
|---|---|---|---|---|
| complex-profile | 0.875 | 0.933 | 0.903 | 15 |
| heavy-profile | 0.957 | 0.971 | 0.964 | 68 |
| l-profile | 1.000 | 0.942 | 0.970 | 274 |
| l-profile-reshoot | 1.000 | 0.973 | 0.986 | 223 |
| square-profile | 0.975 | 0.983 | 0.979 | 355 |

## Pairing (hungarian_aligned)

| arm | P | R | F1 | inv |
|---|---|---|---|---|
| before rescue | 0.995 | 0.953 | 0.973 | 40 |
| after rescue | 0.984 | 0.968 | 0.976 | 40 |

Post-rescue inversions by scene (swap proxy): l-profile-reshoot/standing-05 (10), l-profile-reshoot/standing-06 (10), l-profile-reshoot/tilted-03 (1), l-profile/overlap-01 (1), l-profile/overlap-03 (1), l-profile/overlap-14 (1), l-profile/single-05 (2), l-profile/single-11 (1), l-profile/standing-04 (1), square-profile/single-02 (3), square-profile/single-12 (2), square-profile/single-14 (2), square-profile/single-15 (2), square-profile/single-16 (1), square-profile/single-18 (2).

Rescue attempts (unmatched partners): 59; accepted: 12. Overlays in b2_overlays_pool/ (rescued boxes thick orange).


## Timing (pool)

Wall-clock, single machine, CPU only. Mean is per call, not per scene, where a stage runs once per scene-side.

| stage | calls | total s | mean ms/call |
|---|---|---|---|
| rescue_total | 109 | 4.980 | 45.684 |
