# ccorr detector vs gt counts

Detections: `b1_detections_ccorr.json`, 108 scenes scored (109 annotated, 1 excluded as a byte-identical duplicate, `gt.FLAGGED_SCENES`). Count proxy: TP per side = min(pred, gt); see the correspondence report for the bar-level check.

| arm | P | R | F1 |
|---|---|---|---|
| confirmed | 0.992 | 0.917 | 0.953 |
| confirmed + lost band | 0.726 | 0.985 | 0.836 |

## By profile family

l-profile and l-profile-reshoot are the same profile with overlapping scenarios, not independent samples. complex-profile and heavy-profile have few scenes.

| family | scenes | P | R | F1 | faces |
|---|---|---|---|---|---|
| complex-profile | 4 | 1.000 | 0.733 | 0.846 | 15 |
| heavy-profile | 11 | 1.000 | 0.926 | 0.962 | 68 |
| l-profile | 40 | 1.000 | 0.891 | 0.942 | 274 |
| l-profile-reshoot | 24 | 0.976 | 0.919 | 0.947 | 223 |
| square-profile | 29 | 0.994 | 0.941 | 0.967 | 355 |


## Timing (ccorr)

Wall-clock, single machine, CPU only. Mean is per call, not per scene, where a stage runs once per scene-side.

| stage | calls | total s | mean ms/call |
|---|---|---|---|
| detect_total | 218 | 941.962 | 4320.927 |
| point_attach | 218 | 3.837 | 17.602 |
| projection | 218 | 2.075 | 9.516 |
| template_match | 218 | 933.232 | 4280.881 |

Run with `--workers 4`; per-stage means are per-call wall time.
