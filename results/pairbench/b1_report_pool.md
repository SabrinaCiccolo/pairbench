# pool detector vs gt counts

Detections: `b1_detections_pool.json`, 108 scenes scored (109 annotated, 1 excluded as a byte-identical duplicate, `gt.FLAGGED_SCENES`). Count proxy: TP per side = min(pred, gt); see the correspondence report for the bar-level check.

| arm | P | R | F1 |
|---|---|---|---|
| confirmed | 0.986 | 0.955 | 0.970 |
| confirmed + lost band | 0.585 | 1.000 | 0.738 |

## By profile family

l-profile and l-profile-reshoot are the same profile with overlapping scenarios, not independent samples. complex-profile and heavy-profile have few scenes.

| family | scenes | P | R | F1 | faces |
|---|---|---|---|---|---|
| complex-profile | 4 | 0.875 | 0.933 | 0.903 | 15 |
| heavy-profile | 11 | 0.957 | 0.971 | 0.964 | 68 |
| l-profile | 40 | 1.000 | 0.942 | 0.970 | 274 |
| l-profile-reshoot | 24 | 1.000 | 0.955 | 0.977 | 223 |
| square-profile | 29 | 0.977 | 0.963 | 0.970 | 355 |


## Timing (pool)

Wall-clock, single machine, CPU only. Mean is per call, not per scene, where a stage runs once per scene-side.

| stage | calls | total s | mean ms/call |
|---|---|---|---|
| detect_total | 218 | 924.921 | 4242.758 |
| ownership | 218 | 314.119 | 1440.914 |
| projection | 218 | 2.644 | 12.128 |
| selection | 218 | 33.481 | 153.583 |
| template_match | 218 | 554.471 | 2543.446 |

Run with `--workers 4`; per-stage means are per-call wall time.
