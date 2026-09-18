# Cross-validated selection of the hybrid arm's constants

Regenerate: `python -m pairbench.experiments.param_cv`. `pool` confirmed detections, 108 scenes, 407 resolved claims, 3000 configurations of nine constants and the two extensions (grid in `param_cv.json`).

## In sample

- default constants: recall 0.929; best of the search: 0.929, reached by 14 configurations
- every best configuration keeps both extensions: yes
- `greedy_index_order` (reads none of these constants): 0.872

## Leave-one-profile-out

`tied` counts the configurations with the highest recall on the other profiles; the held-out columns give their correct claims on the held-out one.

| held-out profile | resolved | tied | default among tied | held-out min | mean | max | default |
|---|---|---|---|---|---|---|---|
| complex-profile | 5 | 14 | yes | 5 | 5.0 | 5 | 5 |
| heavy-profile | 33 | 14 | yes | 32 | 32.0 | 32 | 32 |
| l-profile | 213 | 153 | yes | 147 | 181.5 | 194 | 194 |
| square-profile | 156 | 94 | yes | 128 | 138.9 | 147 | 147 |

Cross-validated recall: 0.929 with ties broken towards the default constants, 0.878 expected under a uniform choice among the tied configurations.

## Grouped 5-fold by bundle, 50 repeats

- default constants among the tied configurations in 245/250 folds
- cross-validated recall, ties towards default: mean 0.925
- cross-validated recall, uniform among tied: mean 0.898, range 0.808-0.908

## Reading

Many configurations tie on the training scenes and score differently on the held-out scenes, so the training scenes do not single out the default constants. No row here is an out-of-sample estimate of the default arm.

## Limits

- Three values per constant (six for the distance scale) and a random sample of the grid; a finer grid can only add tied configurations.
- The unit-test invariants that gate `dist_scale` are not applied.
- The two extensions were designed on these scenes; switching them on or off in the search does not make them out-of-sample.
