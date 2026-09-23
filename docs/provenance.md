# Results index

Every file under `results/pairbench/` and the command that produces it. See
[`reproducing.md`](reproducing.md) for data requirements and run order.

## Reports

| What it shows | Files | Command |
|---|---|---|
| Ground truth: face counts, cross-view pairs, scene list | `gt.csv`, `gt_correspondence.csv`, `scenes.csv` | hand annotation |
| Detection precision/recall/F1 | `b1_report_{pool,ccorr}.md`, `b1_counts_{pool,ccorr}.csv` | `detect --detector {pool,ccorr}` |
| Detection after rescue | `b2_rescue_pool.md`, `b2_counts_pool.csv` | `rescue --detector pool` |
| Pairing arms, count metric | `a1_report.md`, `a1_pairing.csv`, `a1_overall.json`, `a1_f1_by_category.png` | `pairing --detector pool` |
| Pairing arms, correspondence accuracy and paired bootstrap | `correspondence_report.md`, `correspondence_by_scene.csv` | `correspondence` |
| Same, on `ccorr` detections | `correspondence_report_b1_detections_ccorr.md`, `correspondence_b1_detections_ccorr_by_scene.csv` | `correspondence --detections results/pairbench/b1_detections_ccorr.json --arm b1` |
| Does the objective rank the true correspondence first | `cost_gap_pool.{md,csv}` | `cost_gap` |
| Which arms survive a change of detector | `detector_transfer.{md,csv,json}` | `detector_transfer` |
| What the bar-length check can resolve | `length_discriminant.{md,csv,json}` | `length_discriminant` |
| Sensitivity to the distance scale | `dist_scale_sensitivity.{md,csv,json}` | `dist_scale` |
| Cross-validated selection of the hybrid arm's constants | `param_cv.{md,json}` | `param_cv` |
| Carriage step fit and self-calibration | `step_fit_report.md`, `fitted_steps.json`, `a2_selfcalib.md`, `a2_calibration.json`, `a2_recalibrated.json` | `fit_step_positions --detector pool`, `selfcalib --detector pool` |
| Learned baseline | `learned_baseline.md`, `learned_folds.csv`, `learned_counts.csv`, `learned_convergence.csv`, `learned_detections_{scene,family}.json` | `learned_baseline --seeds 2 --threads 8` |
| Learned baseline, bootstrap intervals | `learned_significance.md` | `learned_significance` |
| Sensor and scene constants from the real scans | `synthetic_sensor_model.{md,csv,json}` | `sensor_model` |
| Synthetic twin vs real scans | `synthetic_realism.{md,csv,json}`, `synthetic_realism/` | `synth_realism` |
| Is the ambiguity periodic in the bar pitch | `ambiguity_sweep.{md,csv,json}` | `ambiguity_sweep` |
| Synthetic ablation | `synthetic_ablation.{md,csv,json}` | `synth_ablation` |
| Run time vs bar count | `scaling_report.{md,png}` | `scaling` |

Commands are `python -m pairbench.experiments.<command>`.

## Detections

`b1_detections_{pool,ccorr}.json` (from `detect`) and `b2_rescued_pool.json`
(from `rescue`) hold the detections that the pairing and correspondence
experiments read.

## Overlays

`b1_overlays_{pool,ccorr}/`, `b2_overlays_pool/` and `a1_overlays/` show
detections, rescued faces and pairings for the first few scenes each
experiment processes. They are illustrations, not a sample.

## Per-scene correspondence files

`correspondence_by_scene.csv` and its `ccorr` counterpart hold one row per
(scoring, scene, arm). Filter on the `label` column.
