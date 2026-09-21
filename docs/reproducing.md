# Reproducing results

## Data

Committed:

- `data/dxf/*.dxf`: cross-sections of the four profile families
  (`l-profile`, `complex-profile`, `heavy-profile`, `square-profile`).
- `data/calibration_full.json`, `src/pairbench/calibration.json`: scanner calibration.
- `data/scene_manifest.csv`: every scene with its family and category.
- `data/repeat_captures.csv`: scenes that are repeat scans of one unchanged
  bundle; bootstraps resample these bundles together.
- `data/learned/dataset.npz`: projections and face centres for the learned baseline.
- `results/pairbench/gt.csv`, `gt_correspondence.csv`, `scenes.csv`:
  hand-annotated ground truth (face counts, cross-view pairs, checksums).

Not committed: the raw point clouds. Each scene is a directory with
`scanner-1.ply` and `scanner-2.ply`, e.g. `data/l-profile/overlap-01/`.
The raw scans are not redistributable; the code reads them from `data/` or from
`$PAIRBENCH_DATA_ROOT`. Scenes are named `<family>/<category>-<NN>`.
`l-profile-reshoot/overlap-01` is a byte-identical copy of
`l-profile/overlap-01` and is excluded from pooled statistics
(`gt.FLAGGED_SCENES`).

Synthetic scenes are generated into `data/synthetic/` (not committed).

## Experiments

Data column: *raw* needs the point clouds, *committed* reads only files in the
repository, *synthetic* builds its own scenes.

| Experiment | Command | Data |
|---|---|---|
| Detection | `python -m pairbench.experiments.detect --detector {pool,ccorr}` | raw |
| Rescue | `python -m pairbench.experiments.rescue --detector pool` | raw |
| Pairing (count metric) | `python -m pairbench.experiments.pairing --detector pool` | raw |
| Correspondence accuracy | `python -m pairbench.experiments.correspondence` | committed |
| Correspondence, `ccorr` detections | `python -m pairbench.experiments.correspondence --detections results/pairbench/b1_detections_ccorr.json --arm b1` | committed |
| Step-position fit | `python -m pairbench.experiments.fit_step_positions --detector pool` | committed |
| Self-calibration | `python -m pairbench.experiments.selfcalib --detector pool` | committed |
| Objective cost gap | `python -m pairbench.experiments.cost_gap` | committed |
| Detector transfer | `python -m pairbench.experiments.detector_transfer` | committed |
| Length-check resolving power | `python -m pairbench.experiments.length_discriminant` | committed |
| Distance-scale sensitivity | `python -m pairbench.experiments.dist_scale` | committed |
| Cross-validated constants | `python -m pairbench.experiments.param_cv` | committed |
| Learned baseline | `python -m pairbench.experiments.learned_baseline --seeds 2 --threads 8` | committed |
| Learned baseline intervals | `python -m pairbench.experiments.learned_significance` | committed |
| Sensor model | `python -m pairbench.experiments.sensor_model` | raw |
| Synthetic twin build | `python -m pairbench.experiments.build_synthetic_twin --suite <suite>` | synthetic |
| Synthetic realism | `python -m pairbench.experiments.synth_realism` (after `--suite realism`) | raw + synthetic |
| Ambiguity sweep | `python -m pairbench.experiments.ambiguity_sweep` (after `--suite ambiguity`) | synthetic |
| Synthetic ablation | `python -m pairbench.experiments.synth_ablation` (after the other suites) | synthetic |
| Scaling | `python -m pairbench.experiments.scaling` | synthetic |

Run them in the order listed: later experiments read the outputs of earlier
ones from `results/pairbench/`. The learned baseline needs the `learned` extra
(CPU PyTorch).
