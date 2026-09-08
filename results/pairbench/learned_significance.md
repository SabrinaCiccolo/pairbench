# Learned baseline: bootstrap intervals

Regenerate: `python -m pairbench.experiments.learned_significance`. Seed-0 detections from `learned_baseline` and the classical detector's output. 108 scenes in 96 bundles; the bootstrap resamples bundles.

Classical pooled F1 95% CI: [0.958, 0.980].

| arm | F1 | delta vs classical | 95% CI | p | separated |
|---|---|---|---|---|---|
| learned (scene-disjoint) | 0.883 | -0.087 | [-0.143, -0.045] | 0.000 | yes |
| learned (family-disjoint) | 0.355 | -0.615 | [-0.673, -0.556] | 0.000 | yes |
