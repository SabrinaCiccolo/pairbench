# Synthetic twin — sensor and scene constants, measured from real scans

Regenerate: `python -m pairbench.experiments.sensor_model`; 108 scenes, 216 clouds.

## Field of view (scanner-frame tangent coordinates u = x/z, v = y/z)

| side | u range (modal device limit) | scenes on the limit | v envelope (lower bound on the limit) | v median extent |
|---|---|---|---|---|
| 1 | [-0.350988, +0.300222] = [-19.340, +16.711] deg | 104/108 lo, 98/108 hi | [-0.333307, +0.337289] = [-18.434, +18.639] deg | [-17.44, +13.35] deg |
| 2 | [-0.347865, +0.312564] = [-19.181, +17.357] deg | 104/108 lo, 100/108 hi | [-0.311154, +0.341779] = [-17.284, +18.869] deg | [-13.84, +18.37] deg |

`u` reaches the same two values (to 1e-3) in most scenes: a hard limit. `v` never repeats, so its envelope is used as a lower bound.

## Sampling, noise, standoff

- point density on the YZ projection (0.4 mm cells): median 14.41 pts/mm2, IQR [13.06, 17.06]
- median radial standoff: 656 mm, IQR [641, 669]
- range noise, plane fit over a whole real end face (n=18514): sigma = 0.390 mm
- range noise, spread within one projection pixel: complex-profile/single-04 0.123 mm, l-profile/single-01 0.099 mm, square-profile/single-01 0.294 mm (median 0.123 mm)

The plane fit is 3.2x the per-pixel figure because it also absorbs the face's form error; the twin uses the per-pixel figure.

With 1544 pixels along u, the tangent pitch is 4.220e-04 (~0.28 mm ground sampling, 13.0 pts/mm2).

## Bundle geometry (from the correspondence GT anchors)

- end-face centroid height (median anchor Z): 926.3 mm, IQR [917.7, 930.7], n=870 anchors
- bundle centre (median anchor Y): 1727.9 mm
- cross-view transport offset dY: median +3.8 mm, IQR [-22.6, +40.9], range [-73.0, +147.7]
- cross-view dZ: median +0.24 mm (negligible — only Y is treated as transport)

This is the height of an end-face centroid, not of the bed: it sits about half a section height above the bed, so it varies by family. `synth.generator.BED_Z_MM` is the twin's placement parameter, not a measurement of this quantity.

| family | median end-face centroid Z (mm) | n anchors | median bar pitch (mm) | n gaps |
|---|---|---|---|---|
| complex-profile | 957.4 | 11 | 118.4 | 3 |
| heavy-profile | 947.4 | 66 | 96.0 | 44 |
| l-profile | 917.6 | 247 | 69.2 | 167 |
| l-profile-reshoot | 919.1 | 209 | 38.6 | 161 |
| square-profile | 926.7 | 337 | 40.2 | 279 |

## Scan window

World-Y span of each cloud (0.5-99.5 percentile), over 216 clouds, after and before the 850 mm radial gate (`io/calib.py::radial_filter`).

| | median | IQR | p90 | max |
|---|---|---|---|---|
| after the radial gate | 258 mm | [210, 311] | 337 mm | **414 mm** |
| before the radial gate | 479 mm | [426, 502] | 504 mm | **643 mm** |

## Reconstructed bar length

In the offline-calibrated world a 6005 mm bar reconstructs as two faces 4928 mm apart (`synth.generator.APPARENT_LENGTH_MM`).

