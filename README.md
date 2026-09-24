# pairbench

A benchmark for cross-view pairing of identical parts. Two 3D scanners see
opposite ends of a bundle of bars; the task is to decide which end belongs to
which bar when the bars are identical, often symmetric and regularly spaced,
so neither position nor orientation identifies a bar.

The method is described in [`docs/method.md`](docs/method.md).

## Installation

```
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

The learned baseline needs CPU PyTorch:

```
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu -e ".[learned]"
```

## Usage

```
# synthetic, no scan data needed
python -m pairbench.experiments.scaling
python -m pairbench.experiments.build_synthetic_twin --suite ambiguity
python -m pairbench.experiments.ambiguity_sweep

# real scans (not distributed)
python -m pairbench.experiments.detect --detector pool
python -m pairbench.experiments.pairing --detector pool
python -m pairbench.experiments.correspondence
```

## Layout

```
src/pairbench/      calibration and I/O, detection, pairing, synthetic twin, experiments
tests/pairbench/    pytest suite
data/               profile DXFs, calibration, scene manifest
results/pairbench/  generated reports, tables and figures
docs/               method, equations, reproduction guide, results index
```

## License

MIT, see [`LICENSE`](LICENSE).
