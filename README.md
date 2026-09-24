# pairbench

A benchmark for cross-view pairing of identical parts. Two 3D scanners see
opposite ends of a bundle of bars; the task is to decide which end belongs to
which bar when the bars are identical, often symmetric and regularly spaced,
so neither position nor orientation identifies a bar.

The method is described in [`docs/method.md`](docs/method.md).

## Results

**Real scans** (108 scenes in 96 independent bundles, 407 annotated cross-view pairs):

- The `pool` detector reaches 0.986 precision and 0.955 recall on end faces
  ([`b1_report_pool.md`](results/pairbench/b1_report_pool.md)).
- `hybrid_aligned` pairs 0.929 of the bars correctly, ahead of
  `greedy_index_order` (0.872), `monotone_aligned` (0.865) and
  `greedy_index_nolen` (0.835). A paired bootstrap over bundles separates it
  from every arm except `greedy_index_order` (−0.057, 95% CI [−0.133, +0.016])
  ([`correspondence_report.md`](results/pairbench/correspondence_report.md)).
- Every scene `hybrid_aligned` gets wholly wrong is a rigid shift in a scene
  where some detection has no counterpart in the other view (0 of 59 fully
  witnessed scenes, 5 of 43 others). On all 5 the objective itself prefers
  the wrong answer ([`cost_gap_pool.md`](results/pairbench/cost_gap_pool.md)).
- Swapping the detector leaves the translation-aligned arms almost unchanged
  (`hybrid_aligned` 0.929 → 0.911), while the order anchor with twist and
  tilt checks drops from 0.835 to 0.317
  ([`detector_transfer.md`](results/pairbench/detector_transfer.md)).
- With the length check on, `greedy_index` scores 0: the offline calibration's
  world-X bias puts every candidate outside the ±10 mm gate. The carriage
  separation is recoverable from the data, absolute carriage position is not
  ([`a2_selfcalib.md`](results/pairbench/a2_selfcalib.md)).

**Synthetic scenes:**

- Under regular spacing the ambiguity is permanent, not periodic in the bar
  pitch: across 0 to 3 pitches of transport, the margin between the truth and
  a one-bar shift stays at 7.5–8.6 cost units with no dips at whole pitches.
  Almost all of it is the unmatch cost of the two leftover bars; pitch jitter
  is what makes geometry informative
  ([`ambiguity_sweep.md`](results/pairbench/ambiguity_sweep.md)).

[`docs/provenance.md`](docs/provenance.md) lists every result file and the
command that produces it.

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

The full list and run order are in [`docs/reproducing.md`](docs/reproducing.md).

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
