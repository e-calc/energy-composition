# Formula mode

By default `ecalc` uses measurements only: every kernel of a pipeline must have a measured
signature, and a missing one is reported, never guessed. Measuring is the slow part — about ten
seconds per kernel — and the key of a kernel includes its batch size and input size, so every
new batch size is a new set of kernels.

Formula mode trades some accuracy for that time. It estimates kernels that were not measured
from kernels that were, and labels every estimate so it is never mistaken for a measurement.
It is opt-in:

```bash
ecalc estimate --model rf_vgg --batch-size 64 --fill formula      # estimate from batch 32 and 256
ecalc plan --model yolo26n --batch-size 4                         # what formula mode can cover
ecalc measure kernels --models yolo26n --batch-size 4 --fill formula   # measure only what it cannot
```

```python
from ecalc.formula import with_formulas

estimate = evaluate(pipeline.expression, with_formulas(database), platform, on_missing="report")
print(estimate.by_provenance())       # kernel count, time and energy per provenance
```

The measured database is always asked first. A formula is used only where nothing was measured.

## Provenance labels

| Label | Meaning |
|---|---|
| `measured` | this exact kernel was measured |
| `interpolated` | read off a chord between two measured kernels of the same family |
| `formula` | computed from a line fitted on measured kernels, inside the fitted range |
| `extrapolated` | a chord or line extended beyond the measured range, or scaled from a single point |
| `missing` | no source could price it; counted as zero and listed |

`ecalc estimate` prints the kernel count and energy per label; `--kernels` shows the label and
the anchors of every kernel. Treat `extrapolated` energy as a rough figure.
`ecalc measure kernels --fill formula` skips only `interpolated` and `formula` kernels and still
measures `extrapolated` ones.

## Interpolation: kernels with parameters

A **family** is every kernel that differs only in batch size and in spatial extent (the pixels
of an image, the samples of a frame): the same module, parameters, channel counts, dtype, layout
and direction. Inside a family, time and energy are close to linear in both, so a missing point
is read off the chord between its two nearest measured neighbours.

- between two measured points → `interpolated`
- past the measured range → `extrapolated` (the end chord extended, clamped at zero)
- a single measured point → `extrapolated` (scaled in proportion)

When the family has a measured kernel at the same input size, the chord runs over batch size;
when it has one at the same batch size, over extent; otherwise every measured batch size is
first brought to the requested extent and the chord then crosses batch sizes.

Example: with `rf_vgg` measured at batch 32 and 256, every kernel of batch 64 is `interpolated`.

## Light operations: parameter-free kernels

Operations without parameters — `cat`, `add`, `chunk`, `split`, pooling, upsampling,
activations — are memory-bound. For each family (same operation, arguments, dtype and
direction) a line is fitted to the measured rows:

```
T = overhead + bytes / bandwidth          bytes = total size of the inputs
E = power · T                             power = total measured energy / total measured time
```

A view-like operation (`chunk`, `split`) fits a constant. A family needs at least two measured
sizes. Inside the fitted byte range the estimate is labelled `formula`; outside, `extrapolated`.

## Nothing is hard-coded

There are no per-GPU constants in the code. Chords and fits are computed from the database in
use, so they follow your GPU, your precision and your measurement protocol. A fresh database
with a single batch size can only scale in proportion, and says so.

## How accurate is it?

Measure it on your own data: `ecalc.formula.leave_one_out(database, platform)` hides each
measured kernel in turn, estimates it from the rest, and returns the errors.

**Check on the A40.** The database held YOLO26 at batch sizes 1 and 8 (640 × 640) and RFML at
batch sizes 32 and 256. Other configurations were then measured separately and estimated with
formulas only, with none of their kernels visible. Errors are of the whole-model energy against
the whole-model measurement.

| Held-out configuration | Estimated from | Energy error, formulas only | Energy error, its own measured kernels |
|---|---|---:|---:|
| YOLO26 s, m, l, x at batch size 4 | interpolation between 1 and 8 | −0.1% to +3.6% | −0.3% to +1.4% |
| yolo26n at batch size 4 | interpolation between 1 and 8 | +9.8% | +7.3% |
| RFML at batch sizes 64 and 128 | interpolation between 32 and 256 | +3.4% to +13.0% | −1.1% to +10.4% |
| yolo26n, yolo26s at batch size 16 | extrapolation beyond 8 | −11.1%, −2.9% | −2.6%, −0.0% |
| yolo26n at 320 × 320 and 1280 × 1280 | extrapolation from one input size | −62.9%, +81.9% | −11.4%, +2.0% |

Per kernel, the median energy error was 4.9% for `interpolated` (134 kernels), 2.5% for
`formula` (54) and 14.7% for `extrapolated` (69, with a 90th percentile above 100%).
Interpolating between two measured batch sizes is usable; extrapolating is not reliable, and
extrapolating in input size from a single measured size is wrong by a large factor. Treat
`extrapolated` as a prompt to measure.

## Limits

- Interpolation needs the batch to be the first input dimension, and inputs shaped
  `(batch, channels, spatial…)` or `(batch, …, features)`.
- A family is exact about channels: a different channel count is a different family, not a
  point to interpolate.
- Linear chords do not capture a change of algorithm (a convolution switching cuDNN
  implementation between sizes) or launch-bound small kernels whose time does not scale.
- Composite blocks are interpolated as a whole. Their internals are not re-derived.
