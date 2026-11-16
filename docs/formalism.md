# The energy calculus

This page defines what `ecalc.calculus` implements: the energy signature of a kernel, the two
ways signatures compose, the laws that follow, and the file formats. The package needs only the
Python standard library.

## Kernel and signature

A **kernel** is one callable applied to inputs of a given shape, dtype and layout, in a given
direction. In this repository a kernel is a module of a model (a YOLO26 `C3k2` block, an RFML
`ConvPool` stage) or a functional operation between modules (`torch.cat`, a residual `add`).

Its **energy signature** on a platform is

```
E = s·T + d
```

| Symbol | Meaning | Unit |
|---|---|---|
| `E` | energy of one call | joule |
| `T` | duration of one call | second |
| `s` | static power: what the GPU draws while awake and doing nothing | watt |
| `d` | dynamic energy: everything above the static floor | joule |

What is measured is the pair `(T, E)`. Static power is measured separately, once per GPU, and
`d = E − s·T` is derived. Dynamic energy is deliberately not stored: several definitions of
static power exist (see [measurement.md](measurement.md)), and a database of `(T, E)` pairs can
be decomposed under any of them.

```python
from ecalc.calculus import load

database = load("data/signature_database.jsonl")
platform = database.platforms()[0]                    # e.g. A40/default/fp32
signature = database.signatures(platform)[0]
s, T, d = signature.decompose(database.static_power(platform, "active_idle"))
assert abs(s * T + d - signature.energy_joules) < 1e-12
```

### Kernel identity: θ

The parameters of a signature, `θ`, are held by `KernelSpecification`. It wraps a canonical JSON
document, the *structural signature*:

- for a module: its class, the classes and whitelisted attributes of its submodules, and the
  shapes and dtypes of its parameters and buffers;
- for a functional operation: its name and non-tensor arguments;
- the shape and dtype of every input tensor;
- `layout` of an input, written only when it is `channels_last`;
- `direction`, written only when it is `backward` (with `input_gradient: false` when the
  gradient with respect to the inputs is not computed).

The **key** is the first 16 hex digits of the SHA-256 of that document. Where a module sits in a
model is not part of it, so the same block on the same input shape shares one key across models.
Batch size is part of the input shape, so it is part of the key.

Fields written only when they differ from the default keep keys stable: a key computed before a
field existed is still the key of the same kernel.

A **platform** is `(gpu_name, frequency_label, precision)`, for example `A40/default/fp32`.
Signatures of different platforms never mix; `evaluate` raises `PlatformMismatch`.

## Composition

A `Cost` is the time and energy of a kernel or of any composition. There are two operators.

| Operator | Duration | Dynamic energy | Static energy |
|---|---|---|---|
| `sequential(a, b, …)` | `Σ Tᵢ` | `Σ dᵢ` | `s · Σ Tᵢ` |
| `parallel(a, b, …)` | `max Tᵢ` | `Σ dᵢ` | `s · max Tᵢ` — paid once, over the longest |

Sequential composition is kernels back to back. Parallel composition is **overlap**: kernels
that run at the same time share one static window, so against running them back to back overlap
saves

```
s · (Σ Tᵢ − max Tᵢ)          for two kernels:  s · min(T_a, T_b)
```

A `Cost` does not contain `s`. It stores three numbers:

- `time_seconds`: the wall time of the composition;
- `energy_joules`: the sum of the member kernels' energies, as measured;
- `busy_seconds`: the sum of the member kernels' durations, i.e. how long they would take back
  to back. For one kernel, and for anything sequential, it equals `time_seconds`; under overlap
  it is larger.

Each measured energy already includes static power over that kernel's own duration, so the sum
has paid `s` over `busy_seconds`. Static power is supplied when energy is asked for, and
corrects that to the wall time:

```
E = energy_joules + s · (time_seconds − busy_seconds)        which equals  s·T + d
```

For a sequential cost the two times are equal and the energy is the plain sum of measured
energies for any `s`. This is why a sequential estimate does not depend on which definition of
static power you choose, while an overlap estimate does.

```python
from ecalc.calculus import Cost, parallel, sequential

a, b = Cost(0.002, 0.60), Cost(0.003, 0.90)                    # (T, E) of two kernels
assert sequential(a, b).energy(90.0) == 1.5
assert parallel(a, b).time_seconds == 0.003
assert abs(parallel(a, b).energy(90.0) - (1.5 - 90.0 * 0.002)) < 1e-12
```

### Expressions and pipelines

A pipeline is an expression tree: `Kernel` leaves combined by `Sequential` and `Parallel` nodes.
`evaluate(expression, source, platform)` folds it into a `Cost` by looking every kernel up in a
signature source, and keeps each leaf's signature and provenance in the returned `Estimate`.

```python
from ecalc.calculus import Parallel, Sequential, evaluate, pipeline_name

yolo = database.pipeline(pipeline_name("yolo26n", batch_size=1, input_size=640)).expression
vgg = database.pipeline(pipeline_name("rf_vgg", batch_size=32, input_size=1024)).expression
static_power = database.static_power(platform, "active_idle")

alone = evaluate(yolo, database, platform)
together = evaluate(Parallel((yolo, vgg)), database, platform)
print(alone.energy(static_power), together.energy(static_power))
```

A model's forward pass is a `Sequential` of its kernels. Its backward pass is another
`Sequential`; a training-style step is `Sequential((forward, backward))`.

### Laws

All of these are tested in [`tests/test_laws.py`](../tests/test_laws.py) with exact equality,
and on the YOLO26 and RFML model structure in
[`tests/test_laws_models.py`](../tests/test_laws_models.py).

| Law | Statement |
|---|---|
| Closure | every cost has the form `E = s·T + d` |
| Identity | `sequential(a, ZERO) = a`, `parallel(a, ZERO) = a` |
| Commutativity | the order of the arguments does not matter, for both operators |
| Associativity | nested equals flat, for both operators |
| Sequential additivity | `T = Σ Tᵢ`, `d = Σ dᵢ`, and `E = Σ Eᵢ` for every static power |
| Overlap | `T = max Tᵢ`, `d = Σ dᵢ`, `E = s·max Tᵢ + Σ dᵢ` |
| Overlap saving | `E(sequential) − E(parallel) = s·(Σ Tᵢ − max Tᵢ) ≥ 0` |
| Bounds | `max Eᵢ ≤ E(parallel) ≤ Σ Eᵢ` when every `dᵢ ≥ 0` |
| No static power, no saving | with `s = 0`, `parallel` and `sequential` cost the same |
| Affine in `s` | `E(s) − E(0) = s·(T − busy)`, a slope that is zero or negative |
| Coarser overlap saves more | `E(parallel(sequential(a,b), sequential(c,d))) ≤ E(sequential(parallel(a,c), parallel(b,d)))` |

Three things are deliberately **not** laws, and the tests assert the counterexamples:

- `parallel(a, a) ≠ a`: both copies run, so dynamic energy doubles.
- `sequential` does not distribute over `parallel`:
  `sequential(a, parallel(b, c))` runs `a` once, `parallel(sequential(a, b), sequential(a, c))` twice.
- The lower bound `max Eᵢ ≤ E(parallel)` fails when a kernel draws less than the static power.

### Negative dynamic energy

Some real kernels average less than the static power. On the A40 a `torch.chunk` (a view,
almost pure launch overhead) draws about 82 W against an 89.7 W static floor, so its `d` is
negative. The calculus never clamps: clamping would invent energy.

- A sequential estimate does not depend on `s`, so it is never affected.
- `Estimate.energy(...)` raises `NegativeDynamicEnergy` when an **overlapped** kernel has
  `d < 0`, because then the static power does not fit that kernel and the overlap saving is not
  trustworthy. Pass `negative_dynamic="warn"` or `"allow"` to proceed, or use a static power
  definition below the kernel's average power.

### What the calculus assumes

- A kernel's `T` and `d` are the same inside a composition as when measured alone. Contention
  between overlapped kernels (one slowing the other) is not modelled.
- One static power per platform. A pipeline that spans several GPUs is evaluated with their
  static powers summed.

**Check on the A40.** `scripts/measure_overlap.py` runs two models at once on two CUDA streams
and compares the measured energy of an 8 s window with predictions built from each model
measured alone and the calls that happened in the window (one run, two pairs):

| Pair | Slowdown per call when sharing | No overlap: `Σ nᵢ·Eᵢ` | Overlap rule: `s·T + Σ nᵢ·dᵢ` |
|---|---|---:|---:|
| yolo26n at batch size 1 with rf_vgg at batch size 32 | 1.33×, 1.34× | +23.4% | −2.4% |
| yolo26m at batch size 8 with yolo26n at batch size 8 | 1.28×, 3.56× | +6.0% | +4.2% |

When both models are launch-bound, the GPU has room for both and the overlap rule predicts the
shared window well. When each model alone already draws the 300 W power cap, they take turns
rather than overlap: there is little static power to save, and the rule is only slightly
better than adding the energies. In both cases each call takes longer than alone, which the
calculus does not predict; the comparison above uses the measured window.

## Sources and provenance

A signature source answers "what does this kernel cost on this platform?".

| Source | Answers | Provenance |
|---|---|---|
| `SignatureDatabase` | exact key matches only | `measured` |
| `ecalc.formula.Interpolation` | kernels with parameters, from the same family at other batch sizes or input sizes | `interpolated`, `extrapolated` |
| `ecalc.formula.LightOpsFormula` | parameter-free operations, from bytes moved | `formula`, `extrapolated` |
| `SourceChain([...])` | the first source with an answer | that source's |

By default only the database is used, and a pipeline with an unmeasured kernel raises
`MissingKernel` (or lists it in `Estimate.missing` with `on_missing="report"`). Formula mode is
opt-in; see [formula.md](formula.md).

## File formats

Every document carries a `schema` string with a major version. An unknown schema raises
`SchemaVersionError`; unknown fields raise `ValueError`. Floats are written with full precision
and round-trip bit for bit.

**One signature** — `ecalc.signature/1`

```json
{"schema": "ecalc.signature/1", "key": "1a2658234970b8d3",
 "kernel": {"sig_version": 1, "module": {"cls": "Conv", "...": "..."},
            "input": {"kind": "tensor", "shape": [1, 3, 640, 640], "dtype": "float32"}},
 "platform": {"gpu_name": "A40", "frequency_label": "default", "precision": "fp32"},
 "time_seconds": 6.911e-05, "energy_joules": 0.020677, "provenance": "measured"}
```

**One pipeline** — `ecalc.pipeline/1`. `kernels` maps each key to its structural signature;
expression nodes are `{"kernel": key}`, `{"sequential": [...]}` or `{"parallel": [...]}`, each
with an optional `name`.

```json
{"schema": "ecalc.pipeline/1", "name": "two kernels overlapped",
 "kernels": {"1a2658234970b8d3": {"...": "..."}, "83a9d30579dbb458": {"...": "..."}},
 "expression": {"parallel": [{"kernel": "1a2658234970b8d3"}, {"kernel": "83a9d30579dbb458"}]}}
```

For `ecalc compose`, a node may also be `{"pipeline": "<name>"}`, which stands for a pipeline
stored in the database (see [`examples/`](../examples)).

**A signature database** — `ecalc.signature_database/1`, JSON Lines. The first line is the
header; each later line is one record: `kernel` (key and structural signature, once per kernel),
`signature`, `static_power`, `pipeline` or `model_run` (an end-to-end reference measurement).
Records refer to kernels by key. On load, a kernel whose structure does not hash to its key is
rejected.

`dump(obj, path)` and `load(path)` handle all three. The file is measured data for one GPU: keep
it out of version control.
