# Measuring signatures

A signature is a measurement: the time and energy of one kernel call on one GPU. This page
describes how `ecalc` takes that measurement, how static power is defined, how the backward pass
is measured, and what to watch for.

Requirements: a CUDA GPU, `pip install 'ecalc[measure]'` (or `uv sync` in a checkout), and GPU
persistence mode on:

```bash
sudo nvidia-smi -i 0 -pm 1        # once, needs root
```

Without persistence mode the commands refuse to measure unless you pass
`--allow-no-persistence`; rows taken that way are flagged `no_persistence_mode`.

## The three commands

```bash
ecalc measure static-power --models yolo26n --auto-window               # s, once per GPU
ecalc measure kernels --models all --batch-size 1                       # every unique kernel, alone
ecalc measure model   --models all --batch-size 1                       # the whole model: the reference
ecalc export                                                            # store -> signature database
```

`scripts/run_all.sh` runs the same sequence. Results go to a local SQLite store
(`data/ecalc.sqlite`); `ecalc export` writes it as a portable signature database
(`data/signature_database.jsonl`). Both are measured data for your GPU and are git-ignored.

`measure kernels` is resumable: keys already in the store are skipped, and each kernel is
committed as soon as it is measured.

| Option | Meaning |
|---|---|
| `--models` | comma list, `all` (YOLO26 n/s/m/l/x) or `rfml` (rf_vgg, rf_resnet, rf_lstm) |
| `--batch-size`, `--input-size` | part of every kernel key; input size is the image side (YOLO, default 640) or the frame length (RFML, default 1024) |
| `--level module\|leaf` | one kernel per top-level layer, or per leaf module and glue operation |
| `--direction forward\|backward` | backward is module level only |
| `--repeats`, `--measurement-duration`, `--cooldown` | the protocol below |
| `--fill formula` | do not measure kernels that formula mode can already estimate |

## Protocol

Every kernel, and every whole model, is measured the same way with
[`zeus.profile.measure`](https://ml.energy/zeus/):

1. **Capture.** The model runs once on a seeded random input; each kernel is recorded with the
   exact tensors it received.
2. **Calibrate.** 50 warm-up calls, then 1000 calls to estimate the time per call.
3. **Cool down.** The GPU idles (3 s by default) so every trial starts from a similar
   temperature.
4. **Warm up, then measure.** 50 calls, then back-to-back calls for a 5 s window. Energy comes
   from the GPU's cumulative energy counter (NVML) read at both ends of the window, time from
   the wall clock.
5. **Repeat.** With `--repeats n` the trial with the median energy is kept.

The stored values are per call: `time_s = window time / calls` and
`energy_j = window energy / calls`. Time is wall time, so it includes the gaps between GPU
kernels (Python and launch overhead), exactly as a real forward pass does.

Forward kernels run eagerly under `torch.inference_mode()`. cuDNN autotuning is off and TF32 is on
(precision `fp32`). A background sampler records clocks, P-state and temperature during the
window; a temperature rise above 5 °C or a clock dip is written to the row's `notes`.

The whole-model run uses the identical protocol. A composition and its reference are therefore
compared like for like.

## Static power

`s` is measured, not assumed. It is the **active-idle** power: the floor under repeated
inference in a process that holds a CUDA context. `ecalc measure static-power` runs the model
for 2 s, waits 1 s for the transient to decay, then measures a 1 s window while the GPU is
still awake at P0. It repeats this 20 times, rejects the cycles where the GPU had already
ramped down, and stores the median. On the A40 this is 89.7 W.

A sequential estimate does not depend on `s`. The split into static and dynamic energy, and
any overlap saving, do.

## Granularity

- **module** (default): each top-level layer of the model is one kernel — 24 for a YOLO26
  model, 11 / 10 / 3 for rf_vgg / rf_resnet / rf_lstm.
- **leaf**: leaf modules (a fused convolution with its activation, pooling, upsampling,
  attention; the detection head is kept whole) plus the functional glue between them
  (`cat`, `chunk`, `split`, residual `add`) — 126 to 215 per YOLO26 model.

Keys do not depend on the level. A leaf that is also a top-level layer reuses the same row.

Module level is the recommended default: on the A40 it is more accurate and needs fewer
measurements (for YOLO26 at batch size 8, within 1.2% at module level against 3.6% to 5.4% low
at leaf level). A kernel measured alone re-reads cache-resident inputs;
a composite block measured as a whole includes the memory traffic between its parts.

## Backward pass

`--direction backward` measures the gradient computation of each module:

```
torch.autograd.grad(outputs, inputs + parameters, grad_outputs, retain_graph=True)
```

on an autograd graph that is built once and retained, so the call can be repeated inside the
energy window. The rest of the protocol is unchanged.

- A backward kernel has its own key: the forward signature plus `direction: backward`.
- The first layer reads the model input, which needs no gradient; its backward computes
  parameter gradients only, and that is part of its key.
- Where a layer's output feeds several layers, autograd sums the gradients arriving at it. Each
  extra consumer is counted as one `add` kernel of that output's shape.
- The whole-model reference is the gradient of the model's raw outputs with respect to every
  parameter, by the same call. For YOLO26 the raw outputs are the head tensors before box
  decoding and top-k selection.
- Recurrent layers are switched to train mode for the backward measurement (cuDNN refuses a
  recurrent backward in eval mode) and restored afterwards.

**Scope.** This is the backward pass of the model as deployed for inference: convolutions are
fused with their batch normalization, so there is no batch-norm backward, and there is no loss,
no `.grad` accumulation and no optimizer step. It prices the gradient kernels; it is not the
energy of a training step. Backward is available at module level only.

**Check on the A40.** Composed backward kernels against the whole-model backward, composed
relative to measured:

| Models | Batch size | Energy error | Time error |
|---|---:|---:|---:|
| YOLO26 n, s, m, l, x | 8 | −1.3% to +0.5% | −0.1% to +12.5% |
| YOLO26 n, s, m, l, x | 1 | +3.7% to +14.9% | +16.7% to +47.5% |
| rf_vgg, rf_resnet | 256 | +0.2% to +5.1% | +14.1% to +17.4% |
| rf_vgg, rf_resnet | 32 | +24.1% to +34.8% | +45.1% to +67.6% |
| rf_lstm | 32 and 256 | +11.2%, +12.8% | +2.7%, +1.5% |

Backward composes as well as forward when the kernels are large, and overestimates when they
are small. The sum of isolated kernel times is well above the whole-model time there, which
points at a per-call cost of an isolated `torch.autograd.grad` that the whole-model call pays
once; this has not been isolated yet. The rf_lstm gap is in energy, not time, and is also
unexplained.

## Things that go wrong

- **Temperature.** Small models are sensitive to where a trial starts thermally. For the RFML
  models a 10 s cooldown was needed on the A40 for components and the whole model to agree.
  Treat an end-to-end repeat spread above about 1 % as a warning sign.
- **Launch-bound kernels.** A tiny kernel measured alone is dominated by launch overhead and
  sits near the static floor; inside a model part of that overhead is hidden. This is the main
  source of composition error for small models at batch 1.
- **The profiler.** `--profile-kernel-time` adds GPU-active time from `torch.profiler`. Once the
  profiler has run, every later CUDA launch in that process is about 30 % slower, so it always
  runs after all energy measurements.
- **Changed libraries.** Kernel keys follow the module structure, which can change between
  Ultralytics versions. The versions are stored with every row.

## Adding a model

A model needs a loader and a way to capture its kernels with their inputs.

- A model whose forward runs a top-level `layers` list in order can subclass
  `ecalc.rfml.SequentialRFModel`; `capture_sequential` then works as is.
- Ultralytics detection models are captured by `capture_layers` (module) and `capture_leaves`.
- New leaf module types need an entry in `ATTR_WHITELIST` in `ecalc/signature.py`, listing the
  attributes that determine their computation.

`ecalc inspect --model <name>` shows the captured kernels and keys without a GPU.
