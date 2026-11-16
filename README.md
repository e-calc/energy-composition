# e-calc: energy calculus for GPU workloads

`ecalc` estimates the GPU energy of a model from the energy of its parts. Each part — a
*kernel* — is measured once and stored with an energy signature

```
E = s·T + d          static power × duration + dynamic energy
```

and signatures compose: kernels back to back add up; kernels that overlap pay static power once,
over the longest one. A model you have never run end to end is priced by looking up its kernels
and composing them.

What is in the box:

- **The calculus** (`ecalc.calculus`): the signature schema, the two composition operators,
expression trees for pipelines, and JSON serialization. Standard library only.
- **Measurement** (`ecalc measure`): per-kernel time and energy on your GPU with
[Zeus](https://ml.energy/zeus/), measured static power, forward and backward passes.
- **Estimation** (`ecalc estimate`, `ecalc compose`): the energy of a model or of any pipeline
from a signature database. No GPU needed.
- **Formula mode** (optional): interpolate kernels you did not measure, clearly labelled.

Models supported out of the box: YOLO26 n/s/m/l/x (Ultralytics) and three RFML modulation
classifiers (`rf_vgg`, `rf_resnet`, `rf_lstm`).

## Install

```bash
git clone https://github.com/e-calc/energy-composition && cd energy-composition
pip install .                  # the calculus and the estimation commands; no dependencies
pip install '.[capture]'       # + torch and Ultralytics, to capture a model's kernels (CPU is enough)
pip install '.[measure]'       # + Zeus and NVML, to measure on a CUDA GPU
```

or, with [`uv`](https://docs.astral.sh/uv/), everything at once:

```bash
uv sync
uv run pytest tests -m "not gpu"
```

Python 3.12 or newer.

## The calculus in a minute

```python
from ecalc.calculus import Cost, parallel, sequential

a = Cost(time_seconds=0.002, energy_joules=0.60)       # one kernel, as measured
b = Cost(time_seconds=0.003, energy_joules=0.90)
s = 90.0                                               # static power, watts

sequential(a, b).energy(s)      # 1.50 J in 5 ms: energies add
parallel(a, b).energy(s)        # 1.32 J in 3 ms: static power is paid once, saving s · 2 ms = 0.18 J
```

With a signature database, a model is an expression over its kernels:

```python
from ecalc.calculus import evaluate, load, pipeline_name

database = load("data/signature_database.jsonl")
platform = database.platforms()[0]                                  # e.g. A40/default
static_power = database.static_power(platform)

yolo = database.pipeline(pipeline_name("yolo26n", batch_size=8, input_size=640)).expression

evaluate(yolo, database, platform).energy(static_power)
```

The definitions, the algebraic laws and the file formats are in
[docs/formalism.md](docs/formalism.md).

## Command line

| Command | What it does | Needs |
|---|---|---|
| `ecalc estimate --model yolo26n --batch-size 8` | energy of a model, split into static and dynamic, with the error against the whole-model measurement when one is stored | a database |
| `ecalc compose examples/two_models_one_gpu.json` | energy and time of any sequential / parallel pipeline | a database |
| `ecalc plan --model yolo26n --batch-size 4` | which kernels are measured, which formula mode can fill, which are missing | a database |
| `ecalc inspect --model rf_vgg --batch-size 32` | the kernels of a model and their keys | torch (CPU) |
| `ecalc measure static-power \| kernels \| model` | measure on your GPU into `data/ecalc.sqlite` | a CUDA GPU |
| `ecalc export` | write the store as `data/signature_database.jsonl` | — |

Useful options: `--direction backward`, `--level leaf`, `--fill formula`, `--kernels` (per-kernel
table), `--json` (which also carries the times).

```text
$ ecalc estimate --model yolo26n --batch-size 8
yolo26n: batch size 8, input size 640, module level, forward
  platform            A40/default, static power 89.7 W
  kernels             24 (24 unique)
  energy              2,742.7 mJ
    static  s*T       939.5 mJ
    dynamic d         1,803.1 mJ
  by provenance       measured 24 kernels, 2,742.7 mJ
  measured end to end 2,748.8 mJ
    composition error -0.2%
```

`estimate`, `compose` and `plan` read the database given by `--database`, else
`$ECALC_DATABASE`, else `data/signature_database.jsonl`, else `data/ecalc.sqlite`.

## Measure your own GPU

A signature database belongs to one GPU. None is shipped: you build yours, and it stays on your
machine (`data/` is git-ignored).

```bash
sudo nvidia-smi -i 0 -pm 1                                   # once: GPU persistence mode
ecalc measure static-power --models yolo26n --auto-window
ecalc measure kernels --models all --batch-size 8            # each unique kernel, alone; resumable
ecalc measure model   --models all --batch-size 8            # whole models, as the reference
ecalc export                                                 # -> data/signature_database.jsonl
ecalc estimate --model all --batch-size 8
```

Add `--direction backward` to `measure kernels` and `measure model` for the backward pass, and
`--models rfml` for the RFML classifiers. `scripts/run_all.sh` runs the whole sequence. The
protocol, how static power is measured and the pitfalls are in
[docs/measurement.md](docs/measurement.md).

## Formula mode: skip measurements you can estimate

A kernel's key includes its batch size and input size, so every new batch size is a new set of
measurements, at about ten seconds per kernel. Formula mode estimates the kernels you did not
measure from the ones you did, and labels each estimate `interpolated`, `formula` or
`extrapolated`. It is off unless you ask:

```bash
ecalc estimate --model rf_vgg --batch-size 64 --fill formula            # from batch 32 and 256
ecalc measure kernels --models yolo26n --batch-size 4 --fill formula    # measure only what it cannot fill
```

Nothing is hard-coded per GPU: chords and fits come from your own database. Details and limits:
[docs/formula.md](docs/formula.md).

## Results on the NVIDIA A40

Each model's forward pass is estimated as a `sequential` of its kernels, which is the sum of the
measured kernel energies, and compared with a measurement of the whole model. The estimate uses
no whole-model measurement.

![Composed energy per forward pass as stacked static and dynamic bars, with the measured whole-model energy marked above each bar. Left: YOLO26 n, s, m, l and x at batch size 8, errors -0.2%, -1.2%, -1.1%, -1.0% and -0.9%. Right: rf_vgg, rf_resnet and rf_lstm at batch size 256, errors +2.5%, +1.5% and +0.2%.](docs/figures/yolo26_rfml_a40.png)

Left: YOLO26 at batch size 8, 640 × 640 images. Right: RFML at batch size 256, 2 × 1024 IQ
frames. A bar is the composed energy, split into static (`s·T`, with `s` = 89.7 W) and dynamic;
the black line is the measured whole model, and the label is the error of the composed energy.

- **YOLO26.** The composed energy is within 1.2% of the measurement for all five models, from
2.7 J (yolo26n) to 27 J (yolo26x) per forward pass.
- **RFML.** The composed energy is within 2.5% for the three models, from 0.6 J to 5 J.
- **Static power.** Between 30% and 35% of the energy of every model is static.

Setup: one NVIDIA A40 with a 300 W power cap, FP32 with TF32, PyTorch 2.14.

## Repository layout

```text
ecalc/calculus/           signature schema, sequential / parallel, expressions, serialization (standard library only)
ecalc/formula/            optional estimates for unmeasured kernels: interpolation, light ops
ecalc/command_line.py     the ecalc command
ecalc/signature.py        structural signature of a torch module or functional op -> kernel key
ecalc/capture.py          a model's kernels with their real inputs: module level, leaf level, backward
ecalc/measure.py          the Zeus measurement of one kernel or one whole model
ecalc/static_power.py     static power: the GPU awake and idle
ecalc/db.py               SQLite measurement store and its bridge to the calculus
ecalc/estimate.py         per-layer reports, level comparison, plots
ecalc/models.py, rfml.py  YOLO26 and RFML model registry
scripts/                  wrappers for the measurement commands, run_all.sh, estimate_model.py (reports and plots), measure_overlap.py
examples/                 pipeline files and library usage
tests/                    laws, serialization, formula mode, capture, command line; fixtures hold model structure only
docs/                     formalism.md, measurement.md, formula.md, figures/
```

## Acknowledgements

e-calc is based on "Energy Calculus: A Compositional Algebra of Energy in Computational Systems"
(arXiv 2607.11087). Energy is measured with [Zeus](https://ml.energy/zeus/).

The YOLO26 models are from Jocher et al., "Ultralytics YOLO26: Unified Real-Time End-to-End
Vision Models" (arXiv 2606.03748). The RFML models follow O'Shea, Roy and Clancy, "Over-the-Air
Deep Learning Based Radio Signal Classification" (2018), for rf_vgg and rf_resnet, and Rajendran
et al., "Deep Learning Models for Wireless Signal Classification with Distributed Low-Cost
Spectrum Sensors" (2018), for rf_lstm.

## License

Apache License 2.0. See [LICENSE](LICENSE).
