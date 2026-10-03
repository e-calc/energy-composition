# Estimating RFML inference energy from a module database on NVIDIA A40

## Executive summary

This study applies the e-calc composition method, first validated on YOLO26
([report_yolo26_a40.md](report_yolo26_a40.md)), to radio-frequency machine
learning (RFML). Each building block of a model is measured once, stored in a
database, and the energy of a complete inference is estimated as `E_sum`, the
sum of the measured energies of the model's database entries. Static power is
the `active_idle` value of 89.7 W from the YOLO study.

We evaluate three standard automatic-modulation-classification (AMC) networks
on RadioML 2018.01A-style frames (2 × 1024 IQ samples, 24 classes): the VGG-style
CNN and the ResNet of O'Shea et al., and the two-layer LSTM of Rajendran et al.
Each model runs at batch 32 and batch 256 on one A40.

The main results are:

- Module-level `E_sum` predicts all six model/batch cases within 6.2%. The mean
  absolute error is 3.5% at batch 32 and 2.6% at batch 256.
- The LSTM is the easiest case (-1.2% and +0.2%). Its two recurrent layers keep
  the GPU fully busy and dominate its energy. The VGG CNN is the hardest case
  (-6.2% at batch 32, +6.2% at batch 256).
- The two VGG errors have opposite signs. They follow the temperature
  difference between the component and end-to-end measurements, plus, at batch
  256, launch overhead of the small late layers that end-to-end execution hides.
  In a diagnostic re-measurement, the same VGG forward cost 9% more energy at
  66–72 °C than at 54–62 °C. With the components re-measured on a warm GPU, the
  batch-32 VGG error fell from -6.2% to -2.4%.
- The three models contain 24 module instances but need 21 database entries
  per batch size, because VGG and ResNet share the same dense classifier head.
- Moving from batch 32 to batch 256 reduces measured energy per frame by about
  30% for all three models, and the static share of `E_sum` falls from
  48–60% to 31–34%.

## 1. Objective

The YOLO26 study showed that composing a model from independently measured
modules predicts its inference energy within a few percent. RFML models differ
in ways that could break this:

- they are one-dimensional and small (0.16–0.20 M parameters), so many of their
  blocks are too short to keep the GPU busy;
- they include recurrent layers, whose sequential cuDNN kernels behave unlike
  convolutions;
- typical deployments classify streams of short frames in batches, not single
  images.

We ask:

1. Does module-level `E_sum` estimate end-to-end energy for convolutional,
   residual, and recurrent RFML models?
2. How does accuracy change between a partly launch-bound batch (32) and a
   GPU-saturating batch (256)?
3. How much database reuse is possible across RFML architectures?

## 2. Estimation method

### 2.1 Models and database units

All models classify RadioML 2018.01A frames of 1024 complex samples into 24
modulation classes.

| Model | Source | Parameters | Input per frame |
|---|---|---:|---|
| `rf_vgg` | O'Shea, Roy & Clancy, IEEE JSTSP 2018 (VGG CNN) | 159,832 | 2 × 1024 IQ |
| `rf_resnet` | O'Shea, Roy & Clancy, IEEE JSTSP 2018 (ResNet) | 165,144 | 2 × 1024 IQ |
| `rf_lstm` | Rajendran et al., IEEE TCCN 2018 (LSTM, 128 cells × 2) | 202,776 | 1024 × 2 amplitude/phase |

The paper does not give the convolution kernel sizes. We use the common
reimplementation: kernel 3 with `same` padding and max-pooling by 2. Our VGG and
ResNet therefore have fewer parameters than the paper reports (257,099 and
236,344), but they keep the layer structure, widths, and output dimensions of the
paper's tables.

Each model is defined as a list of top-level blocks executed in order. Each
block is one database unit, the counterpart of one Ultralytics graph entry in
YOLO26:

| Model | Module-level units (in execution order) | Count |
|---|---|---:|
| `rf_vgg` | 7 × `ConvPool` (Conv1d k3 + ReLU + MaxPool 2; 2→64, then 64→64), `Flatten` (64 × 8 → 512), `DenseSELU` 512→128, `DenseSELU` 128→128, `Classifier` 128→24 + softmax | 11 |
| `rf_resnet` | 6 × `ResidualStack` (Conv1d k1 → 2 residual units [Conv1d k3 + ReLU → Conv1d k3 → add] → MaxPool 2), `Flatten` (32 × 16 → 512), the same three dense blocks | 10 |
| `rf_lstm` | `LSTMLayer` 2→128, `LSTMLayer` 128→128, `LastStepClassifier` (last time step → Linear 128→24 + softmax) | 3 |

Residual additions, activations, and dropout (an identity at inference) stay
inside their block. This mirrors YOLO's composite blocks such as `C3k2`. The
`Flatten` units are views that launch no GPU kernel. They are kept as units so
that every executed operation of the forward pass is covered.

Entries are identified by the same structural signature as for YOLO: module
configuration, parameter shapes, and input tensor shapes and dtypes. Composing
the captured units reproduces the model output bit-exactly on CPU.

### 2.2 Energy estimator

The estimator is unchanged. For entry \(k\) with measured energy \(e_k\) and
time \(t_k\):

\[
E_{\text{sum}} = \sum_k \left(E_{\text{dyn},k} + P_{\text{active-idle}}t_k\right) = \sum_k e_k
\]

All three models run their blocks sequentially on one stream, so `E_sum` is a
pure database estimate. It uses no end-to-end time or energy.

### 2.3 Active-idle static power

We reuse the A40 `active_idle` measurement from the YOLO study:

\[
P_{\text{active-idle}} = 89.7\ \text{W}
\]

The static power only splits `E_sum` into static and dynamic portions; it does
not change the `E_sum` prediction.

## 3. Experimental setup

| Item | Configuration |
|---|---|
| GPU | NVIDIA A40, 300 W power cap, 1740 MHz maximum SM clock |
| Software | Python 3.12, PyTorch 2.14.0+cu130, Zeus 0.16.0, driver 595.71.05 |
| Models | `rf_vgg`, `rf_resnet`, `rf_lstm`; seeded random weights (seed 0) |
| Input | 1024-sample frames, seeded standard-normal FP32 tensors (complex-AWGN-like IQ) |
| Precision | FP32 with TF32 enabled for convolution and matrix multiplication |
| Batch sizes | 32 and 256 |
| Kernel protocol | 5 s measurement window, 3 s cooldown, 50 warm-up calls, 1000 calibration calls |
| Repeats | 3 for module entries and end-to-end measurements (median-energy trial kept) |
| GPU state | persistence mode off, unlocked clocks, as for the YOLO data |

No official PyTorch checkpoints exist for these models. Weights are seeded
random tensors. Dense FP32 kernels do the same work for any weight values, so
the measured energy does not depend on training. Preprocessing is excluded. For
the LSTM this means the IQ-to-amplitude/phase conversion is excluded, and the
model receives a 1024 × 2 sequence directly.

Every module lookup succeeds for all six model/batch cases. The repeat-to-repeat
energy variation of the 42 component entries has a median of 0.6% and a maximum
of 2.1%. All rows ran at 1740 MHz without clock dips. Most rows carry a
temperature-rise flag (more than 5 °C within a measurement), as many YOLO rows
did.

## 4. Module-level results

### 4.1 Batch 32

| Model | Measured energy / forward | `E_sum` | Estimation error | Measured time | GPU-active share | Measured energy / frame |
|---|---:|---:|---:|---:|---:|---:|
| `rf_vgg` | 107.2 mJ | 100.6 mJ | **-6.2%** | 0.67 ms | 50% | 3.35 mJ |
| `rf_resnet` | 223.3 mJ | 216.4 mJ | **-3.1%** | 1.49 ms | 44% | 6.98 mJ |
| `rf_lstm` | 909.5 mJ | 898.5 mJ | **-1.2%** | 4.81 ms | 100% | 28.42 mJ |

The mean absolute error is 3.5%. At batch 32 the two CNNs are half
launch-bound: the GPU executes kernels for only 44–50% of the forward pass. In
isolation, every block after the first two or three takes 73–75 µs (`ConvPool`)
or 205–239 µs (`ResidualStack`) of wall time, but only 20–130 µs of GPU time,
and draws 100–170 W. The LSTM is the opposite case: its two layers are long
cuDNN recurrent kernels (2.4 ms each) that keep the GPU busy.

Time composes well for VGG (-0.4%) and the LSTM (+0.7%). For ResNet the sum of
isolated block times is 4.4% below the end-to-end time. Its energy error
(-3.1%) follows from the missing time at a moderate power.

### 4.2 Batch 256

| Model | Measured energy / forward | `E_sum` | Estimation error | Measured time | GPU-active share | Measured energy / frame | Change per frame from batch 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rf_vgg` | 610.7 mJ | 648.3 mJ | **+6.2%** | 2.28 ms | 99% | 2.39 mJ | -29% |
| `rf_resnet` | 1240.1 mJ | 1259.3 mJ | **+1.6%** | 4.25 ms | 99% | 4.84 mJ | -31% |
| `rf_lstm` | 5024.4 mJ | 5032.4 mJ | **+0.2%** | 17.37 ms | 100% | 19.63 mJ | -31% |

The mean absolute error is 2.6%. At batch 256 the end-to-end forward keeps the
GPU 99–100% busy. The sum of block times now overshoots end-to-end time for the
CNNs (+7.3% VGG, +8.1% ResNet). The last few CNN blocks and the dense head work
on 8–64-sample feature maps and stay launch-bound when measured alone. For
example, the last VGG `ConvPool` takes 76 µs of wall time for 24 µs of GPU
work. In the full forward pass, the CPU launches these blocks while the GPU is
still busy with the large early layers, so their launch time is hidden. Summing
the wall-minus-GPU time of these blocks gives 177 µs for VGG and 379 µs for
ResNet. That matches the time overshoots of 167 µs and 342 µs.

The overshoot is time spent near the static floor. At 89.7 W it adds about 15 mJ
to VGG's `E_sum` (+2.5%) and about 31 mJ to ResNet's (+2.5%). ResNet's +1.6%
is therefore fully explained by this launch overhang. VGG's +6.2% is not: a
further +3.7% comes from the large early `ConvPool` blocks themselves, which
were measured hotter than the end-to-end runs (Section 6).

### 4.3 Static and dynamic energy portions

| Model | Batch-32 dynamic | Batch-32 static | Batch-256 dynamic | Batch-256 static |
|---|---:|---:|---:|---:|
| `rf_vgg` | 40% | 60% | 66% | 34% |
| `rf_resnet` | 41% | 59% | 67% | 33% |
| `rf_lstm` | 52% | 48% | 69% | 31% |

At batch 32 the static floor accounts for about 60% of the CNN estimates, the
same range as the launch-bound batch-1 YOLO26n and YOLO26s. At batch 256 the static share
falls to 31–34% for all three models, close to the 30% that every YOLO26 size
reached at batch 8. Energy per frame drops by 29–31%.

### 4.4 Energy distribution by module type

| Model | Batch | Main blocks | `Flatten` | Dense head (2 × `DenseSELU` + classifier) |
|---|---:|---:|---:|---:|
| `rf_vgg` | 32 | 90.0% (`ConvPool`) | 0.3% | 9.7% |
| `rf_vgg` | 256 | 98.3% (`ConvPool`) | 0.1% | 1.7% |
| `rf_resnet` | 32 | 95.4% (`ResidualStack`) | 0.2% | 4.6% |
| `rf_resnet` | 256 | 99.1% (`ResidualStack`) | 0.0% | 0.9% |
| `rf_lstm` | 32 | 99.6% (`LSTMLayer`) | – | 0.4% |
| `rf_lstm` | 256 | 99.9% (`LSTMLayer`) | – | 0.1% |

Energy is concentrated in the full-resolution front of each network. The
first `ConvPool` and the first `ResidualStack`, which operate on all 1024
samples, are the largest single modules: 28% and 29% of `E_sum` at batch 32,
40% and 50% at batch 256. The two LSTM layers split energy almost evenly (48%
and 52%). The dense head is a measurable share only at batch 32. There it is
launch-bound and runs at about 100 W.

The LSTM is the most expensive model per frame by a wide margin: 19.6 mJ at
batch 256, against 4.8 mJ for ResNet and 2.4 mJ for VGG. Each LSTM layer is a
sequential recurrence over 1024 time steps.

## 5. Database reuse

The three models contain 24 module instances per batch size but need only 21
unique database entries. VGG and ResNet both flatten to a 512-wide vector and
share the dense head exactly, so their `DenseSELU` 512→128, `DenseSELU`
128→128, and `Classifier` 128→24 entries are measured once. Their `Flatten`
entries remain distinct, because the input shapes differ (64 × 8 and 32 × 16).
The `LastStepClassifier` of the LSTM uses the same Linear layer, but it is a
different module with a different input, so it is a separate entry.

There is no reuse within a model. Every CNN block works at a different sequence
length, and sequence length is part of the input signature. Batch size is part
of the signature too, so batch 256 needs another 21 entries. Building the
database took 8.8 minutes for batch 32 and 9.8 minutes for batch 256, with
three repeats.

For comparison, YOLO26 reached 96 entries for 120 instances across five sizes
of one family. Cross-architecture reuse in RFML is limited to shared heads,
unless the database grows to cover many models built from the same blocks.

## 6. Error sources specific to small models

### 6.1 GPU temperature

These models draw 230–300 W in short kernels, and the 3-second cooldown does
not return the GPU to a fixed temperature. Temperature therefore drifts through
a measurement session. The measurements show the effect directly. For
`rf_vgg` at batch 256, the three end-to-end repeats ran at the same time per
forward (2.28 ms) and the same clock (1740 MHz), but their energy rose from
593 mJ to 611 mJ to 617 mJ as the start temperature rose from 63 °C to 70 °C.

The sign of the VGG errors follows the temperature difference between the
component and end-to-end measurements:

- **Batch 32 (-6.2%).** The session started on a cold GPU. The VGG components
  were measured first, at 56–66 °C, and the end-to-end runs later, at 70–74 °C.
  The cooler components draw less leakage power, so `E_sum` comes out low.
  Time agrees within 0.4%, so the gap is a power difference, not missing work.
- **Batch 256 (+6.2%).** The large `ConvPool` components heated the GPU to
  82–85 °C during their own windows. The median end-to-end trial ran at
  67–78 °C. `E_sum` comes out high.

A separate diagnostic tested this for `rf_vgg` at batch 32, using a scratch
copy of the database so that the primary results above are unchanged. It
measured the end-to-end forward on a cool GPU, then re-measured all 11
components, and then measured the end-to-end forward again on the warm GPU:

| Diagnostic measurement | GPU temperature | Time / forward | Energy / forward |
|---|---:|---:|---:|
| End to end, cool GPU | 54–62 °C | 676 µs | 97.6 mJ |
| Components re-measured (`E_sum`) | 62–77 °C | 668 µs (sum) | 103.8 mJ |
| End to end, warm GPU | 66–72 °C | 673 µs | 106.4 mJ |

The same forward pass costs 9% more energy at 66–72 °C than at 54–62 °C, at
the same time per forward and the same 1740 MHz clock. With the components
measured on a warm GPU, the batch-32 VGG error against the warm end-to-end run
falls from -6.2% to **-2.4%**. Most of the original error therefore comes from
the temperature difference, not from the decomposition. These diagnostic rows
are not used anywhere else in this report.

### 6.2 Hidden launch overhead

At batch 256, the end-to-end forward hides the CPU launch time of the small late
blocks behind the GPU work of earlier blocks. In isolation, the same blocks
show it in full. Section 4.2 quantifies this at about +2.5% of `E_sum` for both
CNNs. For YOLO26 at batch 8, block times summed to within 0.4% of end-to-end time for
s/m/l/x, and this effect was not visible. RFML networks shrink the sequence by 64–128× from input to head, so
their last blocks are much smaller than YOLO's.

A database entry describes a block measured on its own. When a block is
launch-bound alone but not inside the model, its stored time and static energy
are too high for that context. Merging these short blocks with the head into
one larger unit would remove this bias, at the cost of less reuse.

## 7. Conclusions

The module database estimates RFML inference energy within 6.2% for all three
architectures and both batch sizes, with mean absolute errors of 3.5% at batch
32 and 2.6% at batch 256. The estimator works across a 50-fold range in forward
energy, from 0.10 J (VGG, batch 32) to 5.0 J (LSTM, batch 256). Recurrent layers
are the most predictable case (within 1.2%), because they are long kernels that
keep the GPU busy.

Two error sources are larger for these models than for YOLO26:

- the GPU temperature at measurement time: the same forward cost up to 9% more
  energy on a warm GPU than on a cool one;
- launch overhead of tiny late blocks, which end-to-end execution hides at
  large batch.

For future RFML measurements we recommend:

- estimator: **`E_sum`**, with **`active_idle` at 89.7 W** for the static and
  dynamic split;
- database level: **module**, with the blocks listed in Section 2.1;
- warming the GPU to its steady operating temperature before component and
  end-to-end measurements, or interleaving them, so that both are measured at
  the same temperature. Treat end-to-end repeat spreads above about 1% as a
  warning sign.

These conclusions apply to the measured A40, software, FP32/TF32 precision,
1024-sample frames, and batch sizes 32 and 256. Other frame lengths, batch
sizes, or precisions need their own database entries.
