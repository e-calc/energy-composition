# Estimating YOLO26 inference energy from a module database on NVIDIA A40

## Executive summary

This study tests whether the energy of a complete neural-network inference can
be estimated without measuring the model end to end. Instead, each building
block is measured once, stored in a database, and reused whenever the same
block and input shape appear in another model.

We evaluate the five YOLO26 sizes (`n`, `s`, `m`, `l`, and `x`) on an NVIDIA
A40 at batch sizes 1 and 8. The estimator used throughout this report is
`E_sum`, the sum of the measured energies of the database entries selected for
the model. Static power is measured as `active_idle`, the power of an awake GPU
after inference activity has settled.

The main results are:

- Module-level `E_sum` estimates energy well in almost every case. The only
  exception is batch-1 YOLO26n (-5.0%), where the small model and its
  small workloads make execution CPU-launch-bound.
- At batch 1, the other four models are within 2.9% of direct measurement; at
  batch 8, every model is within 1.2%.
- The five batch-1 models contain 120 module instances but require only 96
  unique database entries because some modules are shared across model sizes.
- A finer leaf-level decomposition increases the database from 96 to 230
  unique entries per batch and introduces systematic underestimation,
  especially at batch 8.
- We therefore use the top-level Ultralytics module as the default database
  level. It is more compact and includes the memory traffic and execution
  interactions inside composite blocks.

## 1. Objective

Directly measuring every complete model, batch size, and deployment
configuration does not scale. The goal of e-calc is to build a reusable energy
database from smaller model components and estimate a model by composing those
components.

For this first case study, the target is YOLO26 inference on an NVIDIA A40. We
ask three questions:

1. Can isolated component measurements predict end-to-end inference energy?
2. How does prediction accuracy change between power-limited and
   non-power-limited workloads?
3. What component granularity gives the best balance between accuracy,
   database size, and reuse?

## 2. Estimation method

### 2.1 Database unit

The default database unit is one top-level entry in the Ultralytics YOLO graph.
Each YOLO26 model has 24 such modules:

- 7 `Conv`
- 8 `C3k2`
- 1 `SPPF`
- 1 `C2PSA`
- 2 `Upsample`
- 4 `Concat`
- 1 `Detect`

A module can internally launch multiple CUDA kernels. In this report, “module”
therefore means a reusable graph-level block, not a single CUDA kernel.

Each database entry is identified by a structural signature containing the
module configuration, parameter and buffer shapes, and input tensor shapes and
dtypes. Graph position and runtime caches are excluded, allowing an identical
module used elsewhere to reuse the same measurement.

### 2.2 Energy estimator

For database entry \(k\), let:

- \(e_k\) be its measured total energy;
- \(t_k\) be its measured execution time;
- \(P_{\text{active-idle}}\) be the measured static power of an awake GPU.

The dynamic portion of an entry can be written as:

\[
E_{\text{dyn},k} = e_k - P_{\text{active-idle}}t_k
\]

The YOLO26 forward pass executes its top-level modules sequentially on one GPU
stream. We therefore estimate model energy by summing
the energy of every module in execution order:

\[
E_{\text{sum}}
  = \sum_k \left(E_{\text{dyn},k}
    + P_{\text{active-idle}}t_k\right)
  = \sum_k e_k
\]

Thus, `E_sum` is a pure database estimate for sequential execution: it only
sums component measurements and does not use the measured end-to-end model
time or energy. The static split is used to explain where energy is spent,
while the total prediction remains the sum of measured entry energies.

### 2.3 Active-idle static power

All static-energy analysis in this report uses the `active_idle` measurement.
The GPU first runs full-model inference for 2 seconds, synchronizes, and waits
1 second for post-work activity to settle. Power is then measured over a
1-second window while the GPU remains awake at P0 and at least 95% of its
maximum SM clock.

On the A40, 20 accepted cycles gave:

\[
P_{\text{active-idle}} = 89.7\ \text{W}
\]

with a standard deviation of 1.2 W. This state represents the static baseline
relevant to repeated inference in a process that holds a CUDA context.

## 3. Experimental setup

| Item | Configuration |
|---|---|
| GPU | NVIDIA A40, 300 W power cap, 1740 MHz maximum SM clock |
| Software | Python 3.12, PyTorch 2.14.0+cu130, Ultralytics 8.4.148, Zeus 0.16.0 |
| Models | YOLO26 n/s/m/l/x official weights |
| Model path | Fused, NMS-free end-to-end detection head |
| Input | 640 × 640, seeded random FP32 tensors |
| Precision | FP32 with TF32 enabled for convolution and matrix multiplication |
| Batch sizes | 1 and 8 |
| Kernel protocol | 5 s measurement window, 3 s cooldown, 50 warm-up calls, 1000 calibration calls |
| Repeats | 3 for module entries and end-to-end measurements |

Component and end-to-end measurements use the same Zeus/NVML protocol. Calls
are repeated back to back within each energy window, and the median-energy
trial is stored. A background sampler records clocks, power, P-state, and
temperature so unstable measurements can be identified.

The model is warmed before capture, then each graph module receives the same
input tensors and routing that it receives in a real forward pass. All 24
module lookups succeed for every model.

## 4. Module-level results

### 4.1 Batch 1

| Model | Measured energy | `E_sum` | Estimation error | Measured time | GPU-active share |
|---|---:|---:|---:|---:|---:|
| YOLO26n | 765 mJ | 727 mJ | **-5.0%** | 6.06 ms | 32% |
| YOLO26s | 1018 mJ | 1021 mJ | **+0.4%** | 6.08 ms | 50% |
| YOLO26m | 1847 mJ | 1889 mJ | **+2.3%** | 6.87 ms | 87% |
| YOLO26l | 2414 mJ | 2450 mJ | **+1.5%** | 9.89 ms | 76% |
| YOLO26x | 3788 mJ | 3897 mJ | **+2.9%** | 12.63 ms | 96% |

The mean absolute error is 2.4%. YOLO26n is the most difficult case because it
is strongly launch-bound: the GPU is active for only 32% of the forward pass,
and static energy dominates. Even in this regime, the estimate remains within
5%. The other four models are within 3%.

### 4.2 Batch 8

| Model | Measured energy / forward | `E_sum` | Estimation error | Measured energy / image | Change per image from batch 1 |
|---|---:|---:|---:|---:|---:|
| YOLO26n | 2749 mJ | 2743 mJ | **-0.2%** | 344 mJ | -55% |
| YOLO26s | 5806 mJ | 5738 mJ | **-1.2%** | 726 mJ | -29% |
| YOLO26m | 12,749 mJ | 12,608 mJ | **-1.1%** | 1594 mJ | -14% |
| YOLO26l | 16,548 mJ | 16,376 mJ | **-1.0%** | 2069 mJ | -14% |
| YOLO26x | 27,326 mJ | 27,087 mJ | **-0.9%** | 3416 mJ | -10% |

At batch 8, GPU-active share reaches 98–99% for all model sizes, and the mean
absolute estimation error falls to 0.9%. The module time sum also matches
end-to-end time within 0.4% for s/m/l/x; YOLO26n remains slightly
launch-bound, but its energy estimate is still within 0.2%.

### 4.3 Static and dynamic energy portions

The `active_idle` baseline of 89.7 W separates each `E_sum` estimate into
static and dynamic portions:

| Model | Batch-1 dynamic | Batch-1 static | Batch-8 dynamic | Batch-8 static |
|---|---:|---:|---:|---:|
| YOLO26n | 30% | 70% | 66% | 34% |
| YOLO26s | 46% | 54% | 69% | 31% |
| YOLO26m | 61% | 39% | 70% | 30% |
| YOLO26l | 59% | 41% | 70% | 30% |
| YOLO26x | 67% | 33% | 70% | 30% |

Batching reduces the static-energy cost per image. At batch 1, static energy is
about 70% of the YOLO26n module sum but only 33% for YOLO26x. At batch 8, the
static share converges to roughly 30% across the model family. Consequently,
YOLO26n benefits most from batching, reducing measured energy per image by 55%.

Small models at batch 1 spend a larger fraction of their execution near the
active-idle power floor because they are more CPU-launch-bound. Batch 8 keeps
the GPU busy, so useful computation accounts for about 70% of estimated energy
for every model size.

### 4.4 Energy distribution by module type

| Model | Conv | C3k2 | SPPF | C2PSA | Concat + Upsample | Detect |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | 12% | 54% | 3% | 7% | 1% | 22% |
| YOLO26s | 18% | 53% | 2% | 6% | 3% | 18% |
| YOLO26m | 26% | 53% | 1% | 3% | 3% | 14% |
| YOLO26l | 20% | 62% | 1% | 4% | 1% | 11% |
| YOLO26x | 23% | 60% | 1% | 3% | 1% | 12% |

`C3k2` blocks account for 53–62% of estimated energy across all sizes. The
Detect head is the largest individual module for n/s/m, contributing
22%/18%/14%. For l/x, the first high-resolution `C3k2` block is the largest
individual module at about 15–16%.

## 5. Database reuse

Across the five batch-1 models, 120 module instances collapse to 96 unique
database entries. Eighteen keys are shared by two or three model sizes.
YOLO26m and YOLO26l have the same width, so they reuse many Conv, Concat,
Upsample, SPPF, and Detect entries; their deeper `C3k2` blocks remain distinct.

Batch size is part of the input signature. Batch 8 therefore requires another
96 module entries rather than reusing batch-1 measurements. This is necessary
because changing batch size changes execution time, utilization, power, and
energy.

The database is stored in SQLite and records the component signature,
measurement result, protocol, environment, and model composition. A model
estimate is a database join followed by a sum, and missing entries remain
visible instead of being silently ignored.

## 6. Choosing the composition level

We evaluated two levels:

- **Module level:** one entry for each of the 24 top-level YOLO graph modules.
- **Leaf level:** leaf modules plus functional operations such as `cat`,
  `chunk`, `split`, and residual addition.

| Level | Instances across five models | Unique entries per batch | Batch-1 `E_sum` error | Batch-8 `E_sum` error |
|---|---:|---:|---:|---:|
| Module | 120 | **96** | -5.0% to +2.9% | **-1.2% to -0.2%** |
| Leaf | 822 | **230** | -8.4% to +1.1% | **-5.4% to -3.6%** |

The leaf database is 2.4 times larger even though it uses finer components.
YOLO26 contains many distinct convolution shapes, so splitting composite
modules creates more database keys rather than fewer.

The finer level also introduces a systematic measurement bias. A leaf measured
alone repeatedly reads cache-resident inputs, while the same operation inside
a composite block processes activations produced by other operations and
competes for memory traffic. Isolated leaves therefore miss part of the
inter-operation cost. This is clearest at batch 8, where leaf-level estimates
are 3.6–5.4% low for all five models while module-level estimates remain within
1.2%.

For these reasons, **module is the default level**:

1. it gives the most consistent accuracy across both batch sizes;
2. it directly captures memory traffic and interactions within composite
   blocks;
3. it requires fewer measurements and a smaller database;
4. its 24-entry model representation is easier to inspect and maintain.

Leaf-level data may still be useful when transfer to a different architecture
is more important than database size, but it is not the primary estimator for
this study.

## 7. Conclusions

The experiment shows that inference energy can be composed from independently
measured modules. Using `E_sum`, the module database predicts all five YOLO26
sizes within 5% at batch 1 and within 1.2% at batch 8. The estimator works
across a 36-fold range in forward energy, from 765 mJ for batch-1 YOLO26n to
27.3 J for batch-8 YOLO26x.

The recommended configuration for this project is therefore:

- estimator: **`E_sum`**;
- static-power measurement: **`active_idle` at 89.7 W**;
- default database level: **`module`**.

These conclusions are specific to the measured A40 software and hardware
configuration. Extending the database to another GPU, precision, image size,
clock setting, or batch size requires measurements with those attributes in
the entry signature.
