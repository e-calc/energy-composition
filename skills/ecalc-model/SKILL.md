---
name: ecalc-model
description: Use agents to inspect, decompose, validate, and energy-profile a requested neural network with ecalc. Generate or adapt executable component capture for new architectures, reuse compatible measurements, and compare composition estimates with whole-model energy.
---

# Agent-led model decomposition for ecalc

Carry a model/workload request through executable decomposition, validation,
missing-component measurement, and an honest energy comparison. The agent
chooses and implements the decomposition; a fixed model registry or mandatory
graph extractor is not the architecture of this workflow.

Work in the user's ecalc checkout. Resolve the checkout from the current task
or this skill's repository location. Inspect its actual files and local
instructions: documentation may describe adapters that do not exist locally.

## Request contract and independent evaluation

Use [references/request.md](references/request.md) for the reusable prompt,
scope, budget and success criteria. Structured requests follow
[references/request.schema.json](references/request.schema.json). Read
[references/capture-policy.md](references/capture-policy.md) before interpreting
`module` or `leaf`; level names alone do not specify atomic boundaries.

Honor `scope.stage`: `capture-only` ends with numerical/coverage validation and
any requested reference comparison, without submitting GPU measurement jobs.
`energy-validation` additionally measures missing compatible components and
fresh whole-model energy. Execute only the listed levels and workloads.

For `capture.adapter_mode: independent`, the following isolation rule overrides
the ordinary reuse-first inspection guidance below. Delegate decomposition to
a fresh-context agent, supplying only the request, capture-policy contract,
model source/checkpoint, and neutral input/output interfaces. Do not give it
existing ecalc capture adapters, capture tests, reference counts, keys, traces,
or results, or ask it to import them indirectly. It may read the model/library
implementation; independent means independent of the reference adapter.

Save the agent's implementation, ordered composition, replay checks, boundary
rationale, input manifest and accessed-source list. Freeze those artifacts with
SHA-256 hashes before a separate evaluator reads the reference. Compare both
module and leaf levels only if requested. Preserve the first comparison even
when different: a correct alternative decomposition is not an exact reference
match. Any later reference-informed change is a separate, non-blind revision.
Do not label a previously exposed agent as independent. If fresh-context
isolation is unavailable, report that limitation instead of claiming blindness.

`adapter_mode: reuse` permits inspecting and reusing an existing validated
adapter. Agent orchestration of existing capture is not independent derivation.
After an independent artifact is frozen, a measurement agent may inspect cache
and measurement tools and bridge compatible records into ecalc without changing
the frozen boundaries. Report boundary equivalence and key equivalence separately.
Mark unmet criteria as failed and criteria unsupported by available reference
evidence as inconclusive. A successful count/boundary comparison does not turn
missing identity fields or unverified dependencies into an overall pass.

## Establish the workload and reuse opportunity

Infer the model/checkpoint, representative inputs, inference boundaries,
precision, batch/shape, GPU, and requested accuracy from the user's request.
Use documented model defaults when sufficient and record those assumptions.
Ask only for missing information that prevents a meaningful run. A model name
alone does not define sequence lengths, masks, preprocessing, or input data.

Read existing adapters and reference measurements before adding code or
collecting data. For a validation request with an existing compatible database,
prefer importing/verifying that data and measuring fresh end-to-end energy;
do not automatically repeat the entire original study. Keep imported reference
measurements separate from newly measured rows and preserve provenance.

Inspect these code entrypoints as relevant:

- `ecalc/models.py`: loading, input creation, precision and software metadata.
- `ecalc/capture.py`: executable `LayerRecord`, capture and composition rows.
- `ecalc/signature.py`: identity, supported inputs and synthetic input creation.
- `ecalc/measure.py`, `ecalc/static_power.py`, `ecalc/gpu.py`: measurement tools.
- `ecalc/db.py`, `ecalc/estimate.py`: storage and composition estimation.
- `scripts/_common.py`, `scripts/build_kernel_db.py`,
  `scripts/measure_model_e2e.py`: current GPU orchestration and assumptions.

The existing YOLO code uses Ultralytics routing; it is not a universal adapter.
The default Session and composition identity assume square RGB images. Do not
shoehorn tokens, masks or multi-input workloads into `(batch, imgsz)` identity.

## Inspect, choose and implement the decomposition

Inspect the requested model's actual forward implementation, not only its
module tree or architecture description. Identify executed blocks, shared
module invocations, functional operations, branches, skip connections,
mutations, caching, and input/output structure.

Choose boundaries that preserve executable computation. Prefer meaningful
composite blocks initially when they contain residual paths or substantial
internal interactions. Finer operations can improve reuse but can distort
energy through cache effects and launch overhead. Explain the choice in the
run manifest; do not equate numerical replay with energy accuracy.

Use the technique appropriate to the model: source-informed routing, hooks
with explicit functional-operation capture, FX/export where faithful, or a
small custom adapter. No one technique is mandatory. If automatic tracing
fails, inspect why and implement an alternative rather than declaring the
architecture unsupported solely because tracing failed.

Generate the smallest reusable executable adapter required by the model.
Prefer extending an architecture-family adapter over a separate file for
every model variant. Model-specific code is acceptable when semantics require
it; a whitelist of model names is not an adequate decomposition mechanism.
Keep Python responsible for repeatable execution and measurement, while the
agent performs the source analysis and adapter development.

Retain every executed component occurrence even when identical identities
share one measurement. Include functional glue or keep it inside a component;
do not double-count it. Capture pre-mutation inputs when necessary. Prepare
input cloning/reset outside timed work, and prove repeated replay is faithful.
If input restoration cannot be excluded from timing, redesign the boundary
or explicitly stop; do not hide restoration energy in the component result.

Use `LayerRecord` and existing database/measurement tools where they fit.
Extend their contracts coherently for nested args/kwargs, non-floating tensors,
layout/stride, and workload identity when needed. Component identities must
distinguish implementation and computation-affecting settings, not just class
names and parameter shapes. Version incompatible signatures. Do not silently
reuse measurements with different device, clocks, precision, software,
execution policy or protocol merely because legacy lookup keys match.

## Validate and repair before measuring

Check original forward versus reconstructed execution, including nested
outputs, and isolated versus captured component outputs. Exercise repeated
invocations, functional operations, in-place behavior, changed shapes and
attributes, stable identical identities, and explicit failure paths as relevant.
Use representative inputs; a single successful trace only covers that path.
Demonstrate coverage from the inspected forward and captured execution,
not merely from a successful hooks run or matching final tensor shape.

Run relevant CPU tests first and retain YOLO regression checks after shared-code
changes. Resolve validation failures by inspecting evidence and repairing the
adapter, without routine permission questions for reversible local edits.
When delegation is available, an independent review agent can inspect coverage,
identities and replay tests while the implementation agent prepares the run;
GPU measurements themselves must remain serialized on each allocated GPU.

Stop with a specific explanation if model source/checkpoint or valid inputs
are inaccessible, or semantics cannot be captured faithfully. Do not fabricate
missing energy or bypass correctness checks to finish a report.

## Measure, resume and assess

Record a run manifest containing model/checkpoint provenance, workload,
decomposition and adapter revision, environment, validation evidence, component
occurrences, unique identities, compatible hits, missing measurements, protocol,
and runtime estimate. Respect the user's component/time/resource budget.
Start with the smallest experiment that answers the request; do not expand
one-model validation into all variants, batches and leaf comparisons.

Use actual NVML/Zeus energy measurement on an allocated GPU, not FLOPs or
parameter-count estimates. Check GPU mapping and persistence-mode requirements.
Measure missing unique components and freshly measure the original complete
forward under the same conditions. Measure static power if needed by the chosen
estimator/report; E_sum itself is the sum of measured component energies.

Generate an appropriate resumable runner/Slurm script for the chosen adapter,
using the user's available account and partition. Inspect existing jobs and
locks first; do not overlap measurement processes on the same GPU. Preserve
committed rows across failure/retry. Track job IDs, logs and outputs, diagnose
failures, and resume within the agreed scope. A queued or running job is not
a completed experiment.

Report E_sum, measured whole-model energy, relative error, coverage, reuse,
protocol, variability and relevant environment differences. Mark abbreviated
protocols as smoke tests. Distinguish execution success from meeting the user's
accuracy criterion. An inaccurate decomposition is a result to investigate:
check missing/double-counted operations, mismatched policy, unstable measurements,
and boundary effects. Refine boundaries when evidence supports it, preserving
previous results; do not tune against the reference merely to claim agreement.

Leave executable adapter/runner code, tests, the manifest, measurement data,
and the report so future invocations can reproduce and resume the work.
This skill is executed by the agent; it does not install a background agent
service or promise unattended repair of arbitrary unknown model semantics.
