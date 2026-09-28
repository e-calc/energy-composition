# Reusable agent request

Copy this prompt and edit its request. The agent executes the workflow through
the requested stage; it does not merely propose an implementation.

```text
Use $ecalc-model to execute the following request using agents.
Have the decomposition agent inspect the model implementation, implement the
requested capture, and validate coverage and numerical replay. Honor the
adapter mode, capture policy, scope and budget. For independent mode, freeze
the candidate before a separate evaluator opens reference capture or results.
Report capture correctness, reference agreement and energy accuracy separately.
Do not expand model variants, levels, batches or measurement scope silently.

request:
  version: 1
  model:
    name: yolo26n
    source: ultralytics
    checkpoint: official
    variant: nms-free-end2end
  capture:
    levels: [module, leaf]
    policy: ecalc-yolo-v1
    adapter_mode: independent
    reference_compare: after-freeze
  scope:
    stage: capture-only
    workloads:
      - shape: [1, 3, 640, 640]
        dtype: float32
        input: seeded-random
        distribution: standard-normal
        rng: torch-cpu-generator
        seed: 0
    execution: fused-eval-inference
    tf32: true
    preprocessing: excluded
    postprocessing: excluded
  reference:
    adapter: ecalc.capture.capture
    revision: 3740d543c440f38c15246e225aac44129e26c964
  target:
    gpu: A40
    account: bhvy-delta-gpu
    partition: gpuA40x4
  measurement:
    cache: compatible-only
    repeats: 3
    window_seconds: 5
    cooldown_seconds: 3
    warmup_calls: 50
    calibration_calls: 1000
  budget:
    max_new_components: 0
    gpu_minutes: 0
  acceptance:
    complete_coverage: true
    require_reference_equivalence: true
    replay_rtol: 0.00001
    replay_atol: 0.000001
    max_abs_energy_error_pct: 5
  output: results/agent_requests/yolo26n-independent-capture
```

Validate the value under `request` against `request.schema.json` before execution.
YAML is a convenience; the same mapping can be JSON.

## Scope choices

- `capture-only`: source inspection, adapter generation/reuse, CPU replay,
  coverage checks, frozen artifacts, and optional reference comparison. No
  energy measurement, fabricated energy rows or GPU job submission.
- `energy-validation`: the above plus target-device validation, compatible
  cache lookup, missing-component measurement and fresh whole-model energy.
  Allocate GPU time and permit new components if needed. Stop with a costed partial plan if the
  requested run does not fit; do not silently reduce repeats or coverage.

For a YOLO energy test, change `stage` to `energy-validation` and give an
explicit budget, for example 256 new components and 120 GPU-minutes. That is
a ceiling, not a required allocation; estimate from the captured workload.
`max_new_components` counts new unique energy-measurement identities, deduplicated
across the requested levels/workloads under the compatibility policy. It does
not limit the number of captured occurrences; zero allows capture but forbids
new component energy measurements.
Runtime estimate must include repeats, cooldowns, warmup/calibration and
end-to-end measurement. GPU-minutes are allocated wall minutes multiplied by
GPU count. Queue wait does not consume the GPU budget.

For an initial new-model test, request `[module]`, use
`policy: agent-declared-v1`, `reference_compare: none`, `require_reference_equivalence: false`, and the appropriate
model source, checkpoint and workload (e.g. ResNet18, torchvision, official,
shape `[1, 3, 224, 224]`, execution `eval-inference`). Module/leaf lists are
explicit: `[module, leaf]` means test both and report them separately.
`adapter_mode: reuse` allows existing adapters but cannot answer whether an
agent independently recovers the decomposition.

For non-image or multiple-input workloads, supply `factory` instead of
`shape`; it names an importable function returning representative args/kwargs.
Record token lengths, masks, dtype and distribution in its manifest. Do not
assume random floating-point tensors are valid for arbitrary workloads.

`complete_coverage` is always required. `require_reference_equivalence` requires
matching normalized boundaries, order, arguments and routing after freezing;
literal key equality is reported separately. Replay tolerances are workload-specific.
If the reference lacks evidence needed to verify a requested criterion, report
that criterion as inconclusive and overall reference equivalence as partial;
do not silently relax the criterion. Missing computation-affecting identity
fields must be resolved in a labeled revision before claiming cache compatibility.
The energy threshold applies only to energy-validation; exceeding it is a
reported scientific outcome, not permission to tune the measurement until it
passes. `compatible-only` may still yield zero hits. `remeasure` deliberately
collects all requested unique components in an isolated run.

## Artifacts

Save the resolved request, model/checkpoint/software provenance, agent roles
and accessible-source lists, candidate adapter, ordered composition per level,
validation tests/results, boundary rationale and frozen SHA-256 manifest.
Reference evaluation records its own sources and phase separately. Preserve
the first blind result, all mismatches and any subsequent labeled revisions.

For energy-validation also save cache audit, measured/missing counts, budget
estimate, Slurm script/job IDs, resumable database, logs, per-level E_sum versus
whole-model measurement, repeat variability and interpretation. Do not count
shape/count agreement as complete decomposition equivalence.

## Resolve exact execution and references

For `seeded-random`, `distribution: standard-normal` and
`rng: torch-cpu-generator` mean constructing a new CPU torch.Generator seeded
with the declared seed for each workload, then calling torch.randn with the
full shape and declared floating dtype on CPU, then transferring to the target
device without another random draw or cast. Use a workload factory for integer
or mixed inputs. Record the torch version and resolved inputs.

The YOLO `variant` must be `nms-free-end2end` or `one-to-many`. The former keeps
Detect's internal one-to-one head and internal selection/postprocessing in the
forward. `postprocessing: excluded` excludes external NMS/formatting, not code
inside Detect.forward. The example selects the NMS-free end-to-end variant.

`reference` identifies the evaluator's adapter and its repository revision.
Resolve and record the exact revision/files; do not silently compare to a
changed working tree. Omit it when no comparison is requested. Remove the
reference locator from the decomposition agent's task packet, and expose it
only to the evaluator after the candidate is frozen.

For `agent-declared-v1`, including preprocessing or external postprocessing
requires `scope.pipeline_factory`, an importable factory constructing the full
executable pipeline around the model. Its source, checkpoint binding and exact
boundaries must be recorded; an input-data factory alone is insufficient.
