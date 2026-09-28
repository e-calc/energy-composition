# Capture levels and comparison contract

Module and leaf are decomposition policies, not CUDA kernel definitions. A
captured component may execute several GPU kernels. Both levels must include
all executed computation without gaps or double-counting, retain repeated
occurrences, and preserve functional operations and mutations.

## `ecalc-yolo-v1`

Applies to the fused Ultralytics YOLO detection forward, not preprocessing or
external NMS. Select the requested `nms-free-end2end` or `one-to-many` variant
explicitly; internal Detect processing remains part of the model forward. Derive the actual execution order, routes and shapes from the
requested model implementation and workload, never from expected counts.

- **module:** one component per top-level executable entry in the model's
  architecture graph. Composite blocks retain their complete internal forward.
  Graph routing itself must preserve skips, concatenation inputs and outputs.
- **leaf:** inside each graph entry, treat Ultralytics Conv/DWConv (fused
  convolution plus activation), MaxPool2d, Upsample, Concat, Attention, and
  Detect as atomic boundaries. Attention and Detect remain whole. Do not
  descend into a selected atomic boundary and also count its children.
  Capture functional `cat`, `chunk`, `split`, and residual `add` executed
  outside atomic boundaries, including their scalar arguments and routing.
  Tensor metadata access may be excluded; an unhandled compute operation
  must fail explicitly. This policy is intentionally not literal childless
  PyTorch-module capture.

This policy is public experimental input to both agents and reference code.
Supplying it does not reveal reference counts, ordered records or identities.

## `agent-declared-v1`

For other architectures, the agent declares the concrete boundaries before
comparison or energy measurement. At module level prefer architecture blocks
with internal functional work intact. At leaf level state the atomic class/op
set, handling of fused operations, and any composite exceptions. An exception
must have a rationale and remain visible in the manifest; do not silently call
a block-level decomposition leaf-level. Reuse an existing declared policy
across variants when applicable. A policy mismatch is not a failed model.

## Required evidence and comparison

Save ordered occurrences with parent graph position, component path/type,
input/output tensor shapes and dtypes, operation arguments, routing/dependencies,
and executable replay handles or reconstruction instructions. Save a structural
identity including computation-affecting settings and input layout. Validate
the original forward, composed replay, and repeated isolated replay on declared
inputs with explicit tolerances. Numerical equality alone does not prove
complete coverage: reconcile captured computation with the inspected forward.

After freezing independent artifacts, compare:

1. **Coverage/replay:** does the candidate preserve the requested computation?
2. **Boundaries/order:** do types, parent assignments, occurrences, routes and
   argument/tensor specifications match the reference policy?
3. **Identity:** are signatures equivalent, and are literal cache keys equal?
   Different signature namespaces may encode the same boundary. Do not infer
   incompatibility or a match from counts alone.
4. **Energy** (energy-validation only): compare E_sum and freshly measured
   whole-model energy, protocol, variability and environment. Record the
   accuracy threshold before measurement; do not adjust it to pass.

An exact reference match is a separate finding from correct computation and
acceptable energy error. Report each outcome independently.
