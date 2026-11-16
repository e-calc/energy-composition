"""Energy calculus: kernel signatures ``E = s*T + d`` and their composition.

This package needs only the Python standard library.

    from ecalc.calculus import Kernel, Sequential, Parallel, evaluate, load

    database = load("data/signature_database.jsonl")
    pipeline = database.pipeline("yolo26n/batch_size=1/input_size=640/module/forward")
    platform = database.platforms()[0]
    estimate = evaluate(pipeline.expression, database, platform)
    joules = estimate.energy(database.static_power(platform, "active_idle"))
"""

from .cost import ZERO, Cost, parallel, sequential
from .errors import (EnergyCalculusError, MissingKernel, NegativeDynamicEnergy, PlatformMismatch,
                     SchemaVersionError)
from .expression import (Estimate, Expression, Kernel, LeafResult, Parallel, Pipeline, ProvenanceTotal,
                         Sequential, SignatureSource, evaluate, flatten, kernels)
from .kernel import (BACKWARD, DIRECTIONS, FORWARD, SIGNATURE_VERSION, KernelSpecification,
                     TensorSpecification, canonical_json, kernel_key, short_description)
from .serialization import dump, load
from .signature import (EXTRAPOLATED, FORMULA, INTERPOLATED, MEASURED, PROVENANCES, Platform, Signature,
                        StaticPower, static_power_watts)
from .source import ModelRun, SignatureDatabase, SourceChain, pipeline_name

__all__ = [
    "BACKWARD", "DIRECTIONS", "EXTRAPOLATED", "FORMULA", "FORWARD", "INTERPOLATED", "MEASURED", "PROVENANCES",
    "SIGNATURE_VERSION", "ZERO",
    "Cost", "EnergyCalculusError", "Estimate", "Expression", "Kernel", "KernelSpecification", "LeafResult",
    "MissingKernel", "ModelRun", "NegativeDynamicEnergy", "Parallel", "Pipeline", "Platform", "PlatformMismatch",
    "ProvenanceTotal", "SchemaVersionError", "Sequential", "Signature", "SignatureDatabase", "SignatureSource",
    "SourceChain", "StaticPower", "TensorSpecification",
    "canonical_json", "dump", "evaluate", "flatten", "kernel_key", "kernels", "load", "parallel",
    "pipeline_name", "sequential", "short_description", "static_power_watts",
]
