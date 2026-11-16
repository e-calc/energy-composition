"""Use the energy calculus as a library. Needs a signature database and nothing else:

    python examples/library_usage.py data/signature_database.jsonl
"""

import sys

from ecalc.calculus import Parallel, Sequential, evaluate, load, pipeline_name
from ecalc.formula import with_formulas

database = load(sys.argv[1] if len(sys.argv) > 1 else "data/signature_database.jsonl")
platform = database.platforms()[0]
static_power = database.static_power(platform, "active_idle")

# One model: a sequence of kernels. Its energy is the sum of the kernel energies.
yolo = database.pipeline(pipeline_name("yolo26n", batch_size=1, input_size=640)).expression
estimate = evaluate(yolo, database, platform)
print(f"yolo26n            {estimate.energy(static_power) * 1e3:8.1f} mJ in {estimate.time_seconds * 1e6:8.1f} us")
print(f"  static  s*T      {estimate.static_energy(static_power) * 1e3:8.1f} mJ")
print(f"  dynamic d        {estimate.dynamic_energy(static_power) * 1e3:8.1f} mJ")

# The signature of one kernel, in the form E = s*T + d.
first = estimate.leaves[0].signature
s, t, d = first.decompose(static_power)
print(f"first kernel       {first.kernel.description}: s = {s:.1f} W, T = {t * 1e6:.1f} us, d = {d * 1e3:.3f} mJ")

# Two models overlapped on the same GPU: static power is paid once, over the longer one.
vgg = database.pipeline(pipeline_name("rf_vgg", batch_size=32, input_size=1024)).expression
together = evaluate(Parallel((yolo, vgg)), database, platform)
in_sequence = evaluate(Sequential((yolo, vgg)), database, platform)
saving = in_sequence.energy(static_power) - together.energy(static_power, negative_dynamic="warn")
print(f"yolo26n + rf_vgg   {in_sequence.energy(static_power) * 1e3:8.1f} mJ back to back, "
      f"{together.energy(static_power, negative_dynamic='warn') * 1e3:8.1f} mJ overlapped (saves {saving * 1e3:.1f} mJ)")

# Formula mode: price kernels that were never measured, and say so.
filled = evaluate(yolo, with_formulas(database), platform, on_missing="report")
print("provenance         " + ", ".join(f"{name}: {total.kernel_count} kernels"
                                        for name, total in filled.by_provenance().items()))
