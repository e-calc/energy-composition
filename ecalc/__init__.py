"""ecalc: energy calculus for GPU workloads.

Every kernel has an energy signature

    E = s*T + d        static power x duration + dynamic energy

and signatures compose: in sequence, durations and energies add; in overlap, static power is
paid once over the longest duration and dynamic energies add. ``ecalc.calculus`` implements the
signature schema, its serialization and the two composition operators with the standard library
only. The rest of the package captures a model's kernels (``ecalc.capture``), measures them on a
GPU with Zeus (``ecalc.measure``), stores them (``ecalc.db``) and estimates unmeasured kernels on
request (``ecalc.formula``). The ``ecalc`` command ties them together.
"""

__version__ = "0.2.0"
