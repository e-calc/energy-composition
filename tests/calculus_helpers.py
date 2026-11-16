"""Shared helpers for the calculus tests: synthetic kernels and exactly representable costs.

Law tests assert with ``==``. That is sound only when floating-point arithmetic is exact, so
every time and energy here is a multiple of ``2**-30`` and every static power is an integer:
sums, products and differences of such numbers are exact in double precision.
"""

from __future__ import annotations

import random

from ecalc.calculus import (Cost, Expression, Kernel, KernelSpecification, Parallel, Platform, Sequential,
                            Signature, SignatureDatabase)

PLATFORM = Platform("A40")
GRID = 2.0 ** -30
INTEGER_WATTS = (0, 1, 23, 90, 100, 317)


def operation_specification(name: str, shape=(1, 8, 16, 16), **keyword_arguments) -> KernelSpecification:
    """A functional-operation kernel with a single input of ``shape``."""
    return KernelSpecification.from_signature({
        "sig_version": 1,
        "op": {"name": name, "args": [], "kwargs": dict(sorted(keyword_arguments.items()))},
        "input": {"kind": "list", "items": [{"shape": list(shape), "dtype": "float32"}]},
    })


def grid_cost(rng: random.Random, minimum_watts: int = 0) -> Cost:
    """A single-kernel cost on the exact grid, with average power of at least ``minimum_watts``."""
    time_units = rng.randint(1, 2 ** 20)
    watts = rng.randint(minimum_watts, 400)
    extra_units = rng.randint(0, 2 ** 20)
    time_seconds = time_units * GRID
    return Cost(time_seconds, watts * time_seconds + extra_units * GRID, time_seconds)


def grid_signature(specification: KernelSpecification, rng: random.Random, minimum_watts: int = 0) -> Signature:
    cost = grid_cost(rng, minimum_watts)
    return Signature(specification, PLATFORM, cost.time_seconds, cost.energy_joules)


def synthetic_signature(specification: KernelSpecification, minimum_watts: int = 90) -> Signature:
    """A deterministic signature derived from the kernel's key, on the exact grid."""
    return grid_signature(specification, random.Random(specification.key), minimum_watts)


def synthetic_database(specifications, minimum_watts: int = 90) -> SignatureDatabase:
    database = SignatureDatabase()
    for specification in specifications:
        database.add_signature(synthetic_signature(specification, minimum_watts))
    return database


def random_expression(rng: random.Random, specifications, depth: int = 3) -> Expression:
    """A random tree of Sequential and Parallel nodes over ``specifications``."""
    if depth == 0 or rng.random() < 0.3:
        return Kernel(rng.choice(specifications))
    node_type = rng.choice((Sequential, Parallel))
    return node_type(tuple(random_expression(rng, specifications, depth - 1) for _ in range(rng.randint(1, 4))))
