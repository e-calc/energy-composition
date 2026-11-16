"""Algebraic laws of the energy calculus.

Inputs come from an exactly representable grid (see ``calculus_helpers``), so the laws are
asserted with ``==`` on the production code path. A second, small layer checks realistic
values with a relative tolerance. No torch, no measured data.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest
from calculus_helpers import (INTEGER_WATTS, PLATFORM, grid_cost, grid_signature, operation_specification,
                              random_expression, synthetic_database)

from ecalc.calculus import (ZERO, Cost, Kernel, MissingKernel, NegativeDynamicEnergy, Parallel, Platform,
                            PlatformMismatch, Sequential, Signature, SignatureDatabase, StaticPower, evaluate,
                            flatten, parallel, sequential)

CASES = 300


def _costs(seed: int, count: int, minimum_watts: int = 0) -> list[Cost]:
    rng = random.Random(seed)
    return [grid_cost(rng, minimum_watts) for _ in range(count)]


# ---- the signature form -----------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_every_cost_has_the_form_s_t_plus_d(seed):
    rng = random.Random(seed)
    cost = _random_composite(rng)
    for watts in INTEGER_WATTS:
        assert cost.energy(watts) == cost.static_energy(watts) + cost.dynamic_energy(watts)


def test_a_single_kernel_decomposes_into_s_t_d():
    signature = Signature(operation_specification("cat"), PLATFORM, 0.5, 100.0)
    s, t, d = signature.decompose(90)
    assert (s, t, d) == (90.0, 0.5, 55.0)
    assert s * t + d == signature.energy_joules
    assert signature.cost().energy(90) == signature.energy_joules


def _random_composite(rng: random.Random, depth: int = 3) -> Cost:
    if depth == 0 or rng.random() < 0.3:
        return grid_cost(rng)
    operator = rng.choice((sequential, parallel))
    return operator(*[_random_composite(rng, depth - 1) for _ in range(rng.randint(1, 4))])


# ---- monoid laws of both operators ------------------------------------------------------------

@pytest.mark.parametrize("operator", [sequential, parallel])
def test_identity(operator):
    for cost in _costs(1, CASES):
        assert operator(cost, ZERO) == cost
        assert operator(ZERO, cost) == cost
        assert operator(cost) == cost
    assert operator() == ZERO


@pytest.mark.parametrize("operator", [sequential, parallel])
def test_commutative_and_order_independent(operator):
    rng = random.Random(2)
    for _ in range(CASES):
        costs = [grid_cost(rng) for _ in range(rng.randint(2, 6))]
        reference = operator(*costs)
        shuffled = costs[:]
        rng.shuffle(shuffled)
        assert operator(*shuffled) == reference
    a, b, c = _costs(3, 3)
    for permutation in itertools.permutations((a, b, c)):
        assert operator(*permutation) == operator(a, b, c)


@pytest.mark.parametrize("operator", [sequential, parallel])
def test_associative_nested_equals_flat(operator):
    rng = random.Random(4)
    for _ in range(CASES):
        a, b, c, d = (grid_cost(rng) for _ in range(4))
        flat = operator(a, b, c, d)
        assert operator(operator(a, b), operator(c, d)) == flat
        assert operator(a, operator(b, operator(c, d))) == flat
        assert operator(operator(operator(a, b), c), d) == flat


def test_a_kernel_used_n_times_costs_n_times_its_signature():
    rng = random.Random(5)
    for _ in range(50):
        cost = grid_cost(rng)
        count = rng.randint(1, 64)
        total = sequential(*[cost] * count)
        assert total.time_seconds == count * cost.time_seconds
        assert total.energy_joules == count * cost.energy_joules
        for watts in INTEGER_WATTS:
            assert total.energy(watts) == count * cost.energy(watts)


# ---- sequential composition -------------------------------------------------------------------

def test_sequential_time_and_dynamic_energy_add():
    rng = random.Random(6)
    for _ in range(CASES):
        costs = [grid_cost(rng) for _ in range(rng.randint(1, 8))]
        total = sequential(*costs)
        assert total.time_seconds == math.fsum(c.time_seconds for c in costs)
        for watts in INTEGER_WATTS:
            assert total.dynamic_energy(watts) == math.fsum(c.dynamic_energy(watts) for c in costs)


def test_sequential_energy_is_the_sum_of_measured_energies_for_any_static_power():
    rng = random.Random(7)
    for _ in range(CASES):
        costs = [grid_cost(rng) for _ in range(rng.randint(1, 8))]
        total = sequential(*costs)
        measured = math.fsum(c.energy_joules for c in costs)
        assert not total.has_overlap
        for watts in INTEGER_WATTS:
            assert total.energy(watts) == measured
        assert total.energy(89.69807891254976) == measured  # not on the grid, still exact: s multiplies zero


def test_sequential_energy_adds_even_when_parts_contain_overlap():
    rng = random.Random(8)
    for _ in range(CASES):
        a = parallel(grid_cost(rng), grid_cost(rng))
        b = sequential(grid_cost(rng), parallel(grid_cost(rng), grid_cost(rng), grid_cost(rng)))
        for watts in INTEGER_WATTS:
            assert sequential(a, b).energy(watts) == a.energy(watts) + b.energy(watts)


# ---- parallel composition (overlap) -----------------------------------------------------------

def test_parallel_time_is_the_longest_and_dynamic_energy_adds():
    rng = random.Random(9)
    for _ in range(CASES):
        costs = [grid_cost(rng) for _ in range(rng.randint(1, 8))]
        total = parallel(*costs)
        assert total.time_seconds == max(c.time_seconds for c in costs)
        for watts in INTEGER_WATTS:
            assert total.dynamic_energy(watts) == math.fsum(c.dynamic_energy(watts) for c in costs)


def test_static_power_is_paid_once_over_the_longest_duration():
    rng = random.Random(10)
    for _ in range(CASES):
        costs = [grid_cost(rng) for _ in range(rng.randint(2, 8))]
        total = parallel(*costs)
        longest = max(c.time_seconds for c in costs)
        for watts in INTEGER_WATTS:
            dynamic = math.fsum(c.dynamic_energy(watts) for c in costs)
            assert total.energy(watts) == watts * longest + dynamic


def test_overlap_saving_against_sequential():
    rng = random.Random(11)
    for _ in range(CASES):
        costs = [grid_cost(rng) for _ in range(rng.randint(2, 8))]
        times = [c.time_seconds for c in costs]
        for watts in INTEGER_WATTS:
            saving = sequential(*costs).energy(watts) - parallel(*costs).energy(watts)
            assert saving == watts * (math.fsum(times) - max(times))
            assert saving == parallel(*costs).overlap_saving(watts)
            assert saving >= 0


def test_two_kernel_overlap_saves_static_power_times_the_shorter_duration():
    rng = random.Random(12)
    for _ in range(CASES):
        a, b = grid_cost(rng), grid_cost(rng)
        for watts in INTEGER_WATTS:
            saving = sequential(a, b).energy(watts) - parallel(a, b).energy(watts)
            assert saving == watts * min(a.time_seconds, b.time_seconds)


def test_parallel_energy_bounds_when_dynamic_energy_is_not_negative():
    rng = random.Random(13)
    for _ in range(CASES):
        watts = 90
        costs = [grid_cost(rng, minimum_watts=watts) for _ in range(rng.randint(2, 8))]
        energy = parallel(*costs).energy(watts)
        assert max(c.energy_joules for c in costs) <= energy <= math.fsum(c.energy_joules for c in costs)


def test_with_zero_static_power_overlap_changes_nothing():
    rng = random.Random(14)
    for _ in range(CASES):
        costs = [grid_cost(rng) for _ in range(rng.randint(1, 8))]
        assert parallel(*costs).energy(0) == sequential(*costs).energy(0)


def test_energy_is_affine_in_static_power_with_slope_time_minus_busy():
    rng = random.Random(15)
    for _ in range(CASES):
        cost = _random_composite(rng)
        slope = cost.time_seconds - cost.busy_seconds
        assert slope <= 0
        for watts in INTEGER_WATTS:
            assert cost.energy(watts) - cost.energy(0) == watts * slope


def test_overlapping_whole_pipelines_saves_at_least_as_much_as_overlapping_stage_by_stage():
    rng = random.Random(16)
    for _ in range(CASES):
        a, b, c, d = (grid_cost(rng) for _ in range(4))
        whole = parallel(sequential(a, b), sequential(c, d))
        staged = sequential(parallel(a, c), parallel(b, d))
        for watts in INTEGER_WATTS:
            assert whole.energy(watts) <= staged.energy(watts)
        assert whole.time_seconds <= staged.time_seconds


# ---- things that are deliberately NOT laws ----------------------------------------------------

def test_parallel_is_not_idempotent():
    a = Cost(0.5, 100.0, 0.5)
    assert parallel(a, a) != a
    assert parallel(a, a).energy(90) == 2 * 100.0 - 90 * 0.5  # both run; static is shared once


def test_sequential_does_not_distribute_over_parallel():
    a, b, c = Cost(0.25, 50.0, 0.25), Cost(0.5, 80.0, 0.5), Cost(1.0, 120.0, 1.0)
    left = sequential(a, parallel(b, c))
    right = parallel(sequential(a, b), sequential(a, c))
    assert left.time_seconds == right.time_seconds  # same wall time ...
    assert left.energy(90) != right.energy(90)      # ... but on the right `a` runs twice


def test_lower_bound_fails_when_a_kernel_draws_less_than_the_static_power():
    cold = Cost(1.0, 82.0, 1.0)   # 82 W average, below the 90 W static power: d = -8 J
    hot = Cost(1.0, 300.0, 1.0)
    assert parallel(cold, hot).energy(90) < max(cold.energy_joules, hot.energy_joules)


# ---- the same numbers as the reference rule ---------------------------------------------------

def _reference_segment(times, energies, static_watts):
    """The overlap rule written out directly: T = max T_i, E = s*T + sum(E_i - s*T_i)."""
    return max(times), static_watts * max(times) + math.fsum(e - static_watts * t for e, t in zip(energies, times))


@pytest.mark.parametrize("times, energies, static_watts", [
    ((2.0e-3, 3.0e-3), (0.6, 0.9), 179.4),           # two kernels, static power of two GPUs summed
    ((1.0e-3, 4.0e-3, 2.5e-3), (0.3, 1.1, 0.7), 89.7),
    ((5.0e-4, 5.0e-4), (0.2, 0.25), 89.69807891254976),
])
def test_matches_the_reference_overlap_rule(times, energies, static_watts):
    costs = [Cost(t, e, t) for t, e in zip(times, energies)]
    total = parallel(*costs)
    reference_time, reference_energy = _reference_segment(times, energies, static_watts)
    assert total.time_seconds == reference_time
    assert total.energy(static_watts) == pytest.approx(reference_energy, rel=1e-12)
    saving = sequential(*costs).energy(static_watts) - total.energy(static_watts)
    assert saving == pytest.approx(static_watts * (sum(times) - max(times)), rel=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_laws_hold_to_rounding_on_realistic_values(seed):
    rng = random.Random(seed)
    costs = []
    for _ in range(rng.randint(2, 30)):
        time_seconds = rng.uniform(5e-6, 5e-3)
        costs.append(Cost(time_seconds, time_seconds * rng.uniform(95.0, 300.0), time_seconds))
    watts = 89.69807891254976
    assert sequential(*costs).energy(watts) == pytest.approx(sum(c.energy_joules for c in costs), rel=1e-12)
    half = len(costs) // 2
    nested = sequential(sequential(*costs[:half]), sequential(*costs[half:]))
    assert nested.energy(watts) == pytest.approx(sequential(*costs).energy(watts), rel=1e-12)
    times = [c.time_seconds for c in costs]
    saving = sequential(*costs).energy(watts) - parallel(*costs).energy(watts)
    assert saving == pytest.approx(watts * (sum(times) - max(times)), rel=1e-12)


# ---- expressions evaluate to the same costs ---------------------------------------------------

def _specifications(count: int = 12):
    return [operation_specification("cat", shape=(1, 8, 16, 16 + i)) for i in range(count)]


def test_evaluate_commutes_with_the_operators():
    specifications = _specifications()
    database = synthetic_database(specifications)
    rng = random.Random(17)

    def fold(expression):
        if isinstance(expression, Kernel):
            return database.lookup(expression.specification, PLATFORM).cost()
        operator = sequential if isinstance(expression, Sequential) else parallel
        return operator(*[fold(part) for part in expression.parts])

    for _ in range(100):
        expression = random_expression(rng, specifications)
        assert evaluate(expression, database, PLATFORM).cost == fold(expression)


def test_flatten_preserves_the_cost():
    specifications = _specifications()
    database = synthetic_database(specifications)
    rng = random.Random(18)
    for _ in range(100):
        expression = random_expression(rng, specifications, depth=4)
        assert evaluate(flatten(expression), database, PLATFORM).cost == evaluate(expression, database, PLATFORM).cost


def test_missing_kernels_raise_or_are_reported():
    present, absent = _specifications(2)
    database = synthetic_database([present])
    expression = Sequential((Kernel(present), Kernel(absent)))
    with pytest.raises(MissingKernel):
        evaluate(expression, database, PLATFORM)
    estimate = evaluate(expression, database, PLATFORM, on_missing="report")
    assert estimate.missing == (absent,) and not estimate.is_complete
    assert estimate.cost == database.lookup(present, PLATFORM).cost()
    assert estimate.by_provenance()["missing"].kernel_count == 1


def test_signatures_of_another_platform_are_not_used():
    specification = operation_specification("add")
    database = SignatureDatabase()
    database.add_signature(Signature(specification, Platform("A100"), 0.5, 100.0))
    with pytest.raises(MissingKernel):
        evaluate(Kernel(specification), database, PLATFORM)
    estimate = evaluate(Kernel(specification), database, Platform("A100"))
    with pytest.raises(PlatformMismatch):
        estimate.energy(StaticPower(PLATFORM, 90.0))


def test_negative_dynamic_energy_matters_only_under_overlap():
    cold = operation_specification("chunk")  # like the measured op:chunk, 82 W against 90 W static
    hot = operation_specification("cat")
    database = SignatureDatabase()
    database.add_signature(Signature(cold, PLATFORM, 1.0, 82.0))
    database.add_signature(Signature(hot, PLATFORM, 1.0, 300.0))
    in_sequence = evaluate(Sequential((Kernel(cold), Kernel(hot))), database, PLATFORM)
    assert in_sequence.energy(90) == 382.0  # never raises: the result does not depend on s
    overlapped = evaluate(Parallel((Kernel(cold), Kernel(hot))), database, PLATFORM)
    with pytest.raises(NegativeDynamicEnergy):
        overlapped.energy(90)
    with pytest.warns(UserWarning):
        assert overlapped.energy(90, negative_dynamic="warn") == 382.0 - 90.0
    assert overlapped.energy(90, negative_dynamic="allow") == 292.0
    assert overlapped.energy(80) == 302.0  # a static power below the kernel's 82 W is fine


def test_costs_reject_invalid_numbers():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Cost(bad, 1.0, 1.0)
        with pytest.raises(ValueError):
            Cost(1.0, 1.0, 1.0).energy(bad)
    assert Cost(0.0, 0.0, 0.0) == ZERO  # a zero-cost kernel (a view, a flatten) is valid


def test_busy_time_defaults_to_the_duration_of_a_single_kernel():
    assert Cost(0.5, 100.0) == Cost(0.5, 100.0, 0.5)
    assert not Cost(0.5, 100.0).has_overlap
    overlapped = parallel(Cost(0.5, 100.0), Cost(0.25, 40.0))
    assert (overlapped.time_seconds, overlapped.busy_seconds) == (0.5, 0.75)
    assert sequential(Cost(0.5, 100.0), Cost(0.25, 40.0)).busy_seconds == 0.75


def test_grid_signature_is_on_the_grid():
    signature = grid_signature(operation_specification("cat"), random.Random(0), minimum_watts=90)
    assert (signature.time_seconds * 2 ** 30).is_integer() and (signature.energy_joules * 2 ** 30).is_integer()
    assert signature.dynamic_energy(90) >= 0
