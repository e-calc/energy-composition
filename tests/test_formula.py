"""Formula mode: interpolation inside kernel families and the light-op line fit. Synthetic data,
no torch."""

from __future__ import annotations

import pytest
from calculus_helpers import PLATFORM, operation_specification

from ecalc.calculus import (EXTRAPOLATED, FORMULA, INTERPOLATED, MEASURED, Kernel, KernelSpecification, MissingKernel,
                            Sequential, Signature, SignatureDatabase, evaluate, kernels)
from ecalc.formula import (Interpolation, LightOpsFormula, family_coordinates, formula_sources, leave_one_out,
                           light_family, with_formulas)


def convolution(batch_size: int, side: int, channels: int = 16) -> KernelSpecification:
    return KernelSpecification.from_signature({
        "sig_version": 1, "input": {"kind": "tensor", "shape": [batch_size, channels, side, side], "dtype": "float32"},
        "module": {"cls": "Conv", "buffers": [],
                   "modules": [{"n": "conv", "cls": "Conv2d", "a": {"in_channels": channels, "out_channels": 32}}],
                   "params": [{"n": "conv.weight", "s": [32, channels, 3, 3], "dt": "float32"}]}})


def recurrent(batch_size: int, length: int) -> KernelSpecification:
    return KernelSpecification.from_signature({
        "sig_version": 1, "input": {"kind": "tensor", "shape": [batch_size, length, 2], "dtype": "float32"},
        "module": {"cls": "LSTMLayer", "buffers": [],
                   "modules": [{"n": "lstm", "cls": "LSTM", "a": {"input_size": 2, "hidden_size": 8, "batch_first": True}}],
                   "params": [{"n": "lstm.weight_ih_l0", "s": [32, 2], "dt": "float32"}]}})


def affine(batch_size: int, side: int) -> tuple[float, float]:
    """An exactly affine cost model: 10 us overhead + 1 ns per input element, at 200 W."""
    time_seconds = 10e-6 + 1e-9 * batch_size * side * side
    return time_seconds, 200.0 * time_seconds


def database_of(points) -> SignatureDatabase:
    database = SignatureDatabase()
    for specification, (time_seconds, energy_joules) in points:
        database.add_signature(Signature(specification, PLATFORM, time_seconds, energy_joules))
    return database


# ---- families ---------------------------------------------------------------------------------

def test_family_ignores_batch_size_and_extent_only():
    base = family_coordinates(convolution(1, 64))
    assert base.batch_size == 1 and base.extent == 64 * 64
    assert family_coordinates(convolution(8, 64)).family == base.family
    assert family_coordinates(convolution(1, 32)).family == base.family
    assert family_coordinates(convolution(1, 64, channels=32)).family != base.family
    assert family_coordinates(convolution(1, 64).with_direction("backward")).family != base.family


def test_recurrent_inputs_keep_features_last():
    coordinates = family_coordinates(recurrent(4, 1024))
    assert (coordinates.batch_size, coordinates.extent) == (4, 1024)
    assert family_coordinates(recurrent(4, 512)).family == coordinates.family


def test_parameter_free_kernels_are_not_interpolated_but_are_light():
    cat = operation_specification("cat", shape=(1, 8, 16, 16), dim=1)
    assert family_coordinates(cat) is None and light_family(cat) is not None
    assert light_family(convolution(1, 64)) is None
    assert light_family(KernelSpecification.from_key("0123456789abcdef")) is None


# ---- interpolation ----------------------------------------------------------------------------

def test_chord_between_two_batch_sizes_is_exact_on_affine_costs():
    database = database_of([(convolution(b, 64), affine(b, 64)) for b in (1, 8)])
    source = Interpolation(database)
    estimate = source.lookup(convolution(4, 64), PLATFORM)
    assert estimate.provenance == INTERPOLATED and "between 1 and 8" in estimate.detail
    assert estimate.time_seconds == pytest.approx(affine(4, 64)[0], rel=1e-12)
    assert estimate.energy_joules == pytest.approx(affine(4, 64)[1], rel=1e-12)


def test_beyond_the_measured_range_is_labelled_extrapolated():
    database = database_of([(convolution(b, 64), affine(b, 64)) for b in (1, 8)])
    estimate = Interpolation(database).lookup(convolution(16, 64), PLATFORM)
    assert estimate.provenance == EXTRAPOLATED
    assert estimate.time_seconds == pytest.approx(affine(16, 64)[0], rel=1e-12)


def test_a_single_measured_point_scales_in_proportion():
    database = database_of([(convolution(2, 64), affine(2, 64))])
    estimate = Interpolation(database).lookup(convolution(4, 64), PLATFORM)
    assert estimate.provenance == EXTRAPOLATED and "proportion" in estimate.detail
    assert estimate.time_seconds == pytest.approx(2 * affine(2, 64)[0])


def test_interpolation_over_extent_and_over_both_axes():
    grid = [(convolution(b, side), affine(b, side)) for b in (1, 8) for side in (32, 128)]
    source = Interpolation(database_of(grid))
    over_extent = source.lookup(convolution(1, 64), PLATFORM)
    assert over_extent.provenance == INTERPOLATED and "extent" in over_extent.detail
    assert over_extent.time_seconds == pytest.approx(affine(1, 64)[0], rel=1e-12)
    both = source.lookup(convolution(4, 64), PLATFORM)
    assert both.provenance == INTERPOLATED
    assert both.time_seconds == pytest.approx(affine(4, 64)[0], rel=1e-12)


def test_estimates_are_never_negative():
    database = database_of([(convolution(8, 64), (1e-3, 0.2)), (convolution(16, 64), (5e-3, 1.0))])
    estimate = Interpolation(database).lookup(convolution(1, 64), PLATFORM)  # the extended chord goes below zero
    assert estimate.time_seconds == 0.0 and estimate.energy_joules == 0.0 and estimate.provenance == EXTRAPOLATED


def test_another_family_or_platform_gives_nothing():
    database = database_of([(convolution(b, 64), affine(b, 64)) for b in (1, 8)])
    source = Interpolation(database)
    assert source.lookup(convolution(4, 64, channels=32), PLATFORM) is None
    assert source.lookup(convolution(4, 64), type(PLATFORM)("A100")) is None
    assert source.lookup(operation_specification("cat"), PLATFORM) is None


# ---- light operations -------------------------------------------------------------------------

def cat(elements: int) -> KernelSpecification:
    return operation_specification("cat", shape=(1, elements), dim=1)


def test_line_fit_recovers_overhead_bandwidth_and_power():
    points = [(cat(n), (5e-6 + 4 * n / 100e9, 250.0 * (5e-6 + 4 * n / 100e9))) for n in (1000, 50_000, 400_000)]
    source = LightOpsFormula(database_of(points))
    (fit,) = source.fits(PLATFORM)
    assert fit.point_count == 3 and fit.name == "op:cat"
    assert fit.overhead_seconds == pytest.approx(5e-6, rel=1e-9)
    assert fit.bandwidth_bytes_per_second == pytest.approx(100e9, rel=1e-9)
    assert fit.power_watts == pytest.approx(250.0)
    inside = source.lookup(cat(100_000), PLATFORM)
    assert inside.provenance == FORMULA
    assert inside.time_seconds == pytest.approx(5e-6 + 400_000 / 100e9, rel=1e-9)
    assert inside.energy_joules == pytest.approx(250.0 * inside.time_seconds)
    assert source.lookup(cat(4_000_000), PLATFORM).provenance == EXTRAPOLATED


def test_a_view_like_operation_fits_a_constant():
    chunk = lambda n: operation_specification("chunk", shape=(1, n), chunks=2)  # noqa: E731
    source = LightOpsFormula(database_of([(chunk(n), (3e-6, 3e-6 * 85.0)) for n in (1000, 8000, 64_000)]))
    (fit,) = source.fits()
    assert fit.seconds_per_byte == 0.0 and fit.overhead_seconds == pytest.approx(3e-6)
    assert source.lookup(chunk(20_000), PLATFORM).time_seconds == pytest.approx(3e-6)


def test_a_family_needs_enough_measured_sizes():
    one_size = database_of([(cat(1000), (5e-6, 1e-3))])
    assert LightOpsFormula(one_size).lookup(cat(2000), PLATFORM) is None
    two_sizes = database_of([(cat(1000), (5e-6, 1e-3)), (cat(4000), (6e-6, 1.2e-3))])
    assert LightOpsFormula(two_sizes).lookup(cat(2000), PLATFORM) is not None
    assert LightOpsFormula(two_sizes, minimum_points=3).lookup(cat(2000), PLATFORM) is None


def test_different_arguments_are_different_families():
    database = database_of([(cat(n), (5e-6 + n * 1e-11, 1e-3)) for n in (1000, 4000, 16_000)])
    other_dimension = operation_specification("cat", shape=(1, 2000), dim=0)
    assert LightOpsFormula(database).lookup(other_dimension, PLATFORM) is None


# ---- formula mode as a whole ------------------------------------------------------------------

def test_measured_rows_win_and_every_estimate_is_labelled():
    measured = [(convolution(b, 64), affine(b, 64)) for b in (1, 8)]
    measured += [(cat(n), (5e-6 + 4 * n / 100e9, 250.0 * (5e-6 + 4 * n / 100e9))) for n in (1000, 400_000)]
    database = database_of(measured)
    pipeline = Sequential((Kernel(convolution(1, 64)), Kernel(convolution(4, 64)), Kernel(cat(100_000)),
                           Kernel(convolution(4, 64, channels=32))))
    with pytest.raises(MissingKernel):
        evaluate(pipeline, database, PLATFORM)  # default mode: measured only
    estimate = evaluate(pipeline, with_formulas(database), PLATFORM, on_missing="report")
    assert [leaf.signature.provenance if leaf.signature else None for leaf in estimate.leaves] == [
        MEASURED, INTERPOLATED, FORMULA, None]
    assert len(estimate.missing) == 1
    totals = estimate.by_provenance()
    assert {name: total.kernel_count for name, total in totals.items()} == {
        MEASURED: 1, INTERPOLATED: 1, FORMULA: 1, "missing": 1}
    assert len(formula_sources(database)) == 2


def test_formulas_use_only_measured_rows():
    database = database_of([(convolution(1, 64), affine(1, 64))])
    database.add_signature(Signature(convolution(8, 64), PLATFORM, 1.0, 1.0, INTERPOLATED, "not a measurement"))
    estimate = Interpolation(database).lookup(convolution(4, 64), PLATFORM)
    assert estimate.provenance == EXTRAPOLATED and estimate.time_seconds == pytest.approx(4 * affine(1, 64)[0])


def test_leave_one_out_on_affine_data_has_no_error():
    points = [(convolution(b, 64), affine(b, 64)) for b in (1, 2, 4, 8, 16)]
    points += [(cat(n), (5e-6 + 4 * n / 100e9, 250.0 * (5e-6 + 4 * n / 100e9))) for n in (1000, 8000, 64_000, 512_000)]
    results = leave_one_out(database_of(points), PLATFORM)
    assert len(results) == len(points)
    for result in results:
        assert result.estimated.provenance != MEASURED
        assert abs(result.energy_error) < 1e-9 and abs(result.time_error) < 1e-9


def test_formula_mode_on_a_model_fixture(fixture_pipelines):
    """Hide every other kernel of rf_vgg at one batch size; anchors at two other batch sizes fill them."""
    pipeline = fixture_pipelines["rf_vgg_module_forward"]  # batch size 4
    database = SignatureDatabase()
    for leaf in kernels(pipeline.expression):
        shape = leaf.specification.signature["input"]["shape"]
        for batch_size in (1, 16):
            signature = dict(leaf.specification.signature)
            signature["input"] = {**signature["input"], "shape": [batch_size] + shape[1:]}
            elements = batch_size * max(1, len(shape))
            database.add_signature(Signature(KernelSpecification.from_signature(signature), PLATFORM,
                                             1e-5 + 1e-6 * elements, 2e-3 + 2e-4 * elements))
    estimate = evaluate(pipeline.expression, with_formulas(database), PLATFORM, on_missing="report")
    assert estimate.is_complete
    provenances = {leaf.signature.provenance for leaf in estimate.leaves}
    assert provenances <= {INTERPOLATED, FORMULA} and INTERPOLATED in provenances
