"""The algebraic laws on real model structure: YOLO26 and the RFML classifiers.

The fixtures under ``tests/fixtures`` hold which kernels each model is made of (captured on CPU),
with no measured numbers. Costs are synthetic and exactly representable, so the laws are
asserted with ``==``. Needs no torch.

The last tests use a local measured database when one exists (``data/ecalc.sqlite``) and are
skipped otherwise: measured data is not part of the repository.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from calculus_helpers import PLATFORM, synthetic_database, synthetic_signature

from ecalc.calculus import (Kernel, NegativeDynamicEnergy, Parallel, Platform, Sequential, Signature,
                            SignatureDatabase, evaluate, flatten, kernels)

FORWARD_MODELS = ["yolo26n_module_forward", "yolo26n_leaf_forward", "rf_vgg_module_forward",
                  "rf_resnet_module_forward", "rf_lstm_module_forward"]
ALL_MODELS = FORWARD_MODELS + ["yolo26n_module_backward", "rf_vgg_module_backward", "rf_resnet_module_backward",
                               "rf_lstm_module_backward"]
WATTS = (0, 23, 90)


def _database(*pipelines) -> SignatureDatabase:
    return synthetic_database({leaf.specification for p in pipelines for leaf in kernels(p.expression)})


# ---- structure of the fixtures ----------------------------------------------------------------

def test_kernel_counts_match_the_published_structure(fixture_pipelines):
    def counts(name):
        leaves = list(kernels(fixture_pipelines[name].expression))
        return len(leaves), len({leaf.specification.key for leaf in leaves})

    assert counts("yolo26n_module_forward") == (24, 24)
    assert counts("yolo26n_leaf_forward") == (126, 66)
    assert counts("rf_vgg_module_forward") == (11, 11)
    assert counts("rf_resnet_module_forward") == (10, 10)
    assert counts("rf_lstm_module_forward") == (3, 3)
    assert len(fixture_pipelines["yolo26n_leaf_forward"].expression.parts) == 24  # one group per top-level layer


def test_dense_head_kernels_are_shared_between_vgg_and_resnet(fixture_pipelines):
    vgg = [leaf.specification for leaf in kernels(fixture_pipelines["rf_vgg_module_forward"].expression)]
    resnet = [leaf.specification for leaf in kernels(fixture_pipelines["rf_resnet_module_forward"].expression)]
    shared = {s.key for s in vgg} & {s.key for s in resnet}
    assert shared == {s.key for s in vgg[-3:]} == {s.key for s in resnet[-3:]}
    assert [s.name for s in vgg[-3:]] == ["DenseSELU", "DenseSELU", "Classifier"]


def test_backward_kernels_have_their_own_keys(fixture_pipelines):
    for model in ("yolo26n", "rf_vgg", "rf_resnet", "rf_lstm"):
        forward = {leaf.specification.key for leaf in kernels(fixture_pipelines[f"{model}_module_forward"].expression)}
        backward = [leaf.specification for leaf in kernels(fixture_pipelines[f"{model}_module_backward"].expression)]
        passes = [s for s in backward if s.direction == "backward"]
        assert len(passes) == len(forward)
        assert not ({s.key for s in passes} & forward)
        assert {s.with_direction("forward").key for s in passes} == forward
        assert sum(1 for s in passes if not s.input_gradient) == 1  # only the layer that reads the model input


def test_yolo_backward_counts_gradient_accumulations(fixture_pipelines):
    backward = [leaf.specification for leaf in kernels(fixture_pipelines["yolo26n_module_backward"].expression)]
    accumulations = [s for s in backward if s.direction == "forward"]
    assert len(accumulations) == 6 and {s.name for s in accumulations} == {"op:add"}


# ---- laws -------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", ALL_MODELS)
def test_sequential_energy_is_the_sum_of_kernel_energies_for_any_static_power(fixture_pipelines, model):
    pipeline = fixture_pipelines[model]
    database = _database(pipeline)
    estimate = evaluate(pipeline.expression, database, PLATFORM)
    leaves = [database.lookup(leaf.specification, PLATFORM) for leaf in kernels(pipeline.expression)]
    assert estimate.is_complete and not estimate.cost.has_overlap
    assert estimate.time_seconds == math.fsum(s.time_seconds for s in leaves)
    for watts in WATTS + (89.69807891254976,):
        assert estimate.energy(watts) == math.fsum(s.energy_joules for s in leaves)
    for watts in WATTS:
        assert estimate.energy(watts) == estimate.static_energy(watts) + estimate.dynamic_energy(watts)
        assert estimate.dynamic_energy(watts) == math.fsum(s.dynamic_energy(watts) for s in leaves)


def test_grouped_by_layer_equals_flat(fixture_pipelines):
    pipeline = fixture_pipelines["yolo26n_leaf_forward"]
    database = _database(pipeline)
    grouped = evaluate(pipeline.expression, database, PLATFORM)
    flat = evaluate(flatten(pipeline.expression), database, PLATFORM)
    assert len(flatten(pipeline.expression).parts) == 126
    assert grouped.cost == flat.cost


@pytest.mark.parametrize("model", FORWARD_MODELS)
def test_any_split_into_prefix_and_suffix_adds_up(fixture_pipelines, model):
    pipeline = fixture_pipelines[model]
    database = _database(pipeline)
    parts = flatten(pipeline.expression).parts
    whole = evaluate(Sequential(parts), database, PLATFORM)
    for cut in range(len(parts) + 1):
        head = evaluate(Sequential(parts[:cut]), database, PLATFORM)
        tail = evaluate(Sequential(parts[cut:]), database, PLATFORM)
        assert head.time_seconds + tail.time_seconds == whole.time_seconds
        assert head.energy(90) + tail.energy(90) == whole.energy(90)


def test_a_repeated_kernel_costs_its_count_times_its_signature(fixture_pipelines):
    pipeline = fixture_pipelines["yolo26n_leaf_forward"]
    database = _database(pipeline)
    occurrences: dict = {}
    for leaf in kernels(pipeline.expression):
        occurrences[leaf.specification] = occurrences.get(leaf.specification, 0) + 1
    assert max(occurrences.values()) > 1
    by_count = math.fsum(count * database.lookup(spec, PLATFORM).energy_joules for spec, count in occurrences.items())
    assert evaluate(pipeline.expression, database, PLATFORM).energy(90) == by_count


def test_n_frames_cost_n_times_one_frame(fixture_pipelines):
    pipeline = fixture_pipelines["rf_vgg_module_forward"]
    database = _database(pipeline)
    one = evaluate(pipeline.expression, database, PLATFORM)
    eight = evaluate(Sequential((pipeline.expression,) * 8), database, PLATFORM)
    assert eight.time_seconds == 8 * one.time_seconds and eight.energy(90) == 8 * one.energy(90)


def test_two_models_overlapped_on_one_gpu(fixture_pipelines):
    yolo, vgg = fixture_pipelines["yolo26n_module_forward"], fixture_pipelines["rf_vgg_module_forward"]
    database = _database(yolo, vgg)
    a = evaluate(yolo.expression, database, PLATFORM)
    b = evaluate(vgg.expression, database, PLATFORM)
    together = evaluate(Parallel((yolo.expression, vgg.expression)), database, PLATFORM)
    in_sequence = evaluate(Sequential((yolo.expression, vgg.expression)), database, PLATFORM)
    assert together.time_seconds == max(a.time_seconds, b.time_seconds)
    for watts in WATTS:
        saving = in_sequence.energy(watts) - together.energy(watts)
        assert saving == watts * min(a.time_seconds, b.time_seconds)
        assert together.dynamic_energy(watts) == a.dynamic_energy(watts) + b.dynamic_energy(watts)
        assert max(a.energy(watts), b.energy(watts)) <= together.energy(watts) <= in_sequence.energy(watts)
    assert together.energy(0) == in_sequence.energy(0)
    # the order of the overlapped models does not matter
    assert evaluate(Parallel((vgg.expression, yolo.expression)), database, PLATFORM).cost == together.cost


def test_two_copies_of_one_model_overlapped(fixture_pipelines):
    pipeline = fixture_pipelines["rf_lstm_module_forward"]
    database = _database(pipeline)
    one = evaluate(pipeline.expression, database, PLATFORM)
    two = evaluate(Parallel((pipeline.expression, pipeline.expression)), database, PLATFORM)
    assert two.time_seconds == one.time_seconds
    assert two.energy(90) == 2 * one.energy(90) - 90 * one.time_seconds  # both run; static power is paid once


def test_forward_and_backward_compose_into_one_step(fixture_pipelines):
    forward, backward = fixture_pipelines["rf_vgg_module_forward"], fixture_pipelines["rf_vgg_module_backward"]
    database = _database(forward, backward)
    step = evaluate(Sequential((forward.expression, backward.expression)), database, PLATFORM)
    assert step.energy(90) == (evaluate(forward.expression, database, PLATFORM).energy(90)
                               + evaluate(backward.expression, database, PLATFORM).energy(90))


def test_overlap_with_a_kernel_below_the_static_power_is_refused(fixture_pipelines):
    yolo, vgg = fixture_pipelines["yolo26n_leaf_forward"], fixture_pipelines["rf_vgg_module_forward"]
    database = _database(yolo, vgg)
    chunk = next(leaf.specification for leaf in kernels(yolo.expression) if leaf.specification.name == "op:chunk")
    database.add_signature(Signature(chunk, PLATFORM, 2.0 ** -16, 82 * 2.0 ** -16))  # 82 W, like the measured chunk
    assert evaluate(yolo.expression, database, PLATFORM).energy(90) > 0            # sequential: fine
    overlapped = evaluate(Parallel((yolo.expression, vgg.expression)), database, PLATFORM)
    with pytest.raises(NegativeDynamicEnergy):
        overlapped.energy(90)
    assert overlapped.energy(90, negative_dynamic="allow") < overlapped.cost.energy_joules
    assert overlapped.energy(80) > 0                                                # 80 W is below every kernel


def test_synthetic_costs_are_deterministic(fixture_pipelines):
    specification = next(kernels(fixture_pipelines["rf_lstm_module_forward"].expression)).specification
    assert synthetic_signature(specification) == synthetic_signature(specification)
    assert isinstance(Kernel(specification), Kernel)


# ---- with a local measured database (skipped without one) -------------------------------------

LOCAL_STORE = Path(__file__).resolve().parent.parent / "data" / "ecalc.sqlite"
A40 = Platform("A40")
# model, batch size -> (composed millijoules, measured millijoules) as published for the A40
PUBLISHED = {
    ("yolo26n", 1): (727, 765), ("yolo26s", 1): (1021, 1018), ("yolo26m", 1): (1889, 1847),
    ("yolo26l", 1): (2450, 2414), ("yolo26x", 1): (3897, 3788),
    ("yolo26n", 8): (2743, 2749), ("yolo26s", 8): (5738, 5806), ("yolo26m", 8): (12608, 12749),
    ("yolo26l", 8): (16376, 16548), ("yolo26x", 8): (27087, 27326),
    ("rf_vgg", 32): (103.5, 105.8), ("rf_resnet", 32): (216.4, 223.3), ("rf_lstm", 32): (898.5, 909.5),
    ("rf_vgg", 256): (631.4, 615.7), ("rf_resnet", 256): (1258.7, 1240.1), ("rf_lstm", 256): (5032.4, 5024.4),
}


@pytest.fixture(scope="module")
def local_database():
    if not LOCAL_STORE.exists():
        pytest.skip("no local measured database (data/ecalc.sqlite)")
    from ecalc import db

    database = db.signature_database(db.connect(LOCAL_STORE))
    if A40 not in database.platforms():
        pytest.skip("the local database has no A40 measurements")
    return database


def test_local_database_reproduces_the_published_a40_numbers(local_database):
    from ecalc.calculus import pipeline_name

    checked = 0
    for (model, batch_size), (composed, measured) in PUBLISHED.items():
        size = 1024 if model.startswith("rf_") else 640
        pipeline = local_database.pipeline(pipeline_name(model, batch_size, size))
        run = local_database.model_run(model, batch_size, size, A40)
        if pipeline is None or run is None:
            continue
        estimate = evaluate(pipeline.expression, local_database, A40, on_missing="report")
        if not estimate.is_complete:
            continue
        assert estimate.energy(local_database.static_power(A40, "active_idle")) * 1e3 == pytest.approx(composed, rel=2e-3)
        assert run.energy_joules * 1e3 == pytest.approx(measured, rel=2e-3)
        checked += 1
    if checked == 0:
        pytest.skip("the local database holds none of the published configurations")


def test_stored_keys_match_their_stored_structure(local_database):
    signatures = local_database.signatures()
    assert signatures and all(s.kernel.is_resolved for s in signatures)
