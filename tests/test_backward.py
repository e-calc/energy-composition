"""Backward-pass capture (CPU): keys, gradient accumulation, replayable graphs, and that the
committed structure fixtures still match a fresh capture."""

from __future__ import annotations

import pytest
import torch

from ecalc.calculus import kernels
from ecalc.capture import (BackwardGraph, backward_records, capture, consumer_counts, differentiable_outputs,
                           model_backward_graph, model_composition)
from ecalc.models import load_model, model_input
from ecalc.signature import backward_signature, kernel_key


@pytest.fixture(scope="module")
def vgg():
    model = load_model("rf_vgg", "cpu")
    return model, model_input("rf_vgg", 4, None, "cpu")


@pytest.fixture(scope="module")
def lstm():
    model = load_model("rf_lstm", "cpu")
    return model, model_input("rf_lstm", 2, 64, "cpu")


def test_backward_records_mirror_the_forward_layers(vgg):
    model, x = vgg
    forward = capture(model, x)
    backward = capture(model, x, direction="backward")
    assert [r.layer_idx for r in backward] == [r.layer_idx for r in reversed(forward)]
    assert all(r.direction == "backward" for r in backward)
    assert not ({r.kernel_key for r in backward} & {r.kernel_key for r in forward})
    for f, b in zip(reversed(forward), backward):
        assert b.kernel_key == kernel_key(backward_signature(f.signature, b.input_gradient))
        assert b.short_desc.startswith(f.short_desc) and "<backward" in b.short_desc
    assert [r.input_gradient for r in backward] == [True] * (len(forward) - 1) + [False]


def test_backward_signature_is_the_forward_one_plus_direction(vgg):
    model, x = vgg
    forward = capture(model, x)[3].signature
    backward = backward_signature(forward)
    assert {k: v for k, v in backward.items() if k != "direction"} == forward
    assert backward_signature(forward, input_gradient=False)["input_gradient"] is False
    assert "input_gradient" not in backward


def test_backward_is_module_level_only(records_n):
    from ecalc.capture import capture_leaves
    from ecalc.models import load_fused_model, make_input

    leaves = capture_leaves(load_fused_model("yolo26n", "cpu"), make_input("cpu"), layers=records_n)
    with pytest.raises(ValueError, match="module level only"):
        backward_records(leaves)


def test_yolo_gradient_accumulation_follows_the_layer_graph(records_n):
    counts = consumer_counts(records_n)
    fan_out = {layer: n for layer, n in counts.items() if n > 1}
    assert fan_out == {4: 2, 6: 2, 10: 2, 13: 2, 16: 2, 19: 2}
    backward = backward_records(records_n)
    accumulations = [r for r in backward if r.path == "gradient_accumulation"]
    assert len(backward) == 24 + 6 and len(accumulations) == 6
    for r in accumulations:
        producer = records_n[r.parent_layer_idx]
        assert r.layer_type == "op:add" and r.direction == "forward"
        assert r.input_sig["items"] == [producer.output_specs[0]] * 2
    assert len({r.layer_idx for r in backward}) == len(backward)  # unique indices for the store


def test_a_backward_graph_replays_and_restores_the_model(vgg):
    model, x = vgg
    record = capture(model, x, direction="backward")[2]
    with record.backward_graph() as graph:
        assert all(p.requires_grad for p in record.module.parameters())
        graph.step()
        graph.step()  # the graph is retained: it can be replayed
        gradients = torch.autograd.grad(graph.outputs, graph.wrt, graph.grad_outputs, retain_graph=True,
                                        allow_unused=True)
        assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training


def test_the_first_layer_takes_no_input_gradient(vgg):
    model, x = vgg
    first = capture(model, x, direction="backward")[-1]
    assert not first.input_gradient
    with first.backward_graph() as graph:
        assert len(graph.wrt) == len(list(first.module.parameters()))  # parameters only


def test_recurrent_layers_run_backward_in_train_mode_and_are_restored(lstm):
    model, x = lstm
    record = capture(model, x, direction="backward")[-1]
    recurrent = record.module.lstm
    assert not recurrent.training
    with record.backward_graph() as graph:
        assert recurrent.training
        graph.step()
    assert not recurrent.training


def test_whole_model_backward(vgg, lstm):
    for model, x in (vgg, lstm):
        with model_backward_graph(model, x) as graph:
            assert len(graph.wrt) == len(list(model.parameters()))
            graph.step()
        assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_detect_backward_starts_from_the_raw_head_outputs(records_n):
    detect = backward_records(records_n)[0]
    assert detect.layer_type == "Detect"
    with detect.backward_graph() as graph:
        graph.step()
        shapes = [tuple(o.shape) for o in graph.outputs]
    assert (1, 4, 8400) in shapes and (1, 80, 8400) in shapes   # raw boxes and scores, not the decoded (1, 300, 6)
    assert (1, 300, 6) not in shapes


def test_differentiable_outputs_prefers_tensors_inside_dicts():
    a = torch.ones(2, requires_grad=True)
    b = torch.ones(3, requires_grad=True)
    assert differentiable_outputs((a * 1, {"raw": b * 1}))[0].shape == (3,)
    assert differentiable_outputs(a * 1)[0].shape == (2,)
    assert differentiable_outputs(torch.ones(2)) == []


def test_a_kernel_without_anything_to_differentiate_is_an_error():
    with pytest.raises(RuntimeError, match="no differentiable output"):
        BackwardGraph(lambda t: t.detach(), torch.ones(2), [], [], input_gradient=True)


def test_composition_rows_carry_direction_and_structure(vgg):
    model, x = vgg
    rows = model_composition(capture(model, x, direction="backward"), "rf_vgg", "fp32", 4, 1024, "x")
    assert {r["direction"] for r in rows} == {"backward"}
    assert all(r["signature_json"] for r in rows)
    forward_rows = model_composition(capture(model, x), "rf_vgg", "fp32", 4, 1024, "x")
    assert {r["direction"] for r in forward_rows} == {"forward"}


@pytest.mark.parametrize("model, level, direction, fixture", [
    ("rf_vgg", "module", "forward", "rf_vgg_module_forward"),
    ("rf_resnet", "module", "forward", "rf_resnet_module_forward"),
    ("rf_lstm", "module", "forward", "rf_lstm_module_forward"),
    ("rf_vgg", "module", "backward", "rf_vgg_module_backward"),
    ("rf_lstm", "module", "backward", "rf_lstm_module_backward"),
    ("yolo26n", "module", "forward", "yolo26n_module_forward"),
    ("yolo26n", "leaf", "forward", "yolo26n_leaf_forward"),
    ("yolo26n", "module", "backward", "yolo26n_module_backward"),
])
def test_fixtures_match_a_fresh_capture(fixture_pipelines, model, level, direction, fixture):
    """Kernel keys depend on library versions; when they match the fixture's, the keys must too."""
    import importlib.metadata

    pipeline = fixture_pipelines[fixture]
    recorded = pipeline.attributes.get("captured_with", {})
    package = "ultralytics" if model.startswith("yolo") else "torch"
    if recorded.get(package) != importlib.metadata.version(package):
        pytest.skip(f"fixture captured with {package} {recorded.get(package)}, "
                    f"installed {importlib.metadata.version(package)}: regenerate tests/fixtures")
    batch_size = pipeline.attributes["batch_size"]
    records = capture(load_model(model, "cpu"), model_input(model, batch_size, None, "cpu"), level=level,
                      direction=direction)
    assert sorted(r.kernel_key for r in records) == sorted(leaf.specification.key
                                                           for leaf in kernels(pipeline.expression))
