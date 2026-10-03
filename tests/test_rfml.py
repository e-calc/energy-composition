"""RFML module-level capture (CPU): counts, faithful replay, key stability and cross-model reuse."""

import pytest
import torch

from ecalc.capture import capture, capture_sequential
from ecalc.models import load_model, model_input, parse_model_list, variant_name
from ecalc.rfml import ALL_RFML, load_rfml_model, make_rfml_input
from ecalc.signature import input_signature, synthesize_inputs

BATCH = 4  # keys depend on batch; structure checks do not


@pytest.fixture(scope="session")
def rf_records():
    out = {}
    for name in ALL_RFML:
        model = load_rfml_model(name, "cpu")
        x = make_rfml_input(name, BATCH, device="cpu")
        out[name] = (model, x, capture_sequential(model, x))
    return out


def test_module_counts_and_types(rf_records):
    types = {n: [r.layer_type for r in recs] for n, (_, _, recs) in rf_records.items()}
    assert types["rf_vgg"] == ["ConvPool"] * 7 + ["Flatten", "DenseSELU", "DenseSELU", "Classifier"]
    assert types["rf_resnet"] == ["ResidualStack"] * 6 + ["Flatten", "DenseSELU", "DenseSELU", "Classifier"]
    assert types["rf_lstm"] == ["LSTMLayer", "LSTMLayer", "LastStepClassifier"]


def test_param_counts(rf_records):
    n = {name: sum(p.numel() for p in m.parameters()) for name, (m, _, _) in rf_records.items()}
    assert n == {"rf_vgg": 159832, "rf_resnet": 165144, "rf_lstm": 202776}
    for name, (m, _, recs) in rf_records.items():
        assert sum(r.n_params for r in recs) == n[name]  # every parameter belongs to exactly one component


def test_composed_replay_equals_forward(rf_records):
    for name, (model, x, recs) in rf_records.items():
        with torch.inference_mode():
            ref = model(x)
            cur = x
            for r in recs:
                assert input_signature(cur) == r.input_sig
                assert torch.equal(cur, r.inputs), (name, r.layer_idx)
                cur = r.run(cur)
        assert torch.equal(cur, ref), name
        assert ref.shape == (BATCH, 24)


def test_isolated_replay_deterministic(rf_records):
    for _, (_, _, recs) in rf_records.items():
        for r in recs:
            with torch.inference_mode():
                a, b = r.run(), r.run()
            assert torch.equal(a, b), r.short_desc
            assert [list(a.shape)] == [s["shape"] for s in r.output_specs]


def test_keys_stable_across_loads(rf_records):
    for name, (_, x, recs) in rf_records.items():
        again = capture_sequential(load_rfml_model(name, "cpu"), x)
        assert [r.kernel_key for r in again] == [r.kernel_key for r in recs]


def test_dense_head_shared_between_vgg_and_resnet(rf_records):
    vgg, res = rf_records["rf_vgg"][2], rf_records["rf_resnet"][2]
    assert [r.kernel_key for r in vgg[-3:]] == [r.kernel_key for r in res[-3:]]
    assert vgg[7].kernel_key != res[6].kernel_key  # Flatten on 64x8 vs 32x16
    shared = {r.kernel_key for r in vgg} & {r.kernel_key for r in res}
    assert len(shared) == 3


def test_batch_changes_keys(rf_records):
    model, _, recs = rf_records["rf_resnet"]
    other = capture_sequential(model, make_rfml_input("rf_resnet", BATCH * 2, device="cpu"))
    assert not ({r.kernel_key for r in recs} & {r.kernel_key for r in other})


def test_lstm_layer_returns_sequence(rf_records):
    recs = rf_records["rf_lstm"][2]
    assert recs[0].output_specs == [{"shape": [BATCH, 1024, 128], "dtype": "float32"}]
    assert recs[1].input_sig == {"kind": "tensor", "shape": [BATCH, 1024, 128], "dtype": "float32"}


def test_synthesize_roundtrip(rf_records):
    for _, (_, _, recs) in rf_records.items():
        for r in recs:
            assert input_signature(synthesize_inputs(r.input_sig, "cpu")) == r.input_sig


def test_dispatch_helpers():
    assert parse_model_list("rfml") == ALL_RFML
    assert variant_name("rf_vgg", end2end=False) == "rf_vgg"
    x = model_input("rf_lstm", 2, None, "cpu")
    assert x.shape == (2, 1024, 2)
    recs = capture(load_model("rf_vgg", "cpu"), model_input("rf_vgg", 2, None, "cpu"))
    assert len(recs) == 11
    with pytest.raises(ValueError):
        capture(load_model("rf_vgg", "cpu"), model_input("rf_vgg", 2, None, "cpu"), level="leaf")
