import json

import torch

from ecalc.capture import capture_layers, unique_records
from ecalc.models import load_fused_model
from ecalc.signature import (canonical_json, input_signature, kernel_key, kernel_signature, synthesize_inputs)


def test_yolo26n_has_24_layers(records_n):
    assert len(records_n) == 24
    assert records_n[-1].layer_type == "Detect"
    assert records_n[-1].output_specs[0]["shape"] == [1, 300, 6]


def test_keys_deterministic_across_loads(records_n, cpu_x):
    again = capture_layers(load_fused_model("yolo26n", "cpu"), cpu_x)
    assert [r.kernel_key for r in records_n] == [r.kernel_key for r in again]


def test_layer0_differs_between_sizes(records_n, records_s):
    assert records_n[0].kernel_key != records_s[0].kernel_key


def test_m_and_l_share_position_independent_kernels(records_m, records_l):
    shared_types = {"Conv", "Concat", "Upsample", "SPPF", "Detect"}
    for a, b in zip(records_m, records_l):
        if a.layer_type in shared_types:
            assert a.kernel_key == b.kernel_key, (a.short_desc, b.short_desc)
    # depth differs, so C3k2 blocks must not collide
    assert any(a.kernel_key != b.kernel_key for a, b in zip(records_m, records_l) if a.layer_type == "C3k2")


def test_input_shape_changes_key(records_n):
    r = records_n[0]
    sig2 = kernel_signature(r.module, torch.randn(1, 3, 320, 320))
    assert kernel_key(sig2) != r.kernel_key


def test_signature_excludes_graph_position(records_n):
    for r in records_n:
        top = r.signature["module"]["modules"][0]["a"]
        for bad in ("i", "f", "np", "type", "anchors", "strides", "shape"):
            assert bad not in top, (r.short_desc, bad)
    js = canonical_json(records_n[-1].signature)
    assert '"anchors"' not in js and '"strides"' not in js


def test_same_block_different_position_same_key(records_n):
    # yolo26n layers 3 and 17 are both Conv[64->64,k3,s2] but on different input sizes -> different keys;
    # build the same module signature and compare only the module part.
    a, b = records_n[3], records_n[17]
    assert a.signature["module"] == b.signature["module"]
    assert a.kernel_key != b.kernel_key


def test_synthesize_roundtrip(records_n):
    for r in (records_n[0], records_n[12], records_n[23]):  # tensor, Concat list, Detect list
        synth = synthesize_inputs(r.input_sig, "cpu")
        assert input_signature(synth) == r.input_sig
        with torch.inference_mode():
            out = r.module(synth)
        assert out is not None


def test_unique_records(records_n):
    assert len(unique_records(records_n)) == len({r.kernel_key for r in records_n})


def test_signature_json_serialisable(records_n):
    for r in records_n:
        json.loads(canonical_json(r.signature))
