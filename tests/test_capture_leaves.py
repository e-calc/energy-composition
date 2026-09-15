"""Leaf-level capture (CPU): counts, ordering, key sharing with the module level, op whitelist."""

import collections

import pytest
import torch

from ecalc.capture import RECORDED_OPS, capture_leaves
from ecalc.models import load_fused_model
from ecalc.signature import input_signature, synthesize_inputs


@pytest.fixture(scope="session")
def leaves_n(records_n, cpu_x):
    return capture_leaves(load_fused_model("yolo26n", "cpu"), cpu_x)


@pytest.fixture(scope="session")
def leaves_m(cpu_x):
    return capture_leaves(load_fused_model("yolo26m", "cpu"), cpu_x)


@pytest.fixture(scope="session")
def leaves_l(cpu_x):
    return capture_leaves(load_fused_model("yolo26l", "cpu"), cpu_x)


def test_yolo26n_leaf_counts(leaves_n):
    assert len(leaves_n) == 126
    ops = [r for r in leaves_n if r.op is not None]
    assert len(ops) == 42 and len(leaves_n) - len(ops) == 84
    assert len({r.kernel_key for r in leaves_n}) == 66
    by_cls = collections.Counter(r.layer_type for r in leaves_n)
    assert by_cls == {"Conv": 72, "MaxPool2d": 3, "Attention": 2, "Upsample": 2, "Concat": 4, "Detect": 1,
                      "op:chunk": 8, "op:add": 18, "op:cat": 15, "op:split": 1}


def test_per_block_counts(leaves_n):
    per_layer = collections.Counter(r.parent_layer_idx for r in leaves_n if r.op is None)
    assert per_layer[2] == 4 and per_layer[6] == 9 and per_layer[9] == 5 and per_layer[10] == 5 and per_layer[22] == 7
    assert all(per_layer[i] == 1 for i in (0, 1, 3, 5, 7, 11, 12, 14, 15, 17, 18, 20, 21, 23))
    sppf_cats = [r for r in leaves_n if r.parent_layer_idx == 9 and r.layer_type == "op:cat"]
    assert len(sppf_cats) == 1 and len(sppf_cats[0].inputs) == 4


def test_execution_order_and_seq(leaves_n):
    assert [r.seq for r in leaves_n] == list(range(len(leaves_n)))
    assert [r.layer_idx for r in leaves_n] == [r.seq for r in leaves_n]
    parents = [r.parent_layer_idx for r in leaves_n]
    assert parents == sorted(parents)  # grouped by top-level layer, in graph order
    # inside layer 2 (C3k2): cv1, chunk, m.0.cv1, m.0.cv2, add, cat, cv2
    l2 = [r.path for r in leaves_n if r.parent_layer_idx == 2]
    assert l2 == ["cv1", "op:chunk", "m.0.cv1", "m.0.cv2", "op:add", "op:cat", "cv2"]


def test_deterministic_across_loads(leaves_n, cpu_x):
    again = capture_leaves(load_fused_model("yolo26n", "cpu"), cpu_x)
    assert [r.kernel_key for r in again] == [r.kernel_key for r in leaves_n]
    assert [r.path for r in again] == [r.path for r in leaves_n]


def test_top_level_leaves_share_module_keys(leaves_n, records_n):
    mod = {r.layer_idx: r.kernel_key for r in records_n}
    tops = [r for r in leaves_n if r.path == "" and r.op is None]
    assert len(tops) == 14  # 7 Conv + 2 Upsample + 4 Concat + Detect
    for r in tops:
        assert r.kernel_key == mod[r.parent_layer_idx], r.short_desc


def test_no_leaf_inside_attention_or_detect(leaves_n):
    for r in leaves_n:
        assert ".attn." not in r.path and not r.path.startswith("one2one"), r.path
    assert sum(1 for r in leaves_n if r.layer_type == "Detect") == 1


def test_only_whitelisted_ops(leaves_n):
    names = {r.op["name"] for r in leaves_n if r.op is not None}
    assert names == set(RECORDED_OPS)


def test_ops_roundtrip_and_run(leaves_n):
    with torch.inference_mode():
        for r in leaves_n:
            if r.op is None:
                continue
            synth = synthesize_inputs(r.input_sig, "cpu")
            assert input_signature(synth) == r.input_sig
            out_real = r.run()
            out_synth = r.run(synth)
            ts = out_real if isinstance(out_real, torch.Tensor) else out_real[0]
            ts2 = out_synth if isinstance(out_synth, torch.Tensor) else out_synth[0]
            assert tuple(ts.shape) == tuple(ts2.shape)
            assert [list(t.shape) for t in (out_real if isinstance(out_real, (tuple, list)) else [out_real])] == \
                [s["shape"] for s in r.output_specs]


def test_m_and_l_share_leaf_keys(leaves_m, leaves_l):
    # same width, depth 0.5 vs 1.0: every leaf module of m appears in l except the C3k2 `cv2` convs,
    # whose input width is (2 + n) * c; likewise the C3k2 cat ops take 3 vs 4 tensors.
    km = {r.kernel_key: r for r in leaves_m if r.op is None}
    kl = {r.kernel_key: r for r in leaves_l if r.op is None}
    assert all(km[k].path == "cv2" and km[k].layer_type == "Conv" for k in set(km) - set(kl))
    assert all(kl[k].path == "cv2" and kl[k].layer_type == "Conv" for k in set(kl) - set(km))
    assert len(set(km) - set(kl)) <= 4
    assert len(leaves_l) > len(leaves_m)  # depth 1.0 vs 0.5: more repeats of the same leaves
    om = {r.kernel_key: r for r in leaves_m if r.op is not None}
    ol = {r.kernel_key: r for r in leaves_l if r.op is not None}
    assert all(om[k].op["name"] == "cat" for k in set(om) - set(ol))
    assert all(ol[k].op["name"] == "cat" for k in set(ol) - set(om))


def test_hooks_removed_after_capture(leaves_n):
    for r in leaves_n:
        if r.module is not None:
            assert not r.module._forward_hooks and not r.module._forward_pre_hooks
