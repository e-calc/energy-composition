"""Shared fixtures.

The calculus, serialization, formula and command-line tests need only the standard library and
run without torch. Tests of capture and measurement need torch (and Ultralytics); their files
are skipped at collection when torch is not installed.
"""

import importlib.util
from pathlib import Path

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
FIXTURES = Path(__file__).parent / "fixtures"

TORCH_TEST_FILES = ["test_backward.py", "test_capture_leaves.py", "test_gpu_consistency.py", "test_rfml.py",
                    "test_signature.py"]
collect_ignore = [] if HAS_TORCH else TORCH_TEST_FILES


@pytest.fixture(scope="session")
def fixture_pipelines():
    """The structure-only pipelines under ``tests/fixtures``, by file stem."""
    from ecalc.calculus import load

    return {path.stem: load(path) for path in sorted(FIXTURES.glob("*.json"))}


@pytest.fixture(scope="session")
def cpu_x():
    from ecalc.models import make_input

    return make_input("cpu")


def _module_records(name, cpu_x):
    from ecalc.capture import capture_layers
    from ecalc.models import load_fused_model

    return capture_layers(load_fused_model(name, "cpu"), cpu_x)


@pytest.fixture(scope="session")
def records_n(cpu_x):
    return _module_records("yolo26n", cpu_x)


@pytest.fixture(scope="session")
def records_s(cpu_x):
    return _module_records("yolo26s", cpu_x)


@pytest.fixture(scope="session")
def records_m(cpu_x):
    return _module_records("yolo26m", cpu_x)


@pytest.fixture(scope="session")
def records_l(cpu_x):
    return _module_records("yolo26l", cpu_x)


def pytest_collection_modifyitems(config, items):
    cuda = False
    if HAS_TORCH:
        import torch

        cuda = torch.cuda.is_available()
    if not cuda:
        skip = pytest.mark.skip(reason="no CUDA GPU")
        for it in items:
            if "gpu" in it.keywords:
                it.add_marker(skip)
