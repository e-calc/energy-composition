import pytest
import torch

from ecalc.capture import capture_layers
from ecalc.models import load_fused_model, make_input


@pytest.fixture(scope="session")
def cpu_x():
    return make_input("cpu")


@pytest.fixture(scope="session")
def records_n(cpu_x):
    return capture_layers(load_fused_model("yolo26n", "cpu"), cpu_x)


@pytest.fixture(scope="session")
def records_s(cpu_x):
    return capture_layers(load_fused_model("yolo26s", "cpu"), cpu_x)


@pytest.fixture(scope="session")
def records_m(cpu_x):
    return capture_layers(load_fused_model("yolo26m", "cpu"), cpu_x)


@pytest.fixture(scope="session")
def records_l(cpu_x):
    return capture_layers(load_fused_model("yolo26l", "cpu"), cpu_x)


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA GPU")
        for it in items:
            if "gpu" in it.keywords:
                it.add_marker(skip)
