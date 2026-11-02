"""RFML (radio-frequency ML) automatic-modulation-classification models for RadioML 2018.01A frames.

Three reference architectures on 2 x 1024 IQ frames, 24 modulation classes:

- ``rf_vgg``    VGG-style CNN of O'Shea, Roy & Clancy, "Over-the-Air Deep Learning Based Radio Signal
                Classification" (IEEE JSTSP 2018): 7 x [Conv1d 64, k3 + ReLU + MaxPool 2], FC/SELU 128 x 2,
                FC/Softmax 24.
- ``rf_resnet`` ResNet of the same paper: 6 residual stacks of 32 filters, then the same dense head.
                Residual stack = Conv1d k1 (linear) -> 2 x residual unit -> MaxPool 2; residual unit =
                Conv1d k3 + ReLU -> Conv1d k3 (linear) -> add skip.
- ``rf_lstm``   2-layer LSTM of Rajendran et al., "Deep Learning Models for Wireless Signal Classification
                With Distributed Low-Cost Spectrum Sensors" (IEEE TCCN 2018): LSTM 128 x 2 on
                amplitude/phase frames, last time step -> FC/Softmax 24.

The paper gives no conv kernel sizes; we use the common reimplementation (k3, same padding), so the
VGG/ResNet parameter counts (159,832 / 165,144) differ from the paper's (257,099 / 236,344).

Every model is a ``SequentialRFModel``: a top-level ``nn.ModuleList`` executed in order, each entry
being one module-level DB component (the counterpart of one Ultralytics graph entry). Residual adds,
activations and eval-mode dropout stay inside their block. Weights are seeded random (no official
PyTorch checkpoints exist); dense FP32 kernel energy does not depend on weight values.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import INPUT_SEED

FRAME_LEN = 1024
IQ_CHANNELS = 2
NUM_CLASSES = 24
WEIGHT_SEED = 0


class ConvPool(nn.Module):
    """VGG unit: Conv1d k3 (same) + ReLU + MaxPool1d 2."""

    def __init__(self, c_in: int, c_out: int = 64, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, padding=k // 2)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.act(self.conv(x)))


class ResidualUnit(nn.Module):
    """Conv1d k3 + ReLU -> Conv1d k3 (linear) -> add skip."""

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, k, padding=k // 2)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(c, c, k, padding=k // 2)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class ResidualStack(nn.Module):
    """Conv1d k1 (linear) -> 2 x ResidualUnit -> MaxPool1d 2."""

    def __init__(self, c_in: int, c: int = 32, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c, 1)
        self.unit1 = ResidualUnit(c, k)
        self.unit2 = ResidualUnit(c, k)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.unit2(self.unit1(self.conv(x))))


class DenseSELU(nn.Module):
    """Linear + SELU + AlphaDropout (identity in eval)."""

    def __init__(self, c_in: int, c_out: int = 128, p: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(c_in, c_out)
        self.act = nn.SELU()
        self.drop = nn.AlphaDropout(p)

    def forward(self, x):
        return self.drop(self.act(self.fc(x)))


class Classifier(nn.Module):
    """Linear + Softmax over classes."""

    def __init__(self, c_in: int = 128, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(c_in, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.fc(x))


class LSTMLayer(nn.Module):
    """One batch-first ``nn.LSTM`` layer returning only the output sequence (B, L, hidden)."""

    def __init__(self, c_in: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden, batch_first=True)

    def forward(self, x):
        return self.lstm(x)[0]


class LastStepClassifier(Classifier):
    """Last time step of the sequence -> Linear + Softmax."""

    def forward(self, x):
        return self.softmax(self.fc(x[:, -1]))


class SequentialRFModel(nn.Module):
    """Top-level blocks executed in order; ``layers[i]`` is module-level component ``i``."""

    def __init__(self, layers: list[nn.Module], input_layout: str):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.input_layout = input_layout  # "BCL" (IQ channels first) or "BLC" (batch-first sequence)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


def _dense_head() -> list[nn.Module]:
    return [nn.Flatten(), DenseSELU(512, 128), DenseSELU(128, 128), Classifier(128, NUM_CLASSES)]


def rf_vgg() -> SequentialRFModel:
    convs = [ConvPool(IQ_CHANNELS if i == 0 else 64, 64) for i in range(7)]  # 64 x 1024 -> 64 x 8
    return SequentialRFModel(convs + _dense_head(), "BCL")


def rf_resnet() -> SequentialRFModel:
    stacks = [ResidualStack(IQ_CHANNELS if i == 0 else 32, 32) for i in range(6)]  # 32 x 1024 -> 32 x 16
    return SequentialRFModel(stacks + _dense_head(), "BCL")


def rf_lstm() -> SequentialRFModel:
    return SequentialRFModel([LSTMLayer(IQ_CHANNELS, 128), LSTMLayer(128, 128), LastStepClassifier(128)], "BLC")


RFML = {"rf_vgg": rf_vgg, "rf_resnet": rf_resnet, "rf_lstm": rf_lstm}
ALL_RFML = list(RFML)


def is_rfml(name: str) -> bool:
    return name in RFML


def load_rfml_model(name: str, device: str | torch.device = "cuda") -> SequentialRFModel:
    """Eval-mode model with seeded random weights and all grads disabled."""
    if name not in RFML:
        raise ValueError(f"unknown RFML model {name!r}; known: {ALL_RFML}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(WEIGHT_SEED)
        model = RFML[name]()
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.LSTM):
            m.flatten_parameters()
    return model


def rfml_input_shape(name: str, batch: int, frame_len: int = FRAME_LEN) -> tuple[int, int, int]:
    layout = "BLC" if name == "rf_lstm" else "BCL"
    return (batch, frame_len, IQ_CHANNELS) if layout == "BLC" else (batch, IQ_CHANNELS, frame_len)


def make_rfml_input(name: str, batch: int, frame_len: int = FRAME_LEN, device: str | torch.device = "cuda",
                    seed: int = INPUT_SEED) -> torch.Tensor:
    """Seeded standard-normal frames (complex-AWGN-like IQ; amplitude/phase pair for the LSTM)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(rfml_input_shape(name, batch, frame_len), generator=g, dtype=torch.float32).to(device)
