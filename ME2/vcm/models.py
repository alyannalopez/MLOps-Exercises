"""Compact neural architectures for the VCM.

Five candidate architectures, all consuming a (B, 1, n_mels, n_frames) log-mel
input and emitting logits over `n_classes`. Kept deliberately small so they run
in real time on a Raspberry Pi 4/5. Parameter counts are reported at build time.

Architectures:
  - CNN:        depthwise-separable conv stack
  - CRNN:       conv stack + BiLSTM (the classic baseline)
  - TCN:        dilated causal convolutions
  - ConformerS: tiny conformer (single layer)
  - MLP:        global-average-pooled spectrogram + dense net (small baseline)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = None):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class CNN(nn.Module):
    def __init__(self, n_classes: int = 20, n_mels: int = 40, n_frames: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, 3, s=2),
            ConvBlock(32, 64, 3, s=2),
            ConvBlock(64, 96, 3, s=2),
            ConvBlock(96, 128, 3, s=2),
        )
        # After 4 stride-2 blocks: (H/16, W/16)
        h, w = n_mels // 16, n_frames // 16
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CRNN(nn.Module):
    def __init__(self, n_classes: int = 20, n_mels: int = 40, n_frames: int = 100,
                 lstm_hidden: int = 64, lstm_layers: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, 3, s=2),
            ConvBlock(32, 64, 3, s=2),
            ConvBlock(64, 64, 3, s=1),
        )
        h = n_mels // 4  # spatial height after two stride-2 blocks
        self.lstm = nn.LSTM(64 * h, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, n_classes),
        )

    def forward(self, x):
        x = self.features(x)              # (B, 64, H, W)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (B, W, C*H)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.head(x)


class DilatedConv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, dilation: int = 1):
        super().__init__()
        p = dilation * (k - 1) // 2
        self.conv = nn.Conv2d(cin, cout, k, padding=p, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class TCN(nn.Module):
    def __init__(self, n_classes: int = 20, n_mels: int = 40, n_frames: int = 100, width: int = 48):
        super().__init__()
        self.stem = ConvBlock(1, width, 3, s=2)
        self.blocks = nn.Sequential(
            DilatedConv(width, width, 3, dilation=1),
            DilatedConv(width, width, 3, dilation=2),
            DilatedConv(width, width, 3, dilation=4),
            DilatedConv(width, width, 3, dilation=8),
            DilatedConv(width, width, 3, dilation=16),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(width, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x) + x  # residual
        return self.head(x)


class _TinyConformerLayer(nn.Module):
    def __init__(self, d: int, nhead: int, ff: int, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.ff = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d), nn.Dropout(dropout)
        )
        self.sa = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.sa_norm = nn.LayerNorm(d)
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, 31, padding=15), nn.GELU(),
            nn.Conv1d(d, d, 1), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.ff(x)
        hx = self.sa_norm(x)
        sa, _ = self.sa(hx, hx, hx, need_weights=False)
        x = x + sa
        # Conv block on (B, D, T): normalize over D (dim 1), then Conv1d.
        # (Manual channel-wise LN: this torch build rejects partial-shape
        # normalized_shape for 3-D inputs, so compute it explicitly.)
        xt = x.transpose(1, 2)
        mu = xt.mean(dim=1, keepdim=True)
        sd = xt.std(dim=1, keepdim=True, unbiased=False) + 1e-5
        xt = (xt - mu) / sd
        x = x + self.conv(xt).transpose(1, 2)
        return x


class ConformerS(nn.Module):
    """Two-head conformer: intent head + shared slot head.

    The slot head predicts one of 19 slot classes (18 slot values across
    6 slotted intents + NO_SLOT).  Non-slotted intents always get NO_SLOT.
    This follows the tiny-alexa two-head design: a single shared slot
    vocabulary, joint CE loss, no ASR.
    """

    def __init__(self, n_classes: int = 20, n_slots: int = 19,
                 n_mels: int = 40, n_frames: int = 100, d: int = 64, nhead: int = 4):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(1, d, 3, s=2), ConvBlock(d, d, 3, s=2))
        self.proj = nn.Linear(d * (n_mels // 4), d)
        self.layer = _TinyConformerLayer(d, nhead, d * 2)
        self.norm = nn.LayerNorm(d)
        self.intent_head = nn.Linear(d, n_classes)
        self.slot_head = nn.Linear(d, n_slots)

    def forward(self, x):
        x = self.stem(x)                       # (B, d, H, W)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (B, T, d*H)
        x = self.proj(x)
        x = self.layer(x)
        x = x.mean(dim=1)                      # (B, d)
        x = self.norm(x)
        return self.intent_head(x), self.slot_head(x)


class MLP(nn.Module):
    def __init__(self, n_classes: int = 20, n_mels: int = 40, n_frames: int = 100):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.pool(x))


ARCHS = {
    "cnn": CNN,
    "crnn": CRNN,
    "tcn": TCN,
    "conformer_s": ConformerS,
    "mlp": MLP,
}


def build(name: str, n_classes: int = 20, n_mels: int = 40, n_frames: int = 100,
          n_slots: int | None = None) -> nn.Module:
    if name not in ARCHS:
        raise KeyError(f"unknown arch {name!r}; choose from {sorted(ARCHS)}")
    if name == "conformer_s":
        return ARCHS[name](n_classes=n_classes, n_slots=n_slots or 19,
                           n_mels=n_mels, n_frames=n_frames)
    return ARCHS[name](n_classes=n_classes, n_mels=n_mels, n_frames=n_frames)
