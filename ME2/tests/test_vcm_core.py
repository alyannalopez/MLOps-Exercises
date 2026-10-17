"""Unit tests for the VCM core (features, models, losses).

Run: python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.features import log_mel_spectrogram, prepare_clip, feature_shape
from vcm.models import build, count_params, ARCHS
from vcm.losses import class_weights_from_counts, cross_entropy_smooth


def test_logmel_shape():
    w = np.random.randn(16000).astype(np.float32)
    f = log_mel_spectrogram(w)
    assert f.shape == (40, 100), f.shape
    assert np.isfinite(f).all()


def test_prepare_clip_constant_shape():
    for n in [1600, 8000, 16000, 40000]:  # short, half, 1s, long
        w = np.random.randn(n).astype(np.float32)
        f = prepare_clip(w)
        assert f.shape == feature_shape(), f.shape


def test_prepare_clip_empty():
    f = prepare_clip(np.zeros(0, dtype=np.float32))
    assert f.shape == feature_shape()


def test_all_archs_forward():
    x = torch.randn(4, 1, 40, 100)
    for name in ARCHS:
        m = build(name, n_classes=20)
        out = m(x)
        # conformer_s is two-head (intent + slot); the others are intent-only.
        if name == "conformer_s":
            intent, slot = out
            assert intent.shape == (4, 20), (name, intent.shape)
            assert slot.shape == (4, 19), (name, slot.shape)
        else:
            assert out.shape == (4, 20), (name, out.shape)
        n = count_params(m)
        assert n < 2_000_000, (name, n)  # must stay small for RPi


def test_class_weights_mean_one():
    counts = np.array([100, 50, 10, 5])
    w = class_weights_from_counts(counts)
    assert abs(w.mean().item() - 1.0) < 1e-5
    assert w[3].item() > w[0].item()  # rare class weighted higher


def test_smooth_ce_finite():
    logits = torch.randn(8, 20)
    targets = torch.randint(0, 20, (8,))
    loss = cross_entropy_smooth(logits, targets, 20, eps=0.1)
    assert torch.isfinite(loss)
