"""Loss helpers: class weighting, label smoothing, and a real-voice emphasis
weighting used to counteract the synthetic-heavy training mix.
"""
from __future__ import annotations

import numpy as np
import torch


def class_weights_from_counts(counts: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.clip(counts, 1, None)
    w = 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def smoothed_target(y: int, n_classes: int, eps: float = 0.1) -> torch.Tensor:
    t = torch.full((n_classes,), eps / (n_classes - 1))
    t[y] = 1.0 - eps + eps / (n_classes - 1)
    return t


def cross_entropy_smooth(logits: torch.Tensor, targets: torch.LongTensor,
                         n_classes: int, eps: float = 0.1) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    t = torch.stack([smoothed_target(y.item(), n_classes, eps) for y in targets])
    return -(t * logp).sum(dim=-1).mean()
