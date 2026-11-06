"""Export a trained PyTorch VCM to ONNX and verify parity with onnxruntime.

The exported graph takes a single input `logmel` of shape (1, 1, n_mels,
n_frames) float32 and emits `logits` (1, n_classes). This is the artifact the
Raspberry Pi runtime loads.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, out_path: str, *, n_mels: int = 40,
                n_frames: int = 100, opset: int = 13) -> str:
    model.eval()
    dummy = torch.randn(1, 1, n_mels, n_frames)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["logmel"], output_names=["logits"],
        dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    return out_path


def verify_onnx(pt_model: torch.nn.Module, onnx_path: str, *, n_mels: int = 40,
                n_frames: int = 100, atol: float = 1e-4) -> dict:
    import onnxruntime as ort

    pt_model.eval()
    x = np.random.randn(1, 1, n_mels, n_frames).astype(np.float32)
    with torch.no_grad():
        ref = pt_model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"logmel": x})[0]
    max_diff = float(np.abs(ref - got).max())
    return {"onnx_path": onnx_path, "max_abs_diff": max_diff, "ok": bool(max_diff < atol)}
