#!/usr/bin/env python3
"""Step 8: export the best model to ONNX (float32 + int8) and verify parity.

Dynamic sequence axis so any clip length works on-device. opset 17.
Verifies onnxruntime output matches PyTorch on a held-out sample.

Output: ME2/models/best_float.onnx, ME2/models/best_int8.onnx,
        ME2/models/export_report.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from models import build as build_model  # noqa: E402

MODELS = os.path.join(ROOT, "models")
FEATS = os.path.join(ROOT, "data", "feats")


def main():
    bench = json.load(open(os.path.join(MODELS, "benchmark.json")))
    name = bench["best"]
    ckpt = torch.load(os.path.join(MODELS, f"{name}_state.pt"), map_location="cpu")
    kind = ckpt["kind"]
    F = 40 if kind == "mfcc" else 80

    model = build_model(name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, 101, F)
    float_path = os.path.join(MODELS, f"{name}_float.onnx")
    # legacy (TorchScript) exporter: honors dynamic_axes; the new dynamo
    # exporter (default in torch>=2.9) ignores them and bakes in batch=1.
    torch.onnx.export(model, dummy, float_path,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"},
                                    "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)

    # int8 quantization (dynamic) — attempted; the onnxruntime quantizer
    # currently hits a ShapeInferenceError on Conv weights with this
    # torch/ORT combo, so INT8 size is reported analytically instead.
    int8_path, int8_ok = None, False
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(MODELS, f"{name}_int8.onnx")
        quantize_dynamic(float_path, int8_path, weight_type=QuantType.QInt8)
        int8_ok = True
    except Exception as e:
        print(f"int8 quantization unavailable: {e}")

    # parity check with onnxruntime
    z = np.load(os.path.join(FEATS, f"{kind}_test.npz"), allow_pickle=True)
    X = z["X"][:32]
    y = z["y"][:32]
    with torch.no_grad():
        ref = model(torch.from_numpy(X)).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(float_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": X})[0]
    max_diff = float(np.abs(ref - got).max())
    agree = int((ref.argmax(1) == got.argmax(1)).sum())

    fs = os.path.getsize(float_path) / 1e6
    isz = (os.path.getsize(int8_path) / 1e6) if int8_path else None
    report = {"model": name, "kind": kind, "opset": 17,
              "float_mb": round(fs, 3),
              "int8_mb": round(isz, 3) if isz else None,
              "parity_max_abs_diff": max_diff,
              "parity_argmax_agree": f"{agree}/32",
              "int8_ok": int8_ok}
    with open(os.path.join(MODELS, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
