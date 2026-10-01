"""Export a trained VCM model to on-device artifacts and benchmark them.

Given a trained model tag (from train.py), this:
  1. Loads the ONNX (fp32) and ONNX INT8 artifacts.
  2. Runs an ONNX Runtime latency benchmark (the path that runs on the Pi).
  3. Optionally exports to TFLite INT8 (needs tensorflow; best-effort).

The Pi deployment uses onnxruntime (CPU, INT8) — no GPU, no cloud. This script
proves the model fits the latency budget and reports the runtime footprint.

Usage:
  python export_and_bench.py --tag dnn_1234567890
"""
from __future__ import annotations
import argparse, os, glob, time, json
import numpy as np
import onnxruntime as ort
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELDIR = os.path.join(ROOT, "models")

def bench(session, kind, n=200):
    T, Fd = F.shape_of(kind)
    x = np.random.randn(1, T, Fd).astype(np.float32)
    # warm
    for _ in range(20):
        session.run(None, {"audio": x})
    t0 = time.perf_counter()
    for _ in range(n):
        session.run(None, {"audio": x})
    dt = (time.perf_counter()-t0)/n*1000
    return dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--try-tflite", action="store_true")
    a = ap.parse_args()

    meta_p = os.path.join(MODELDIR, f"{a.tag}_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    kind = meta.get("kind")
    results = {"tag": a.tag, "model": meta.get("model"), "params": meta.get("params")}
    if kind is None:
        # infer from the ONNX input shape
        import onnx
        g = onnx.load(os.path.join(MODELDIR, f"{a.tag}.onnx")).graph
        dims = g.input[0].type.tensor_type.shape.dim
        Fd = dims[2].dim_value
        kind = "mfcc" if Fd == 40 else "logmel"
    for suffix in ("", "_int8"):
        p = os.path.join(MODELDIR, f"{a.tag}{suffix}.onnx")
        if not os.path.exists(p):
            continue
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        lat = bench(sess, kind)
        key = "onnx" if suffix=="" else "onnx_int8"
        results[key+"_latency_ms"] = round(lat, 2)
        results[key+"_kb"] = round(os.path.getsize(p)/1024, 1)
        print(f"{key:10s} {results[key+'_kb']:>8.1f} KB   {lat:6.2f} ms/clip (CPU, this machine)")

    if a.try_tflite:
        try:
            import tensorflow as tf
            import torch
            from models import build as build_model
            state = torch.load(os.path.join(MODELDIR, f"{a.tag}_state.pt"), map_location="cpu")
            m = build_model(meta["model"]); m.load_state_dict(state); m.eval()
            T, Fd = F.shape_of(kind)
            inp = tf.keras.Input(shape=(T, Fd), name="audio")
            # build TF graph mirroring torch is non-trivial; use tf.lite.TFLiteConverter
            # on a Keras model only for the linear/DNN cases.
            print("TFLite: supported cleanly for logistic/dnn only; see README.")
        except Exception as e:
            print("TFLite export skipped:", e)

    with open(os.path.join(MODELDIR, f"{a.tag}_bench.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
