"""Re-export a saved model state dict to ONNX (fp32 + INT8) without retraining.

Usage:
  python reexport.py --model dnn --state models/dnn_XXXX_state.pt --tag dnn_XXXX
"""
from __future__ import annotations
import argparse, os, json
import numpy as np, torch
import features as F
from models import build as build_model, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELDIR = os.path.join(ROOT, "models")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["logistic","dnn","cnn1d","crnn"])
    ap.add_argument("--state", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-acc", type=float, default=None)
    ap.add_argument("--val-acc", type=float, default=None)
    a = ap.parse_args()

    kind = "mfcc" if a.model in ("logistic","dnn") else "logmel"
    m = build_model(a.model)
    sd = torch.load(a.state, map_location="cpu")
    m.load_state_dict(sd); m = m.cpu().eval()
    T, Fd = F.shape_of(kind)
    dummy = torch.randn(1, T, Fd)
    onnx_p = os.path.join(MODELDIR, f"{a.tag}.onnx")
    torch.onnx.export(m, dummy, onnx_p, opset_version=13, dynamo=False,
                      input_names=["audio"], output_names=["logits"],
                      dynamic_axes={"audio":{0:"batch"},"logits":{0:"batch"}})
    from onnxruntime.quantization import quantize_dynamic, QuantType
    q_p = os.path.join(MODELDIR, f"{a.tag}_int8.onnx")
    quantize_dynamic(onnx_p, q_p, weight_type=QuantType.QInt8)
    meta = {"model":a.model,"kind":kind,"params":count_params(m),
            "test_acc":a.test_acc,"val_acc":a.val_acc,
            "onnx":onnx_p,"onnx_int8":q_p,
            "onnx_kb":round(os.path.getsize(onnx_p)/1024,1),
            "onnx_int8_kb":round(os.path.getsize(q_p)/1024,1),
            "feature_T":T,"feature_F":Fd}
    json.dump(meta, open(os.path.join(MODELDIR,f"{a.tag}_meta.json"),"w"), indent=2)
    print(f"{a.tag}: onnx={meta['onnx_kb']}KB int8={meta['onnx_int8_kb']}KB params={meta['params']:,}")

if __name__ == "__main__":
    main()
