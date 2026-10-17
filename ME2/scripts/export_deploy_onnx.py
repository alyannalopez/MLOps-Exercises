"""Stage 6: export the Stage-5 winning VCM (conformer_s) to ONNX for the
Raspberry Pi runtime, and verify numerical parity against the ORIGINAL
PyTorch model on real held-out test clips.

Export path (torch 2.14 + onnxscript 0.7.2):
  - Use the default dynamo-based exporter (torch.onnx.export with
    dynamo=True, the default). It decomposes nn.MultiheadAttention's fused
    SDPA into plain ONNX ops and handles the stem's BatchNorm2d correctly.
  - Requires `pip install onnxscript` (now in requirements.txt).
  - Dynamic batch axis so the runtime can feed any-length clips (the
    feature pipeline crops/pads to 100 frames internally).

CRITICAL: parity is checked against the ORIGINAL model (unmodified
weights, no module swaps). An earlier attempt swapped the MHA for a
hand-unrolled module that was NOT numerically identical (29/50 argmax
agreement) and validated the ONNX graph against that broken reference,
producing a silently-wrong graph. Do not reintroduce module swaps.

Outputs:
  models/conformer_s_float.onnx   (primary, fp32)
  models/conformer_s_int8.onnx    (optional, dynamic int8 quantization)

Verification (all numbers real, written to results/onnx_parity.json):
  - max abs diff on logits vs original PyTorch (random + real clips)
  - argmax agreement on 128 real held-out test clips (batch=1)
  - file sizes
  - int8 argmax agreement on the same 128 clips
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vcm.models import build, count_params  # noqa: E402
from vcm.features import log_mel_spectrogram  # noqa: E402
from vcm.slots import N_SLOTS  # noqa: E402

N_CLASSES, N_MELS, N_FRAMES = 20, 40, 100
CKPT = ROOT / "models" / "conformer_s_seed0" / "best.pt"
OUT_FLOAT = ROOT / "models" / "conformer_s_float.onnx"
OUT_INT8 = ROOT / "models" / "conformer_s_int8.onnx"
OUT_JSON = ROOT / "results" / "onnx_parity.json"


def load_model() -> torch.nn.Module:
    """The ORIGINAL model, unmodified. This is the parity reference."""
    model = build("conformer_s", n_classes=N_CLASSES, n_mels=N_MELS, n_frames=N_FRAMES,
                  n_slots=N_SLOTS)
    sd = torch.load(CKPT, weights_only=True)["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def real_feats(n: int) -> np.ndarray:
    """Log-mel features for n real held-out test clips: (n, 1, 40, 100)."""
    z = np.load(ROOT / "data" / "packs" / "test.npz", allow_pickle=True)
    ws = z["waveforms"][:n]
    out = np.empty((n, 1, N_MELS, N_FRAMES), dtype=np.float32)
    for i, w in enumerate(ws):
        out[i] = log_mel_spectrogram(w, n_mels=N_MELS)[None]
    return out


def main():
    t0 = time.time()
    model = load_model()
    n_params = count_params(model)

    import onnxruntime as ort

    # --- export float (dynamo exporter, dynamic batch) ---
    dummy = torch.randn(1, 1, N_MELS, N_FRAMES)
    OUT_FLOAT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(OUT_FLOAT),
        input_names=["logmel"], output_names=["intent_logits", "slot_logits"],
        dynamic_shapes={"x": {0: "batch"}},
        opset_version=18,
    )
    print("onnx convert OK")

    # The dynamo exporter externalizes weights to a .onnx.data companion and
    # bumps to opset 18. Merge the weights back in for a single self-contained
    # file (what the Pi bundle ships).
    import onnx as _onnx
    _m = _onnx.load(str(OUT_FLOAT), load_external_data=True)
    for _init in _m.graph.initializer:
        _init.ClearField("external_data")
    _onnx.save(_m, str(OUT_FLOAT))
    _data = OUT_FLOAT.with_suffix(".onnx.data")
    if _data.exists():
        _data.unlink()
    print(f"merged -> {OUT_FLOAT.stat().st_size} bytes (single file)")

    sess = ort.InferenceSession(str(OUT_FLOAT), providers=["CPUExecutionProvider"])

    # --- parity on random input ---
    x = np.random.randn(1, 1, N_MELS, N_FRAMES).astype(np.float32)
    with torch.no_grad():
        ref_intent, ref_slot = model(torch.from_numpy(x))
        ref_intent, ref_slot = ref_intent[0].numpy(), ref_slot[0].numpy()
    got_intent, got_slot = sess.run(None, {"logmel": x})
    got_intent, got_slot = got_intent[0], got_slot[0]
    rand_diff = float(max(np.abs(ref_intent - got_intent).max(),
                          np.abs(ref_slot - got_slot).max()))

    # --- parity on real clips, batch=1 (as the Pi runtime runs) ---
    X = real_feats(128)
    agree, max_diff = 0, 0.0
    for i in range(len(X)):
        with torch.no_grad():
            ref_i, ref_s = model(torch.from_numpy(X[i:i + 1]))
        got_i, got_s = sess.run(None, {"logmel": X[i:i + 1]})
        max_diff = max(max_diff, float(np.abs(ref_i[0].numpy() - got_i[0]).max()),
                       float(np.abs(ref_s[0].numpy() - got_s[0]).max()))
        if int(ref_i.argmax()) == int(got_i[0].argmax()):
            agree += 1
    real_check = {
        "n_clips": int(len(X)),
        "argmax_agreement": agree / len(X),
        "max_abs_diff": max_diff,
    }

    # --- int8 dynamic quantization (optional artifact) ---
    # Workaround for ORT quantize_dynamic shape-inference bug: the dynamo
    # exporter embeds value_info with static batch=1 shapes that conflict
    # with the dynamic batch axis during the quantizer's internal shape
    # inference. Fix: strip all value_info and make output dims symbolic.
    int8_info = {}
    try:
        import onnx as _onnx3
        from onnxruntime.quantization import QuantType, quantize_dynamic

        _qm = _onnx3.load(str(OUT_FLOAT))
        del _qm.graph.value_info[:]  # remove static-shape annotations
        for _out in _qm.graph.output:
            del _out.type.tensor_type.shape.dim[:]
            for _ in range(2):
                _d = _out.type.tensor_type.shape.dim.add()
                _d.dim_param = "dyn"
        _tmp_q = ROOT / "models" / "_quant_stripped.onnx"
        _onnx3.save(_qm, str(_tmp_q))
        quantize_dynamic(str(_tmp_q), str(OUT_INT8), weight_type=QuantType.QInt8)
        _tmp_q.unlink()
        sess8 = ort.InferenceSession(str(OUT_INT8), providers=["CPUExecutionProvider"])
        agree8, maxdiff8 = 0, 0.0
        for i in range(len(X)):
            with torch.no_grad():
                ref_i, ref_s = model(torch.from_numpy(X[i:i + 1]))
            got8_i, got8_s = sess8.run(None, {"logmel": X[i:i + 1]})
            maxdiff8 = max(maxdiff8,
                           float(np.abs(ref_i[0].numpy() - got8_i[0]).max()),
                           float(np.abs(ref_s[0].numpy() - got8_s[0]).max()))
            if int(ref_i.argmax()) == int(got8_i[0].argmax()):
                agree8 += 1
        int8_info = {
            "path": str(OUT_INT8),
            "size_bytes": OUT_INT8.stat().st_size,
            "argmax_agreement": agree8 / len(X),
            "max_abs_diff": maxdiff8,
        }
    except Exception as e:  # quantization is optional; record the failure honestly
        int8_info = {"error": repr(e)}

    out = {
        "arch": "conformer_s",
        "checkpoint": str(CKPT),
        "n_params": n_params,
        "export_method": "torch.onnx.export dynamo exporter (opset 14, dynamic batch), onnxscript 0.7.2",
        "reference": "ORIGINAL PyTorch model (no module swaps)",
        "float_onnx": {
            "path": str(OUT_FLOAT),
            "size_bytes": OUT_FLOAT.stat().st_size,
            "random_max_abs_diff": rand_diff,
            **real_check,
        },
        "int8_onnx": int8_info,
        "export_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
