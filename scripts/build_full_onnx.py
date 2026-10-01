"""Compose the exact librosa-matching feature extractor with the trained CRNN
into a SINGLE PCM -> logits ONNX model for on-device deployment.

The feature module (proto_feat.LogMel) reproduces librosa's log-mel exactly
(n_fft=2048, win_length=400, hop=160, center=True, norm='slaney', top_db=80),
so the Pi needs only onnxruntime + numpy — no librosa, no torch.
"""
from __future__ import annotations
import os, sys, glob, json, time
import numpy as np, torch, onnx, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import proto_feat as P
import models as M

TAG = "crnn_1790586409"
MODELDIR = os.path.join(HERE, "..", "models")
N = P.N  # 16000

CLASS_NAMES = {0:"REJECT",1:"PLAY_MUSIC",2:"QUESTION_SEARCH",3:"LIGHTS_ON_OFF",
               4:"DIM_COLOR_LIGHTS",5:"SET_TIMER",6:"SET_ALARM",7:"THERMOSTAT",
               8:"MEDIA_CONTROL",9:"REMINDERS_LISTS",10:"CALLS_MESSAGING"}


class Full(torch.nn.Module):
    def __init__(self, crnn):
        super().__init__()
        self.feat = P.LogMel()
        self.crnn = crnn
    def forward(self, y):          # y: (B, N) float32 in [-1,1]
        return self.crnn(self.feat(y))


def load_crnn(tag):
    state = torch.load(os.path.join(MODELDIR, f"{tag}_state.pt"), map_location="cpu")
    m = M.build("crnn")
    m.load_state_dict(state)
    m.eval()
    return m


def main():
    crnn = load_crnn(TAG)
    full = Full(crnn).eval()
    x = torch.randn(1, N)
    with torch.no_grad():
        out = full(x)
    print("Full model out:", tuple(out.shape))

    out_path = os.path.join(MODELDIR, "crnn_full.onnx")
    try:
        torch.onnx.export(full, (x,), out_path, opset_version=17,
                          input_names=["pcm"], output_names=["logits"],
                          dynamic_axes={"pcm": {0: "B"}, "logits": {0: "B"}},
                          dynamo=False)
    except TypeError:
        # older torch without dynamo flag
        torch.onnx.export(full, (x,), out_path, opset_version=17,
                          input_names=["pcm"], output_names=["logits"],
                          dynamic_axes={"pcm": {0: "B"}, "logits": {0: "B"}})
    mm = onnx.load(out_path)
    mm.ir_version = 9
    onnx.save(mm, out_path)
    print("saved", out_path, os.path.getsize(out_path) / 1024, "KB")

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])

    # ---- accuracy on real Option B clips ----
    wavs = sorted(glob.glob(os.path.join(HERE, "..", "data", "raw", "optionB", "**", "*.wav"), recursive=True))
    fi = {"PLAY_MUSIC":"PLAY_MUSIC","WEATHER":"QUESTION_SEARCH","TIME":"QUESTION_SEARCH",
          "LIGHT_ON":"LIGHTS_ON_OFF","LIGHT_OFF":"LIGHTS_ON_OFF","DIM":"DIM_COLOR_LIGHTS",
          "COLOR":"DIM_COLOR_LIGHTS","TIMER":"SET_TIMER","ALARM":"SET_ALARM",
          "TEMPERATURE":"THERMOSTAT","PAUSE":"MEDIA_CONTROL","STOP":"MEDIA_CONTROL",
          "NEXT":"MEDIA_CONTROL","VOLUME_UP":"MEDIA_CONTROL","VOLUME_DOWN":"MEDIA_CONTROL",
          "REMIND":"REMINDERS_LISTS","LIST":"REMINDERS_LISTS","CALL":"CALLS_MESSAGING",
          "MESSAGE":"CALLS_MESSAGING"}
    import wave
    ok = tot = 0
    lat = []
    for wp in wavs[:200]:
        with wave.open(wp, "rb") as w:
            sr = w.getframerate(); n = w.getnframes()
            pcm = np.frombuffer(w.readframes(n), dtype=np.int16)
        if sr != 16000:
            idx = np.linspace(0, len(pcm)-1, int(len(pcm)*16000/sr)).astype(int)
            pcm = pcm[idx]
        y = pcm.astype(np.float32) / 32768.0
        if len(y) >= N:
            y = y[:N]
        else:
            y = np.pad(y, (0, N-len(y)))
        t0 = time.perf_counter()
        lg = sess.run(None, {"pcm": y[None]})[0][0]
        lat.append((time.perf_counter()-t0)*1e3)
        c = int(np.argmax(lg)); pred = CLASS_NAMES[c]
        exp = None
        for seg in wp.split(os.sep):
            for k, v in fi.items():
                if k in seg.upper():
                    exp = v; break
            if exp: break
        if exp:
            tot += 1
            if pred == exp: ok += 1
    lat = np.array(lat)
    print(f"\nFULL ONNX (PCM->logits) on {tot} real clips: accuracy={ok/tot:.3f}")
    print(f"latency ms: mean={lat.mean():.1f} p50={np.percentile(lat,50):.1f} p95={np.percentile(lat,95):.1f} max={lat.max():.1f}")


if __name__ == "__main__":
    main()
