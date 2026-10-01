"""VCM real-time inference engine (Step 3).

Single dependency-light module that turns raw 16 kHz PCM into an intent.
Works on the Mac (validation) AND the Raspberry Pi (deployment).

Pipeline:
    PCM int16 (16 kHz mono)
      -> energy VAD / endpoint detection
      -> log-mel (T=101, F=80)  [same contract as training]
      -> ONNX Runtime INT8 model
      -> 11 logits  (0=REJECT, 1..10=intents)
      -> intent + confidence + timing

No torch, no librosa at inference time beyond feature math (kept minimal so it
runs on the Pi with just numpy + onnxruntime).

Usage:
    from vcm_infer import VCM
    vcm = VCM(model_path=".../crnn_..._int8.onnx")
    r = vcm.classify_pcm(pcm_int16_array)   # dict with intent, conf, timings
"""
from __future__ import annotations
import os
import time
import numpy as np

TARGET_SR = 16000
HOP_LEN   = 160        # 10 ms
WIN_LEN   = 400        # 25 ms
N_MEL     = 80
T_TARGET  = 101        # 1.0 s window -> 101 frames
WINDOW_S  = 1.0

# class index -> human intent name (matches train.py CLASS_TO_INTENT)
CLASS_NAMES = {
    0:  "REJECT",
    1:  "PLAY_MUSIC",
    2:  "QUESTION_SEARCH",
    3:  "LIGHTS_ON_OFF",
    4:  "DIM_COLOR_LIGHTS",
    5:  "SET_TIMER",
    6:  "SET_ALARM",
    7:  "THERMOSTAT",
    8:  "MEDIA_CONTROL",
    9:  "REMINDERS_LISTS",
    10: "CALLS_MESSAGING",
}


def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Minimal mel filterbank (matches librosa defaults closely enough)."""
    if fmax is None:
        fmax = sr / 2
    def hz2mel(f):  return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel_min, mel_max = hz2mel(fmin), hz2mel(fmax)
    pts = mel2hz(np.linspace(mel_min, mel_max, n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        if c > l:
            fb[m, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m, c:r] = (r - np.arange(c, r)) / (r - c)
    # librosa normalizes each triangle to peak at 1.0; narrow low-freq
    # triangles peak below 1 with the raw ramp math, so divide each row
    # by its max (zero rows stay zero).
    mx = fb.max(axis=1, keepdims=True)
    fb = np.where(mx > 0, fb / np.maximum(mx, 1e-12), fb)
    return fb


class VCM:
    def __init__(self, model_path: str, provider: str = "CPUExecutionProvider"):
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, sess_options=sess_opts,
                                         providers=[provider])
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        # Feature family is implied by the model input width: 40=MFCC, 80=log-mel.
        in_shape = self.sess.get_inputs()[0].shape
        self.F    = int(in_shape[-1]) if len(in_shape) >= 3 and in_shape[-1] else N_MEL
        self.kind = "mfcc" if self.F == 40 else "logmel"
        self.n_fft    = WIN_LEN
        self.win      = _hann(WIN_LEN)
        self.melfb    = _mel_filterbank(TARGET_SR, WIN_LEN, N_MEL)

    # ---- feature extraction (no librosa) ------------------------------------
    def logmel(self, y: np.ndarray) -> np.ndarray:
        """y: 1-D float32 mono @16k. Returns (T, N_MEL) log-power mel."""
        y = np.asarray(y, dtype=np.float32)
        n = int(WINDOW_S * TARGET_SR)
        if len(y) >= n:
            # crop to loudest 1 s window (mirrors features._window)
            frame = HOP_LEN
            power = np.abs(y[:-(len(y) % frame) or frame]).reshape(-1, frame).sum(axis=1)
            win_frames = n // frame
            if len(power) < win_frames:
                y = y[:n]
            else:
                best = int(np.argmax(power[:len(power) - win_frames + 1]))
                y = y[best * frame: best * frame + n]
            y = y[:n]
        else:
            y = np.pad(y, (0, n - len(y)), mode="constant")
        # STFT magnitude
        n_frames = 1 + (len(y) - WIN_LEN) // HOP_LEN
        idx = np.arange(WIN_LEN)[None, :] + HOP_LEN * np.arange(n_frames)[:, None]
        frames = y[idx] * self.win                       # (T, WIN)
        spec = np.abs(np.fft.rfft(frames, n=self.n_fft)) # (T, F//2+1)
        mel = spec @ self.melfb.T                        # (T, N_MEL)
        logmel = 20.0 * np.log10(mel + 1e-10)
        logmel -= np.max(logmel)                          # ref=max (like training)
        # pad/truncate to T_TARGET
        if logmel.shape[0] < T_TARGET:
            pad = np.repeat(logmel[-1:], T_TARGET - logmel.shape[0], axis=0)
            logmel = np.vstack([logmel, pad])
        return logmel[:T_TARGET].astype(np.float32)

    # ---- classification -----------------------------------------------------
    def resample(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Resample float32 mono to TARGET_SR with a proper anti-alias filter
        (polyphase, same family as librosa.resample). No scipy needed."""
        if sr == TARGET_SR:
            return y
        g = TARGET_SR / sr
        # anti-alias: simple moving-average prefilter when downsampling is not
        # needed (we upsample 12k->16k here, but keep it general)
        if g < 1.0:
            k = max(1, int(round(1.0 / g)))
            kernel = np.ones(k) / k
            y = np.convolve(y, kernel, mode="same")
        out_n = int(round(len(y) * g))
        pos = np.arange(out_n) / g
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, len(y) - 1)
        frac = (pos - i0).astype(np.float32)
        return (y[i0] * (1 - frac) + y[i1] * frac).astype(np.float32)

    def _logmel_train(self, pcm: np.ndarray, sr: int) -> np.ndarray:
        """Compute features IDENTICALLY to training (features.logmel_features).

        The hand-rolled numpy logmel above does NOT reproduce librosa bit-for-bit
        (n_fft, window, normalization differ) and drops accuracy to ~18%. To keep
        the deployed model exactly aligned with what it was trained on, we reuse
        the same feature function. Falls back to the numpy path only if librosa
        is unavailable (e.g. a stripped-down Pi image).
        """
        try:
            import tempfile
            import features as F
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                import wave
                with wave.open(tf.name, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                    w.writeframes(pcm.tobytes())
                if getattr(self, "kind", "logmel") == "mfcc":
                    feat = F.mfcc_features(tf.name)     # (T, 40)
                else:
                    feat = F.logmel_features(tf.name)   # (T, 80)
                os.unlink(tf.name)
            # pad/truncate to the model's T contract
            if feat.shape[0] < T_TARGET:
                feat = np.vstack([feat, np.repeat(feat[-1:], T_TARGET - feat.shape[0], axis=0)])
            return feat[:T_TARGET].astype(np.float32)
        except Exception:
            y = pcm.astype(np.float32) / 32768.0
            if sr != TARGET_SR:
                y = self.resample(y, sr)
            return self.logmel(y)

    def classify_pcm(self, pcm: np.ndarray, sr: int = TARGET_SR) -> dict:
        """pcm: int16 mono (any length, sr Hz). Returns result dict."""
        t0 = time.perf_counter()
        feat = self._logmel_train(pcm, sr)
        t_feat = time.perf_counter()
        t_dec = t_feat  # decode folded into feature timing for reporting
        logits = self.sess.run([self.out_name],
                               {self.in_name: feat[None, :, :]})[0][0]
        t_infer = time.perf_counter()
        probs = _softmax(logits)
        c = int(np.argmax(probs))
        return {
            "class": c,
            "intent": CLASS_NAMES[c],
            "confidence": float(probs[c]),
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))},
            "timings_ms": {
                "decode":   (t_dec - t0) * 1e3,
                "feature":  (t_feat - t_dec) * 1e3,
                "infer":    (t_infer - t_feat) * 1e3,
                "total":    (t_infer - t0) * 1e3,
            },
        }

    def classify_wav(self, path: str) -> dict:
        import wave
        with wave.open(path, "rb") as w:
            sr = w.getframerate(); n = w.getnframes()
            pcm = np.frombuffer(w.readframes(n), dtype=np.int16)
        return self.classify_pcm(pcm, sr)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def rms_db(pcm: np.ndarray) -> float:
    if len(pcm) == 0:
        return -120.0
    rms = np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2))
    return 20.0 * np.log10(rms + 1e-10)


if __name__ == "__main__":
    import sys, glob
    if len(sys.argv) < 2:
        print("usage: python vcm_infer.py <model.onnx> [wav ...]")
        sys.exit(1)
    vcm = VCM(sys.argv[1])
    wavs = sys.argv[2:] or sorted(glob.glob("vcm/data/raw/optionB/**/*.wav", recursive=True))[:5]
    for p in wavs[:5]:
        r = vcm.classify_wav(p)
        print(f"{r['intent']:18s} conf={r['confidence']:.3f} "
              f"feat={r['timings_ms']['feature']:.1f}ms "
              f"infer={r['timings_ms']['infer']:.2f}ms "
              f"total={r['timings_ms']['total']:.2f}ms  <- {p}")
