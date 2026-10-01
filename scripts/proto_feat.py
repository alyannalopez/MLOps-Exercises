"""Build a PyTorch feature-extractor module that EXACTLY reproduces
librosa's log-mel (16k, 25ms/10ms, 80 mel, ref=max), then export to ONNX.

This is the canonical, reliable way to get a portable feature graph — no
hand-rolled ONNX ops.
"""
import numpy as np, torch, torch.nn as nn, librosa, tempfile, os
import onnx, onnxruntime as ort

SR=16000; WIN=400; HOP=160; NFFT=2048; N_MELS=80; T_TARGET=101; N=16000
# librosa center=True yields 1 + N//HOP = 101 frames for N=16000
T=1+N//HOP

def mel_filters_np():
    # EXACT librosa melspectrogram defaults: n_fft=2048, norm='slaney', fmax=sr/2
    import librosa.filters as lf
    fb = lf.mel(sr=SR, n_fft=NFFT, n_mels=N_MELS, fmin=0.0, fmax=SR/2,
                htk=False, norm='slaney')
    return fb.astype(np.float32)

class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        # librosa default window is scipy 'hann' (symmetric), not torch.hann_window
        import scipy.signal as sp
        self.register_buffer("win", torch.from_numpy(sp.get_window('hann', WIN).astype(np.float32)))
        self.register_buffer("melfb", torch.from_numpy(mel_filters_np()))
    def forward(self, y):            # y: (B, N)
        S = torch.stft(y, n_fft=NFFT, hop_length=HOP, win_length=WIN,
                       window=self.win, return_complex=True, center=True)
        S = S[..., :NFFT//2+1]              # (B, F, T')
        S = S.permute(0,2,1)               # (B, T', F)
        if S.size(1) < T:
            pad = S.new_zeros(S.size(0), T-S.size(1), S.size(2))
            S = torch.cat([S, pad], dim=1)
        S = S[:, :T]                       # (B, T, F)
        mel = torch.matmul(S.abs().pow(2), self.melfb.t())   # (B,T,N_MELS)
        logmel = 20*torch.log10(mel + 1e-10)
        peak = logmel.max(dim=1, keepdim=True).values
        logmel = logmel - peak
        logmel = torch.clamp(logmel, min=-80.0)   # power_to_db top_db=80
        return logmel

def main():
    m=LogMel().eval()
    x=torch.randn(1,N)
    with torch.no_grad():
        out=m(x)
    print("torch out", tuple(out.shape))
    # compare to librosa
    y=x.numpy()[0]
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_mels=N_MELS,hop_length=HOP,win_length=WIN)
    ref=librosa.power_to_db(S,ref=np.max).T.astype(np.float32)
    mm=min(out.shape[1],ref.shape[0])
    d=np.abs(out.numpy()[0,:mm]-ref[:mm])
    print(f"torch vs librosa maxdiff={d.max():.4f} mean={d.mean():.4f}")

    # export to ONNX
    p=tempfile.mktemp(suffix=".onnx")
    torch.onnx.export(m, (x,), p, opset_version=17,
                      input_names=["y"], output_names=["logmel"],
                      dynamic_axes={"y":{0:"B"},"logmel":{0:"B"}})
    mm2=onnx.load(p); mm2.ir_version=9
    onnx.save(mm2,p)
    sess=ort.InferenceSession(p,providers=["CPUExecutionProvider"])
    rng=np.random.default_rng(0); worst=0
    for i in range(5):
        yn=(rng.standard_normal(N).astype(np.float32))*0.05
        mine=sess.run(None,{"y":yn[None]})[0][0]
        S=librosa.feature.melspectrogram(y=yn,sr=SR,n_mels=N_MELS,hop_length=HOP,win_length=WIN)
        ref=librosa.power_to_db(S,ref=np.max).T.astype(np.float32)
        mm=min(mine.shape[0],ref.shape[0])
        dd=np.abs(mine[:mm]-ref[:mm]); worst=max(worst,float(dd.max()))
        print(f"clip{i}: onnx{mine.shape} maxdiff={dd.max():.4f} mean={dd.mean():.4f}")
    print("WORST",worst)
    print("SAVED",p)

if __name__=="__main__":
    main()
