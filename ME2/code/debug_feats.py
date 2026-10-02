import numpy as np, io, soundfile as sf, librosa, pandas as pd
SR = 16000

def loudest_crop(y, sr=SR, dur=1.0):
    win = int(sr * dur)
    print(f"  input len={len(y)}, win={win}")
    if len(y) <= win:
        print(f"  -> padding to {win}")
        return np.pad(y, (0, win - len(y))) if len(y) < win else y
    hop = win // 4
    best_start, best_amp = 0, -1
    for start in range(0, len(y) - win + 1, hop):
        amp = np.max(np.abs(y[start:start+win]))
        if amp > best_amp:
            best_amp, best_start = amp, start
    result = y[best_start:best_start+win]
    print(f"  -> cropped to {len(result)} (start={best_start})")
    return result

df = pd.read_parquet("/Users/yannamii/sandbox/ME2/data/hf/data/holdout-00000-of-00001.parquet")
a = df["audio"][0]
data, sr = sf.read(io.BytesIO(a["bytes"]))
print("raw:", len(data), "sr:", sr)
data = loudest_crop(data.astype(np.float32))
print("after crop:", len(data))
mfc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=40, hop_length=512)
print("mfcc shape:", mfc.shape)
# Test pad
N_FRAMES = 100
def pad1(x):
    if x.shape[1] >= N_FRAMES:
        return x[:, :N_FRAMES].T
    return np.pad(x.T, ((0,0),(0,N_FRAMES-x.shape[1])))
print("padded:", pad1(mfc).shape)
