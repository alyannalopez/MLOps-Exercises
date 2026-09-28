"""Five candidate VCM architectures, ordered smallest-first.

Every model shares ONE contract so they're directly comparable and the on-device
inference path is identical:

    INPUT : (batch, T, F)  float32 feature tensor
    OUTPUT: (batch, 11)    logits  ->  classes 0..9 = intents 1..10, 10 = REJECT

Feature kinds:
    "mfcc"   -> (T, 40)   used by: logistic, dnn
    "logmel" -> (T, 80)   used by: cnn1d, crnn, (also usable by dnn)

Parameter budgets (approx, fp32):
    A. Logistic (MFCC mean-pool)        ~ 40*11      =   440      (<1 KB)
    B. DNN  (MFCC mean-pool)            ~ 40->128->64->11  ~ 9.5k (~40 KB)
    C. 1-D CNN (log-mel)                ~ 4 conv blocks + fc  ~ 110k (~450 KB)
    D. CRNN  (log-mel)                  ~ CNN + BiLSTM(128)    ~ 480k (~2 MB)
    E. PocketSphinx-style grammar       ~ GMM-HMM, exported separately (see pocketsphinx_note.md)

All deep nets use the SAME trunk style so the only variable is capacity.
"""
from __future__ import annotations
import torch
import torch.nn as nn

N_CLASSES = 11          # 10 intents + rejection
REJECT    = 10

def _mean_pool(x: torch.Tensor) -> torch.Tensor:
    """(B,T,F) -> (B,F) temporal mean pooling (cheap, robust to length)."""
    return x.mean(dim=1)

# --------------------------------------------------------------------------- #
# A. LOGISTIC REGRESSION on mean-pooled MFCC                                  #
# --------------------------------------------------------------------------- #
class LogisticMFCC(nn.Module):
    def __init__(self, in_dim: int = 40):
        super().__init__()
        self.fc = nn.Linear(in_dim, N_CLASSES)
    def forward(self, x):                       # x: (B,T,F)
        return self.fc(_mean_pool(x))

# --------------------------------------------------------------------------- #
# B. SMALL DNN on mean-pooled MFCC                                            #
# --------------------------------------------------------------------------- #
class DNNSmall(nn.Module):
    def __init__(self, in_dim: int = 40, h1: int = 128, h2: int = 64, drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h1, h2),     nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h2, N_CLASSES),
        )
    def forward(self, x):
        return self.net(_mean_pool(x))

# --------------------------------------------------------------------------- #
# C. 1-D CNN on log-mel (temporal convolutions over the spectrogram)          #
# --------------------------------------------------------------------------- #
class CNN1D(nn.Module):
    """Input (B, F, T). 4 stacked conv blocks with residual-ish skips, global
    avg pool, FC head. ~110k params."""
    def __init__(self, in_ch: int = 80, base: int = 32, depth: int = 4,
                 ks: int = 5, drop: float = 0.3):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(depth):
            out_ch = base * (2 ** i)
            layers += [
                nn.Conv1d(ch, out_ch, ks, padding=ks//2),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(drop),
            ]
            ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(ch, 64), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(64, N_CLASSES))
    def forward(self, x):                       # x: (B,T,F)
        x = x.transpose(1, 2)                   # -> (B,F,T)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# --------------------------------------------------------------------------- #
# D. CRNN on log-mel (CNN trunk + BiLSTM sequence layer)                      #
# --------------------------------------------------------------------------- #
class CRNN(nn.Module):
    """CNN extracts local spectral patterns, BiLSTM models temporal dynamics.
    ~480k params. Input (B,T,F)."""
    def __init__(self, in_ch: int = 80, base: int = 32, lstm_h: int = 128,
                 ks: int = 5, drop: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, base,     ks, padding=ks//2), nn.BatchNorm1d(base), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(base, base*2,    ks, padding=ks//2), nn.BatchNorm1d(base*2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(base*2, base*4,  ks, padding=ks//2), nn.BatchNorm1d(base*4), nn.ReLU(),
        )
        self.lstm = nn.LSTM(base*4, lstm_h, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(lstm_h*2, N_CLASSES))
    def forward(self, x):                       # x: (B,T,F)
        x = x.transpose(1, 2)                   # (B,F,T)
        x = self.cnn(x)                         # (B,base*4,T')
        x = x.transpose(1, 2)                   # (B,T',C)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)                       # mean over time
        return self.head(x)

MODELS = {
    "logistic": (LogisticMFCC, "mfcc"),
    "dnn":      (DNNSmall,     "mfcc"),
    "cnn1d":    (CNN1D,        "logmel"),
    "crnn":     (CRNN,         "logmel"),
}

def build(name: str, in_dim: int | None = None) -> nn.Module:
    cls, kind = MODELS[name]
    if name in ("logistic","dnn"):
        return cls(in_dim=in_dim or 40)
    return cls()

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

if __name__ == "__main__":
    for name,(cls,kind) in MODELS.items():
        m = build(name)
        T = 61
        x = torch.randn(2, T, 40 if kind=="mfcc" else 80)
        out = m(x)
        print(f"{name:9s} {kind:6s} params={count_params(m):>8,}  out={tuple(out.shape)}")
