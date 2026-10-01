"""Comprehensive benchmark: retrained 4 models on full 2,883-clip test set.

Metrics per model:
  - Overall / per-intent accuracy (recall), precision, F1
  - Command recall per sub-command (filename-derived, Option B)
  - Task completion / success rate (clean vs noisy)
  - WER (option B clips: filename words vs whisper transcription)
  - Latency: feature ms, forward ms, total ms (p50/p95/max)
  - Efficiency: params, MB, MACs/clip, throughput clips/s

Outputs:
  models_retrain/benchmark_results.json
  models_retrain/VCM_Benchmark.xlsx
  models_retrain/plots/benchmark_*.png
"""
import sys, os, time, json, re, io, shutil, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE, "models_retrain")
TEST_CSV = os.path.join(BASE, "data/processed/splits/test.csv")
OUT_JSON = os.path.join(MODELS_DIR, "benchmark_results.json")
OUT_XLSX = os.path.join(MODELS_DIR, "VCM_Benchmark.xlsx")
PLOTS = os.path.join(MODELS_DIR, "plots")
os.makedirs(PLOTS, exist_ok=True)

CLASS_NAMES = {0:"REJECT",1:"PLAY_MUSIC",2:"QUESTION_SEARCH",3:"LIGHTS_ON_OFF",
               4:"DIM_COLOR_LIGHTS",5:"SET_TIMER",6:"SET_ALARM",7:"THERMOSTAT",
               8:"MEDIA_CONTROL",9:"REMINDERS_LISTS",10:"CALLS_MESSAGING"}
N_CLASSES = 11
TARGET_SR, HOP, WIN, N_MEL = 16000, 160, 400, 80
N_MFCC, T_TARGET = 40, 101

# ---------------------------------------------------------------- models
from models import LogisticMFCC, DNNSmall, CNN1D, CRNN  # noqa: E402

def make_model(name, kind, T=T_TARGET):
    if name == "logistic":  return LogisticMFCC()
    if name == "dnn":       return DNNSmall()
    if name == "cnn1d":     return CNN1D()
    if name == "crnn":      return CRNN()
    raise ValueError(name)

def load_best(name):
    """Load best-val checkpoint from models_retrain."""
    for f in os.listdir(MODELS_DIR):
        if f.startswith(name + "_retrain_") and f.endswith("_state.pt"):
            ck = torch.load(os.path.join(MODELS_DIR, f), map_location="cpu")
            kind = "mfcc" if name in ("logistic", "dnn") else "logmel"
            model = make_model(name, kind)
            sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
            model.load_state_dict(sd)
            model.eval()
            return model, f
    raise FileNotFoundError(name)

# ---------------------------------------------------------------- features (training path, librosa)
import librosa  # noqa: E402

def feat_mfcc(path):
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    n = TARGET_SR
    if len(y) > n:
        fr = HOP
        pw = np.abs(y).reshape(-1, fr).sum(axis=1)
        wf = n // fr
        b = int(np.argmax(pw[:max(1, len(pw)-wf+1)]))
        y = y[b*fr:b*fr+n]
    else:
        y = np.pad(y, (0, n-len(y)))
    f = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC, hop_length=HOP, win_length=WIN)
    return f.T.astype(np.float32)[:T_TARGET]

def feat_logmel(path):
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    n = TARGET_SR
    if len(y) > n:
        fr = HOP
        pw = np.abs(y).reshape(-1, fr).sum(axis=1)
        wf = n // fr
        b = int(np.argmax(pw[:max(1, len(pw)-wf+1)]))
        y = y[b*fr:b*fr+n]
    else:
        y = np.pad(y, (0, n-len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MEL, hop_length=HOP, win_length=WIN)
    return librosa.power_to_db(S, ref=np.max).T.astype(np.float32)[:T_TARGET]

# ---------------------------------------------------------------- MACs estimate
def count_macs(model, x):
    """MACs = sum over Conv1d/Linear of (in_features * out_features) per position.
    For Conv1d the per-position cost is in_ch*out_ch*ksize; the hook sees the
    input at the layer's native resolution, which is exactly what we want."""
    total = [0]
    hooks = []
    def hook_fn(mod, inp, out):
        i = inp[0]
        if mod.__class__.__name__ == "Conv1d":
            w = mod.weight.data.numel() // i.shape[-1]   # out_ch*ksize
            total[0] += int(i.shape[-2]) * int(i.shape[1]) * w
        elif mod.__class__.__name__ == "Linear":
            total[0] += int(i.shape[-1]) * mod.out_features
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))
    with torch.no_grad():
        model(x)
    for h in hooks: h.remove()
    return total[0]

# ---------------------------------------------------------------- WER helpers
def words_from_filename(rel_path):
    """Derive ground-truth words from Option B filename."""
    base = os.path.basename(rel_path)
    s = re.sub(r"\.wav$", "", base)
    s = re.sub(r"_s\d+_v\d+_(clean|noisy)$", "", s)
    s = s.replace("_", " ")
    return re.findall(r"[a-z0-9]+", s.lower())

def ref_words_from_sc(rel_path):
    """SpeechCommands: folder name is the word."""
    parts = rel_path.replace("\\", "/").split("/")
    return [parts[-2].lower()] if len(parts) >= 2 else []

def wer(ref, hyp):
    if not ref: return None
    r, h = ref, hyp
    d = np.zeros((len(r)+1, len(h)+1), dtype=int)
    for i in range(len(r)+1): d[i,0] = i
    for j in range(len(h)+1): d[0,j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    return d[len(r),len(h)]/len(r)

# ---------------------------------------------------------------- whisper.cpp CLI (lazy)
_whisper_cli = None
def get_whisper():
    """Return a callable transcribe(path)->words using whisper-cli + tiny.en.

    Text is parsed from stdout (the CLI writes transcript lines there);
    the -of file output is unreliable across versions.
    """
    global _whisper_cli
    if _whisper_cli is None:
        import subprocess
        cli = shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli"
        model = os.path.join(BASE, "asr", "ggml-tiny.en.bin")
        def transcribe(path):
            r = subprocess.run([cli, "-m", model, "-f", path, "-l", "en"],
                               capture_output=True, text=True, timeout=120)
            words = []
            for line in (r.stdout or "").splitlines():
                line = line.strip()
                if not line or line.startswith("["):
                    continue
                words.extend(re.findall(r"[a-z0-9']+", line.lower()))
            return words
        _whisper_cli = transcribe
    return _whisper_cli

# ---------------------------------------------------------------- main
def main():
    t = pd.read_csv(TEST_CSV, low_memory=False).dropna(subset=["abs_path"]).reset_index(drop=True)
    t["gold"] = t["intent_10"].apply(lambda i: 0 if int(i) == -1 else int(i))
    n = len(t)
    print(f"test clips: {n}")

    # --- WER ground truth (computed once)
    print("computing WER references...")
    refs = []
    for _, r in t.iterrows():
        if r.source == "optionB":
            refs.append(words_from_filename(r.rel_path))
        elif r.source == "speechcommands":
            refs.append(ref_words_from_sc(r.rel_path))
        else:
            refs.append([])
    t["ref_words"] = refs

    # --- whisper transcription (once, shared across models; cached to disk)
    use_asr = "--skip-asr" not in sys.argv
    CACHE = os.path.join(MODELS_DIR, "asr_cache.json")
    if use_asr:
        cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
        todo = [i for i in range(n) if str(i) not in cache]
        if todo:
            print(f"transcribing {len(todo)} clips with whisper tiny.en (cached: {len(cache)})...")
            wh = get_whisper()
            for k, i in enumerate(todo):
                try:
                    cache[str(i)] = wh(t.loc[i, "abs_path"])
                except Exception:
                    cache[str(i)] = []
                if (k+1) % 200 == 0:
                    json.dump(cache, open(CACHE, "w"))
                    print(f"  asr {k+1}/{len(todo)}")
            json.dump(cache, open(CACHE, "w"))
        t["hyp_words"] = [cache.get(str(i), []) for i in range(n)]
    else:
        t["hyp_words"] = [[] for _ in range(n)]

    results = {}
    for name in ["logistic", "dnn", "cnn1d", "crnn"]:
        print(f"\n===== {name} =====")
        model, ckpt_file = load_best(name)
        kind = "mfcc" if name in ("logistic","dnn") else "logmel"
        n_params = sum(p.numel() for p in model.parameters())
        size_mb = os.path.getsize(os.path.join(MODELS_DIR, ckpt_file))/1e6
        x_probe = torch.randn(1, 40, T_TARGET) if kind=="mfcc" else torch.randn(1, N_MEL, T_TARGET)
        if name in ("logistic", "dnn"):
            x_probe = torch.randn(1, T_TARGET, 40)
        macs = count_macs(model, x_probe)

        feats, preds, confs = [], [], []
        feat_ms, fwd_ms = [], []
        with torch.no_grad():
            for i, (_, r) in enumerate(t.iterrows()):
                t0 = time.perf_counter()
                f = feat_mfcc(r.abs_path) if kind=="mfcc" else feat_logmel(r.abs_path)
                tf = (time.perf_counter()-t0)*1000
                if name in ("logistic", "dnn"):
                    x = torch.from_numpy(f[None,:,:])
                else:
                    x = torch.from_numpy(f[None,:,:])
                t1 = time.perf_counter()
                out = model(x)
                tfw = (time.perf_counter()-t1)*1000
                p = int(out.argmax(dim=1).item())
                c = float(torch.softmax(out, dim=1).max().item())
                feats.append(f); preds.append(p); confs.append(c)
                feat_ms.append(tf); fwd_ms.append(tfw)
                if (i+1) % 500 == 0:
                    print(f"  {i+1}/{n}")
        preds = np.array(preds); gold = t["gold"].values
        acc = float((preds==gold).mean())
        print(f"  acc={acc:.4f}  feat p50={np.median(feat_ms):.1f}ms  fwd p50={np.median(fwd_ms):.2f}ms")

        # confusion matrix
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for g, p in zip(gold, preds): cm[g,p] += 1

        # per-intent precision/recall/F1
        per = {}
        for k in range(N_CLASSES):
            tp = cm[k,k]; fp = cm[:,k].sum()-tp; fn = cm[k,:].sum()-tp
            rec = tp/cm[k].sum() if cm[k].sum() else 0.0
            prec = tp/(tp+fp) if (tp+fp) else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
            per[CLASS_NAMES[k]] = dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, n=int(cm[k].sum()))

        # command recall (Option B sub-commands from filename)
        cmd = {}
        ob_idx = t.index[t.source=="optionB"]
        for i in ob_idx:
            base = os.path.basename(t.loc[i,"rel_path"])
            s = re.sub(r"\.wav$","",base); s = re.sub(r"_s\d+_v\d+_(clean|noisy)$","",s)
            key = s.split("_s")[0]
            c = cmd.setdefault(key, [0,0]); c[1]+=1
            if preds[i]==t.loc[i,"gold"]: c[0]+=1
        cmd_recall = {k: {"correct":v[0], "n":v[1], "recall":round(v[0]/v[1],4)} for k,v in sorted(cmd.items())}

        # task success by condition
        cond = {}
        for c in ["clean","noisy"]:
            m = (t.source=="optionB") & (t.condition==c)
            if m.sum():
                cond[c] = {"n":int(m.sum()), "success":int(((preds[m.values])==t.loc[m,"gold"].values).sum()),
                           "rate":round(float((preds[m.values]==t.loc[m,"gold"].values).mean()),4)}
        # overall task completion = non-reject intents correctly recognized
        nr = gold > 0
        task_rate = float((preds[nr]==gold[nr]).mean()) if nr.sum() else 0.0

        # WER
        wers = []
        for i in range(n):
            rw = t.loc[i,"ref_words"]
            if not rw: continue
            hw = t.loc[i,"hyp_words"] if use_asr else []
            w = wer(rw, hw)
            if w is not None: wers.append(w)
        wer_mean = float(np.mean(wers)) if wers else None
        wer_p50 = float(np.median(wers)) if wers else None

        # latency
        tot = np.array(feat_ms)+np.array(fwd_ms)
        lat = dict(feat_p50=float(np.percentile(feat_ms,50)), feat_p95=float(np.percentile(feat_ms,95)),
                   fwd_p50=float(np.percentile(fwd_ms,50)), fwd_p95=float(np.percentile(fwd_ms,95)),
                   total_p50=float(np.percentile(tot,50)), total_p95=float(np.percentile(tot,95)),
                   total_max=float(tot.max()),
                   throughput_per_s=float(n/tot.sum()*1000))

        results[name] = dict(checkpoint=ckpt_file, kind=kind, params=n_params, size_mb=round(size_mb,2),
                             macs_per_clip=macs, acc=round(acc,4), per_intent=per,
                             command_recall=cmd_recall, task_completion=cond,
                             task_success_nonreject=round(task_rate,4),
                             wer_mean=(round(wer_mean,4) if wer_mean is not None else None),
                             wer_p50=(round(wer_p50,4) if wer_p50 is not None else None),
                             latency=lat, confusion=cm.tolist(),
                             conf_mean=float(np.mean(confs)), conf_median=float(np.median(confs)))
        print(f"  saved {name}")

    json.dump(results, open(OUT_JSON,"w"), indent=2)
    print(f"\nwrote {OUT_JSON}")

    # ---------------------------------------------------------------- Excel
    print("building Excel...")
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        # Sheet 1: summary
        rows = []
        for m, r in results.items():
            rows.append(dict(Model=m, Arch=r["kind"], Params=r["params"], Size_MB=r["size_mb"],
                             MACs_per_clip=r["macs_per_clip"], Test_Accuracy=r["acc"],
                             Task_Success_NonReject=r["task_success_nonreject"],
                             WER_Mean=r["wer_mean"], WER_P50=r["wer_p50"],
                             Feat_p50_ms=r["latency"]["feat_p50"], Feat_p95_ms=r["latency"]["feat_p95"],
                             Fwd_p50_ms=r["latency"]["fwd_p50"], Fwd_p95_ms=r["latency"]["fwd_p95"],
                             Total_p50_ms=r["latency"]["total_p50"], Total_p95_ms=r["latency"]["total_p95"],
                             Total_max_ms=r["latency"]["total_max"],
                             Throughput_clips_per_s=r["latency"]["throughput_per_s"],
                             Conf_mean=r["conf_mean"]))
        pd.DataFrame(rows).to_excel(xw, sheet_name="Summary", index=False)

        # Sheet 2: per-intent P/R/F1
        rows = []
        for m, r in results.items():
            for intent, v in r["per_intent"].items():
                rows.append(dict(Model=m, Intent=intent, N=v["n"], TP=v["tp"], FP=v["fp"], FN=v["fn"],
                                 Precision=round(v["precision"],4), Recall=round(v["recall"],4),
                                 F1=round(v["f1"],4)))
        pd.DataFrame(rows).to_excel(xw, sheet_name="PerIntent_PRF1", index=False)

        # Sheet 3: command recall (sub-commands)
        rows = []
        for m, r in results.items():
            for cmd, v in r["command_recall"].items():
                rows.append(dict(Model=m, Command=cmd, Correct=v["correct"], N=v["n"], Recall=v["recall"]))
        pd.DataFrame(rows).to_excel(xw, sheet_name="CommandRecall", index=False)

        # Sheet 4: task completion by condition
        rows = []
        for m, r in results.items():
            for c, v in r["task_completion"].items():
                rows.append(dict(Model=m, Condition=c, N=v["n"], Success=v["success"], Rate=v["rate"]))
        pd.DataFrame(rows).to_excel(xw, sheet_name="TaskCompletion", index=False)

        # Sheet 5: confusion matrices (one block per model)
        buf = io.StringIO()
        hdr = "Gold \\ Pred\t" + "\t".join(CLASS_NAMES[k] for k in range(N_CLASSES))
        for m, r in results.items():
            buf.write(f"\n### {m}\n{hdr}\n")
            cm = np.array(r["confusion"])
            for g in range(N_CLASSES):
                buf.write(CLASS_NAMES[g] + "\t" + "\t".join(str(x) for x in cm[g]) + "\n")
        pd.read_csv(buf, sep="\t").to_excel(xw, sheet_name="Confusion", index=False)

    print(f"wrote {OUT_XLSX}")

    # ---------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results.keys())
    accs = [results[m]["acc"] for m in models]
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(models, [a*100 for a in accs], color=["#9aa0a6","#fbbc04","#4285f4","#34a853"])
    for b, a in zip(bars, accs): ax.text(b.get_x()+b.get_width()/2, a*100+0.5, f"{a*100:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Test accuracy (%)"); ax.set_title("Retrained models — full 2,883-clip test set")
    ax.set_ylim(0,105); plt.tight_layout(); plt.savefig(f"{PLOTS}/benchmark_accuracy.png", dpi=130); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    ax = axes[0]
    ax.bar(models, [results[m]["latency"]["feat_p50"] for m in models], label="feature p50")
    ax.bar(models, [results[m]["latency"]["fwd_p50"] for m in models], label="forward p50", bottom=[results[m]["latency"]["feat_p50"] for m in models])
    ax.set_ylabel("ms"); ax.set_title("Latency breakdown (p50)"); ax.legend()
    ax = axes[1]
    ax.bar(models, [results[m]["latency"]["throughput_per_s"] for m in models])
    for i, m in enumerate(models): ax.text(i, results[m]["latency"]["throughput_per_s"], f'{results[m]["latency"]["throughput_per_s"]:.0f}', ha="center", fontsize=9)
    ax.set_ylabel("clips/sec"); ax.set_title("Throughput (end-to-end)")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/benchmark_latency.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(9,5))
    x = np.arange(len(models)); wdt = 0.35
    ax.bar(x-wdt/2, [results[m]["task_success_nonreject"]*100 for m in models], wdt, label="non-REJECT success")
    cl = [results[m]["task_completion"].get("clean",{}).get("rate",0)*100 for m in models]
    no = [results[m]["task_completion"].get("noisy",{}).get("rate",0)*100 for m in models]
    ax.bar(x, cl, wdt, label="OptionB clean")
    ax.bar(x+wdt/2, no, wdt, label="OptionB noisy")
    ax.set_xticks(x, models); ax.set_ylabel("%"); ax.set_title("Task completion / success rate"); ax.legend(); ax.set_ylim(0,105)
    plt.tight_layout(); plt.savefig(f"{PLOTS}/benchmark_task_success.png", dpi=130); plt.close()

    # per-intent heatmap for best model
    best = max(models, key=lambda m: results[m]["acc"])
    fig, ax = plt.subplots(figsize=(9,7))
    im = ax.imshow(np.array(results[best]["confusion"]), cmap="Blues")
    ax.set_xticks(range(N_CLASSES), [CLASS_NAMES[k][:6] for k in range(N_CLASSES)], rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES), [CLASS_NAMES[k][:6] for k in range(N_CLASSES)])
    cm = np.array(results[best]["confusion"])
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i,j]: ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=7,
                                color="white" if cm[i,j] > cm.max()*0.5 else "black")
    ax.set_title(f"Confusion matrix — {best}"); plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(f"{PLOTS}/benchmark_confusion_{best}.png", dpi=130); plt.close()

    print(f"plots written to {PLOTS}/")
    print("\nDONE.")

if __name__ == "__main__":
    main()
