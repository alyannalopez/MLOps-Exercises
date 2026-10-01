"""Plot train/val accuracy curves from models_retrain/*_history.json.

Outputs (into models_retrain/plots/):
  train_curves_<model>.png   per-model train vs val accuracy
  train_curves_all.png       2x2 grid, one panel per model
  comparison.png             val-acc curves overlaid (separate panels for small vs big models)
Prints a final markdown table with test accuracy + per-intent breakdown.
"""
from __future__ import annotations
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(HERE, "models_retrain")
OUT = os.path.join(MD, "plots")
os.makedirs(OUT, exist_ok=True)

histories = {}
for p in sorted(glob.glob(os.path.join(MD, "*_history.json"))):
    with open(p) as f:
        d = json.load(f)
    histories[d["result"]["model"]] = d

def curve(d, key):
    return [r[key] for r in d["history"]]

# ---- per-model plots -------------------------------------------------------
for name, d in histories.items():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve(d, "train_acc"), label="train acc", lw=1.6)
    ax.plot(curve(d, "val_acc"), label="val acc", lw=1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
    ax.set_title(f"{name} — train vs val accuracy  (best val {d['result']['best_val_acc']:.4f} @ ep {d['result']['best_val_epoch']}, test {d['result']['test_acc']:.4f})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"train_curves_{name}.png"), dpi=130)
    plt.close(fig)

# ---- 2x2 grid ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (name, d) in zip(axes.flat, histories.items()):
    ax.plot(curve(d, "train_acc"), label="train", lw=1.4)
    ax.plot(curve(d, "val_acc"), label="val", lw=1.4)
    ax.set_title(f"{name}  (test {d['result']['test_acc']:.1%})")
    ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("VCM retraining — 100 epochs, per-epoch accuracy", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "train_curves_all.png"), dpi=130)
plt.close(fig)

# ---- comparison: small (mfcc) vs big (logmel) ------------------------------
small = [n for n in histories if histories[n]["result"]["kind"] == "mfcc"]
big = [n for n in histories if histories[n]["result"]["kind"] == "logmel"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, group, title in [(axes[0], small, "MFCC models (logistic, dnn)"),
                         (axes[1], big, "log-mel models (cnn1d, crnn)")]:
    for name in group:
        d = histories[name]
        ax.plot(curve(d, "val_acc"), lw=1.5, label=f"{name} (test {d['result']['test_acc']:.1%})")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "comparison.png"), dpi=130)
plt.close(fig)

# ---- markdown summary table -------------------------------------------------
print("\n| Model | Params | Best val (epoch) | Final train | **Test acc** | Wall (s) |")
print("|-------|--------|------------------|-------------|--------------|----------|")
for name in ["logistic", "dnn", "cnn1d", "crnn"]:
    if name not in histories: continue
    r = histories[name]["result"]
    print(f"| {name} | {r['params']:,} | {r['best_val_acc']:.4f} (ep {r['best_val_epoch']}) | {r['final_train_acc']:.4f} | **{r['test_acc']:.4f}** | {r['wall_seconds']} |")
print("\nPer-intent test accuracy:")
print("| Intent | " + " | ".join(n for n in ["logistic","dnn","cnn1d","crnn"] if n in histories) + " |")
print("|--------|" + "------|" * len(histories))
intents = list(next(iter(histories.values()))["result"]["per_intent_test_acc"].keys())
for it in intents:
    row = []
    for name in ["logistic","dnn","cnn1d","crnn"]:
        if name not in histories: continue
        v = histories[name]["result"]["per_intent_test_acc"][it]
        row.append("—" if v is None else f"{v:.1%}")
    print(f"| {it} | " + " | ".join(row) + " |")
print(f"\nPlots written to {OUT}")
