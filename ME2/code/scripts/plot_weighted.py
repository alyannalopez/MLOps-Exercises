"""Plot the Option-2 weighted CRNN run.

Reads models_retrain/crnn_wob_*_{history,metrics}.json + _confusion.npy and
writes into models_retrain/plots/:
  wob_training_curves.png   train/val acc + loss
  wob_per_intent.png        grouped precision/recall/f1 bars per intent
  wob_confusion.png         normalized confusion heatmap
  wob_metrics_table.png     rendered PRF metric table
Prints a markdown summary.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(HERE, "models_retrain")
OUT = os.path.join(MD, "plots")
os.makedirs(OUT, exist_ok=True)

# latest crnn_wob artifacts
hp = sorted(glob.glob(os.path.join(MD, "crnn_wob_*_history.json")))
mp = sorted(glob.glob(os.path.join(MD, "crnn_wob_*_metrics.json")))
cp = sorted(glob.glob(os.path.join(MD, "crnn_wob_*_confusion.npy")))
assert hp and mp and cp, "missing weighted-run artifacts"
with open(hp[-1]) as f: hist = json.load(f)["history"]
with open(mp[-1]) as f: res = json.load(f)
cm = np.load(cp[-1])

INTENTS = list(res["per_intent_test"].keys())
SHORT = {"REJECT":"REJ","PLAY_MUSIC":"PLAY","QUESTION_SEARCH":"SEARCH",
         "LIGHTS_ON_OFF":"LIGHTS","DIM_COLOR_LIGHTS":"COLOR","SET_TIMER":"TIMER",
         "SET_ALARM":"ALARM","THERMOSTAT":"THERMO","MEDIA_CONTROL":"MEDIA",
         "REMINDERS_LISTS":"REMIND","CALLS_MESSAGING":"CALLS"}
t = res["test"]
pi = res["per_intent_test"]

# ---- 1. training curves (acc + loss) ---------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 5.5))
eps = [r["epoch"] for r in hist]
ax1.plot(eps, [r["train_acc"] for r in hist], label="train acc", lw=1.8, color="#1f77b4")
ax1.plot(eps, [r["val_acc"] for r in hist], label="val acc", lw=1.8, color="#ff7f0e")
ax1.axhline(res["best_val_acc"], ls="--", lw=1, color="gray",
            label=f"best val {res['best_val_acc']:.4f} (ep {res['best_val_epoch']})")
ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy"); ax1.set_ylim(0.5, 1.02)
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(eps, [r["train_loss"] for r in hist], lw=1.2, color="#d62728", alpha=0.6, label="train loss")
ax2.set_ylabel("loss", color="#d62728"); ax2.tick_params(axis="y", labelcolor="#d62728")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
ax1.set_title(f"CRNN Option-2 (source-weighted, ob_mult={res['ob_mult']}) — training curves\n"
              f"test accuracy {t['accuracy']:.4f}   kappa {t['cohen_kappa']:.4f}   "
              f"macro-F1 {t['macro_f1']:.4f}", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "wob_training_curves.png"), dpi=130)
plt.close(fig)

# ---- 2. per-intent grouped bars --------------------------------------------
x = np.arange(len(INTENTS)); wdt = 0.27
prec = [pi[i]["precision"] for i in INTENTS]
rec  = [pi[i]["recall"] for i in INTENTS]
f1   = [pi[i]["f1"] for i in INTENTS]
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - wdt, prec, wdt, label="precision", color="#1f77b4")
b2 = ax.bar(x, rec, wdt, label="recall", color="#2ca02c")
b3 = ax.bar(x + wdt, f1, wdt, label="F1", color="#ff7f0e")
for bars in (b1, b2, b3):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=7, rotation=0)
ax.set_xticks(x); ax.set_xticklabels([SHORT[i] for i in INTENTS], rotation=30, ha="right")
ax.set_ylim(0, 1.12); ax.set_ylabel("score"); ax.grid(alpha=0.3, axis="y")
ax.set_title("Per-intent precision / recall / F1 (held-out test set)", fontsize=12)
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "wob_per_intent.png"), dpi=130)
plt.close(fig)

# ---- 3. confusion heatmap ---------------------------------------------------
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(cmn, cmap="Blues")
labels = [SHORT[i] for i in INTENTS]
ax.set_xticks(range(len(INTENTS))); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(len(INTENTS))); ax.set_yticklabels(labels)
for i in range(len(INTENTS)):
    for j in range(len(INTENTS)):
        ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if cmn[i,j] > 0.5 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("Confusion matrix (rows normalized, true→predicted)", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "wob_confusion.png"), dpi=130)
plt.close(fig)

# ---- 4. metrics table image -------------------------------------------------
rows = [["metric", "value"]]
rows += [["accuracy", f"{t['accuracy']:.4f}"],
         ["Cohen's kappa", f"{t['cohen_kappa']:.4f}"],
         ["macro precision", f"{t['macro_precision']:.4f}"],
         ["macro recall", f"{t['macro_recall']:.4f}"],
         ["macro F1", f"{t['macro_f1']:.4f}"],
         ["micro precision", f"{t['micro_precision']:.4f}"],
         ["micro recall", f"{t['micro_recall']:.4f}"],
         ["micro F1", f"{t['micro_f1']:.4f}"],
         ["weighted precision", f"{t['weighted_precision']:.4f}"],
         ["weighted recall", f"{t['weighted_recall']:.4f}"],
         ["weighted F1", f"{t['weighted_f1']:.4f}"]]
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=None, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#ecf0f1")
ax.set_title("Held-out test metrics — CRNN Option-2 (source-weighted)", fontsize=11, pad=20)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "wob_metrics_table.png"), dpi=130)
plt.close(fig)

# ---- markdown summary -------------------------------------------------------
print("## Option-2 weighted CRNN — test metrics")
print(f"- accuracy **{t['accuracy']:.4f}**, kappa {t['cohen_kappa']:.4f}")
print(f"- macro P/R/F1 {t['macro_precision']:.4f}/{t['macro_recall']:.4f}/{t['macro_f1']:.4f}")
print(f"- micro P/R/F1 {t['micro_precision']:.4f}/{t['micro_recall']:.4f}/{t['micro_f1']:.4f}")
print(f"- weighted P/R/F1 {t['weighted_precision']:.4f}/{t['weighted_recall']:.4f}/{t['weighted_f1']:.4f}")
print("\n| intent | precision | recall | F1 | support |")
print("|---|---|---|---|---|")
for i in INTENTS:
    d = pi[i]
    print(f"| {i} | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} | {d['support']} |")
print(f"\nPlots -> {OUT}/wob_training_curves.png, wob_per_intent.png, wob_confusion.png, wob_metrics_table.png")
