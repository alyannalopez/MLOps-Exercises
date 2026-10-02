#!/usr/bin/env python3
"""Step 7: generate all report figures.

Reads ME2/models/*_history.json, benchmark.json, eval_results.json and writes
PNGs to ME2/report/plots/:
  01_benchmark_accuracy.png     - test acc per architecture
  02_params_vs_accuracy.png     - efficiency frontier (log params vs acc)
  03_training_curves_5models.png- train/val acc per arch (5 panels incl. E)
  04_best_training_curves.png   - best model zoom (loss + acc)
  05_confusion_matrix.png       - best model 11x11
  06_per_intent_prf.png         - best model per-intent P/R/F1
  07_reliability.png            - calibration diagram
  08_task_completion.png        - command/reject success bars
  09_robustness.png             - by condition / duration / value
  10_dataset_balance.png        - OptionB per-intent train distribution
  11_architecture_diagram.png   - best model block diagram
  12_class_labels.png           - 10 intents + REJECT with examples
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS = os.path.join(ROOT, "models")
SPLITS = os.path.join(ROOT, "data", "splits")
PLOTS = os.path.join(ROOT, "report", "plots")
os.makedirs(PLOTS, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10})

bench = json.load(open(os.path.join(MODELS, "benchmark.json")))
ev = json.load(open(os.path.join(MODELS, "eval_results.json")))
best = bench["best"]
res = {r["name"]: r for r in bench["results"]}
ORDER = ["logistic", "dnn", "cnn1d", "crnn"]
SHORT = {"logistic": "A. Logistic\n(MFCC)", "dnn": "B. DNN\n(MFCC)",
         "cnn1d": "C. 1-D CNN\n(log-mel)", "crnn": "D. CRNN\n(log-mel)",
         "pocketsphinx": "E. PocketSphinx\n(grammar)"}


def hist(name):
    return json.load(open(os.path.join(MODELS, f"{name}_history.json")))


# 01 benchmark accuracy -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
names = [r["name"] for r in sorted(res.values(), key=lambda x: x["test_acc"])]
vals = [res[n]["test_acc"] for n in names]
bars = ax.bar([SHORT[n] for n in names], vals,
              color=["#ccc", "#9ad", "#6cb", "#2a9"][:len(names)])
bars[names.index(best)].set_color("#e4572e")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
            ha="center", fontsize=9)
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.4, 1.0)
ax.set_title("Option B only — architecture benchmark (4 trained + PocketSphinx baseline)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/01_benchmark_accuracy.png")
plt.close()

# 02 params vs accuracy ------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for n in ORDER:
    r = res[n]
    c = "#e4572e" if n == best else "#4878a8"
    ax.scatter(r["params"], r["test_acc"], s=120, color=c, zorder=3)
    ax.annotate(f"{n}\n{r['test_acc']:.3f}", (r["params"], r["test_acc"]),
                xytext=(8, 6), textcoords="offset points", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("Parameters (log scale)")
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.4, 1.0)
ax.set_title("Efficiency frontier — accuracy vs model size")
plt.tight_layout()
plt.savefig(f"{PLOTS}/02_params_vs_accuracy.png")
plt.close()

# 03 training curves, all models --------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
axes = axes.flatten()
for i, n in enumerate(ORDER):
    h = hist(n)
    ax = axes[i]
    ax.plot(h["epoch"], h["train_acc"], label="train", lw=1.5)
    ax.plot(h["epoch"], h["val_acc"], label="val", lw=1.5)
    ax.set_title(f"{SHORT[n].splitlines()[0]}  (test={res[n]['test_acc']:.3f})")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8)
ax = axes[4]  # E: PocketSphinx placeholder
ax.text(0.5, 0.5, "E. PocketSphinx (GMM-HMM grammar)\nno training — classical\nbaseline, reported separately",
        ha="center", va="center", transform=ax.transAxes, fontsize=10)
ax.axis("off")
axes[5].axis("off")
fig.suptitle("Training curves — all architectures (seed 42, MPS, early stopping)", y=1.0)
plt.tight_layout()
plt.savefig(f"{PLOTS}/03_training_curves_5models.png")
plt.close()

# 04 best model curves -------------------------------------------------------
h = hist(best)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.plot(h["epoch"], h["train_loss"], lw=1.5, color="#4878a8")
a1.set_title(f"{best} — training loss")
a1.set_xlabel("epoch")
a2.plot(h["epoch"], h["train_acc"], lw=1.5, label="train", color="#4878a8")
a2.plot(h["epoch"], h["val_acc"], lw=1.5, label="val", color="#e4572e")
a2.set_title(f"{best} — accuracy (test={res[best]['test_acc']:.3f})")
a2.set_xlabel("epoch")
a2.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/04_best_training_curves.png")
plt.close()

# 05 confusion matrix --------------------------------------------------------
cm = np.array(ev["confusion"])
labs = ev["classes"] + ["REJECT"]
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labs)))
ax.set_yticks(range(len(labs)))
ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=8)
ax.set_yticklabels(labs, fontsize=8)
for i in range(len(labs)):
    for j in range(len(labs)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted")
ax.set_ylabel("Gold")
ax.set_title(f"{best} — confusion matrix (n={ev['n_test']})")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(f"{PLOTS}/05_confusion_matrix.png")
plt.close()

# 06 per-intent P/R/F1 -------------------------------------------------------
pi = pd.DataFrame(ev["per_intent"])
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(pi))
w = 0.26
ax.bar(x - w, pi["precision"], w, label="Precision", color="#4878a8")
ax.bar(x, pi["recall"], w, label="Recall", color="#e4572e")
ax.bar(x + w, pi["f1"], w, label="F1", color="#6aa84f")
ax.set_xticks(x)
ax.set_xticklabels(pi["intent"], rotation=40, ha="right", fontsize=8)
ax.set_ylim(0.85, 1.01)
ax.legend()
ax.set_title(f"{best} — per-intent precision / recall / F1")
plt.tight_layout()
plt.savefig(f"{PLOTS}/06_per_intent_prf.png")
plt.close()

# 07 reliability -------------------------------------------------------------
rel = ev["reliability"]
fig, ax = plt.subplots(figsize=(5.5, 5.5))
xs = [(r["bin"][0] + r["bin"][1]) / 2 for r in rel]
ys = [r["acc"] for r in rel]
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
ax.bar(xs, ys, width=0.09, align="center", alpha=0.75, color="#4878a8",
       label="model")
ax.plot(xs, ys, "o-", color="#e4572e", ms=4)
ax.set_xlabel("Mean confidence")
ax.set_ylabel("Accuracy")
ax.set_title(f"Calibration (ECE={ev['ece']:.3f}, Brier={ev['brier']:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/07_reliability.png")
plt.close()

# 08 task completion ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
vals = [ev["command_recall"], ev["reject_recall"], ev["task_completion"]]
bars = ax.bar(["Command recall\n(commands correctly recognized)",
               "Reject recall\n(noise correctly rejected)",
               "Task completion\n(overall correctness)"],
              vals, color=["#4878a8", "#e4572e", "#6aa84f"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center")
ax.set_ylim(0.8, 1.01)
ax.set_title(f"{best} — command success & rejection")
plt.tight_layout()
plt.savefig(f"{PLOTS}/08_task_completion.png")
plt.close()

# 09 robustness --------------------------------------------------------------
rob = ev["robustness"]
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.5))
bc = rob["by_condition"]
a1.bar(list(bc.keys()), list(bc.values()), color=["#6aa84f", "#e4572e"])
a1.set_title(f"By condition (Δ={rob['clean_to_noisy_delta']:+.3f})")
a1.set_ylim(0.8, 1.01)
bd = rob["by_duration_bucket"]
a2.bar(list(bd.keys()), list(bd.values()), color="#4878a8")
a2.set_title("By duration bucket")
a2.set_ylim(0.8, 1.01)
pv = list(rob["per_value_acc"].items())[:12]
a3.barh([k.replace("_", " ") for k, _ in pv], [v for _, v in pv], color="#9ad")
a3.set_title("Per-value accuracy (lowest 12)")
a3.set_xlim(0.8, 1.01)
fig.suptitle(f"{best} — robustness")
plt.tight_layout()
plt.savefig(f"{PLOTS}/09_robustness.png")
plt.close()

# 10 dataset balance ---------------------------------------------------------
tr = pd.read_csv(os.path.join(SPLITS, "train.csv"))
tr = tr[tr.intent_name != "REJECT"]
cnt = tr.intent_name.value_counts()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(cnt.index, cnt.values, color="#4878a8")
ax.set_ylabel("Train clips")
ax.set_title(f"Option B train split — per-intent balance "
             f"({len(tr)} clips, {tr.speaker.nunique()} speakers, speaker-disjoint)")
plt.xticks(rotation=40, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOTS}/10_dataset_balance.png")
plt.close()

# 11 architecture diagram ----------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")
blocks = [
    ("Input\nlog-mel (T=101, F=80)", "#d9e8f5"),
    ("Conv1d ×4\n64→128→128→128\nReLU + MaxPool", "#bcd7ee"),
    ("Flatten\n(128·T')", "#d9e8f5"),
    ("BiLSTM\nhidden 128\n(last timestep)", "#bcd7ee"),
    ("Dropout 0.3\nLinear → 11", "#d9e8f5"),
    ("Softmax\n10 intents + REJECT", "#f5d9d9"),
]
bw, bh, gap = 1.55, 1.1, 0.35
x0 = 0.15
for i, (txt, col) in enumerate(blocks):
    box = FancyBboxPatch((x0 + i * (bw + gap), 1.9), bw, bh,
                         boxstyle="round,pad=0.06", fc=col, ec="#333", lw=1.2)
    ax.add_patch(box)
    ax.text(x0 + i * (bw + gap) + bw / 2, 1.9 + bh / 2, txt,
            ha="center", va="center", fontsize=8.5)
    if i < len(blocks) - 1:
        ar = FancyArrowPatch((x0 + i * (bw + gap) + bw, 1.9 + bh / 2),
                             (x0 + (i + 1) * (bw + gap), 1.9 + bh / 2),
                             arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#333")
        ax.add_patch(ar)
ax.text(0.15, 3.4, f"Best model: {best.upper()}  —  {res[best]['params']:,} params, "
                   f"test acc {res[best]['test_acc']:.3f}", fontsize=11, weight="bold")
ax.set_xlim(0, 12.2)
ax.set_ylim(1.5, 3.8)
plt.tight_layout()
plt.savefig(f"{PLOTS}/11_architecture_diagram.png")
plt.close()

# 12 class labels ------------------------------------------------------------
EXAMPLES = {
    "PLAY_MUSIC": '"play music"',
    "QUESTION_SEARCH": '"what\'s the weather" / "what time is it"',
    "LIGHTS_ON_OFF": '"turn on the lights" / "turn off the lights"',
    "DIM_COLOR_LIGHTS": '"dim lights to 60%" / "lights to blue"',
    "SET_TIMER": '"set a timer for 30 seconds"',
    "SET_ALARM": '"set an alarm for 8 am"',
    "THERMOSTAT": '"set temperature to 22 degrees"',
    "MEDIA_CONTROL": '"pause" / "next" / "volume up"',
    "REMINDERS_LISTS": '"remind me to study" / "list reminders"',
    "CALLS_MESSAGING": '"call mom" / "send a message"',
}
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.axis("off")
ax.set_title("Class labels — 10 intents + REJECT", fontsize=12, weight="bold", pad=12)
yy = 0.97
for i, cn in enumerate(ev["classes"]):
    n_tr = int(cnt.get(cn, 0))
    ax.text(0.02, yy, f"{i}. {cn}", fontsize=10, weight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.30, yy, EXAMPLES[cn], fontsize=9, transform=ax.transAxes, va="top",
            style="italic")
    ax.text(0.98, yy, f"train: {n_tr}", fontsize=8, color="#555",
            transform=ax.transAxes, va="top", ha="right")
    yy -= 0.085
ax.text(0.02, yy, "10. REJECT", fontsize=10, weight="bold", color="#a33",
        transform=ax.transAxes, va="top")
ax.text(0.30, yy, "ambient noise / no command (synthesized)", fontsize=9,
        transform=ax.transAxes, va="top", style="italic", color="#a33")
ax.text(0.98, yy, f"train: {int((tr.intent_name == 'REJECT').sum())}", fontsize=8,
        color="#555", transform=ax.transAxes, va="top", ha="right")
plt.tight_layout()
plt.savefig(f"{PLOTS}/12_class_labels.png")
plt.close()

print(f"Wrote 12 figures -> {PLOTS}")
