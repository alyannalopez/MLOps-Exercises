"""Benchmark + best-model graphs for the VCM report.

Produces (models_retrain/plots/report/):
  1. bench_accuracy.png      - test accuracy bars, 4 architectures
  2. bench_params_acc.png    - params (log) vs test accuracy scatter
  3. bench_per_intent.png    - per-intent recall grouped bars, 4 archs
  4. bench_training_curves.png - train/val accuracy curves, 4 archs
  5. best_training_curves.png  - balanced CRNN train loss + val acc
  6. best_confusion.png      - balanced CRNN confusion matrix (heatmap)
  7. best_per_intent.png     - balanced CRNN P/R/F1 per intent
  8. best_calibration.png    - reliability diagram (from comprehensive_eval.json)
  9. best_task_completion.png- task completion / success rates
  10. best_robustness.png    - accuracy by condition/source
"""
from __future__ import annotations
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MR = os.path.join(ROOT, "models_retrain")
OUT = os.path.join(MR, "plots", "report")
os.makedirs(OUT, exist_ok=True)

ARCHS = ["logistic", "dnn", "cnn1d", "crnn"]
COLORS = {"logistic": "#9aa0a6", "dnn": "#f2b705", "cnn1d": "#34a853", "crnn": "#4285f4"}

# ---- load 4-arch retrain histories ----
hist = {}
for m in ARCHS:
    fs = sorted(glob.glob(os.path.join(MR, f"{m}_retrain_*_history.json")))
    h = json.load(open(fs[-1]))
    hist[m] = h

best = json.load(open(os.path.join(MR, "crnn_bal_1790835176_history.json")))
comp = json.load(open(os.path.join(MR, "comprehensive_eval.json")))

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10})

# 1. test accuracy bars
fig, ax = plt.subplots(figsize=(8, 5))
accs = [hist[m]["result"]["test_acc"] for m in ARCHS]
bars = ax.bar(ARCHS, accs, color=[COLORS[m] for m in ARCHS], width=0.6)
ax.set_ylabel("Test accuracy"); ax.set_title("VCM Architecture Benchmark — Test Accuracy")
ax.set_ylim(0, 1.05)
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width()/2, a + 0.01, f"{a*100:.1f}%", ha="center", fontweight="bold")
ax.annotate("BEST", xy=(3, accs[3]), xytext=(2.2, accs[3]-0.12),
            arrowprops=dict(arrowstyle="->", color="#4285f4"), fontsize=11, color="#4285f4", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "bench_accuracy.png")); plt.close()

# 2. params vs accuracy
fig, ax = plt.subplots(figsize=(8, 5))
params = [hist[m]["result"]["params"] for m in ARCHS]
for m, p, a in zip(ARCHS, params, accs):
    ax.scatter(p, a, s=120, color=COLORS[m], zorder=3)
    ax.annotate(f"{m}\n({p:,} p)", (p, a), textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Parameters (log scale)"); ax.set_ylabel("Test accuracy")
ax.set_title("Efficiency frontier — Parameters vs Test Accuracy")
ax.set_ylim(0.4, 1.0)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "bench_params_acc.png")); plt.close()

# 3. per-intent recall grouped bars (use per_intent_test_acc)
INTENTS = ["REJECT","PLAY_MUSIC","QUESTION_SEARCH","LIGHTS_ON_OFF","DIM_COLOR_LIGHTS",
           "SET_TIMER","SET_ALARM","THERMOSTAT","MEDIA_CONTROL","REMINDERS_LISTS","CALLS_MESSAGING"]
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(INTENTS)); w = 0.2
for i, m in enumerate(ARCHS):
    vals = [hist[m]["result"]["per_intent_test_acc"].get(k, 0) for k in INTENTS]
    ax.bar(x + (i-1.5)*w, vals, w, label=m, color=COLORS[m])
ax.set_xticks(x); ax.set_xticklabels(INTENTS, rotation=35, ha="right")
ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
ax.set_title("Per-Intent Recall — 4 Architectures")
ax.legend(ncol=4); plt.tight_layout(); plt.savefig(os.path.join(OUT, "bench_per_intent.png")); plt.close()

# 4. training curves all 4
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for m in ARCHS:
    hh = hist[m]["history"]
    ep = [e["epoch"] for e in hh]
    tr = [e["train_acc"] for e in hh]; va = [e["val_acc"] for e in hh]
    axes[0].plot(ep, tr, alpha=0.35, color=COLORS[m])
    axes[0].plot(ep, va, lw=2, color=COLORS[m], label=m)
    axes[1].plot(ep, tr, alpha=0.35, color=COLORS[m])
    axes[1].plot(ep, va, lw=2, color=COLORS[m], label=m)
axes[0].set_title("Validation Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1.05); axes[0].legend()
axes[1].set_title("Train Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1.05)
plt.suptitle("Training Dynamics — 4 Architectures (100 epochs)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "bench_training_curves.png")); plt.close()

# 5. best training curves
hh = best["history"]
ep = [e["epoch"] for e in hh]; tl = [e["train_loss"] for e in hh]
tr = [e["train_acc"] for e in hh]; va = [e["val_acc"] for e in hh]
bev = best["result"]["best_val_epoch"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].plot(ep, tl, color="#4285f4"); axes[0].set_yscale("log")
axes[0].set_title("Training Loss (log)"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[1].plot(ep, tr, alpha=0.4, color="#4285f4", label="train acc")
axes[1].plot(ep, va, color="#34a853", lw=2, label="val acc")
axes[1].axvline(bev, color="red", ls="--", lw=1, label=f"best val ep {bev}")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1.02); axes[1].legend()
plt.suptitle("Best Model — Balanced CRNN (Option 2b), 100 epochs")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "best_training_curves.png")); plt.close()

# 6. best confusion matrix
cm = np.array(comp["confusion_matrix"]); order = comp["cm_order"]
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
for i in range(len(order)):
    for j in range(len(order)):
        v = cm[i, j]
        if v:
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if v > cm.max()*0.5 else "black", fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.colorbar(im)
ax.set_title(f"Balanced CRNN — Confusion Matrix (test n={comp['n_test']})")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "best_confusion.png")); plt.close()

# 7. best per-intent P/R/F1
pi = comp["intent_recognition"]["per_intent"]
names = [k for k in INTENTS if k in pi]
P = [pi[k]["precision"] for k in names]; R = [pi[k]["recall"] for k in names]; F = [pi[k]["f1"] for k in names]
x = np.arange(len(names)); w = 0.27
fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x-w, P, w, label="Precision", color="#4285f4")
ax.bar(x, R, w, label="Recall", color="#34a853")
ax.bar(x+w, F, w, label="F1", color="#f2b705")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
ax.set_ylim(0.85, 1.01); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Balanced CRNN — Per-Intent Precision / Recall / F1 (test)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "best_per_intent.png")); plt.close()

# 8. reliability diagram
cal = comp["calibration"]
fig, ax = plt.subplots(figsize=(6.5, 6))
xs, ys = [], []
for b in cal["reliability"]:
    xs.append(b["conf"]); ys.append(b["acc"])
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
ax.plot(xs, ys, "-o", color="#4285f4", lw=2, label="model")
ax.fill_between(xs, ys, xs, alpha=0.15, color="#4285f4")
ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Empirical accuracy")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title(f"Reliability Diagram\nECE={cal['ece_15']:.3f}  MCE={cal['mce']:.3f}  Brier={cal['brier']:.3f}")
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "best_calibration.png")); plt.close()

# 9. task completion
tk = comp["task_completion"]
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Overall\nsuccess", "Command\nsuccess", "Reject\nsuccess", "False\ntrigger rate", "Missed\ncommand rate"]
vals = [tk["overall_success_rate"], tk["command_success_rate"], tk["reject_success_rate"],
        tk["false_trigger_rate"], tk["missed_command_rate"]]
cols = ["#4285f4", "#34a853", "#34a853", "#ea4335", "#ea4335"]
bars = ax.bar(labels, vals, color=cols, width=0.6)
ax.set_ylabel("Rate"); ax.set_ylim(0, 1.05); ax.set_title("Task Completion / Success Rates")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v*100:.1f}%", ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "best_task_completion.png")); plt.close()

# 10. robustness
rb = comp["robustness"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cond = rb["by_condition"]
cl = list(cond.keys()); cv = [cond[c]["acc"] for c in cl]
b = axes[0].bar(cl, cv, color=["#34a853" if c=="clean" else "#f2b705" for c in cl], width=0.5)
axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0.85, 1.01); axes[0].set_title("By Condition (noise)")
for bb, v in zip(b, cv): axes[0].text(bb.get_x()+bb.get_width()/2, v+0.005, f"{v*100:.1f}%", ha="center")
src = rb["by_source"]; sl = list(src.keys()); sv = [src[s]["acc"] for s in sl]
b2 = axes[1].bar(sl, sv, color="#4285f4", width=0.5)
axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0.85, 1.01); axes[1].set_title("By Source")
for bb, v in zip(b2, sv): axes[1].text(bb.get_x()+bb.get_width()/2, v+0.005, f"{v*100:.1f}%", ha="center")
plt.suptitle("Robustness — Balanced CRNN")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "best_robustness.png")); plt.close()

print(f"wrote 10 plots -> {OUT}")
for f in sorted(os.listdir(OUT)): print(" ", f)
