#!/usr/bin/env python3
"""Generate all report figures for the VCM ME2 project."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
PLOT_DIR = REPORT_DIR / "plots"
MODEL_DIR = ROOT / "models"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "PLAY_MUSIC","WEATHER","TIME","LIGHT_ON","LIGHT_OFF","PAUSE","STOP","NEXT",
    "VOLUME_UP","VOLUME_DOWN","CALL","MESSAGE","LIST_REMINDERS","TIMER","ALARM",
    "TEMPERATURE","BRIGHTNESS","COLOR","CREATE_REMINDER","OUT_OF_SCOPE",
]
SHORT = [
    "Play\nMusic","Weather","Time","Light\nOn","Light\nOff","Pause","Stop","Next",
    "Vol\nUp","Vol\nDown","Call","Message","List\nRem.","Timer","Alarm",
    "Temp.","Bright.","Color","Create\nRem.","OOS",
]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})

def fig_architecture():
    """Block diagram of the best model (CRNN)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("VCM Architecture — CRNN (CNN + BiLSTM)", fontsize=13, fontweight="bold")
    
    blocks = [
        (0.3, 1.5, 1.4, 1.0, "Input\nMFCC(40)\n× 100 frames", "#4ECDC4"),
        (2.2, 1.5, 1.6, 1.0, "Conv1d\n64 filters\nk=5, pool2", "#45B7D1"),
        (4.2, 1.5, 1.6, 1.0, "Conv1d\n128 filters\nk=5, pool2", "#96CEB4"),
        (6.2, 1.5, 1.6, 1.0, "BiLSTM\n2 layers\n128 hidden", "#FFEAA7"),
        (8.2, 1.5, 1.4, 1.0, "FC\n20 classes", "#DDA0DD"),
    ]
    for x, y, w, h, txt, color in blocks:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                         facecolor=color, edgecolor="#333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=8, fontweight="bold")
    
    for x1, x2 in [(1.7, 2.2), (3.8, 4.2), (5.8, 6.2), (7.8, 8.2)]:
        ax.annotate("", xy=(x2, 2.0), xytext=(x1, 2.0),
                     arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    
    # Dimensions
    dims = ["(B, 100, 40)", "(B, 64, 50)", "(B, 128, 25)", "(B, 25, 256)", "(B, 20)"]
    xs = [1.0, 3.0, 5.0, 7.0, 8.9]
    for x, d in zip(xs, dims):
        ax.text(x, 1.2, d, ha="center", va="top", fontsize=7, color="#666", style="italic")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "architecture_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  architecture_diagram.png")

def fig_training_curves():
    """Training curves for all 5 models."""
    with open(MODEL_DIR / "benchmark_summary.json") as f:
        bench = json.load(f)
    
    colors = {"logistic":"#999", "dnn":"#45B7D1", "cnn1d":"#96CEB4", "crnn":"#FFEAA7", "transformer":"#DDA0DD"}
    markers = {"logistic":"o", "dnn":"s", "cnn1d":"^", "crnn":"D", "transformer":"v"}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for name, r in bench.items():
        h = r["history"]
        c = colors.get(name, "#333")
        m = markers.get(name, "o")
        axes[0].plot(h["epoch"], h["train_loss"], color=c, marker=m, markersize=3, label=name, alpha=0.8)
        axes[1].plot(h["epoch"], h["val_acc"], color=c, marker=m, markersize=3, label=name, alpha=0.8)
    
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Accuracy")
    axes[1].set_title("Holdout Validation Accuracy"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    
    plt.suptitle("Training Curves — 5 Architectures", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_curves_5models.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  training_curves_5models.png")

def fig_benchmark_accuracy():
    """Bar chart: test accuracy per model."""
    with open(MODEL_DIR / "benchmark_summary.json") as f:
        bench = json.load(f)
    
    names = list(bench.keys())
    accs = [bench[n]["test_acc"] for n in names]
    params = [bench[n]["params"] for n in names]
    colors = ["#999","#45B7D1","#96CEB4","#FFEAA7","#DDA0DD"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    bars = ax1.barh(names, accs, color=colors, edgecolor="#333")
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f"{acc:.4f}", va="center", fontsize=9)
    ax1.set_xlim(0, 1.05); ax1.set_xlabel("Test Accuracy")
    ax1.set_title("Test Accuracy by Architecture"); ax1.grid(axis="x", alpha=0.3)
    
    bars2 = ax2.barh(names, [p/1e6 for p in params], color=colors, edgecolor="#333")
    for bar, p in zip(bars2, params):
        ax2.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f"{p:,}", va="center", fontsize=8)
    ax2.set_xlabel("Parameters (M)"); ax2.set_title("Model Size"); ax2.grid(axis="x", alpha=0.3)
    
    plt.suptitle("5-Architecture Benchmark", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "benchmark_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  benchmark_accuracy.png")

def fig_per_intent_recall():
    """Per-intent recall bar chart."""
    with open(REPORT_DIR / "evaluation_results.json") as f:
        ev = json.load(f)
    
    intents = [r["intent"] for r in ev["per_intent"]]
    recalls = [r["recall"] for r in ev["per_intent"]]
    f1s = [r["f1"] for r in ev["per_intent"]]
    
    colors = ["#FF6B6B" if r < 0.9 else "#FFEAA7" if r < 0.95 else "#96CEB4" for r in recalls]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(intents))
    bars1 = ax.bar(x - 0.15, recalls, 0.3, label="Recall", color=colors, edgecolor="#333")
    bars2 = ax.bar(x + 0.15, f1s, 0.3, label="F1", color="#45B7D1", edgecolor="#333", alpha=0.7)
    
    ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Per-Intent Recall and F1 (Test Set)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "per_intent_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  per_intent_recall.png")

def fig_confusion_matrix():
    """Confusion matrix heatmap."""
    with open(REPORT_DIR / "evaluation_results.json") as f:
        ev = json.load(f)
    cm = np.array(ev["confusion_matrix"])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.YlOrRd)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(SHORT, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    
    thresh = cm.max() / 2
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j] > 0:
                color = "white" if cm[i,j] > thresh else "black"
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", color=color, fontsize=6)
    
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  confusion_matrix.png")

def fig_calibration():
    """Reliability diagram + histogram."""
    with open(REPORT_DIR / "evaluation_results.json") as f:
        ev = json.load(f)
    # We don't have raw probs saved, so plot ECE/Brier as summary
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["ECE", "Brier"], [ev["calibration"]["ece"], ev["calibration"]["brier"]],
           color=["#45B7D1", "#96CEB4"], edgecolor="#333")
    ax.set_ylabel("Error")
    ax.set_title("Calibration Errors")
    for i, v in enumerate([ev["calibration"]["ece"], ev["calibration"]["brier"]]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  calibration.png")

def fig_latency():
    """Latency distribution."""
    with open(REPORT_DIR / "evaluation_results.json") as f:
        ev = json.load(f)
    lat = ev["latency"]
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["mean", "p50", "p95", "p99", "max"]
    vals = [lat[f"{m}_ms"] for m in metrics]
    bars = ax.bar(metrics, vals, color="#45B7D1", edgecolor="#333")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}ms", ha="center", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Inference Latency (CPU, batch=1, n={lat['n_runs']})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  latency.png")

def fig_class_labels():
    """Class labels with example phrases."""
    import pandas as pd
    var = pd.read_csv(ROOT / "data" / "hf" / "variations.csv")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    ax.set_title("VCM Class Labels — 19 Intents + REJECT (93 command variations)", fontsize=13, fontweight="bold")
    
    y_pos = 0.95
    for _, grp in var.groupby("label", sort=False):
        label = grp["label"]
        phrases = grp["phrase"].tolist()
        ax.text(0.02, y_pos, f"{label}", fontsize=9, fontweight="bold", color="#2C3E50",
                transform=ax.transAxes, va="top")
        ax.text(0.25, y_pos, "  |  ".join(phrases[:3]), fontsize=7, color="#666",
                transform=ax.transAxes, va="top")
        y_pos -= 0.045
    
    ax.text(0.02, y_pos - 0.02, "OUT_OF_SCOPE (REJECT)", fontsize=9, fontweight="bold", color="#E74C3C",
            transform=ax.transAxes, va="top")
    ax.text(0.25, y_pos - 0.02, "Synthetic negatives: noise, babble, reversed, truncated, silence", fontsize=7,
            color="#666", transform=ax.transAxes, va="top")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_labels.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  class_labels.png")

def fig_rejection():
    """Rejection performance."""
    with open(REPORT_DIR / "evaluation_results.json") as f:
        ev = json.load(f)
    rej = ev["rejection"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["OOS Recall", "False Reject Rate"], [rej["oos_recall"], rej["false_reject"]],
                  color=["#96CEB4", "#FF6B6B"], edgecolor="#333")
    for bar, v in zip(bars, [rej["oos_recall"], rej["false_reject"]]):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylabel("Rate"); ax.set_ylim(0, 1.1)
    ax.set_title("Rejection Performance (OUT_OF_SCOPE)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "rejection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  rejection.png")

def fig_params_vs_acc():
    """Params vs accuracy scatter."""
    with open(MODEL_DIR / "benchmark_summary.json") as f:
        bench = json.load(f)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"logistic":"#999", "dnn":"#45B7D1", "cnn1d":"#96CEB4", "crnn":"#FFEAA7", "transformer":"#DDA0DD"}
    for name, r in bench.items():
        ax.scatter(r["params"], r["test_acc"], s=100, color=colors.get(name,"#333"),
                   edgecolor="#333", zorder=5, label=name)
        ax.annotate(name, (r["params"], r["test_acc"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Parameter Count vs Test Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "params_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  params_vs_accuracy.png")

def main():
    print("Generating figures...")
    fig_architecture()
    fig_training_curves()
    fig_benchmark_accuracy()
    fig_per_intent_recall()
    fig_confusion_matrix()
    fig_calibration()
    fig_latency()
    fig_class_labels()
    fig_rejection()
    fig_params_vs_acc()
    print(f"All figures saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
