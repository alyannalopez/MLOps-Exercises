#!/usr/bin/env python3
"""Plots for the v3 evaluation: training curves + metric comparisons."""
from __future__ import annotations
import json, os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "models_v3")
PLOTS = os.path.join(OUTDIR, "plots")
os.makedirs(PLOTS, exist_ok=True)
MODELS = ["logistic", "dnn", "cnn1d", "crnn"]
COLORS = {"logistic": "#888888", "dnn": "#f4a261", "cnn1d": "#2a9d8f", "crnn": "#e76f51"}

def load_hist(m):
    fs = glob.glob(os.path.join(OUTDIR, f"{m}_v3_*_history.json"))
    if not fs: return None
    with open(fs[0]) as f:
        return json.load(f)

def main():
    with open(os.path.join(OUTDIR, "eval_results.json")) as f:
        R = json.load(f)
    H = {m: load_hist(m) for m in MODELS}

    # ---- 1. training curves (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, m in zip(axes.flat, MODELS):
        h = H[m]["history"]
        eps = [r["epoch"] for r in h]
        tr = [r["train_acc"]*100 for r in h]
        va = [r["val_acc"]*100 for r in h]
        ax.plot(eps, tr, color=COLORS[m], alpha=0.55, lw=1.2, label="train")
        ax.plot(eps, va, color=COLORS[m], lw=2, label="val")
        be = H[m]["result"]["best_val_epoch"]
        bv = H[m]["result"]["best_val_acc"]*100
        ax.axvline(be, color="k", ls=":", lw=0.8)
        ax.annotate(f"best {bv:.1f}% @ ep{be}", (be, bv), fontsize=8,
                    xytext=(be+3, bv-4))
        ax.set_title(f"{m}  (test {H[m]['result']['test_accuracy']*100:.1f}%)")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy %")
        ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("VCM retraining — 100 epochs, seed 42", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "training_curves.png"), dpi=130)
    plt.close(fig)

    # ---- 2. grouped bar: test acc + task success ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(MODELS)); w = 0.38
    ta = [R[m]["test_accuracy"]*100 for m in MODELS]
    ts = [R[m]["task_completion"]["overall_success_rate"]*100 for m in MODELS]
    b1 = axes[0].bar(x-w/2, ta, w, label="test accuracy", color=[COLORS[m] for m in MODELS])
    axes[0].bar(x+w/2, ts, w, label="task success rate", color=[COLORS[m] for m in MODELS], alpha=0.45)
    for xi, v in zip(x-w/2, ta): axes[0].text(xi, v+0.5, f"{v:.1f}", ha="center", fontsize=9)
    for xi, v in zip(x+w/2, ts): axes[0].text(xi, v+0.5, f"{v:.1f}", ha="center", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(MODELS)
    axes[0].set_ylim(0, 105); axes[0].set_ylabel("%")
    axes[0].set_title("Accuracy vs task completion"); axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    lat = [R[m]["latency"]["single_p50_ms"] for m in MODELS]
    thr = [R[m]["latency"]["batch_clips_per_sec"] for m in MODELS]
    b2 = axes[1].bar(x-w/2, lat, w, label="latency p50 (ms)", color=[COLORS[m] for m in MODELS])
    for xi, v in zip(x-w/2, lat): axes[1].text(xi, v+0.3, f"{v:.1f}ms", ha="center", fontsize=9)
    ax2 = axes[1].twinx()
    ax2.bar(x+w/2, thr, w, label="throughput (clips/s)", color=[COLORS[m] for m in MODELS], alpha=0.45)
    for xi, v in zip(x+w/2, thr): ax2.text(xi, v*1.01, f"{v:.0f}", ha="center", fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(MODELS)
    axes[1].set_ylabel("latency p50 (ms)"); ax2.set_ylabel("throughput (clips/s)")
    axes[1].set_title("Latency vs throughput (MPS, warm)")
    h1, l1 = axes[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1+h2, l1+l2, fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "metrics_comparison.png"), dpi=130)
    plt.close(fig)

    # ---- 3. per-intent recall heatmap (best model + worst model) ----
    CLASSES = ["REJECT","PLAY_MUSIC","QUESTION_SEARCH","LIGHTS_ON_OFF","DIM_COLOR_LIGHTS",
               "SET_TIMER","SET_ALARM","THERMOSTAT","MEDIA_CONTROL","REMINDERS_LISTS","CALLS_MESSAGING"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, m in zip(axes, ["logistic", "crnn"]):
        mat = np.array([[R[m]["per_intent_recall"][c]["recall"]*100
                          if R[m]["per_intent_recall"][c] else np.nan for c in CLASSES]])
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([0]); ax.set_yticklabels([m])
        for j, v in enumerate(mat[0]):
            if not np.isnan(v):
                ax.text(j, 0, f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if v < 50 else "black")
        ax.set_title(f"{m} — per-intent recall (%)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Command recall per intent (weakest vs strongest model)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "per_intent_recall.png"), dpi=130)
    plt.close(fig)

    # ---- 4. efficiency scatter: size vs accuracy ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in MODELS:
        sz = R[m]["size_int8_mb"]*1000  # KB
        acc = R[m]["test_accuracy"]*100
        ax.scatter(sz, acc, s=180, color=COLORS[m], zorder=3)
        ax.annotate(f"{m}\n{sz:.0f} KB INT8", (sz, acc), textcoords="offset points",
                    xytext=(10, 8), fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("INT8 model size (KB, log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Efficiency frontier — smaller model, same accuracy?")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "efficiency_frontier.png"), dpi=130)
    plt.close(fig)

    print(f"plots -> {PLOTS}")

if __name__ == "__main__":
    main()
