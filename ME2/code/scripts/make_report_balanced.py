#!/usr/bin/env python3
"""Build the Excel evaluation report for the DEPLOYED balanced CRNN (Option 2b).

Reads:
  models_retrain/crnn_bal_1790835176_metrics.json   (per-intent P/R/F1, confusion, test metrics)
  models_retrain/crnn_bal_1790835176_history.json   (training curves)
  models_retrain/crnn_wob_1790834310_metrics.json   (Option 2 comparison)
  models_retrain/summary.json                       (unweighted baseline)
  validation_full.json                              (full-set latency + per-intent)
Writes:
  models_retrain/VCM_Model_Evaluation_Report.xlsx
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MR = os.path.join(ROOT, "models_retrain")
OUT_XLSX = os.path.join(MR, "VCM_Model_Evaluation_Report.xlsx")

CLASSES = ["REJECT","PLAY_MUSIC","QUESTION_SEARCH","LIGHTS_ON_OFF","DIM_COLOR_LIGHTS",
           "SET_TIMER","SET_ALARM","THERMOSTAT","MEDIA_CONTROL","REMINDERS_LISTS","CALLS_MESSAGING"]

HDR_FILL = PatternFill("solid", fgColor="1F4E79")
HDR_FONT = Font(color="FFFFFF", bold=True)
BEST_FILL = PatternFill("solid", fgColor="C6EFCE")
BAD_FILL  = PatternFill("solid", fgColor="FFC7CE")
WARN_FILL = PatternFill("solid", fgColor="FFEB9C")
TITLE_FONT = Font(bold=True, size=12)
THIN = Border(*[Side(style="thin", color="D9D9D9")]*4)


def load(p):
    with open(p) as f:
        return json.load(f)


def style_header(ws, row=1, ncols=None):
    ncols = ncols or ws.max_column
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def autofit(ws, widths=None):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        mx = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[letter].width = min(max(mx + 2, 10), 46)
    if widths:
        for i, w in widths.items():
            ws.column_dimensions[get_column_letter(i)].width = w


def border_block(ws, r0, r1, c0, c1):
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            ws.cell(row=r, column=c).border = THIN


def main():
    bal = load(os.path.join(MR, "crnn_bal_1790835176_metrics.json"))
    hist = load(os.path.join(MR, "crnn_bal_1790835176_history.json"))["history"]
    wob = load(os.path.join(MR, "crnn_wob_1790834310_metrics.json"))
    summ = load(os.path.join(MR, "summary.json"))[0]
    vf = load(os.path.join(ROOT, "validation_full.json"))

    t = bal["test"]
    pi = bal["per_intent_test"]
    cm = np.array(bal["confusion_matrix"])

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        wb = xw.book

        # ============================ 1. SUMMARY ============================
        rows = [
            ["Model", "CRNN (CNN -> BiLSTM -> 10-way head)"],
            ["Training scheme", "Option 2b: source-weight 3.0 (Option B) + inverse-freq loss weights (tf=0.5) + cap downsampling"],
            ["Epochs / batch / lr", f"{bal['epochs']} / {bal['bs']} / {bal['lr']}"],
            ["Device", bal["device"]],
            ["Parameters", f"{bal['params']:,}"],
            ["ONNX size INT8", "0.34 MB"],
            ["Train clips (after cap)", f"{bal['train_after_cap']:,}"],
            ["Best val epoch", bal["best_val_epoch"]],
            ["Best val accuracy", f"{bal['best_val_acc']*100:.2f}%"],
            ["Final train accuracy", f"{bal['final_train_acc']*100:.2f}%"],
            ["TEST ACCURACY", f"{t['accuracy']*100:.2f}%"],
            ["Cohen's kappa", f"{t['cohen_kappa']:.4f}"],
            ["Macro precision", f"{t['macro_precision']:.4f}"],
            ["Macro recall", f"{t['macro_recall']:.4f}"],
            ["Macro F1", f"{t['macro_f1']:.4f}"],
            ["Micro precision / recall / F1", f"{t['micro_precision']:.4f} / {t['micro_recall']:.4f} / {t['micro_f1']:.4f}"],
            ["Weighted precision / recall / F1", f"{t['weighted_precision']:.4f} / {t['weighted_recall']:.4f} / {t['weighted_f1']:.4f}"],
            ["Wall-clock train time", f"{bal['wall_seconds']:.0f} s"],
        ]
        df = pd.DataFrame(rows, columns=["Metric", "Value"])
        df.to_excel(xw, sheet_name="Summary", index=False, startrow=1)
        ws = wb["Summary"]
        ws.cell(row=1, column=1, value="VCM Model Evaluation — Deployed Balanced CRNN (Option 2b)").font = TITLE_FONT
        style_header(ws, row=2)
        border_block(ws, 3, 2 + len(rows), 1, 2)
        # highlight the test-accuracy row
        for r in range(3, 3 + len(rows)):
            if ws.cell(row=r, column=1).value == "TEST ACCURACY":
                ws.cell(row=r, column=2).fill = BEST_FILL
                ws.cell(row=r, column=2).font = Font(bold=True)
        autofit(ws, {1: 30, 2: 60})

        # ============================ 2. PER-INTENT P/R/F1 ============================
        rows = []
        for c in CLASSES:
            d = pi[c]
            rows.append([c, f"{d['precision']:.4f}", f"{d['recall']:.4f}",
                         f"{d['f1']:.4f}", d["support"], f"{d['acc']*100:.1f}%"])
        rows.append(["MACRO", f"{t['macro_precision']:.4f}", f"{t['macro_recall']:.4f}",
                     f"{t['macro_f1']:.4f}", sum(pi[c]['support'] for c in CLASSES),
                     f"{t['accuracy']*100:.2f}%"])
        df = pd.DataFrame(rows, columns=["Intent", "Precision", "Recall", "F1", "Support", "Acc"])
        df.to_excel(xw, sheet_name="Per-Intent", index=False, startrow=1)
        ws = wb["Per-Intent"]
        ws.cell(row=1, column=1, value="Per-intent precision / recall / F1 (held-out test set)").font = TITLE_FONT
        style_header(ws, row=2)
        for r in range(3, 3 + len(CLASSES)):
            f1 = float(ws.cell(row=r, column=4).value)
            fill = BEST_FILL if f1 >= 0.97 else (WARN_FILL if f1 >= 0.95 else BAD_FILL)
            for c in range(1, 7):
                ws.cell(row=r, column=c).fill = fill
                ws.cell(row=r, column=c).border = THIN
        # macro row bold
        mr = 3 + len(CLASSES)
        for c in range(1, 7):
            ws.cell(row=mr, column=c).font = Font(bold=True)
            ws.cell(row=mr, column=c).border = THIN
        autofit(ws)

        # ============================ 3. CONFUSION MATRIX ============================
        df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
        df.index.name = "true \\ pred"
        df.to_excel(xw, sheet_name="Confusion Matrix", startrow=1)
        ws = wb["Confusion Matrix"]
        ws.cell(row=1, column=1, value="Confusion matrix (rows=true, cols=predicted; test set n=%d)" % int(cm.sum())).font = TITLE_FONT
        # header row is row 2 (index label col + class names)
        for c in range(2, 2 + len(CLASSES)):
            cell = ws.cell(row=2, column=c)
            cell.fill = HDR_FILL; cell.font = HDR_FONT
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
        for r in range(2, 2 + len(CLASSES)):
            cell = ws.cell(row=r, column=1)
            cell.fill = HDR_FILL; cell.font = HDR_FONT
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                cell = ws.cell(row=3 + i, column=2 + j)
                v = int(cm[i, j])
                cell.alignment = Alignment(horizontal="center")
                cell.border = THIN
                if i == j:
                    cell.fill = BEST_FILL
                elif v > 0:
                    cell.fill = BAD_FILL
        autofit(ws, {1: 18})

        # ============================ 4. LATENCY & EFFICIENCY ============================
        lat = vf["latency_ms"]
        rows = [
            ["Validation set size", vf["n"]],
            ["Overall accuracy (full set)", f"{vf['overall_acc']*100:.2f}%"],
            ["Latency mean (ms)", round(lat["mean"], 2)],
            ["Latency p50 (ms)", round(lat["p50"], 2)],
            ["Latency p95 (ms)", round(lat["p95"], 2)],
            ["Latency p99 (ms)", round(lat["p99"], 2)],
            ["Latency max (ms)", round(lat["max"], 2)],
            ["Note", "Measured on the exact Pi inference path (INT8 ONNX, onnxruntime). First clip includes ~1.2s cold decode; figures above are warm."],
        ]
        df = pd.DataFrame(rows, columns=["Metric", "Value"])
        df.to_excel(xw, sheet_name="Latency & Efficiency", index=False, startrow=1)
        ws = wb["Latency & Efficiency"]
        ws.cell(row=1, column=1, value="On-device latency & efficiency (INT8 ONNX)").font = TITLE_FONT
        style_header(ws, row=2)
        border_block(ws, 3, 2 + len(rows), 1, 2)
        autofit(ws, {1: 28, 2: 70})

        # ============================ 5. TRAINING CURVES ============================
        tdf = pd.DataFrame(hist)[["epoch", "train_loss", "train_acc", "val_acc"]]
        tdf.to_excel(xw, sheet_name="Training Curves", index=False, startrow=1)
        ws = wb["Training Curves"]
        ws.cell(row=1, column=1, value="Per-epoch training history (Option 2b)").font = TITLE_FONT
        style_header(ws, row=2)
        border_block(ws, 3, 2 + len(tdf), 1, 4)
        autofit(ws)

        # ============================ 6. MODEL COMPARISON ============================
        rows = [
            ["Unweighted baseline", "—", f"{summ['best_val_acc']*100:.2f}%", f"{summ['test_acc']*100:.2f}%"],
            ["Option 2 (source-weight only)", f"{wob['best_val_acc']*100:.2f}%", f"{wob['test']['accuracy']*100:.2f}%", "Option B rows ×3.0"],
            ["Option 2b (balanced, DEPLOYED)", f"{bal['best_val_acc']*100:.2f}%", f"{t['accuracy']*100:.2f}%", "+ inverse-freq loss weights (tf=0.5)"],
        ]
        df = pd.DataFrame(rows, columns=["Configuration", "Best val acc", "Test acc", "Notes"])
        df.to_excel(xw, sheet_name="Model Comparison", index=False, startrow=1)
        ws = wb["Model Comparison"]
        ws.cell(row=1, column=1, value="Progression across training schemes (same 100-epoch budget)").font = TITLE_FONT
        style_header(ws, row=2)
        border_block(ws, 3, 5, 1, 4)
        # highlight deployed row
        for c in range(1, 5):
            ws.cell(row=5, column=c).fill = BEST_FILL
            ws.cell(row=5, column=c).font = Font(bold=True)
        autofit(ws, {1: 34, 4: 40})

    print(f"WROTE {OUT_XLSX}")
    print(f"Sheets: Summary, Per-Intent, Confusion Matrix, Latency & Efficiency, Training Curves, Model Comparison")


if __name__ == "__main__":
    main()
