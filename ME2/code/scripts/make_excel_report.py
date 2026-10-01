#!/usr/bin/env python3
"""Build the multi-sheet Excel evaluation report from models_v3/eval_results.json."""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "models_v3")

CLASSES = ["REJECT","PLAY_MUSIC","QUESTION_SEARCH","LIGHTS_ON_OFF","DIM_COLOR_LIGHTS",
           "SET_TIMER","SET_ALARM","THERMOSTAT","MEDIA_CONTROL","REMINDERS_LISTS","CALLS_MESSAGING"]
MODELS = ["logistic", "dnn", "cnn1d", "crnn"]

HDR_FILL = PatternFill("solid", fgColor="1F4E79")
HDR_FONT = Font(color="FFFFFF", bold=True)
BEST_FILL = PatternFill("solid", fgColor="C6EFCE")
BAD_FILL  = PatternFill("solid", fgColor="FFC7CE")
WARN_FILL = PatternFill("solid", fgColor="FFEB9C")
THIN = Border(*[Side(style="thin", color="D9D9D9")]*4)

def style_header(ws, row=1, ncols=None):
    ncols = ncols or ws.max_column
    for c in range(1, ncols+1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HDR_FILL; cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def autofit(ws, widths=None):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        mx = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[letter].width = min(max(mx + 2, 10), 42)
    if widths:
        for i, w in widths.items():
            ws.column_dimensions[get_column_letter(i)].width = w

def mark_best(ws, col_idx, values, higher_is_better=True):
    """Highlight the best cell in a column of numbers."""
    nums = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if not nums: return
    best = max(nums, key=lambda t: t[1])[1] if higher_is_better else min(nums, key=lambda t: t[1])[1]
    for i, v in nums:
        if v == best:
            ws.cell(row=i+2, column=col_idx).fill = BEST_FILL

def main():
    with open(os.path.join(OUTDIR, "eval_results.json")) as f:
        R = json.load(f)
    out_xlsx = os.path.join(OUTDIR, "VCM_Model_Evaluation_Report.xlsx")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        wb = xw.book

        # ============================ 1. SUMMARY ============================
        rows = []
        for m in MODELS:
            r = R[m]
            tc = r["task_completion"]; lat = r["latency"]
            rows.append({
                "Model": m,
                "Features": r["kind"],
                "Params": f"{r['params']:,}",
                "Size FP32 (MB)": r["size_fp32_mb"],
                "Size INT8 (MB)": r["size_int8_mb"],
                "MACs/clip (M)": r["macs_millions"],
                "Train time (s)": r["train_wall_seconds"],
                "Best val acc": f"{r['best_val_acc']*100:.2f}% (ep {r['best_val_epoch']})",
                "Final train acc": f"{r['final_train_acc']*100:.2f}%",
                "TEST ACCURACY": f"{r['test_accuracy']*100:.2f}%",
                "Task success rate": f"{tc['overall_success_rate']*100:.2f}%",
                "Command success": f"{tc['command_success_rate']*100:.2f}%",
                "False triggers": tc["false_triggers"],
                "Missed commands": tc["missed_commands"],
                "Latency p50 (ms)": lat["single_p50_ms"],
                "Latency p95 (ms)": lat["single_p95_ms"],
                "Latency p99 (ms)": lat["single_p99_ms"],
                "Latency max (ms)": lat["single_max_ms"],
                "Throughput (clips/s)": lat["batch_clips_per_sec"],
                "WER (ASR leg)": f"{R['logistic']['wer'].get('overall_wer', float('nan'))*100:.1f}%",
            })
        df = pd.DataFrame(rows)
        df.to_excel(xw, sheet_name="Summary", index=False, startrow=1)
        ws = wb["Summary"]
        ws.cell(row=1, column=1, value="VCM Model Evaluation — 4 architectures, 100 epochs, seed 42, test set n=2,883").font = Font(bold=True, size=12)
        style_header(ws, row=2)
        for c in range(1, ws.max_column+1):
            for rr in range(3, ws.max_row+1):
                ws.cell(row=rr, column=c).border = THIN
        # highlight best test accuracy column
        vals = [float(R[m]["test_accuracy"])*100 for m in MODELS]
        mark_best(ws, 10, vals, True)
        autofit(ws, {1: 11, 20: 14})

        # ============================ 2. PER-INTENT RECALL ============================
        rows = []
        for c in CLASSES:
            row = {"Intent": c}
            for m in MODELS:
                pr = R[m]["per_intent_recall"][c]
                row[m] = f"{pr['recall']*100:.1f}%  ({pr['correct']}/{pr['n']})" if pr else "—"
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_excel(xw, sheet_name="Per-Intent Recall", index=False, startrow=1)
        ws = wb["Per-Intent Recall"]
        ws.cell(row=1, column=1, value="Command recall per intent (held-out test set)").font = Font(bold=True, size=12)
        style_header(ws, row=2)
        # color-code: <85% red, 85-93% yellow
        for rr in range(3, ws.max_row+1):
            for ci in range(2, ws.max_column+1):
                cell = ws.cell(row=rr, column=ci)
                v = str(cell.value)
                if v.startswith("—"): continue
                pct = float(v.split("%")[0])
                if pct < 85: cell.fill = BAD_FILL
                elif pct < 93: cell.fill = WARN_FILL
                else: cell.fill = BEST_FILL
                cell.border = THIN
        autofit(ws)

        # ============================ 3. CONFUSION MATRICES ============================
        for m in MODELS:
            cm = np.array(R[m]["confusion_matrix"])
            cm_pct = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
            df = pd.DataFrame(cm_pct, index=CLASSES, columns=CLASSES)
            df.index.name = "true \\ pred"
            df.to_excel(xw, sheet_name=f"Confusion-{m}", startrow=2)
            ws = wb[f"Confusion-{m}"]
            ws.cell(row=1, column=1, value=f"{m} — confusion matrix (% of each true class)").font = Font(bold=True, size=12)
            style_header(ws, row=2, ncols=len(CLASSES)+1)
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    cell = ws.cell(row=3+i, column=2+j)
                    v = cell.value
                    if i == j:
                        cell.fill = BEST_FILL if v >= 93 else (WARN_FILL if v >= 85 else BAD_FILL)
                    elif v >= 5:
                        cell.fill = BAD_FILL
                    cell.border = THIN
            autofit(ws, {1: 20})

        # ============================ 4. LATENCY & EFFICIENCY ============================
        rows = []
        for m in MODELS:
            r = R[m]; lat = r["latency"]
            rows.append({
                "Model": m,
                "Params": f"{r['params']:,}",
                "FP32 (MB)": r["size_fp32_mb"],
                "INT8 (MB)": r["size_int8_mb"],
                "MACs/clip (M)": r["macs_millions"],
                "Train wall (s)": r["train_wall_seconds"],
                "Single p50 (ms)": lat["single_p50_ms"],
                "Single p95 (ms)": lat["single_p95_ms"],
                "Single p99 (ms)": lat["single_p99_ms"],
                "Single max (ms)": lat["single_max_ms"],
                "Batch throughput (clips/s)": lat["batch_clips_per_sec"],
                "n measured": lat["n_measured"],
                "Model size / test-acc ratio (KB/%)": round(r["size_int8_mb"]*1000 / (r["test_accuracy"]*100), 2),
            })
        df = pd.DataFrame(rows)
        df.to_excel(xw, sheet_name="Latency & Efficiency", index=False, startrow=1)
        ws = wb["Latency & Efficiency"]
        ws.cell(row=1, column=1, value="Inference latency (single-clip, warm, Apple Silicon MPS) + efficiency").font = Font(bold=True, size=12)
        style_header(ws, row=2)
        for c in range(1, ws.max_column+1):
            for rr in range(3, ws.max_row+1):
                ws.cell(row=rr, column=c).border = THIN
        autofit(ws)

        # ============================ 5. TASK COMPLETION ============================
        rows = []
        for m in MODELS:
            tc = R[m]["task_completion"]
            rows.append({
                "Model": m,
                "Commands (true intent)": tc["commands_total"],
                "Commands correctly executed": tc["commands_ok"],
                "Command success rate": f"{tc['command_success_rate']*100:.2f}%",
                "Rejects (non-commands)": tc["rejects_total"],
                "Rejects correctly ignored": tc["rejects_ok"],
                "False triggers (noise -> action)": tc["false_triggers"],
                "Missed commands (cmd -> reject)": tc["missed_commands"],
                "Wrong-intent executions": tc["commands_total"] - tc["commands_ok"] - tc["missed_commands"],
                "OVERALL TASK SUCCESS RATE": f"{tc['overall_success_rate']*100:.2f}%",
            })
        df = pd.DataFrame(rows)
        df.to_excel(xw, sheet_name="Task Completion", index=False, startrow=1)
        ws = wb["Task Completion"]
        ws.cell(row=1, column=1, value="Task completion / success rate (command executed correctly OR noise correctly rejected)").font = Font(bold=True, size=12)
        style_header(ws, row=2)
        for c in range(1, ws.max_column+1):
            for rr in range(3, ws.max_row+1):
                ws.cell(row=rr, column=c).border = THIN
        mark_best(ws, 9, [float(R[m]["task_completion"]["overall_success_rate"])*100 for m in MODELS], True)
        autofit(ws)

        # ============================ 6. WER (ASR LEG) ============================
        wer_csv = os.path.join(OUTDIR, "wer_details.csv")
        if os.path.exists(wer_csv):
            wdf = pd.read_csv(wer_csv)
            # per-value summary
            pv = wdf.groupby(["intent", "value"]).agg(
                n=("wer", "size"), mean_wer=("wer", "mean"), max_wer=("wer", "max")).reset_index()
            pv["mean_wer"] = (pv["mean_wer"]*100).round(1)
            pv["max_wer"] = (pv["max_wer"]*100).round(1)
            pv.to_excel(xw, sheet_name="WER by Value", index=False, startrow=1)
            ws = wb["WER by Value"]
            ws.cell(row=1, column=1, value="Word Error Rate of the ASR leg (whisper-tiny) per value group — identical across all 4 intent models").font = Font(bold=True, size=12)
            style_header(ws, row=2)
            for rr in range(3, ws.max_row+1):
                cell = ws.cell(row=rr, column=3)
                v = cell.value
                if isinstance(v, (int, float)):
                    cell.fill = BAD_FILL if v > 50 else (WARN_FILL if v > 25 else BEST_FILL)
                for c in range(1, ws.max_column+1):
                    ws.cell(row=rr, column=c).border = THIN
            autofit(ws)
            # detail
            wdf.to_excel(xw, sheet_name="WER Details", index=False, startrow=1)
            ws = wb["WER Details"]
            ws.cell(row=1, column=1, value="Per-utterance ASR transcripts vs canonical references").font = Font(bold=True, size=12)
            style_header(ws, row=2)
            autofit(ws, {4: 40, 5: 40, 7: 60})

        # ============================ 7. TRAINING HISTORY ============================
        hist_frames = []
        for m in MODELS:
            hf = glob_hist(m)
            if hf is not None:
                hf["model"] = m
                hist_frames.append(hf)
        if hist_frames:
            hdf = pd.concat(hist_frames, ignore_index=True)
            # wide pivot for plotting sheet
            piv = hdf.pivot(index="epoch", columns="model", values="val_acc") * 100
            piv.to_excel(xw, sheet_name="Val Acc History", startrow=1)
            ws = wb["Val Acc History"]
            ws.cell(row=1, column=1, value="Validation accuracy per epoch (%)").font = Font(bold=True, size=12)
            style_header(ws, row=2, ncols=piv.shape[1]+1)
            autofit(ws, {1: 8})
            piv2 = hdf.pivot(index="epoch", columns="model", values="train_acc") * 100
            piv2.to_excel(xw, sheet_name="Train Acc History", startrow=1)
            ws = wb["Train Acc History"]
            ws.cell(row=1, column=1, value="Training accuracy per epoch (%)").font = Font(bold=True, size=12)
            style_header(ws, row=2, ncols=piv2.shape[1]+1)
            autofit(ws, {1: 8})

    print(f"wrote {out_xlsx}")

def glob_hist(m):
    import glob
    fs = glob.glob(os.path.join(OUTDIR, f"{m}_v3_*_history.json"))
    if not fs: return None
    with open(fs[0]) as f:
        d = json.load(f)
    return pd.DataFrame(d["history"])

if __name__ == "__main__":
    main()
