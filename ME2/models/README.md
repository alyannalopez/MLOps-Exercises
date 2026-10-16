# Models

This directory holds the **deployable ONNX artifacts** for the ME2 voice-command
model (v7). Raw PyTorch checkpoints (`*.pt`) are intentionally **not** committed —
see `.gitignore` — because the ONNX files are the actual deployment artifact and
are byte-for-byte reproducible from the checkpoint via `scripts/export_deploy_onnx.py`.

## Winner: `conformer_s` (two-head)

A small Conformer encoder with **two classification heads**:

- `intent_head` → 20-way intent logits (10 command categories × on/off variants)
- `slot_head` → 19-way slot logits (18 slot values + `NO_SLOT`)

This is the "new architecture" for v7: slots are decoded **directly by the neural
network** (no ASR anywhere in the pipeline).

| Property | Value |
|---|---|
| Parameters | **246,055** |
| Checkpoint | `conformer_s_seed0/best.pt` (0.99 MB, gitignored — regenerate below) |
| Test accuracy | **85.2 %** (95 % CI [84.2 %, 86.2 %]) |
| Balanced accuracy | **83.5 %** (95 % CI [82.3 %, 84.7 %]) |
| Macro F1 | 83.6 % |
| Real-voice accuracy | 43.5 % (synthetic 94.1 %) |
| Slot exact-match | **69.3 %** overall / **77.3 %** given correct intent |
| OOS false-accept | 31.0 % |

### Slot metrics (per slotted intent)

| Intent | Exact match | Given intent | MAE | Unit | Rel. err |
|---|---|---|---|---|---|
| TIMER | 75.7 % | 84.0 % | 4.59 | s | 20.1 % |
| ALARM | 79.7 % | 90.6 % | 1.20 | h | 14.3 % |
| COLOR | 74.0 % | 85.3 % | — | — | phonetic 0.22 |
| CREATE_REMINDER | 72.6 % | 77.9 % | — | — | phonetic 0.27 |
| BRIGHTNESS | 60.1 % | 67.7 % | 16.6 | % | 47.1 % |
| TEMPERATURE | 53.9 % | 58.9 % | 2.17 | °C | 9.7 % |

## Deployable ONNX (committed)

| File | Size | Outputs | Parity vs PyTorch |
|---|---|---|---|
| `conformer_s_float.onnx` | 1.03 MB | `intent_logits [B,20]`, `slot_logits [B,19]` | **128/128 (100 %)** argmax, max diff 5.7e-06 |
| `conformer_s_int8.onnx` | 0.32 MB | same | **126/128 (98.4 %)** argmax, max diff 0.48 |

- Export: `torch.onnx.export` (dynamo exporter, opset 14, dynamic batch), onnxscript 0.7.2.
- Quantization: ORT `quantize_dynamic` (int8). Requires stripping the dynamo
  `value_info` annotations + making output dims symbolic before quantizing (see
  `scripts/export_deploy_onnx.py`).
- Runtime: ONNX Runtime, **~0.67 ms/clip** end-to-end (≈1,494 clips/s) on Apple
  Silicon; sub-millisecond p50 on both float and int8.

## Regenerating from a checkpoint

```bash
./.venv/bin/python scripts/export_deploy_onnx.py \
  --checkpoint models/conformer_s_seed0/best.pt
```

Produces both ONNX files, runs the parity check, and copies them into
`pi_bundle/models/`.

## Other seed checkpoints (gitignored)

`cnn_seed0/`, `crnn_seed0/`, `mlp_seed0/`, `tcn_seed0/` hold the losing
architectures from the 5-model bake-off (see `results/comparison_top3.json`).
Their `*.pt` files are excluded from git; the comparison numbers live in `results/`.
