import json, os
archs = ["mlp", "tcn", "cnn", "conformer_s", "crnn"]
hdr = f"{'arch':<14}{'epochs':>7}{'best_ep':>9}{'best_val_ba':>13}{'final_val_acc':>15}{'train_s':>10}{'ckpt_MB':>10}"
print(hdr)
print("-" * len(hdr))
for a in archs:
    p = f"models/{a}_seed0/train_log.json"
    cp = f"models/{a}_seed0/best.pt"
    d = json.load(open(p))
    assert isinstance(d, list)
    n = len(d)
    # best by val_balanced_acc
    best = max(d, key=lambda r: r["val_balanced_acc"])
    last = d[-1]
    total_s = sum(r["epoch_seconds"] for r in d)
    ck = os.path.getsize(cp) / 1e6 if os.path.exists(cp) else None
    row = (f"{a:<14}{n:>7}{best['epoch']:>9}"
           f"{best['val_balanced_acc']:>13.4f}{last['val_acc']:>15.4f}"
           f"{total_s:>10,.0f}{(f'{ck:.2f}' if ck else '-'):>10}")
    print(row)
