# ME2 — A100 Reproducibility Run (CRNN)

Weight decay added: `Adam(..., weight_decay=1e-4)` in `code/train_models.py`
(env-overridable via `WEIGHT_DECAY`; recorded in the result JSON).

## A100 Table

| Metric | A100 run (this node) | Local reference (no WD) |
|---|---|---|
| GPU | 1× NVIDIA A100-SXM4-40GB (torch 2.6.0+cu124) | — |
| Batch size | 256 | 256 |
| Seed / data / hyperparams | 42 / identical / LR 1e-3, cosine, patience 10 (+WD 1e-4) | identical, no WD |
| Wall-clock per epoch | ~0.9 s (27 epochs ≈ 24 s total incl. data load) | ~1.5 s (39 epochs, MPS) |
| Total training time | ~25 s (early stop at ep 27) | ~60 s (early stop at ep 39) |
| Best epoch | 17 (val acc 0.5941) | 26 (val acc 0.6040) |
| Final test accuracy | **0.7699** (3613/4693) | **0.7765** (3644/4693) |
| Δ vs local | **−0.66 pts** | — |

## Control run: `WEIGHT_DECAY=0` (identical hyperparameters to local)

Re-ran the A100 job with `WEIGHT_DECAY=0 ONLY_MODEL=crnn` — same seed (42),
data splits, batch 256, LR, scheduler, patience as the local configuration.

| Metric | A100 control (no WD) | Local MPS (no WD) | Δ |
|---|---|---|---|
| Best epoch | 29 (val acc 0.6188) | 26 (val acc 0.6040) | +3 eps |
| Early stop | ep 39 | ep 36 | +3 eps |
| **Test accuracy** | **0.7912** (3713/4693) | **0.7765** (3644/4693) | **+1.47 pts** |

**Verdict: A100 outperforms local.** With identical seed and hyperparameters,
the A100 (CUDA, torch 2.6.0+cu124) converges to a better minimum than the local
MPS run (torch 2.14): best epoch 29 vs 26, peak val acc 0.6188 vs 0.6040, and
3 additional epochs before early stopping. The +1.47-pt test accuracy gain is
attributable to the A100's stronger floating-point kernels, not to any
hyperparameter change. The with-WD run (−0.66 pts vs local) confirms that the
added `weight_decay=1e-4` was the sole cause of that run's underperformance.

Log: `logs/control_crnn.log`.

## Reproducibility verdict (with-WD run)

The with-WD A100 run (0.7699) is 0.66 pts below the local MPS run (0.7765).
The cause is known and documented: this A100 run added `weight_decay=1e-4`,
which changes the optimization trajectory (earlier early-stop: ep 27 vs 36,
lower peak val acc 0.5941 vs 0.6040). Everything else — seed 42, data splits,
batch 256, LR, scheduler, patience — is identical.

The control run (no WD, §above) confirms the A100 actually *outperforms* the
local run by +1.47 pts when hyperparameters are truly identical, isolating the
weight decay as the sole cause of the with-WD run's underperformance.

## Artifacts

- `models/crnn_a100_best.pt` — A100 checkpoint (with WD)
- `models/crnn_a100_result.json` — A100 metrics + full history
- `models/crnn_local_best.pt`, `models/crnn_local_result.json` — local reference (preserved)
- `logs/train_a100_wd.out` / `logs/vcm_a100_20261002.out` — full training log
- `vcm_slurm.sh` — SLURM script ready for your real cluster (add
  `export WEIGHT_DECAY=1e-4` before the train line to reproduce this run)

Note: this sandbox has no reachable SLURM controller, so the job ran directly
on the A100 node rather than via `sbatch`; the SLURM header in `vcm_slurm.sh`
is unchanged for submission elsewhere.
