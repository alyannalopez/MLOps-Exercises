#!/bin/bash
#SBATCH --job-name=vcm_crnn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00
#SBATCH --output=logs/vcm_%j.out

# NOTE: This sandbox has no reachable SLURM controller (sinfo fails), so this
# job was executed directly on an interactive A100 node instead of via sbatch.
# The pipeline below is identical to what the job would run.

set -e
VENV=/mnt/jfs_hpc/home/alyanna.may.lopez/sandbox/venv
PY=$VENV/bin/python
cd /mnt/jfs_hpc/home/alyanna.may.lopez/sandbox/ME2
mkdir -p logs models

echo "=== GPU ==="
nvidia-smi | head -12
$PY -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== Retrain CRNN (seed 42, same as local) ==="
CUDA_VISIBLE_DEVICES=0 ONLY_MODEL=crnn $PY code/train_models.py 2>&1 | tee logs/train.log

echo "=== Evaluate ==="
$PY code/evaluate.py 2>&1 | tee logs/eval.log

echo "=== Export ONNX (so it's ready for the Pi) ==="
$PY code/export_crnn_onnx.py 2>&1 | tee logs/export.log

echo "DONE"
