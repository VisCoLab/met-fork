#!/bin/bash
# Re-score the FULL-benchmark painting slice under the committed Classification=="Paintings"
# definition (148 queries) for the 6 mix models + the step-1 reference. CPU-only (faiss-CPU
# kNN over the 397k DB); reuses already-extracted descriptors (no GPU / no re-extraction).
# usage: sbatch eval_paint_cls.slurm
#SBATCH --account=pl0896-03
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=met-paint-cls
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
echo "start=$(date)"
.venv/bin/python scripts/eval_paintings_cls.py
echo "end=$(date)"
