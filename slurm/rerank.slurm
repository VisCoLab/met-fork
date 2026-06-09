#!/bin/bash
# Step 3: decisive geometric-rerank-for-GAP experiment (ViT-L stage-1, 100% patch coverage).
# Computes C2 PatchMatch scores on the DINOv3 top-50, fuses them into OUR canonical
# softmax-temperature GAP (lambda=0 = pure-CLS baseline built-in check). CPU-only
# (cv2 RANSAC + numpy), streams patch chunks (~17 GB peak) -> standard partition.
#SBATCH --account=pl0896-03
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=met-rerank
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs data/rerank
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
echo "host=$(hostname) start=$(date)"

.venv/bin/python scripts/patchmatch_rerank.py \
    --model dinov3_vitl16 --size 224 \
    --info-dir data/ground_truth --out-dir data/rerank

echo "end=$(date)"
