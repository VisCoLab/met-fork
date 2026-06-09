#!/bin/bash
# Step 2: reproduce the DINOv3 Met GAP IN OUR PIPELINE. Assemble the raw DINOv3-7B
# CLS features (already extracted by the art-research repo over the byte-identical
# Met split) into our descriptors.pkl, then run OUR eval_fullgrid.py (our PCAw +
# faiss kNN + GAP/GAP-/ACC). Target: ~50 GAP (DINOv3 paper = DINOv2 40.0 + 10.8 = 50.8).
# CPU-only (faiss-CPU + numpy) -> standard partition, leaves the GPUs for training.
#SBATCH --account=pl0896-03
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --job-name=met-dinov3-eval
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs data/dinov3_descriptors
export PYTHONUNBUFFERED=1
AR=/mnt/storage_6/project_data/pl0896-03/art-research
FEATS="$AR/experiments/met_protocol/features/dinov3_vit7b16_aspect512"

echo "host=$(hostname) start=$(date)"

echo "=== assemble raw DINOv3-7B (aspect512, 4096-d) -> our descriptors.pkl ==="
.venv/bin/python scripts/build_dinov3_pkl.py \
    --features-dir "$FEATS" --model dinov3_vit7b16 --variant aspect512 \
    --out-dir data/dinov3_descriptors || { echo "build failed"; exit 1; }

echo "=== eval with OUR full K x tau grid (PCAw 4096->512 + faiss kNN + GAP) ==="
.venv/bin/python scripts/eval_fullgrid.py data/dinov3_descriptors data/ground_truth 512

echo "end=$(date)"
