#!/bin/bash
# Extract frozen DINOv3 ViT-L CLS features over the 24,760 synthetic gallery renders,
# identical aspect512 preprocessing to art-research's Met extraction -> directly
# comparable to the already-extracted real Met ViT-L features. Batched (all 512x512).
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --job-name=met-synth-dino
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs data/synth_dino
# reuse the DINOv3 weights already downloaded by the art-research repo
export HF_HOME=/mnt/storage_6/project_data/pl0896-03/art-research/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

.venv-dino/bin/python scripts/extract_synth_dino.py \
    --model dinov3_vitl16 --long-side 512 --dtype bfloat16 \
    --out-dir data/synth_dino --batch-size 32 --num-workers 8

echo "end=$(date)"
