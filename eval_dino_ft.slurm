#!/bin/bash
# Eval an adapted DINOv3 model on the FULL Met benchmark + paintings, exactly like
# every other model: extract over the ORIGINAL 397k studio DB + real queries
# (.venv-dino, transformers), then full-grid eval + paintings eval (proven .venv).
# usage: sbatch --job-name=met-dinoeval-head-synth eval_dino_ft.slurm head_synth
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out

VARIANT="${1:?usage: eval_dino_ft.slurm <head_synth|lora_synth|head_studio|lora_studio>}"
LORA=""; [[ "$VARIANT" == lora_* ]] && LORA="--lora"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export HF_HOME=/mnt/storage_6/project_data/pl0896-03/art-research/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
CKPT=$(ls data/models/dinov3L_${VARIANT}/*_epoch:_3 2>/dev/null | head -1)
OUT="data/descriptors_dino_${VARIANT}"
echo "variant=$VARIANT lora=[$LORA] ckpt=$CKPT start=$(date)"
[ -z "$CKPT" ] && { echo "ERROR: no epoch-3 checkpoint in data/models/dinov3L_${VARIANT}/"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== extract over original 397k studio DB + real queries (.venv-dino) ==="
.venv-dino/bin/python scripts/extract_dino_ckpt.py --net dinov3_vitl16 $LORA \
    --netpath "$CKPT" --out-dir "$OUT" --info-dir data/ground_truth --im-root data/ \
    || { echo "extract failed"; exit 1; }

echo "=== full benchmark eval (PCAw 1024->512 + kNN K x tau grid), proven .venv ==="
.venv/bin/python scripts/eval_fullgrid.py "$OUT" data/ground_truth 512 | tee "$OUT/fullgrid.log"

K=$(grep "BEST by val GAP" "$OUT/fullgrid.log" | grep -oP "'K': \K[0-9]+"); K=${K:-5}
TAU=$(grep "BEST by val GAP" "$OUT/fullgrid.log" | grep -oP "'t': \K[0-9.]+"); TAU=${TAU:-20}
echo "=== paintings eval (strict/broad) at tuned K=$K tau=$TAU ==="
.venv/bin/python scripts/eval_paintings.py "$OUT" data/ground_truth data/MetObjects.csv "$K" "$TAU" 512
echo "end=$(date)"
