#!/bin/bash
# Evaluate a fine-tuned model (epoch 5) like steps 1-2: extract MS descriptors over the
# ORIGINAL studio DB + real queries, then full-grid eval + paintings eval.
# usage: sbatch --job-name=met-fteval-synth slurm/extract_eval_ft.slurm synth
#        sbatch --job-name=met-fteval-combined slurm/extract_eval_ft.slurm combined
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out

VARIANT="${1:?usage: sbatch slurm/extract_eval_ft.slurm <combined|synth>}"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"; export PYTHONUNBUFFERED=1
CKPT=$(ls data/models/r18SWSL_ft_${VARIANT}/*_epoch:_5 2>/dev/null | head -1)
OUT="data/descriptors_ft_${VARIANT}"; D="$OUT/r18_contr_loss_gem_fc_swsl_ms"
echo "variant=$VARIANT start=$(date)"; echo "ckpt=$CKPT"
[ -z "$CKPT" ] && { echo "ERROR: no epoch-5 checkpoint found"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== extract multi-scale descriptors (DB = original 397k studio + real queries) ==="
.venv/bin/python -m code.examples.extract_descriptors "$OUT" \
    --net r18_contr_loss_gem_fc_swsl --netpath "$CKPT" --ms \
    --info_dir data/ground_truth --im_root data/ --gpuid 0
echo "=== full benchmark eval (full K x tau grid) ==="
.venv/bin/python scripts/eval_fullgrid.py "$D" data/ground_truth 512
echo "=== paintings eval (Classification==Paintings, 148) ==="
.venv/bin/python scripts/eval_paintings.py "$D"
echo "end=$(date)"
