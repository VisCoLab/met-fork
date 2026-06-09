#!/bin/bash
# Evaluate any checkpoint on the FULL Met retrieval benchmark: extract MS descriptors over
# the original 397k studio DB + the full test/val queries (incl. distractors), then full-grid
# eval (GAP / GAP- / ACC, tuned on val) + the paintings-slice eval. Comparable to the paper
# (GAP 36.1) and to EXP-2's painting slice.
# usage: sbatch --job-name=met-full-<tag> eval_full.slurm <model_dir> [epoch=10] [tag=full]
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"; export PYTHONUNBUFFERED=1
MODELDIR="${1:?usage: sbatch eval_full.slurm <model_dir> [epoch] [tag]}"; EP="${2:-10}"; TAG="${3:-full}"
CKPT=$(ls $MODELDIR/*_epoch:_$EP 2>/dev/null | head -1)
OUT="data/descriptors_full_${TAG}"; D="$OUT/r18_contr_loss_gem_fc_swsl_ms"
echo "model=$MODELDIR epoch=$EP tag=$TAG ckpt=$CKPT start=$(date)"
[ -z "$CKPT" ] && { echo "ERROR: no epoch-$EP checkpoint in $MODELDIR/"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== extract MS descriptors (FULL 397k studio DB + 19,319 queries) ==="
.venv/bin/python -m code.examples.extract_descriptors "$OUT" \
    --net r18_contr_loss_gem_fc_swsl --netpath "$CKPT" --ms \
    --info_dir data/ground_truth --im_root data/ --gpuid 0
echo "=== full benchmark eval (full K x tau grid, tuned on val) ==="
.venv/bin/python scripts/eval_fullgrid.py "$D" data/ground_truth 512
echo "=== paintings-slice eval (strict/broad, full 397k DB) ==="
.venv/bin/python scripts/eval_paintings.py "$D"
echo "end=$(date)"
