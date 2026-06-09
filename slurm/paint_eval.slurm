#!/bin/bash
# Closed-world painting eval: extract MS descriptors over the PAINTING DB (data/gt_paint)
# from a checkpoint, then 2-fold-CV eval (val=1 can't tune K/tau). The eval DB is ALWAYS
# the closed painting world, so this also serves as the control for any model.
# usage: sbatch slurm/paint_eval.slurm [model_dir=data/models/r18SWSL_paint] [epoch=10] [tag=paint]
#   control (full step-1 model on the closed world):
#     sbatch slurm/paint_eval.slurm data/models/r18SWSL_con-syn+real-closest 10 step1_on_paint
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --job-name=met-paint-eval
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"; export PYTHONUNBUFFERED=1
MODELDIR="${1:-data/models/r18SWSL_paint}"; EP="${2:-10}"; TAG="${3:-paint}"
CKPT=$(ls $MODELDIR/*_epoch:_$EP 2>/dev/null | head -1)
OUT="data/descriptors_${TAG}"; D="$OUT/r18_contr_loss_gem_fc_swsl_ms"
echo "model=$MODELDIR epoch=$EP tag=$TAG ckpt=$CKPT start=$(date)"
[ -z "$CKPT" ] && { echo "ERROR: no epoch-$EP checkpoint in $MODELDIR/"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== extract MS descriptors (DB = 12,403 painting studio imgs + 148 painting queries) ==="
.venv/bin/python -m code.examples.extract_descriptors "$OUT" \
    --net r18_contr_loss_gem_fc_swsl --netpath "$CKPT" --ms \
    --info_dir data/gt_paint --im_root data/ --gpuid 0
echo "=== closed-world paintings eval (2-fold CV) ==="
.venv/bin/python scripts/eval_paintings_closed.py "$D" data/gt_paint 512
echo "end=$(date)"
