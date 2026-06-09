#!/bin/bash
# Close step 4: evaluate the FROM-SCRATCH +synth run (r18SWSL_scratch_synth, epoch 10) with the
# IDENTICAL protocol to step 1 (extract_eval.slurm) — extract MS descriptors over the ORIGINAL
# studio DB + real queries, then full-grid eval + paintings eval. This is the clean A/B vs step 1
# (same 10-epoch recipe, only +synthetic), so it must use the same epoch (10) and same eval. Submit:
#   sbatch extract_eval_scratch.slurm
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --job-name=met-eval-scratch-synth
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"; export PYTHONUNBUFFERED=1
CKPT=$(ls data/models/r18SWSL_scratch_synth/*_epoch:_10 2>/dev/null | head -1)
OUT="data/descriptors_scratch_synth"; D="$OUT/r18_contr_loss_gem_fc_swsl_ms"
echo "start=$(date)"; echo "ckpt=$CKPT"
[ -z "$CKPT" ] && { echo "ERROR: no epoch-10 checkpoint found"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== extract multi-scale descriptors (DB = original 397k studio + real queries) ==="
.venv/bin/python -m code.examples.extract_descriptors "$OUT" \
    --net r18_contr_loss_gem_fc_swsl --netpath "$CKPT" --ms \
    --info_dir data/ground_truth --im_root data/ --gpuid 0
echo "=== full benchmark eval (full K x tau grid) ==="
.venv/bin/python scripts/eval_fullgrid.py "$D" data/ground_truth 512
echo "=== paintings eval (strict/broad) ==="
.venv/bin/python scripts/eval_paintings.py "$D"
echo "end=$(date)"
