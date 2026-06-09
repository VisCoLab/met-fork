#!/bin/bash
# CLOSED-WORLD painting-only run: the SAME step-1 recipe (R18-SWSL Con-Syn+Real-closest)
# trained on the painting subset only -- train DB, val, and test are all paintings
# (Classification=="Paintings", 4,898 classes / 12,403 imgs; data/gt_paint, built by
# scripts/build_paintings_data.py). NOT comparable to the full-DB GAP 36.1; a
# self-contained painting benchmark aligned with the synthetic experiment.
# Submit:  sbatch slurm/paint_train.slurm                                  # real baseline (data/gt_paint)
#   real/synth mix run (eval still on real via slurm/paint_eval.slurm):
#     sbatch --job-name=met-tr-40r60s slurm/paint_train.slurm data/gt_paint_mix_40r60s data/aug paint_40r60s
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00             # ~12k imgs (~32x less than 397k) -> ~1h; generous headroom
#SBATCH --job-name=met-paint
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/models

INFO="${1:-data/gt_paint}"; IMROOT="${2:-data/}"; TAG="${3:-paint}"   # mix sweep: <mix_dir> data/aug <tag>
AUG="${4:-base}"   # EXP-9 phone-photo augmentation arm: base|jpeg|blur|sensor|phoneall

export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  info=$INFO  im_root=$IMROOT  tag=$TAG  aug=$AUG  start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Identical to slurm/train.slurm; ONLY --info_dir (closed painting world) and --aug (EXP-9) change.
.venv/bin/python -m code.examples.train_contrastive \
    ./data/models/r18SWSL_${TAG} \
    --net r18_sw-sup --pretrained \
    --pairs_type new_pos+new_neg \
    --emb_proj --pca \
    --seed 0 \
    --aug "$AUG" \
    --info_dir "$INFO" \
    --im_root "$IMROOT" \
    --gpuid 0

echo "end=$(date)"
