#!/bin/bash
# Finish step 1: extract multi-scale descriptors from our trained epoch-10 checkpoint,
# then evaluate with the full K x tau grid. Submit: sbatch slurm/extract_eval.slurm
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --job-name=met-extract-eval
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/descriptors
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"
export PYTHONUNBUFFERED=1

CKPT="data/models/r18SWSL_con-syn+real-closest/method:_contrastive_net:_r18_sw-sup_bckbn_lr:1e-07_b_size:_64_epochs:_10_wdecay:_1e-06_margin:_1.8_schedstep:_6_schedgamma:_0.1_imsize:_500_pairs_type:_new_pos+new_neg_pca_emb_proj_pretrained_seed:_0_epoch:_10"

echo "host=$(hostname) start=$(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== extract multi-scale descriptors (epoch 10) ==="
.venv/bin/python -m code.examples.extract_descriptors \
    data/descriptors --net r18_contr_loss_gem_fc_swsl --netpath "$CKPT" --ms \
    --info_dir data/ground_truth --im_root data/ --gpuid 0

echo "=== eval (full K x tau grid) ==="
.venv/bin/python scripts/eval_fullgrid.py data/descriptors/r18_contr_loss_gem_fc_swsl_ms data/ground_truth 512

echo "end=$(date)"
