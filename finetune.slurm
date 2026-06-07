#!/bin/bash
# Fine-tune the step-1 model (epoch 10) with synthetic gallery data, LR 1e-7.
# usage: sbatch --job-name=met-ft-combined finetune.slurm combined
#        sbatch --job-name=met-ft-synth    finetune.slurm synth
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x-%j.out

VARIANT="${1:?usage: sbatch finetune.slurm <combined|synth>}"
case "$VARIANT" in
  combined) INFO=data/gt_aug;   OUT=data/models/r18SWSL_ft_combined ;;
  synth)    INFO=data/gt_synth; OUT=data/models/r18SWSL_ft_synth ;;
  *) echo "unknown variant: $VARIANT"; exit 1 ;;
esac

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$OUT"
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"; export PYTHONUNBUFFERED=1
CKPT="data/models/r18SWSL_con-syn+real-closest/method:_contrastive_net:_r18_sw-sup_bckbn_lr:1e-07_b_size:_64_epochs:_10_wdecay:_1e-06_margin:_1.8_schedstep:_6_schedgamma:_0.1_imsize:_500_pairs_type:_new_pos+new_neg_pca_emb_proj_pretrained_seed:_0_epoch:_10"

echo "host=$(hostname) variant=$VARIANT info=$INFO start=$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Fine-tune from epoch-10 weights (fresh optimizer @ 1e-7), same Con-Syn+Real-closest recipe.
.venv/bin/python -m code.examples.train_contrastive "$OUT" \
    --net r18_sw-sup --pretrained --init_weights "$CKPT" \
    --pairs_type new_pos+new_neg --emb_proj --pca \
    --backbone_lr 1e-7 --epochs 5 --sched_step 3 --seed 0 \
    --info_dir "$INFO" --im_root data/aug --gpuid 0
echo "end=$(date)"
