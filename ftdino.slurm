#!/bin/bash
# DINOv3 adaptation REUSING /met's contrastive recipe (Embedder+ContrastiveLoss+pair
# mining+validate), backbone swapped to DINOv3-ViT-L. head = frozen backbone + trained
# projector; lora = + LoRA adapters. Data: studio (Met train) or synth (gallery renders).
# usage: sbatch --job-name=met-dinoL-head-studio ftdino.slurm studio head
#        sbatch --job-name=met-dinoL-lora-synth   ftdino.slurm synth  lora
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

DATA="${1:?usage: ftdino.slurm <studio|synth> <head|lora>}"
METHOD="${2:?usage: ftdino.slurm <studio|synth> <head|lora>}"
case "$DATA" in
  studio) INFO=data/ground_truth; IMROOT=data/ ;;
  synth)  INFO=data/gt_synth;     IMROOT=data/aug ;;
  *) echo "bad DATA: $DATA"; exit 1 ;;
esac
LORA=""; [ "$METHOD" = "lora" ] && LORA="--lora"
OUT="data/models/dinov3L_${METHOD}_${DATA}"

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$OUT"
# reuse the DINOv3 weights already downloaded by the art-research repo (data, not code)
export HF_HOME=/mnt/storage_6/project_data/pl0896-03/art-research/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"
echo "host=$(hostname) data=$DATA method=$METHOD out=$OUT start=$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Paper recipe (Con-Syn+Real-closest mining, FC projector + PCAw) on DINOv3-ViT-L.
# Small physical batch + virtual batching keeps ViT-L @512 within H100 memory; imsize=512 -> 32x32 patches.
.venv-dino/bin/python -m code.examples.train_contrastive "$OUT" \
    --net dinov3_vitl16 --pretrained $LORA --emb_proj --pca \
    --pairs_type new_pos+new_neg --imsize 512 \
    --bsize 8 --vbsizemul 8 --epochs 3 --sched_step 2 \
    --seed 0 --info_dir "$INFO" --im_root "$IMROOT" --gpuid 0
echo "end=$(date)"
