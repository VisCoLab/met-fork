#!/bin/bash
# Step 1 of the VISART plan: reproduce the paper's best single model
#   R18-SWSL Con-Syn+Real-closest  (target GAP 36.1 / GAP- 52.4 / ACC 55.0).
# Submit from the repo root:  sbatch slurm/train.slurm
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16         # 8 dataloader workers + threads for CPU faiss (kNN + hard-neg mining)
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00          # generous; checkpoints every epoch, resume with --resume
#SBATCH --job-name=met-r18swsl-con
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/models

export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"   # pre-cached SWSL weights + hub repo -> no internet needed
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Paper-matching defaults are implicit (bsize=64 pairs=128 imgs, epochs=10, margin=1.8,
# backbone_lr=1e-7, sched_step=6 gamma=0.1, wdecay=1e-6, imsize=500).
# The flags below are what selects the SWSL backbone + Con-Syn+Real-closest pairs + FC/PCAw.
.venv/bin/python -m code.examples.train_contrastive \
    ./data/models/r18SWSL_con-syn+real-closest \
    --net r18_sw-sup --pretrained \
    --pairs_type new_pos+new_neg \
    --emb_proj --pca \
    --seed 0 \
    --info_dir ./data/ground_truth \
    --im_root ./data/ \
    --gpuid 0

echo "end=$(date)"
