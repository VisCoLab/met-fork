#!/bin/bash
# Step 5 (new method): CROSS-DOMAIN pair mining. Identical recipe to train_synth.slurm
# (from-SWSL, 10 epochs, studio+synthetic combined manifest) EXCEPT positives are mined
# cross-domain (studio<->synthetic, same class) via --pairs_type cross_domain_pos+new_neg.
# => clean A/B vs the scratch+synth run (same data, standard new_pos+new_neg mining).
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=met-crossdomain
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs data/models
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"; export PYTHONUNBUFFERED=1
echo "host=$(hostname) start=$(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

# Paper defaults (10 epochs, sched_step 6, lr 1e-7, ...); only pairs_type differs from train_synth.slurm.
.venv/bin/python -m code.examples.train_contrastive ./data/models/r18SWSL_crossdomain \
    --net r18_sw-sup --pretrained --pairs_type cross_domain_pos+new_neg --emb_proj --pca \
    --seed 0 --info_dir data/gt_aug --im_root data/aug --gpuid 0
echo "end=$(date)"
