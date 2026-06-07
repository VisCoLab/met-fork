#!/bin/bash
# Step 3 rescue: fuse the geometric (PatchMatch) score into the CONFIDENCE (GAP's
# global ranking), not the pre-softmax similarity. Operates on the SAVED PM scores
# -> no re-matching, ~1 min. Tests add / gate / RRF schemes vs the CLS baseline.
#SBATCH --account=pl0896-03
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --job-name=met-conffusion
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export PYTHONUNBUFFERED=1
echo "host=$(hostname) start=$(date)"
.venv/bin/python scripts/rerank_confidence_fusion.py \
    --model dinov3_vitl16 --pm data/rerank/pm_scores_dinov3_vitl16.npz \
    --info-dir data/ground_truth --out data/rerank/conffusion_dinov3_vitl16.json
echo "end=$(date)"
