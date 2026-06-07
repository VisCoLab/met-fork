#!/bin/bash
# Step 3: extract synthetic-image descriptors (GPU) then retrieve vs the Met DB (CPU).
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --job-name=met-synth-eval
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"; export PYTHONUNBUFFERED=1
echo "host=$(hostname) start=$(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== extract synthetic descriptors (24,760 imgs, multi-scale) ==="
.venv/bin/python scripts/extract_synthetic.py
echo "=== retrieval vs Met DB (recall@k, all angles + per angle) ==="
.venv/bin/python scripts/eval_synthetic_retrieval.py
echo "end=$(date)"
