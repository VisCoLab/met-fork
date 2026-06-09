#!/bin/bash
# Re-score the 4 R18 models' FULL-benchmark painting slice under the COMMITTED
# Classification=="Paintings" (148 queries, data/gt_paint/testset.json), K=7 / tau=50.
# CPU-only (faiss-CPU kNN over the 397k DB); reuses already-extracted descriptors. Submit: sbatch eval_paint148.slurm
#SBATCH --account=pl0896-03
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=met-paint148
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
echo "start=$(date)"
for m in \
  "baseline      :data/descriptors/r18_contr_loss_gem_fc_swsl_ms" \
  "combined_ft   :data/descriptors_ft_combined/r18_contr_loss_gem_fc_swsl_ms" \
  "synth_ft      :data/descriptors_ft_synth/r18_contr_loss_gem_fc_swsl_ms" \
  "scratch_synth :data/descriptors_scratch_synth/r18_contr_loss_gem_fc_swsl_ms" ; do
  tag="${m%%:*}"; ddir="${m##*:}"
  echo "######## ${tag} ########"
  .venv/bin/python scripts/eval_paintings.py "$ddir" data/ground_truth 7 50 512
done
echo "end=$(date)"
