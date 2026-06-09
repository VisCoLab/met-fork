#!/bin/bash
# CPU job: assemble the real DINOv3 ViT-L reference clouds (reuse art-research feats)
# then analyze synthetic embedding structure (angle vs floor) + domain separation.
# Chained after the synthetic extraction job via --dependency=afterok.
#SBATCH --account=pl0896-03
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=met-synth-dino-analysis
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs data/synth_dino
# keep all caches OFF the 1GB $HOME (matplotlib/fontconfig/etc. -> quota errors there)
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/data/.mplcache"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/data/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export PYTHONUNBUFFERED=1
echo "host=$(hostname) start=$(date)"

echo "=== assemble real reference features (studio sources + painting queries) ==="
if [ -f data/synth_dino/real_dino_vitl16_aspect512.npz ]; then
  echo "real features already exist -> skipping assemble"
else
  .venv-dino/bin/python scripts/assemble_real_dino.py --out-dir data/synth_dino || { echo "assemble FAILED"; exit 1; }
fi

echo "=== analyze embedding structure + domain gap ==="
.venv-dino/bin/python scripts/analyze_synth_dino.py --dir data/synth_dino

echo "end=$(date)"
