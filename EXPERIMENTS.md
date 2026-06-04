# Experiments log — Met / VISART fork

Running lab notebook: what we've done, the exact settings, and how to continue.
Goal: beat the paper's best single model (**R18-SWSL Con-Syn+Real-closest, GAP 36.1**) by adding a
synthetic gallery phone-photo dataset + a new method. Full plan in `reference/README.md`.

_Last updated: 2026-06-04._

## Status snapshot

| Step | What | State |
|---|---|---|
| Eval pipeline | Validate our eval reproduces the paper | ✅ **Done — GAP 36.10 / GAP⁻ 52.41 / ACC 55.03** on authors' descriptors |
| 1 | Reproduce best result from scratch (full benchmark) | 🟡 Training (job 7313742, epoch 10/10); extract+eval pending |
| 2 | Same model, test only on paintings | ⬜ Not started (173 strict / 221 broad painting test queries) |
| 3 | Same model, test on synthetic images | ⬜ Blocked on synthetic set (location/count/labels TBD) |
| 4 | Train from scratch on paintings only, test on paintings | ⬜ Not started (train 5,143 cls / 12,955 imgs strict) |
| 5 | New method | ⬜ TBD |

## Environment (proven, on PCSS Eagle)

- **Cluster/SLURM:** account `pl0896-03`, QOS `normal,tesla`. GPU jobs → `--partition=tesla --gres=gpu:h100:1` (H100; use GRES type, **not** `--constraint`). CPU-only jobs → `--partition=standard`.
- **venv (`.venv/`, git-ignored):** built on **Python 3.9** (system base — faiss-gpu wheels don't cover 3.13). Plain `venv` ensurepip is broken here, so:
  ```bash
  python3 -m venv --without-pip .venv
  curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | .venv/bin/python -
  .venv/bin/pip install torch torchvision faiss-gpu-cu12 numpy pillow
  ```
  → torch **2.8.0+cu128**, torchvision 0.23, faiss-gpu-cu12 1.12, numpy 1.26. (torch cu128 needs H100/driver-570; a V100 would need a cu124 torch.)
- **faiss runs on CPU.** The prebuilt faiss-gpu wheel has no H100/sm_90 kernels (CUDA err 209), so the GPU-move lines in `code/classifiers/knn_classifier.py` (`fit`) and `code/utils/train_utils.py` (`mine_negatives`) are commented out → CPU `IndexFlatIP` (identical exact inner-product results, slightly slower mining). Re-enable with a faiss build that has sm_90.
- **TORCH_HOME=`data/torch_home`** caches the SWSL hub repo + weights → compute nodes need no internet.

## Data

- Dataset (already downloaded): `/mnt/storage_6/project_data/pl0896-03/met-dataset` (397,121 train imgs / 224,408 classes; test 19,319 = 1,003 Met + 10,352 other-art + 7,964 non-art; val 2,165). Full layout/schema in `CLAUDE.md` → "Dataset".
- Wired via git-ignored symlinks `data/images` and `data/ground_truth` (both → the dataset dir) ⇒ run with `--im_root ./data/ --info_dir ./data/ground_truth`. (JSON `path` has no `images/` segment but the loaders prepend it — that's why the symlink is named `images`.)
- Composition: paintings ≈ **5,143** classes (strict) / 7,310 (broad) — only ~3% of classes, but **~17–22% of scored Met test queries** (173 strict / 221 broad of 1,003). Reproduce with `data/count_paintings.py`. **Val has only 2 painting queries** → a paintings-only experiment can't tune k,τ on val.

## Experiment log

### EXP-0 — eval pipeline validation (authors' released descriptors) ✅
- **Input:** authors' `r18SWSL_con-syn+real-closest` descriptors → `data/authors/descriptors/descriptors.pkl` (also checkpoint at `data/authors/models/r18SWSL_con-syn+real-closest`).
- **Command:** `.venv/bin/python data/eval_fullgrid.py data/authors/descriptors data/ground_truth 512`
- **Result:** **GAP 36.10 / GAP⁻ 52.41 / ACC 55.03** (best **K=10, τ=50**). Matches paper (36.1 / 52.4 / 55.0) ✓.
- **Lesson:** `knn_eval.py --autotune` defaults `--k 1` and only tunes τ → degenerate τ=500, **GAP ≈ 23**. Must sweep the full K grid; `eval_fullgrid.py` does this. ACC is unaffected by the bug (matched 55.0 either way).

### EXP-1 — step 1, from-scratch reproduction 🟡
- **Submit:** `sbatch train.slurm` → job **7313742** (H100 `gpu13`).
- **Exact settings** (R18-SWSL Con-Syn+Real-closest):
  ```
  python -m code.examples.train_contrastive ./data/models/r18SWSL_con-syn+real-closest \
      --net r18_sw-sup --pretrained --pairs_type new_pos+new_neg --emb_proj --pca \
      --seed 0 --info_dir ./data/ground_truth --im_root ./data/ --gpuid 0
  ```
  Paper-matching defaults (implicit): 64 pairs/batch (=128 imgs), 10 epochs, margin 1.8, backbone_lr 1e-7, sched step 6 ×0.1, wdecay 1e-6, imsize 500.
  **`--net r18_sw-sup` is required** — without it the default `resnet18` trains the ImageNet model (GAP 32.5), not SWSL (36.1).
- **Training-progress val GAP⁻** (single-scale, no-PCAw, K=1 — quick metric, not comparable to tuned test GAP): e1 .502, e2 .503, e3 .519, e4 .512, e5 .524, e6 .479, e7 .529, e8 .536, **e9 .544**, e10 pending. Trending up.
- **Checkpoints:** `data/models/r18SWSL_con-syn+real-closest/method:_..._epoch:_N` (epochs 1–9 saved; per-epoch `train_descriptors_epoch:_N.pkl` too).
- **Result:** pending (extract + eval after epoch 10).

## How to finish step 1 (after training completes)

1. Pick best epoch (by val GAP⁻/ACC in `logs/met-r18swsl-con-7313742.out`; likely epoch 9 or 10).
2. Extract multi-scale descriptors (GPU job — wrap in srun/sbatch on `tesla`):
   ```bash
   TORCH_HOME=$PWD/data/torch_home .venv/bin/python -m code.examples.extract_descriptors \
       data/descriptors --net r18_contr_loss_gem_fc_swsl \
       --netpath "data/models/r18SWSL_con-syn+real-closest/method:_..._epoch:_<N>" \
       --ms --info_dir data/ground_truth --im_root data/ --gpuid 0
   # writes data/descriptors/r18_contr_loss_gem_fc_swsl_ms/descriptors.pkl
   ```
3. Eval with the full grid (CPU node):
   ```bash
   .venv/bin/python data/eval_fullgrid.py data/descriptors/r18_contr_loss_gem_fc_swsl_ms data/ground_truth 512
   ```
   Target ≈ GAP 36 / GAP⁻ 52 / ACC 55.

## Gotchas & decisions
- `--net r18_sw-sup` for training, `r18_contr_loss_gem_fc_swsl` for extraction (SWSL variant).
- Eval must tune the **full K grid** (use `eval_fullgrid.py`), not the README's `--autotune` default.
- faiss on **CPU** (H100 sm_90 gap); identical results.
- GPU = `--gres=gpu:h100:1` on `tesla` (not `--constraint`).
- Authors' artifacts live under `data/authors/` (kept separate from our run's `data/models/`).

## Repo artifacts (all under git-ignored `data/` unless noted)
- `train.slurm` (tracked) — step-1 training job.
- `data/eval_fullgrid.py` — full-K-grid eval (use for every model).
- `data/count_paintings.py` — painting counts via Met Open Access join.
- `data/smoke_gpu.py` — GPU/faiss env smoke test.
- `data/authors/` — authors' checkpoint + descriptors.
- `data/torch_home/` — cached SWSL weights.
- `data/MetObjects.csv` — Met Open Access metadata (~303 MB).
- Targets-to-beat tables + method↔`pairs_type` mapping: `reference/README.md`.
