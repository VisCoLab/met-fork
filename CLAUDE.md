# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Official code for **The Met dataset** (NeurIPS paper) — instance-level recognition / image retrieval over ~224k classes of museum exhibits. The pipeline trains a CNN embedding model with contrastive learning, extracts global descriptors, and classifies query images against the training set with a non-parametric kNN classifier. There is no build system, test suite, or linter config — this is a research codebase.

## Research context (this fork)

This is a **fork of [nikosips/met](https://github.com/nikosips/met)** being extended for a paper
submission to **VISART** (the Vision for Art workshop). Two planned contributions, on top of the
original dataset/benchmark:

1. **A synthetic dataset of phone photos of paintings in a gallery.** The hypothesis is that adding
   this data *by itself* improves recognition. This directly attacks the benchmark's core
   difficulty — the **distribution shift** between studio-condition exhibit images (training) and
   real visitor photos (queries), plus the long tail (60.8% of classes have a single training image).
2. **A new method** (details TBD) that improves the numbers further, on top of the synthetic data.

When making changes, the goal is to **beat the original paper's results** (primary baseline:
R18-SWSL Con-Syn+Real-closest, **GAP 36.1**) using the *same task definition, metrics, and
protocol*. Preserve the GAP/GAP⁻/ACC evaluation so results stay comparable.

**The original paper is the main reference.** Its LaTeX source is in [`reference/`](reference/),
and [`reference/README.md`](reference/README.md) summarizes the task, methods, the full result
tables (numbers to beat), and the crucial `--pairs_type` ⇄ paper-method-name mapping. Read it before
designing experiments or writing up results.

## Running code

All scripts are run as **Python modules from the repository root** (the directory containing `code/`), because every file uses absolute imports rooted at `code.` (e.g. `from code.utils.train_utils import *`). Running a script by path (`python code/examples/knn_eval.py`) will fail with import errors.

```bash
# Train embedding model with contrastive loss (writes checkpoints + per-epoch descriptor pkls to EXPORT_DIR)
python -m code.examples.train_contrastive ./data/models/<run_name> --seed 0 --pretrained \
    --pairs_type new_pos+new_neg --emb_proj --pca --info_dir ./data/ground_truth --im_root ./data/ --gpuid 0

# Extract descriptors with a backbone (writes descriptors.pkl into EXPORT_DIR/<exp_name>/)
python -m code.examples.extract_descriptors ./data/descriptors --net r18_contr_loss_gem_fc_swsl \
    --netpath ./data/models/<checkpoint> --ms --info_dir ./data/ground_truth --im_root ./data/ --gpuid 0

# Evaluate pre-extracted descriptors with the kNN classifier (reads descriptors.pkl from EXPORT_DIR)
python -m code.examples.knn_eval ./data/descriptors/<exp_name>/ --autotune --info_dir ./data/ground_truth/

# Every script supports -h for the full option list
python -m code.examples.<script> -h
```

**Prerequisites** (no `requirements.txt` exists): Python 3, NumPy, PyTorch, torchvision, `faiss-gpu`, PIL. A **CUDA GPU is mandatory** — training, descriptor extraction, and the faiss kNN index all call `.cuda()` unconditionally, and each script sets `CUDA_VISIBLE_DEVICES` from `--gpuid`.

**Data is not in the repo.** It is downloaded separately from the [official site](http://cmp.felk.cvut.cz/met/). Scripts expect:
- `--info_dir`: ground-truth JSONs — `MET_database.json` (+ `mini_MET_database.json`), `testset.json`, `valset.json`.
- `--im_root`: image root; images are read from `<im_root>/images/<path>`. If `--im_root` is omitted, the image root defaults to the parent of `--info_dir`.

## Architecture

The three example scripts are stages of one pipeline; descriptors are the interchange format, passed between stages as pickled NumPy arrays (`descriptors.pkl` with keys `train_descriptors` / `test_descriptors` / `val_descriptors`).

```
images + GT JSON ──> Embedder (CNN backbone → GeM pool → L2-norm → [optional FC projector])
                       │
                       └─> descriptors ──> PCA-whitening ──> faiss kNN classifier ──> GAP / accuracy
```

**Embedder** (`code/networks/backbone.py`) is the core descriptor extractor: a fully-convolutional ResNet trunk → `GeM` generalized-mean pooling → L2 normalization, optionally followed by an FC `projector` that is L2-normalized again. The projector can be **initialized from PCA-whitening statistics** (`init_projector`), so PCA whitening is used in two distinct places: as train-time FC initialization (`--pca`) and as eval-time descriptor post-processing in `knn_eval.py`. `OUTPUT_DIM` maps architecture name → descriptor dimension. Multi-scale extraction (`extract_ms`, scales `[1, 1/√2, 1/2]`) is selected by `--ms`.

**siamese_network** (`code/networks/SiameseNet.py`) is a thin training wrapper that runs `Embedder` on two augmented views and returns both descriptors for `ContrastiveLoss` (`code/utils/losses.py`, margin-based). Only `.backbone` is used at inference time.

**Datasets** (`code/utils/datasets.py`):
- `MET_database` — training-set images (label = exhibit `id`).
- `MET_queries` — val/test queries; missing `MET_id` becomes label **`-1`, the distractor label** (query depicts no known exhibit).
- `MET_pairs_dataset` — yields `(img1, img2), label` pairs for contrastive training (label 1 = same class, 0 = different).

**Online pair mining is the heart of training.** `MET_pairs_dataset.create_epoch_pairs` is called once per epoch and rebuilds all pairs from the *current* model's descriptors using faiss (`mine_negatives` = hard negatives among nearest neighbors of a different class; `mine_positive` = closest same-class sample). `--pairs_type` selects the strategy: `sim_siam_pos`, `sim_siam_pos+new_neg`, `pos+new_neg`, `new_pos+new_neg`. The **first** epoch mines from ImageNet (or `--init_descr`) descriptors; **subsequent** epochs reuse the descriptors returned by `validate()` (`code/utils/train_utils.py`), which runs a full kNN eval each epoch and feeds its train descriptors back into mining — a tight train↔validate coupling.

**Classifier & metrics:**
- `KNN_Classifier` (`code/classifiers/knn_classifier.py`) — faiss-GPU inner-product search over L2-normalized descriptors; per-class scores are softmax-weighted by temperature `t`, yielding a prediction + confidence. `tune_KNN` grid-searches `K`/`t` on the val set (used by `--autotune`).
- `evaluate` (`code/utils/utils.py`) reports three numbers: **GAP** (Global Average Precision, the primary metric, ranks all queries by confidence and accounts for distractors), **GAP without distractors**, and accuracy (distractors excluded). Distractors are identified by label `-1`.

## Conventions & gotchas

- **Wildcard imports** (`from X import *`) are used throughout; new public helpers in `utils/` become available transitively.
- Training **freezes all BatchNorm layers** to ImageNet running statistics (`set_batchnorm_eval`) because effective batch size per anchor is tiny.
- `--vbsizemul` implements **virtual (gradient-accumulation) batching**; the *effective* batch size is `bsize * vbsizemul`, which is what gets baked into the checkpoint filename.
- Several argparse options (`--bsize`, `--epochs`, `--backbone_lr`) have **no `type=`**, so they arrive as strings and are cast inline (`int(args.bsize)`, `float(args.backbone_lr)`); preserve the casts when editing.
- Checkpoint and descriptor filenames are auto-generated from hyperparameters (long `method:_..._epoch:_N` strings); the consuming script must be pointed at the matching directory.
- `--net` accepts two families in `extract_descriptors.py`: ImageNet/self-supervised backbones built from scratch (e.g. `r18INgem`, `r50_swav_gem`, `r18_sw-sup_gem`) and **checkpoint-loadable** variants requiring `--netpath` (`r18_contr_loss_gem`, `r18_contr_loss_gem_fc`, `r18_contr_loss_gem_fc_swsl`).
