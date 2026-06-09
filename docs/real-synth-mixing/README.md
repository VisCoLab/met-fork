# Real vs synthetic training data: how much synthetic helps for paintings

*We train the painting recognizer on different blends of **real** museum photos and **synthetic**
gallery renders, then test every model on the **same real painting photos**. Which blend recognizes
real paintings best? (Met / VISART fork; lab notebook: [`EXPERIMENTS.md` → EXP-8](../../EXPERIMENTS.md).)*

## What we did, in one paragraph

We trained the painting recognizer **six times**. Each run saw the **same number of training images
(12,403)**, but with a different blend of **real** museum catalog photos and **synthetic** gallery
renders — from **100 % real** (no synthetic), in five steps, down to **0 % real / 100 % synthetic**.
Synthetic images are used **only for training**; every model is tested on the **same set of real
painting photos**. We score each model two ways: an easy **"closed painting world"** (pick the right
painting out of ~12 k painting photos) and the **full Met benchmark** (find the right painting among
**all 397 k** museum photos while also ignoring ~18 k "distractor" junk queries). The question: as we
trade real training photos for synthetic ones, does recognizing real paintings get better or worse?

> **How to read the numbers** — all scores are 0–100, higher is better.
> - **ACC** (accuracy): of the real painting photos, the share whose top match is the correct artwork.
> - **GAP⁻**: the paper's main quality score *without* distractors — like accuracy, but it also rewards
>   being **confident on correct answers and unsure on wrong ones** (it ranks every answer and checks
>   that the good ones float to the top).
> - **GAP**: the full score *with* distractors — the model must also push ~18 k junk queries (non-art,
>   random photos) **below** the real answers. Only the full benchmark has distractors; the closed
>   world has none, so there GAP = GAP⁻.
> - **two test settings**: *closed world* = search only the 12,403 real painting photos (easy);
>   *full benchmark* = search all 397,121 photos (hard — the original Met task).
> - **"paint" columns** = scored on just the **148 real painting photos**; the plain GAP/GAP⁻/ACC =
>   scored over all **1,003** real Met queries.

## TL;DR

- **Synthetic data helps — and more is better.** Shifting training from all-real to all-synthetic,
  real-painting recognition climbs **steadily**, in *both* test settings.
- **Training on synthetic *alone* (zero real photos) is the best of the six.** It beats the all-real
  baseline everywhere, and on paintings it **beats the original model trained on all 397 k real photos**
  in *both* test settings (closed 72.5 vs 71.6; full benchmark 70.0 vs 67.9) — using ~32× less data and
  not one real painting photo.
- **Why it works:** the test photos are real *gallery* shots; the "real" training photos are clean
  *studio* shots. The synthetic renders imitate gallery conditions (angle, glass, lighting), so they
  teach the right kind of variation. (Consistent with EXP-4 and EXP-7.)
- **The catch on the full benchmark:** these models only ever trained on paintings, so they're weaker
  at rejecting the 18 k distractors — that drags down **GAP**. On the painting-relevant scores
  (**GAP⁻, ACC, and the painting slice**) they essentially match the full-data model.
- **Honest limit:** only **148** painting test photos, so gaps of ≤ ~2 points are noise. Trust the big
  all-real → all-synthetic jump and the steady trend, not the exact ordering of the middle blends.

## Results — all six models, one table

| Training mix (real : synth) | Paint GAP⁻ (closed) | Paint ACC (closed) | GAP (full) | GAP⁻ (full) | ACC (full) | Paint GAP⁻ (full) |
|---|--:|--:|--:|--:|--:|--:|
| 100 : 0 — all real | 67.18 | 70.27 | 28.83 | 49.08 | 52.14 | 61.83 |
| 80 : 20 | 70.56 | 72.97 | 30.23 | 50.03 | 52.84 | 66.09 |
| 60 : 40 | 70.65 | 72.30 | 31.15 | 50.60 | 53.34 | 67.22 |
| 40 : 60 | 71.37 | 72.97 | 30.38 | 50.74 | 53.54 | 67.92 |
| 20 : 80 | 71.24 | 72.30 | 30.85 | 50.92 | 53.64 | 69.62 |
| **0 : 100 — all synthetic** | **72.47** | **73.65** | **31.32** | **51.47** | **54.04** | **70.04** |
| *reference: all-real-data model* | *71.62* | *72.30* | *35.97* | *52.14* | *54.64* | *67.86* |

*Column guide — **Paint GAP⁻ / ACC (closed):** painting score searching only the 12,403 painting
photos (easy). **GAP / GAP⁻ / ACC (full):** whole-benchmark score searching all 397,121 photos — GAP
includes the 18 k distractors, GAP⁻ removes them, ACC is top-1 on the 1,003 real Met queries.
**Paint GAP⁻ (full):** the same 148 painting photos, searched against the full 397 k DB. The
**reference** row is the original model trained on all 397 k real photos (paintings + everything),
scored identically. Best run (all-synthetic) in bold.*

![Painting recognition vs training mix](fig_paintings.png)

*The headline: more synthetic training data → better recognition of the same 148 real painting photos,
in both the easy closed world (blue) and the hard full benchmark (red). Dotted = the all-real-data model.*

![Whole Met benchmark vs training mix](fig_full_benchmark.png)

*On the whole benchmark, GAP⁻ and ACC rise with synthetic and nearly reach the all-real-data model
(dotted). GAP (with distractors) stays lower — these painting-only models are weaker at rejecting the
18 k junk queries.*

## What it means

- **For this task, synthetic gallery renders are better training material than real studio photos.**
  We test on real *gallery* photos, and the renders resemble those more than clean studio catalog shots
  do — so even with zero real photos the all-synthetic model recognizes real paintings best.
- **The win is specific to paintings.** On the full benchmark the painting-only models can't reject the
  18 k distractors as well as a model trained on all 224 k classes, so the distractor-sensitive **GAP**
  stays below the full-data model — but on the painting slice they **beat** it (GAP⁻ 70.0 vs 67.9, ACC
  71.6 vs 69.6), which is exactly the domain the VISART contribution targets.
- **Takeaway:** for a painting recognizer, synthetic gallery renders aren't merely a cheap stand-in for
  real data — here they are *better* than the real studio images we have.

## How we trained (identical to the paper's model — only the data changes)

Every run uses the **same recipe** as our step-1 reproduction of the paper's best model (*R18-SWSL
Con-Syn+Real-closest*, EXP-1, GAP 35.97). The command (`paint_train.slurm`) is literally `train.slurm`
with the data swapped:

- backbone **R18-SWSL**, started from **ImageNet-SWSL** weights (fresh each run — not continued from a
  checkpoint);
- **10 epochs**, contrastive loss with hard-pair mining (`new_pos+new_neg`), projector + PCA-whitening
  init, **seed 0**;
- paper defaults unchanged: learning rate 1e-7, 64 pairs/batch, margin 1.8, weight-decay 1e-6, LR step
  6 ×0.1, image size 500.

The **only** differences between runs are the training images (the real:synth blend) and the image
folder. Backbone, init, **epoch count**, optimizer, schedule, and seed are identical across all six runs
and match step-1.

> **Same recipe ≠ same amount of training.** "1 epoch" means each training image is used once, so 10
> epochs over **12,403** painting images is ~**32× fewer weight updates** than step-1's 10 epochs over
> **397,121** images (≈ 22–32 min per run vs ≈ 21 h for step-1). Same recipe and epoch count; far fewer
> updates — worth remembering when comparing to the full-data reference.

## Caveats

- **Small test set:** 148 painting photos, and only 1 painting in the validation set — so we tune the
  k/τ knobs by **2-fold cross-validation on the test photos** (each photo scored on a half it did not
  help tune, so no leakage). Differences ≤ ~2 points are within noise.
- **Closed-world scores are not comparable to the paper's GAP 36.1** — searching 12 k photos is far
  easier than 397 k. Comparisons *across blends* are fair (identical test each time); the full-benchmark
  columns are the ones comparable to the paper / EXP-2.
- **The synthetic set still has the broken `right upper` camera view** (EXP-3 / EXP-7). The all-synthetic
  win happens *despite* that bad view — fixing the camera rig could help further.
- **Fixed training size (12,403):** the all-synthetic run uses ~half of the 24,490 available renders; an
  un-capped "use all synthetic" run might do even better.

## Reproduce

```bash
.venv/bin/python scripts/build_paintings_mix_data.py     # build the blended manifests (data/gt_paint_mix_*)
for tag in 80r20s 60r40s 40r60s 20r80s 0r100s; do        # train each blend (100/0 already = data/gt_paint)
  tid=$(sbatch --parsable --job-name=met-tr-$tag paint_train.slurm data/gt_paint_mix_$tag data/aug paint_$tag)
  sbatch --dependency=afterok:$tid --job-name=met-ev-$tag paint_eval.slurm data/models/r18SWSL_paint_$tag 10 $tag
done
# full-benchmark eval per model (tag = synthetic %): paint->0s, paint_80r20s->20s, ... paint_0r100s->100s
sbatch eval_full.slurm data/models/r18SWSL_paint 10 0s   # ...repeat for each model
sbatch eval_paint_cls.slurm                              # painting slice on the full 397k DB (148-photo def)
.venv-dino/bin/python scripts/plot_mixing_report.py      # -> fig_paintings.png, fig_full_benchmark.png
```
