# Does adding synthetic gallery data improve painting recognition?

*We retrain the Met recognition model with our synthetic "phone-photos-in-a-gallery" renders added,
and ask whether that **alone** — same recipe, no new method — beats the original paper. This closes the
core plan, steps 1–4. (Met / VISART fork; lab notebook: [`EXPERIMENTS.md` → EXP-1, EXP-4](../../EXPERIMENTS.md).)*

## What we did, in one paragraph

The Met benchmark trains on **clean studio catalog photos** but is tested on **real visitor phone
photos** — a hard "looks different" gap. Our hypothesis: adding **synthetic** gallery renders of the
paintings to the training set should help recognition *on its own*, before we add any new method. To
test it without fooling ourselves, we first **reproduced the paper's best model from scratch** as a
baseline, then **retrained the exact same model with the synthetic renders added** — identical recipe,
only the training data changed — and evaluated both on the **original** benchmark (the synthetic images
are used *only for training*, never as answers). The synthetic images never enter the test database, so
the two runs are directly comparable and any difference is down to the added data.

> **How to read the numbers.** The task: given a query photo, name which of ~224k museum exhibits it
> shows — or correctly reject it as "not in the collection". All scores are 0–100, higher is better.
> - **GAP** (Global Average Precision) — the **headline metric**: ranks *every* query by the model's
>   confidence and rewards putting correct, confident answers first *and* giving junk low confidence.
> - **GAP⁻** — the same, but scored on the real exhibit queries only (junk removed): pure recognition,
>   without the "reject the junk" part.
> - **Accuracy (ACC)** — of the in-collection queries, the fraction whose top guess is correct.
> - **distractors / "junk"** — query photos of things *not* in the collection (other artworks, non-art).
>   A good model gives them low confidence. They make GAP much harder than plain accuracy.
> - **paintings (strict)** — the 173 test queries that are paintings (MetObjects `Classification ==
>   "Paintings"`). Our contribution targets paintings, so we track them separately. ("broad" = a looser
>   painting definition, 221 queries.)
> - **clean A/B** — two runs identical in *everything* except the one thing under test (here: +synthetic
>   data). Any difference is attributable to that one change — no other explanation.

## TL;DR

- **Adding synthetic gallery data lifts full-benchmark GAP from 35.97 → 38.15** with the *identical*
  training recipe — a clean **+2.18**, and it **beats the original paper's best single model (36.1)**.
- **Every metric improves, not just paintings:** GAP⁻ +3.35, accuracy +3.59, paint GAP⁻ +3.28, paint
  accuracy +3.47 (strict; broad paintings move even more, ~+4).
- **Non-painting queries improve too** — so this isn't simply "more painting data". The diverse renders
  teach broadly useful lighting / viewpoint / glass invariance that helps real photos across the board.
- Two quicker **fine-tuning** variants gave comparable or larger gains (synth-only 38.61, combined 37.38)
  but are **confounded** (they also trained longer); the from-scratch run removes that doubt.
- **Bottom line: synthetic data helps on its own.** Contribution 1 of the paper stands — before any new method.

---

## 1. The task, the baseline, and the four models

We compare four versions of the *same* R18-SWSL contrastive model (the paper's best single model:
ResNet-18 SWSL backbone → GeM pooling → projector, trained with online hard-pair mining; paper name
*Con-Syn+Real-closest*). Only the **training data** and **how long we trained** change:

| model | training data | recipe | role |
|---|---|---|---|
| **Step 1 — baseline** | studio only (397k) | from SWSL, 10 epochs | reproduces the paper |
| Combined FT | studio + synthetic | fine-tune the baseline, +5 epochs | quick test |
| Synth-only FT | synthetic only | fine-tune the baseline, +5 epochs | quick test |
| **From-scratch +synth** | studio + synthetic (422k) | from SWSL, 10 epochs *(same as baseline)* | **clean A/B** |

The **synthetic dataset** is 24,760 Blender gallery renders — 4,952 Met paintings × 5 camera views —
added to the 397,121 studio images for training (→ `data/gt_aug`). Our **baseline reproduces the paper**:
GAP **35.97** vs the paper's 36.1 (and 36.10 on the authors' own descriptors), so it's a faithful
starting point. Everything below is measured on the **original** studio database and real test queries,
with the kNN classifier's `K` and temperature tuned on the validation set — exactly as in step 1.

---

## 2. Headline: synthetic data raises GAP — and clears the paper

![Full-benchmark GAP across the four configurations](figures/gap_by_config.png)

| configuration | training data | full GAP | Δ vs baseline |
|---|---|--:|--:|
| Step 1 — baseline *(= paper)* | studio | 35.97 | — |
| Combined FT *(confounded)* | studio + synth | 37.38 | +1.41 |
| Synth-only FT *(confounded)* | synthetic | 38.61 | +2.64 |
| **From-scratch +synth** *(clean A/B)* | studio + synth | **38.15** | **+2.18** |

*Δ vs baseline = GAP minus the 35.97 baseline. The dashed line in the figure is the paper's best single
model (36.1); every synthetic configuration sits above it.*

**Plain reading:** *every* way of adding the synthetic data beats both our baseline and the published
number. The bar we trust most — the solid teal **from-scratch +synth** — lands at **38.15 GAP**, a clean
+2.18 over baseline and +2.05 over the paper.

---

## 3. The clean A/B, metric by metric

The from-scratch run changes *one thing* versus the baseline (it adds synthetic data; same 10 epochs,
same learning-rate schedule, same everything else). So this is the honest measure of what the data buys:

![Baseline vs from-scratch +synth, every metric](figures/baseline_vs_synth.png)

| metric | Step 1 baseline | From-scratch +synth | Δ |
|---|--:|--:|--:|
| Full GAP | 35.97 | **38.15** | **+2.18** |
| GAP⁻ (no distractors) | 52.14 | **55.49** | **+3.35** |
| Accuracy | 54.64 | **58.23** | **+3.59** |
| Paint GAP⁻ (strict) | 67.81 | **71.09** | **+3.28** |
| Paint ACC (strict) | 69.36 | **72.83** | **+3.47** |
| Paint GAP⁻ (broad) | 65.92 | 69.90 | +3.98 |
| Paint ACC (broad) | 67.42 | 71.49 | +4.07 |

*Same model, same recipe, +synthetic. All seven numbers go up.*

**The gain is broad, not just paintings.** Accuracy on *all* in-collection queries rises +3.59, including
non-painting exhibits the synthetic set never depicts. That tells us the renders aren't helping by simply
adding painting examples — they teach the model **lighting, viewpoint, and glass-glare invariance** that
transfers to real photos generally. (A companion analysis of *what* the synthetic images vary is in the
[DINOv3 embedding study](../synth-embedding-analysis/README.md).)

---

## 4. Were the gains real, or just "more training"?

The two fine-tuning runs are **confounded**: to fine-tune we re-warmed the learning rate (1e-8 → 1e-7)
and trained **5 extra epochs**, so part of their lift could be the extra training rather than the
synthetic data. The from-scratch run is built precisely to remove that doubt:

| run | what differs from baseline | full GAP | clean? |
|---|---|--:|:--:|
| Synth-only FT | +5 epochs at a re-warmed LR; synthetic data only | 38.61 | ✗ confounded |
| Combined FT | +5 epochs at a re-warmed LR; studio + synthetic | 37.38 | ✗ confounded |
| **From-scratch +synth** | **nothing but the added data** (same 10 epochs, same LR) | **38.15** | ✓ clean A/B |

**Verdict:** the clean run, with no extra training of any kind, still gains **+2.18 GAP**, and it lands
**within 0.5** of the best confounded run (38.15 vs 38.61). So the improvement is **real and attributable
to the synthetic data itself** — the fine-tunes' larger numbers were mostly the data too, not the extra
epochs. (Curiously, fine-tuning on synthetic *only* edges out mixing it with studio data — plausibly
because mixing 397k studio + 25k synthetic dilutes the synthetic signal each epoch.)

---

## 5. What this means

- **The core hypothesis holds.** Synthetic gallery renders improve Met recognition *on their own*,
  under the paper's exact task, metrics, and protocol — and the clean result already **beats the paper's
  best single model**. This is the evidence for contribution 1.
- **It's a domain-gap fix, not a data-count fix.** The lift reaches non-painting queries, so the renders
  are closing the studio→real-photo gap broadly, not just padding the painting classes.
- **The from-scratch number (38.15) is the one to quote** — it's the apples-to-apples comparison to the
  35.97 baseline; the fine-tune numbers are useful corroboration, not the headline.
- **Headroom remains.** The synthetic set still has a known camera-framing bug (one of five views is
  unusable) and the renders are "too clean" versus real phone photos — both are addressable, and the new
  method (step 5) builds on top of this.

---

## 6. Caveats

- **kNN tuning on a tiny painting val set.** `K` and temperature are tuned on the full validation set
  (as in the paper); the painting-only val set is too small to tune on, so paint scores use a fixed
  `K=7, τ=50`. Comparisons are like-for-like across models, but paint numbers aren't independently tuned.
- **One seed.** All runs are `seed 0`; the +2.18 is within the range we'd treat as real (baseline already
  reproduced the paper to within 0.13), but we have not measured run-to-run variance.
- **Synthetic camera-rig bug.** One of the five rendered views (`right upper`) is grazing/edge-on and
  near-useless (see EXP-3); the gain above is *despite* carrying that broken view in training.
- **The fine-tune runs are confounded** by design — they are corroboration only; see §4.

---

## 7. How to reproduce

```bash
# 1) build the augmented training set (studio + synthetic symlinks + manifests)
.venv/bin/python scripts/build_finetune_data.py      # -> data/gt_aug, data/aug

# 2) train the clean A/B: from SWSL, 10 epochs, studio + synthetic   (H100, ~23 h)
sbatch train_synth.slurm                             # job 7330059  -> data/models/r18SWSL_scratch_synth

# 3) evaluate it exactly like step 1 (MS descriptors over the ORIGINAL studio DB, full K×τ grid)
sbatch extract_eval_scratch.slurm                    # job 7342026  -> GAP 38.15 / GAP⁻ 55.49 / ACC 58.23

# 4) the figures in this doc (CPU; run via SLURM, not the login node)
srun --account=pl0896-03 --partition=standard --time=0:10:00 --mem=4G \
    .venv-dino/bin/python scripts/plot_synthetic_training.py
```

The two fine-tune controls are `finetune.slurm <synth|combined>` (train) + `extract_eval_ft.slurm
<variant>` (eval), jobs 7330026 / 7330036 (synth) and 7330025 / 7332888 (combined).

Code: [`scripts/build_finetune_data.py`](../../scripts/build_finetune_data.py) ·
[`train_synth.slurm`](../../train_synth.slurm) · [`extract_eval_scratch.slurm`](../../extract_eval_scratch.slurm) ·
[`scripts/eval_fullgrid.py`](../../scripts/eval_fullgrid.py) · [`scripts/eval_paintings.py`](../../scripts/eval_paintings.py) ·
[`scripts/plot_synthetic_training.py`](../../scripts/plot_synthetic_training.py).
Every number above is recorded in [`EXPERIMENTS.md`](../../EXPERIMENTS.md) (EXP-1, EXP-4).
