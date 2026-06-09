# Can the Met model recognize our synthetic gallery renders?

*We take the recognition model trained **only on the original Met data** (it has never seen a single
synthetic image), use each of our **24,760 synthetic gallery renders as a query**, and check whether it
retrieves the **correct painting** from the original 397k studio database. This measures how
recognizable the renders are across the synthetic→real gap — and, view by view, which camera angles are
well-framed. (Met / VISART fork; lab notebook: [`EXPERIMENTS.md` → EXP-3](../../EXPERIMENTS.md).)*

## What we did, in one paragraph

The Met task is **instance recognition**: given a query photo, find the same artwork among ~224k museum
exhibits, each represented by clean **studio catalog photos**. We have a synthetic dataset — Blender
renders of Met paintings hung in a gallery, shot from 5 camera angles — and we want a quick, honest read
on whether those renders actually look like the paintings they came from. So we ran a **retrieval test**:
each synthetic render becomes a **query**, the **database** is the original **397,121 studio photos**, and
a render scores a **hit** if its nearest studio neighbour(s) belong to its **own source painting**. The
model is our **step-1 reproduction** of the paper's best model — trained *purely on real Met studio
photos*, with **zero exposure to synthetic data** — so this is a clean, independent probe: if the model
can still find the right painting from a render, the render genuinely depicts that painting. We report
**recall@k** overall, **per camera angle**, and on the **painting** subset.

> **How to read the numbers.** All scores are 0–100, higher is better.
> - **query / database** — a *query* is one image we look up; the *database* is the 397,121 studio photos
>   we search through. Here every query is a synthetic render.
> - **recall@k (R@1 / R@5 / R@10)** — look at the **k most similar** studio photos to the query. It's a
>   **hit** if *any* of those k is a photo of the render's **own painting**. R@1 = the single best match is
>   correct; R@10 = the correct painting is somewhere in the top 10.
> - **correct** — the render retrieves a studio photo whose Met class id equals the painting the render was
>   built from (recovered from each render folder's `metadata.json`).
> - **camera angle** — the 5 gallery viewpoints each painting is rendered from: `front`, `left upper`,
>   `right upper`, `left bottom`, `right bottom`.
> - **no distractors here** — unlike the main GAP benchmark, every query *has* a correct answer in the
>   database, so this is **pure recognition (recall)**, not the harder "also reject the junk" task. These
>   recall numbers are therefore **not** comparable to the GAP scores elsewhere.

## TL;DR

- **A model that has never seen synthetic data still recognizes the renders.** Across all 24,760 renders,
  the single nearest studio photo is the right painting **36.9%** of the time (R@1), rising to **46.8%** in
  the top 10.
- **Camera angle dominates the result, by a huge margin.** Well-framed views work well — `left upper`
  **75.9%** and `front` **64.8%** R@1 — while the broken `right upper` view is essentially useless at
  **1.4%**. This is the **camera-framing bug** from EXP-3, here measured directly.
- **A well-framed render is about as recognizable as a real photo.** On paintings, `front` renders hit
  **70.5%** R@1 — almost exactly the **69.6%** accuracy this same model gets on *real* visitor photos of
  paintings (the committed 148-query painting set). The clean views are doing their job.
- **Paintings score higher** than the full synthetic set (all-angles R@1 **40.0%** vs 36.9%; `front`
  **70.5%** vs 64.8%), as expected.
- **Independent confirmation of the framing bug:** a completely different method (DINOv3 embeddings, EXP-7)
  flags the exact same `right upper` view as broken.

---

## 1. Setup

- **Model — the step-1 reproduction, trained on real Met data only.** R18-SWSL backbone → GeM pool →
  L2-norm → FC projector, trained with contrastive loss on the **original Met studio photos**. It
  reproduces the paper's best single model (our **GAP ≈ 36.0**, paper 36.1) and has **never seen any
  synthetic render** — this experiment only ever *queries* it.[^name] Checkpoint:
  `data/models/r18SWSL_con-syn+real-closest/…_epoch:_10`.
- **Queries — the whole synthetic dataset.** All **24,760** renders = **4,952 paintings × 5 camera views**
  (Blender/Cycles gallery scenes: framed canvas + glass + placard, randomized lighting/floor/camera; 512²
  RGBA PNG). Each render's source Met painting comes from its folder's `metadata.json`.
- **Database — the original 397,121 studio photos** (the Met training set), exactly as in the main
  benchmark. Synthetic images are *only* queries; they never enter the database.
- **Pipeline.** Extract multi-scale descriptors for every render with the step-1 model → apply the **same
  PCA-whitening** (learned on the training set) used by the main eval → search with faiss inner-product
  over L2-normalized descriptors → take each query's top-10 studio neighbours → score recall@1/5/10.

---

## 2. Results — overall and by camera view

All 24,760 renders, queried against the 397k studio database (rows sorted best→worst R@1):

| camera view | N | R@1 | R@5 | R@10 |
|---|--:|--:|--:|--:|
| **ALL angles** | 24,760 | **36.93** | **43.97** | **46.81** |
| left upper | 4,952 | 75.85 | 83.10 | 85.20 |
| front | 4,952 | 64.84 | 74.64 | 78.68 |
| right bottom | 4,952 | 21.95 | 31.10 | 34.77 |
| left bottom | 4,952 | 20.62 | 28.55 | 31.91 |
| **right upper** | 4,952 | **1.41** | 2.48 | 3.49 |

**Plain reading:** the overall 36.9% is an *average over wildly different views*. The two well-framed
views (`left upper`, `front`) recognize the right painting most of the time; the foreshortened `*bottom`
views land near 20%; and `right upper` is essentially a coin-flip away from zero. The spread is **~54×**
between best and worst view — almost entirely a framing artifact, not a property of the paintings (see §4).

---

## 3. Paintings only (the subset we care about)

The synthetic data exists to help recognize **paintings**, so we restrict to the renders whose source
painting is actually a **painting test query**. We use the project's single committed painting definition
— Met Open Access **`Classification == "Paintings"`** — which yields **148 painting test queries** spanning
**122 distinct classes**; the painting subset is the renders of those 122 classes (122 × 5 views = 610
renders). Same model, same database.

| group | N | R@1 (all angles) | R@1 `front` | R@1 `left upper` |
|---|--:|--:|--:|--:|
| all synthetic | 24,760 | 36.93 | 64.84 | 75.85 |
| **paintings (`Classification=="Paintings"`, 122 classes)** | 610 | **40.00** | **70.49** | **81.97** |

Paintings score **~3 points higher** on all-angles R@1 (40.0 vs 36.9) and higher on every view — the same
picture, slightly sharper. The key cross-check is the **`front` view: R@1 = 70.49%**, essentially equal to
the **69.59% accuracy this same model achieves on *real* visitor photos of paintings** (the same 148-query
`Classification=="Paintings"` slice, scored on the full 397k DB). In other words, **a well-framed render is
about as recognizable as a real photo** — the synthetic content is faithful; the limitation is the camera
rig, not the rendering.

---

## 4. The camera-framing bug

The per-view spread in §2 is not telling us "some angles are intrinsically harder." It is telling us the
**camera rig frames the painting very unevenly**:

- `left upper` / `front` — the painting fills the frame → highly recognizable (65–76% R@1).
- `left bottom` / `right bottom` — foreshortened, painting smaller/skewed → ~20% R@1.
- `right upper` — **grazing / edge-on**: the painting is a barely-visible sliver, so there is almost
  nothing to recognize → **1.4% R@1**.

This is the **known camera-rig bug** flagged in EXP-3. It is corroborated by a completely independent
method in **[EXP-7 / the DINOv3 embedding analysis](../synth-embedding-analysis/README.md)**: there, the
`right upper` render sits far from its *own* studio source in embedding space (cosine 0.44 vs `front`'s
0.84), bottoming out at exactly the view that scores worst here. Two different models, two different tests,
same conclusion → **the bug is real and lives in the camera poses, not in any one model.**

**Implication:** do **not** read the per-angle numbers as "synthetic renders are only 37% recognizable."
The well-framed views are 65–76%; the average is dragged down by the broken views. Per-angle synthetic
claims should wait until the `right upper` (and `*bottom`) camera poses are fixed and the dataset
regenerated.

---

## 5. What this means

- **The renders genuinely depict the right paintings.** A model trained only on real studio photos finds
  the correct painting from a render up to 76% of the time (best view) — the synthetic→real content gap is
  crossable, which is the prerequisite for the synthetic data being useful at all.
- **Well-framed render ≈ real photo.** `front` paintings (70.5%) ≈ real-photo accuracy (69.6%): the clean
  views are as informative as actual visitor photos, so the dataset's *content* is sound.
- **The rig, not the renderer, is the bottleneck.** The only thing standing between "37% overall" and
  "~70% overall" is the camera framing on 3 of the 5 views.
- **Consistent with the bigger story.** Adding this synthetic data to *training* already beats the paper
  (EXP-4: GAP 35.97 → 38.15). This retrieval test explains *why it can work* (the renders are recognizable)
  and *where the easy headroom is* (fix the framing).

---

## 6. Caveats

- **Recall, not GAP.** Every query has a correct answer in the database (no distractors), so these numbers
  are a pure recognition probe and are **not comparable** to the GAP/GAP⁻/ACC scores elsewhere.
- **Per-angle spread is a framing artifact**, not a clean measurement of "domain difficulty by viewpoint" —
  the rig must be fixed first (§4).
- **"Correct" = source class.** A render is judged against the single Met painting it was built from; if a
  painting genuinely resembles another collection item, that isn't modelled here.
- **Renders are "too clean."** EXP-7 found the renders look like clean studio shots, not messy phone photos
  — so high recall here means *recognizable*, not *realistic*.
- **Step-1 model only.** This uses the reproduction R18-SWSL model, not the +synthetic model (EXP-4) or the
  DINOv3 backbone (EXP-6).

---

## 7. How to reproduce

```bash
# 1) GPU: extract multi-scale descriptors for all 24,760 renders with the step-1 model, then
#    retrieve vs the 397k studio DB (recall@k, overall + per angle + the committed painting subset).
sbatch synth_eval.slurm                     # job 7342800: COMPLETED in ~7 min on an H100
# 2) CPU re-score only (descriptors already exist — no GPU); reads data/gt_paint/testset.json for the
#    committed Classification=="Paintings" subset. Run via a standard-partition SLURM job, NOT the login node.
.venv/bin/python scripts/eval_synthetic_retrieval.py   # job 7342900: ~5 min on a CPU node
```

Outputs (git-ignored `data/`):
- `data/descriptors/synthetic/synth_descriptors.pkl` — per-render descriptors + source Met id + camera angle.
- `data/descriptors/synthetic/retrieval_summary.json` — **every number in this doc**.

Code: [`scripts/extract_synthetic.py`](../../scripts/extract_synthetic.py) (step-1 model → render descriptors) ·
[`scripts/eval_synthetic_retrieval.py`](../../scripts/eval_synthetic_retrieval.py) (retrieval + recall@k + the JSON dump).

[^name]: The checkpoint is named after the paper's method, **Con-Syn+Real-closest**. There, "Syn" means
the contrastive loss's *augmented-view* positive (a standard data-augmentation trick), **not** our
synthetic gallery dataset. The model is trained entirely on real Met studio photos.
