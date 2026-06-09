# Can the Met model recognize our synthetic gallery renders?

*We take the recognition model trained **only on the original Met data** (it has never seen a synthetic
image), use our **24,760 synthetic gallery renders as queries**, and measure how well it finds the
**correct painting** — against **two databases**: the **full 397k Met benchmark** and a **paintings-only**
gallery. Everything is scored the paper's way (**GAP / GAP⁻ / ACC**) plus **recall@k**, broken down by camera
angle, with the **paper's real-photo numbers as the baseline**. (Met / VISART fork; lab notebook:
[`EXPERIMENTS.md` → EXP-3](../../EXPERIMENTS.md).)*

## What we did, in one paragraph

The Met task is **instance recognition**: given a query photo, find the same artwork in a database of clean
**studio catalog photos**. We have a synthetic dataset — Blender renders of Met paintings hung in a gallery,
shot from 5 camera angles — and want an honest read on how recognizable they are. So we use each render as a
**query**, with the model = our **step-1 reproduction** of the paper's best model, trained *purely on real Met
studio photos* (**zero synthetic exposure**). We retrieve against **two databases**: **(A)** the full original
benchmark — all **397,121** studio photos / 224,408 classes — and **(B)** a **paintings-only** gallery — the
**12,403** photos of the **4,898** `Classification=="Paintings"` classes. For each we report the paper's
**GAP / GAP⁻ / ACC** and **recall@k**, per camera angle, alongside the **paper's real-photo benchmark** and our
reproduction's real-photo numbers as baselines.

> **How to read the numbers.** All scores are 0–100, higher is better; metric definitions (**GAP**,
> **GAP⁻**, **ACC**, **R@1/5/10**) are in the [experiments README](../README.md). Two query types appear:
> **real** photos (the paper's test queries) and our **synthetic** renders. Specific to this experiment:
> - **the two databases** — what we search against. **(A) full Met:** 397,121 studio photos / 224,408 classes
>   (the original benchmark). **(B) paintings-only:** 12,403 photos of the 4,898 `Classification=="Paintings"`
>   classes (a smaller, easier gallery).
> - **GAP includes the 18,316 distractors** (open-set); the renders carry none of their own, so **GAP⁻ equals
>   the closed-world GAP** — the two columns show both treatments at once.
> - **R@1 == ACC** here (the τ=50 kNN vote follows the single nearest neighbour); R@5/R@10 add the "in the top-k" view.
> - **correct** — the query retrieves a studio photo of the same Met class (for a render, its source painting).
> - **camera angle** — the 5 gallery viewpoints: `front`, `left/right upper`, `left/right bottom`.

## TL;DR

- **A model that has never seen synthetic data still recognizes the renders** — and the well-framed views do
  it about as well as *real* photos.
- **Camera angle dominates everything.** On the full DB, `front`/`left upper` renders hit ACC 65–76% while the
  broken `right upper` view collapses to ~1% — the **camera-framing bug**, identical across GAP/GAP⁻/ACC/recall.
- **A well-framed render ≈ a real photo.** `front` renders land within a few points of the real painting
  queries (full DB: ACC 64.8 vs 69.6; paintings-only DB: 79.0 vs 72.3 — *above*).
- **The paintings-only database is uniformly easier** (fewer classes to confuse): e.g. `front`-render ACC
  64.8 → 79.0, all-angles 36.9 → 45.2.
- **Paper baseline in every table:** our reproduction matches the paper on the real benchmark (GAP 35.97 vs
  36.1); the synthetic numbers sit beside it, scored identically.

---

## 1. Setup

- **Model — the step-1 reproduction, trained on real Met data only.** R18-SWSL → GeM → L2-norm → FC projector,
  contrastive loss on the **original Met studio photos**. Reproduces the paper's best single model (our **GAP
  35.97**, paper 36.1) and has **never seen a synthetic render**.[^name] Checkpoint:
  `data/models/r18SWSL_con-syn+real-closest/…_epoch:_10`.
- **Queries.** The **24,760** synthetic renders (= 4,952 paintings × 5 views; 512² Blender/Cycles gallery
  scenes). Reference rows use the **real** Met test queries (1,003 total; 148 of them paintings).
- **Two databases.** **(A)** full Met — **397,121** studio photos / 224,408 classes; **(B)** paintings-only —
  **12,403** photos / **4,898** `Classification=="Paintings"` classes (a subset of the same studio photos).
- **Scoring.** Multi-scale descriptors → PCA-whitening (re-learned per database) → faiss + kNN classifier
  (**K=7, τ=50**, the EXP-2 protocol). **GAP includes the 18,316 real distractors** (open-set); GAP⁻/ACC
  exclude them. Database (B) just subsets database (A)'s descriptors — no re-extraction.

---

## 2. Table A — against the full Met benchmark database

Database = all **397,121** studio photos (224,408 classes). The first three rows are the **real-photo
baselines** (paper + our reproduction); the rest are the **synthetic renders** per camera view (sorted
best→worst). GAP includes the 18,316 distractors.

| query (DB = full Met, 397k / 224k cls) | N | GAP | GAP⁻ | ACC | R@1 | R@5 | R@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| *real — paper R18-SWSL Con-Syn+Real-closest* | 1,003 | 36.1 | 52.4 | 55.0 | — | — | — |
| *real — our step-1 reproduction* | 1,003 | 35.97 | 52.14 | 54.64 | — | — | — |
| *real — our paintings only (148)* | 148 | 39.50 | 67.86 | 69.59 | — | — | — |
| **synthetic — ALL angles** | 24,760 | 31.42 | 33.69 | 36.93 | 36.93 | 43.97 | 46.81 |
| synthetic — left upper | 4,952 | 65.49 | 74.72 | 75.85 | 75.85 | 83.10 | 85.20 |
| synthetic — front | 4,952 | 54.54 | 62.55 | 64.84 | 64.84 | 74.64 | 78.68 |
| synthetic — right bottom | 4,952 | 9.58 | 16.23 | 21.95 | 21.95 | 31.10 | 34.77 |
| synthetic — left bottom | 4,952 | 9.57 | 15.43 | 20.62 | 20.62 | 28.55 | 31.91 |
| **synthetic — right upper** | 4,952 | 0.02 | 0.09 | 1.41 | 1.41 | 2.48 | 3.49 |

*(Real rows: GAP/GAP⁻/ACC are the benchmark metrics; recall@k wasn't reported there and R@1==ACC, so those
cells are blank. Our reproduction matches the paper, GAP 35.97 vs 36.1.[^authors])*

**Reading it:** the well-framed `front`/`left upper` renders (ACC 65–76) come within a few points of the real
painting queries (ACC 69.6); the `*bottom` views fall to ~20, and `right upper` collapses to ~1 — the
**framing bug** (§4), identical across GAP/GAP⁻/ACC/recall. The all-angles average (ACC 36.9) is just that —
an average dragged down by the broken views.

---

## 3. Table B — against a paintings-only database

Same queries and model, but the database is restricted to the **4,898 `Classification=="Paintings"` classes**
(**12,403** studio photos). The synthetic queries are the **24,490** renders whose source class is in this DB
(the 54 non-painting-classification synthetic classes are dropped as unanswerable). GAP still uses the same
18,316 distractors, so **GAP = open-set** and **GAP⁻ = the closed-world painting GAP**.

| query (DB = paintings only, 12,403 / 4,898 cls) | N | GAP | GAP⁻ | ACC | R@1 | R@5 | R@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| *real — our paintings only (148)* | 148 | 45.69 | 71.48 | 72.30 | 72.30 | 79.73 | 83.78 |
| **synthetic — ALL angles** | 24,490 | 39.14 | 42.19 | 45.15 | 45.15 | 53.61 | 56.98 |
| synthetic — left upper | 4,898 | 70.97 | 80.12 | 80.93 | 80.93 | 87.83 | 89.98 |
| synthetic — front | 4,898 | 68.55 | 77.90 | 78.99 | 78.99 | 87.34 | 89.96 |
| synthetic — right bottom | 4,898 | 15.01 | 27.28 | 32.50 | 32.50 | 45.35 | 50.73 |
| synthetic — left bottom | 4,898 | 14.91 | 25.25 | 30.60 | 30.60 | 42.08 | 47.10 |
| **synthetic — right upper** | 4,898 | 0.06 | 0.35 | 2.74 | 2.74 | 5.45 | 7.15 |

*(No paper row — the paper publishes no painting-only / painting-query numbers. The real painting queries'
GAP⁻ 71.48 / ACC 72.30 match the project's recorded closed-world control, 71.62 / 72.30.)*

**Reading it:** the smaller gallery is **uniformly easier** (fewer classes to confuse) — synthetic recognition
climbs across the board (all-angles ACC 36.9→45.2, `front` 64.8→79.0) and the real painting queries rise to
ACC 72.3. Here the well-framed `front`/`left upper` renders (ACC 79–81) actually **beat** the real painting
queries (72.3). The framing bug is unchanged — `right upper` still ~3%. This smaller, easier benchmark is
**not** comparable to Table A or the paper's GAP 36.1.

---

## 4. The camera-framing bug

The per-view spread in **both tables** isn't "some angles are intrinsically harder" — it is the **camera rig
framing the painting very unevenly**:

- `left upper` / `front` — the painting fills the frame → highly recognizable.
- `left bottom` / `right bottom` — foreshortened, painting smaller/skewed → ~20–33% ACC.
- `right upper` — **grazing / edge-on**: the painting is a barely-visible sliver → ~1–3% ACC.

This is the **known camera-rig bug** flagged in EXP-3, corroborated by a completely independent method in
**[EXP-7 / the DINOv3 embedding analysis](../dinov3-embedding-analysis/README.md)**: there, the `right upper`
render sits far from its *own* studio source (cosine 0.44 vs `front`'s 0.84), bottoming out at exactly the
view that scores worst here. Two models, two tests, same conclusion → **the bug lives in the camera poses, not
in any one model.**

**Implication:** do **not** read the all-angles average as "renders are only ~37–45% recognizable." The
well-framed views are far higher; the average is dragged down by the broken views. Per-angle synthetic claims
should wait until the `right upper` (and `*bottom`) poses are fixed and the dataset regenerated.

---

## 5. What this means

- **The renders genuinely depict the right paintings.** A model that never saw synthetic data finds the
  correct painting from a well-framed render most of the time (`front` ACC 65 on the full DB, 79 on the
  paintings-only DB) — the synthetic→real content gap is crossable.
- **Well-framed render ≈ real photo.** `front` renders land within a few points of the real painting queries
  (full DB 64.8 vs 69.6 ACC; paintings-only DB 79.0 vs 72.3 — above) — the synthetic *content* is faithful.
- **The rig, not the renderer, is the bottleneck** — the only thing between the ~37–45% all-angles average and
  the ~65–80% of the good views is the framing on 3 of the 5 cameras.
- **The paintings-only DB is a smaller, easier benchmark** — every number is higher, but it is **not**
  comparable to the full-DB results or the paper's GAP 36.1.
- **Consistent with the bigger story.** Adding this synthetic data to *training* already beats the paper
  (EXP-4: GAP 35.97 → 38.15); this retrieval test explains *why* (the renders are recognizable) and *where the
  easy headroom is* (fix the framing).

---

## 6. Caveats

- **GAP open vs. GAP⁻ closed.** The synthetic renders carry no distractors of their own; we borrow the **same
  18,316 real test distractors** for GAP (open-set), while GAP⁻ excludes them (= the closed-world GAP). So both
  treatments show at once, and the synthetic vs. real-query GAP/GAP⁻/ACC are directly comparable.
- **ACC == R@1 throughout** — the τ=50 kNN vote follows the single nearest neighbour, so the classifier's top-1
  and 1-NN retrieval agree exactly (which is why those two columns match).
- **Per-angle spread is a framing artifact**, not "viewpoint difficulty" — the rig must be fixed first (§4).
- **Paintings-only DB is a smaller, easier benchmark** (12,403 vs 397,121 images) — its numbers are **not**
  comparable to the full-DB results or the paper's GAP 36.1.
- **"Correct" = source class**; **renders are "too clean"** (EXP-7 — recognizable, not photo-realistic);
  **step-1 model only** (not the +synthetic model of EXP-4 or the DINOv3 backbone of EXP-6).

---

## 7. How to reproduce

```bash
# 1) GPU: extract multi-scale descriptors for all 24,760 renders with the step-1 model.
sbatch slurm/synth_eval.slurm                              # job 7342800: ~7 min on an H100 (extract + recall@k)
# 2) CPU re-scores (descriptors already exist — no GPU). Run via a standard-partition SLURM job, NOT login.
.venv/bin/python scripts/eval_synthetic_retrieval.py # recall@k (full DB)            — job 7342900, ~5 min
.venv/bin/python scripts/eval_synthetic_gap.py       # GAP/GAP-/ACC, full DB (Table A) — job 7342973, ~8 min
.venv/bin/python scripts/eval_painting_db.py         # GAP/GAP-/ACC + recall, paint DB (Table B) — job 7342987, ~1 min
```

Outputs (git-ignored `data/`):
- `synth_descriptors.pkl` — per-render descriptors + source Met id + camera angle.
- `retrieval_summary.json` — recall@k (full DB).
- `gap_summary.json` — GAP/GAP⁻/ACC, **full DB** (Table A synthetic rows; all + per view).
- `painting_db_summary.json` — GAP/GAP⁻/ACC + recall, **paintings-only DB** (Table B; all + per view).

Code: [`scripts/extract_synthetic.py`](../../scripts/extract_synthetic.py) (render descriptors) ·
[`scripts/eval_synthetic_retrieval.py`](../../scripts/eval_synthetic_retrieval.py) (recall@k) ·
[`scripts/eval_synthetic_gap.py`](../../scripts/eval_synthetic_gap.py) (Table A GAP/GAP⁻/ACC) ·
[`scripts/eval_painting_db.py`](../../scripts/eval_painting_db.py) (Table B, paintings-only DB).
Real-query baselines (GAP/GAP⁻/ACC) are from `EXPERIMENTS.md` (EXP-1/EXP-2) and the paper.

[^name]: The checkpoint is named after the paper's method, **Con-Syn+Real-closest**. There, "Syn" means the
contrastive loss's *augmented-view* positive (a standard data-augmentation trick), **not** our synthetic
gallery dataset. The model is trained entirely on real Met studio photos.

[^authors]: Sanity check on our eval pipeline: re-scoring the authors' *released* descriptors gives GAP 36.10 /
GAP⁻ 52.41 / ACC 55.03 — matching the paper's 36.1 / 52.4 / 55.0.
