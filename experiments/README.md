# Experiments

Per-experiment writeups for the Met / VISART fork. Each subdirectory is one self-contained
experiment: a `README.md` plus its plots in `figures/`. The running lab notebook (raw log, job ids,
exact commands) is [`../EXPERIMENTS.md`](../EXPERIMENTS.md); the task/method reference (paper targets,
`pairs_type` ⇄ method mapping) is [`../reference/README.md`](../reference/README.md).

**This page defines the metrics and the shared evaluation protocol once, so the individual experiment
docs don't repeat them.**

## The task

The Met benchmark is **instance-level recognition**: given a query photo, identify which of the
~224k museum-exhibit classes it shows — or correctly reject it as **not in the collection**. A
non-parametric **kNN classifier** over global descriptors gives each query a predicted class **and a
confidence**. Training images are clean **studio catalog photos**; the test queries are **real visitor
photos** plus **distractors** (photos of things not in the collection) — so the core difficulty is the
studio→real-photo distribution shift *and* rejecting the distractors.

- **Test set:** 19,319 queries = **1,003 real Met queries** + **18,316 distractors** (10,352
  other-artwork + 7,964 non-art). Validation: 2,165 queries.
- **Database (default):** all **397,121** studio photos / **224,408** classes. Some experiments also use
  a **paintings-only** database (**12,403** photos / **4,898** `Classification=="Paintings"` classes) —
  smaller and easier, and **not** comparable to the full-DB numbers.
- **Classifier:** faiss kNN over L2-normalized, PCA-whitened descriptors; per-class scores are
  softmax-weighted by temperature **τ** over the **K** nearest neighbours. K and τ are tuned on the
  validation set (full K×τ grid — see [`../EXPERIMENTS.md`](../EXPERIMENTS.md)).

## The metrics

All scores are **0–100, higher is better**. Each query gets a predicted class + a confidence; the three
metrics read that output differently.

- **ACC** (accuracy) — of the **real (non-distractor) queries**, the fraction whose **top-1** predicted
  class is correct. Ignores confidence and distractors. *"Did we name the exhibit right?"*
- **GAP⁻** (Global Average Precision, no distractors) — rank the **real queries** by confidence and take
  the average precision over the correct ones. Rewards correct answers also being the **confident** ones.
  *"Are the right answers also the confident ones?"*
- **GAP** (the headline, open-set) — Global Average Precision over **all 19,319 queries** (real +
  18,316 distractors): rank *everything* by confidence; distractors always count as wrong. Rewards
  correct queries being **confident** *and* distractors being **unconfident**. This is the realistic
  open-set metric, hence primary — and the lowest of the three. *"Are the right answers ranked above
  everything, including 18k things that aren't in the catalogue?"*
- **R@k** (recall@k; e.g. R@1 / R@5 / R@10) — is the correct class among the **k** nearest database items
  for the query. A retrieval-style readout. With the project's τ=50 kNN vote, **R@1 == ACC** (the vote
  follows the single nearest neighbour).

**The GAP⁻ − GAP gap = distractor-rejection quality.** A model can rank true matches well (high GAP⁻)
yet hand distractors high confidence (low GAP); the size of that gap is exactly how badly distractors
pollute the ranking.

> **Closed world vs. open set.** With **no distractors**, GAP and GAP⁻ coincide — so a closed-world
> (paintings-only) GAP is the same quantity as that benchmark's GAP⁻. Numbers from the smaller
> paintings-only database are **not** comparable to the full 397k-DB results or the paper's GAP 36.1.

The original paper's best single model scores **GAP 36.1 / GAP⁻ 52.4 / ACC 55.0** — the baseline to beat.

## The experiments

| Experiment | Question |
|---|---|
| [`synthetic-retrieval/`](synthetic-retrieval/README.md) | Can the Met model (trained on real data only) recognize our synthetic gallery renders, used as queries? |
| [`training-with-synthetic/`](training-with-synthetic/README.md) | Does adding synthetic data to training beat the paper on its own — then, a stronger backbone (DINOv3 + geometric re-rank)? |
| [`real-synth-mixing/`](real-synth-mixing/README.md) | Real vs synthetic training mix: how much does synthetic help for paintings, and does *more* synthetic keep helping? |
| [`synth-embedding-analysis/`](synth-embedding-analysis/README.md) | How does a frozen DINOv3 organize the renders (camera angle / scene settings / painting identity), and how far is synthetic from real? |

Painting experiments use the committed definition `Classification == "Paintings"` (4,898 classes /
148 test queries); see [`../EXPERIMENTS.md`](../EXPERIMENTS.md).
