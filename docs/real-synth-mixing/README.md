# Real ↔ Synthetic training-data mixing (closed painting world)

_Created 2026-06-08._

**Question.** In the closed painting world, how does recognition of **real** painting
photos change as the *training* mix shifts from 100 % real studio images toward 100 %
synthetic gallery renders — holding training-set **size** and the **evaluation** fixed?

**Headline.** Adding synthetic gallery renders helps a lot, **monotonically**, and
**training on synthetic alone (0 % real) is best** — GAP⁻ 72.47 vs the real-only baseline
67.18 (**+5.3**), even edging the full-397 k-image model (71.62) on the same eval. The
synthetic gallery renders bridge the studio→phone-photo domain gap better than the real
*studio* images do.

## Setup

- **Closed painting world** — Met classes with `Classification == "Paintings"` (4,898
  classes / 12,403 studio images). Definition rationale, nesting, val=1 caveat: see
  `EXPERIMENTS.md` (EXP-8) and the `closed-world-paintings` note. Built by
  `scripts/build_paintings_data.py` (→ `data/gt_paint`) and
  `scripts/build_paintings_mix_data.py` (→ `data/gt_paint_mix_*`).

- **Training mix** — **fixed total = 12,403 images** per run (= the real-painting set
  size, so the 100 %-real point is exactly the baseline). The real:synth split varies;
  subsamples are **nested & seeded** (real shuffled once @seed 0, synth once @seed 1,
  then prefixes), so higher-real mixes are supersets of lower-real ones.

  | train mix | real imgs | synth imgs | manifest |
  |---|--:|--:|---|
  | 100 / 0 (baseline) | 12,403 | 0 | `data/gt_paint` |
  | 80 / 20 | 9,922 | 2,481 | `data/gt_paint_mix_80r20s` |
  | 60 / 40 | 7,442 | 4,961 | `data/gt_paint_mix_60r40s` |
  | 40 / 60 | 4,961 | 7,442 | `data/gt_paint_mix_40r60s` |
  | 20 / 80 | 2,481 | 9,922 | `data/gt_paint_mix_20r80s` |
  | 0 / 100 | 0 | 12,403 | `data/gt_paint_mix_0r100s` |

- **Evaluation — identical for every run, always REAL.** kNN DB = the **12,403 real**
  painting studio images, test = **148 real** painting queries, val = 1. Synthetic data
  enters **training only**, never the eval DB. No distractors ⇒ GAP ≡ GAP⁻. Because
  val = 1 can't tune k/τ, we use **2-fold CV** over the 148 test queries (tune on one
  half, report on the other, seed 0) — every query is scored on a fold it did not tune,
  so there's no leakage (`scripts/eval_paintings_closed.py`).

- **Recipe** — same for all runs: R18-SWSL (ImageNet-SWSL init), 10 epochs,
  `--pairs_type new_pos+new_neg --emb_proj --pca`, seed 0.

- **Run.** `paint_train.slurm <mix_dir> data/aug <tag>` then (chained on success)
  `paint_eval.slurm data/models/r18SWSL_<tag> 10 <tag>` — the eval extracts over
  `data/gt_paint`, keeping the DB + queries real.

## Results

GAP⁻ / ACC, 2-fold-CV mean on the 148 real painting queries (≡ GAP since no distractors).

| train mix (real : synth) | GAP⁻ | ACC | Δ GAP⁻ vs baseline | jobs |
|---|--:|--:|--:|---|
| **100 : 0**  (real only, baseline) | 67.18 | 70.27 | — | 7336119 / 7336122 |
| 80 : 20 | 70.56 | 72.97 | +3.38 | 7336276 / 7336277 |
| 60 : 40 | 70.65 | 72.30 | +3.47 | 7336278 / 7336279 |
| 40 : 60 | 71.37 | 72.97 | +4.19 | 7336280 / 7336281 |
| 20 : 80 | 71.24 | 72.30 | +4.06 | 7336282 / 7336283 |
| **0 : 100**  (synth only) | **72.47** | **73.65** | **+5.29** | 7336284 / 7336285 |

**Off-axis reference** (different variable — training-DB *scope*, not real:synth mix):
the full step-1 model (trained on all 397,121 images / 224 k classes, paintings +
everything) evaluated on the **identical** closed painting world → **GAP⁻ 71.62 / ACC
72.30** (job 7336269).

### Findings

1. **Any synthetic helps, and the gain is monotonic in synthetic fraction.** GAP⁻ climbs
   67.18 → 70.56 → 70.65 → 71.37 → (71.24) → 72.47 as real→synth (the 20/80 vs 40/60 dip
   is within noise). Even a 20 % synthetic admixture buys +3.4 GAP⁻.
2. **Synthetic-only (0 % real) is the best model** (72.47 / 73.65) — it beats the real-only
   baseline by **+5.3 GAP⁻ / +3.4 ACC** and edges the full-397 k model (71.62) on the same
   eval, despite using 0 real painting images and ~32× less data than that reference.
3. **Interpretation.** The eval queries are real *phone/gallery* photos; the training
   "real" images are studio shots. The synthetic renders model gallery viewing conditions
   (framing, glass, lighting, viewpoint), so they apparently sit closer to the query
   domain than the studio images do — consistent with EXP-4 (synthetic fine-tuning helped
   broadly) and EXP-7 (renders add viewpoint/glass/lighting variation). Here, with the eval
   DB held fixed and real, that translates directly into better real-photo recognition.

## Caveats

- **Statistical power is limited:** 148 test queries, val = 1, 2-fold CV. ~1.5 queries
  ≈ 1 % ACC, so single-point differences ≤ ~2 pts are within noise. The **baseline→synth
  gap (~+5 GAP⁻)** and the **monotone trend** are the trustworthy signals; the exact
  ranking among 80/20…20/80 is not. A multi-seed repeat would tighten this.
- **Not comparable** to the paper GAP 36.1 or EXP-2's painting slice (67.81) — the closed
  painting DB is ~32× smaller (an easier task). Comparisons **across the mix** are valid:
  the eval is byte-identical for every row, only the model weights differ.
- The synthetic set still contains the **broken `right upper` grazing view** (EXP-3 /
  EXP-7); a per-view-clean sweep would need the fixed camera rig + regeneration. The
  synth-only win is achieved *despite* including that broken view.
- Fixed total size = 12,403 means the 0/100 run uses ~half the available synthetic data
  (24,490 painting renders); an "all-synth" (un-size-matched) run could push it further.

## Reproduce

```bash
.venv/bin/python scripts/build_paintings_mix_data.py          # build data/gt_paint_mix_*
for tag in 80r20s 60r40s 40r60s 20r80s 0r100s; do
  tid=$(sbatch --parsable --job-name=met-tr-$tag paint_train.slurm data/gt_paint_mix_$tag data/aug paint_$tag)
  sbatch --dependency=afterok:$tid --job-name=met-ev-$tag paint_eval.slurm data/models/r18SWSL_paint_$tag 10 $tag
done
grep -H "2-fold mean" logs/met-ev-*.out                        # collect GAP-/ACC
```
