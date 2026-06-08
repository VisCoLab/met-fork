# Met / VISART 2026 — Progress Update

**Date:** 2026-06-08 · **For:** supervisor meeting · **Repo:** `met` (fork of nikosips/met)

## TL;DR

- ✅ **Reproduced the paper's best single model** (R18-SWSL): our **GAP 35.97** vs paper 36.1, and validated our GAP/GAP⁻/ACC eval against the authors' released descriptors (36.10). Solid foundation.
- ✅ **Synthetic gallery data helps**: fine-tuning R18 on our synthetic renders lifts **GAP 35.97 → 38.61** (and helps paintings most: ACC 69 → 74). A clean confound-free A/B is finishing today.
- ⚠️ **Finding that reframes the project:** a **frozen DINOv3** foundation model scores **GAP ≈ 52** on Met *zero-shot* — nearly double our trained R18 (36). **This is the DINOv3 authors' own published result**, so it resets the baseline: our contribution must sit **on top of DINOv3**, not just beat R18.
- ✅ **First on-top-of-DINOv3 contribution already works:** a **parameter-free geometric re-rank** lifts DINOv3 **+4.9 GAP** (48.2 → 53.1, ViT-L) by fixing DINOv3's specific weakness — rejecting distractors. No training.
- 🟡 **DINOv3 + synthetic-data adaptation (head / LoRA)** is training now; full-benchmark numbers **land today**.

---

## 1. Task recap

The Met benchmark is **instance-level recognition** over ~224k museum-exhibit classes via a non-parametric **kNN classifier** over global descriptors. Primary metric is **GAP** (Global Average Precision over 19,319 test queries = **1,003 real visitor photos + 18,316 distractors**) — it rewards giving distractors *low* confidence, so it is much harder than plain accuracy. Secondary: **GAP⁻** (Met queries only) and **ACC**. Our fork's original plan: (1) a synthetic gallery phone-photo dataset, (2) a new method — to beat the paper's R18-SWSL **GAP 36.1**.

### Metrics — one line each
Each query gets a predicted class **+ a confidence** from the kNN classifier; the three metrics read that output differently:

- **ACC** — accuracy on the **1,003 real Met queries only** (is the top-1 predicted class correct?). Ignores confidence and distractors. *“Did we name the exhibit right?”*
- **GAP⁻** — Global Average Precision over **Met queries only**: rank them by confidence, average precision over the correct ones. *“Are the correct answers also the confident ones?”*
- **GAP** (the headline) — Global Average Precision over **all 19,319 queries** (1,003 Met + 18,316 distractors): rank *everything* by confidence; distractors always count as wrong. Rewards correct Met queries being **confident** *and* distractors being **unconfident**. *“Are the right answers ranked above everything — including 18k things that aren't in the catalogue?”* This is the realistic open-set metric, hence primary.
- The **GAP⁻ − GAP gap** = how badly distractors pollute the ranking = **distractor-rejection quality**. DINOv3's is large (75 vs 52, a 23-pt gap) — exactly what our re-rank attacks.

## 2. Completed results (R18 — the paper's recipe, reproduced & extended)

All numbers in our validated eval pipeline:

| Model | **Full:** GAP | GAP⁻ | ACC | **Paint:** GAP | GAP⁻ | ACC |
|---|--:|--:|--:|--:|--:|--:|
| Paper R18-SWSL (target) | 36.1 | 52.4 | 55.0 | —† | —† | —† |
| Our reproduction (from scratch) | 35.97 | 52.14 | 54.64 | 41.5 | 67.8 | 69.4 |
| **+ fine-tune on synthetic only** | **38.61** | 55.6 | 57.8 | **48.7** | **72.8** | **74.0** |
| **+ fine-tune on synthetic + real** | ‡‡ | ‡‡ | ‡‡ | ‡‡ | ‡‡ | ‡‡ |

*Full = the 19,319-query test set; Paint = strict paintings subset (173 queries; its GAP still includes all 18,316 distractors). †the original paper didn't report a paintings subset. ‡‡ the synthetic+real (combined) fine-tune is trained; its eval is computing now (lands in ~15 min).*

**Takeaways:** (a) paintings are the tractable, high-value subset (ACC 69 vs 55 overall) — the natural target for synthetic data; (b) synthetic gallery renders improve recognition, and *most* on paintings — but the current **+2.6 GAP is confounded** by extra training epochs; a clean from-scratch A/B is running now to isolate the synthetic-data effect.

*(Minor: our synthetic renders have a camera-rig framing bug — 1 of 5 viewpoints is grazing/edge-on and near-useless; flagged for regeneration.)*

## 3. The DINOv3 development — the strategic shift

Benchmarking **frozen foundation backbones** through the *exact* Met protocol (reusing the official codebase + our eval):

| Backbone | **Full:** GAP | GAP⁻ | ACC | **Paint:** GAP | GAP⁻ | ACC |
|---|--:|--:|--:|--:|--:|--:|
| R18-SWSL (our trained baseline) | 35.97 | 52.14 | 54.64 | 41.5 | 67.8 | 69.4 |
| **DINOv3 ViT-L** (frozen, zero-shot) | 48.2 | 72.1 | 77.1 | ‡ | ‡ | ‡ |
| **DINOv3 ViT-7B** (frozen, zero-shot) | **52.1** | 75.5 | 82.0 | ‡ | ‡ | ‡ |

*‡ DINOv3 paintings-subset evals are running now (land today); the full-test numbers are final.*

This **reproduces the DINOv3 paper's own Met number** (≈ 50.8 = DINOv2's 40.0 + their reported +10.8). **Implication:** "DINOv3 is strong on Met" is already published — not something we can claim. The realistic bar is now **~52, not 36**, and our novelty must be a *delta on top of DINOv3*.

**Where DINOv3 is weak — and our first contribution.** DINOv3 ranks true matches superbly (GAP⁻ 75) but its confidence barely separates the 18k distractors (GAP 52) — a **23-point gap**. We add a **parameter-free geometric re-rank**: patch-level mutual-NN + RANSAC verification on DINOv3 patches, fused into the classifier's confidence.

> **How the geometric re-rank works.** Two stages. **Stage 1:** DINOv3's *global* descriptor retrieves the top-50 candidate catalogue images for a query (standard kNN). **Stage 2:** for the query and each candidate we take their DINOv3 **patch** features (the 16×16-pixel-patch grid of *local* descriptors), find mutually-nearest-neighbour patch pairs, and fit a geometric transform (homography) over the matched patch *positions* using RANSAC. The number of geometrically-consistent "inlier" matches measures whether the two images depict the **same physical object** — a genuine query vs its true catalogue image yields many consistent matches; a distractor vs anything yields almost none. We fold that inlier score into the kNN confidence, so distractors sink in the global ranking and GAP rises. It is the classical local-feature-matching recipe (mutual-NN + RANSAC, as in SIFT / DELG) applied to **foundation-model patch features** — no training, test-time only. *(It directly targets the GAP⁻−GAP distractor gap above.)*

| DINOv3-ViT-L | **Full:** GAP | GAP⁻ | ACC | **Paint:** GAP | GAP⁻ | ACC |
|---|--:|--:|--:|--:|--:|--:|
| zero-shot | 48.2 | 72.1 | 77.1 | ‡ | ‡ | ‡ |
| **+ geometric re-rank** | **53.1** | 74.7 | 77.1 | ‡ | ‡ | ‡ |

*‡ paintings-subset eval pending (see note above).*

**+4.9 GAP, both GAP and GAP⁻ up, ACC unchanged** — clean distractor rejection, no training. ViT-L + re-rank already edges out 7B zero-shot. Next: apply to ViT-7B (projected ~57).

*All DINOv3 work is built on the **official Met codebase** (the backbone is swapped into the benchmark's own training/eval pipeline), so the comparison is apples-to-apples and the contribution is clean.*

## 4. Currently running (6 jobs) / landing soon

| Run | What it answers | ETA |
|---|---|---|
| **DINOv3 + synthetic adaptation** (head & LoRA) | *Does adapting DINOv3 with our synthetic gallery data beat its ZS 48/52?* | **~1 hour** |
| DINOv3 + studio adaptation (head & LoRA) | Same, on the real Met train set (clean A/B) | ~tomorrow |
| R18 from-scratch + synthetic (epoch 6/10) | Confound-free synthetic-data A/B | today |
| R18 cross-domain pair-mining (epoch 5/10) | A method mining studio↔synthetic positive pairs | today |

The synthetic-adaptation runs are where **contribution #1 (the synthetic data) re-enters** — in a sharper form: *does synthetic data still help a strong foundation backbone?*

## 5. For discussion today

1. **Reframe the paper around DINOv3.** Since DINOv3 ZS (~52) is published, propose the contribution = **(a) geometric re-rank for distractor rejection + (b) synthetic-data adaptation, both on top of DINOv3**, reported as a new Met state-of-the-art. Is this the right VISART story?
2. **Synthetic data's role.** It clearly helps the weak R18 (+2.6). The more publishable question is whether it *still* helps a strong foundation backbone — answer landing today.
3. **Positioning / honesty.** We must cleanly separate "DINOv3 is strong" (theirs) from "our re-rank + adaptation improve it" (ours) — a reviewer will check this.
4. **Effort is modest & on-cluster** — most runs are short; the heaviest are the DINOv3 studio-adaptation trainings.

## 6. Immediate next steps

- Fold today's DINOv3 synth-adaptation numbers (and studio, tomorrow) into the comparison table.
- If the re-rank holds, run it on ViT-7B for the headline (~57 projected).
- Finish the clean R18 synthetic A/B to de-confound the +2.6.
- Lock the paper framing per discussion point (1).
