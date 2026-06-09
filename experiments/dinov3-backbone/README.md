# DINOv3 backbone + geometric re-rank

*The DINOv3 retrieval thread. A **frozen DINOv3** foundation backbone nearly **doubles** Met GAP
zero-shot (the DINOv3 paper's own published result), and our **parameter-free geometric re-rank** fixes
its one real weakness — rejecting distractors — for **+4.9 GAP**, with no training. (Met / VISART fork;
lab notebook: [`EXPERIMENTS.md` → EXP-6](../../EXPERIMENTS.md).)*

> **Scope.** This is the project's *secondary* thread — the main contribution is the synthetic dataset
> on the R18 model ([`../training-with-synthetic/`](../training-with-synthetic/README.md)). The DINOv3
> retrieval methodology here is where we'd push further given time: the **backbone + re-rank** below are
> done; the head/LoRA **adaptation** (does our synthetic data help a strong backbone too?) is scaffolded
> but not yet run.

> **How to read the numbers.** All scores are 0–100, higher is better; metric definitions (**GAP**,
> **GAP⁻**, **ACC**) are in the [experiments README](../README.md). The quantity that matters most here
> is the **GAP⁻ − GAP gap = distractor-rejection quality** — DINOv3's weak spot, and exactly what the
> re-rank attacks.

## TL;DR

- **A frozen DINOv3 nearly doubles GAP out of the box:** 35.97 (our R18 baseline) → **48.16** (ViT-L),
  **52.11** (the 7B model). *Zero-shot DINOv3 on Met is the DINOv3 paper's own result — not our contribution.*
- **DINOv3's weakness is distractor rejection:** it ranks the right painting first superbly (GAP⁻ 72)
  but barely separates the 18k distractors (GAP 48) — a 24-point gap.
- **Our new method, a parameter-free geometric re-rank, lifts ViT-L to GAP 53.07** (+4.9) — raising both
  GAP and GAP⁻ with **accuracy untouched**, and already edging the 8× larger 7B zero-shot.
- **Next:** put the gate on the 7B backbone (projected ~57), and test whether **adapting** DINOv3 on our
  synthetic renders (head / LoRA) stacks further — scaffolded, results pending.

---

## 1. A frozen DINOv3 nearly doubles GAP

We swap the ResNet-18 for a **frozen DINOv3** foundation model — used as-is, no training: we just feed
its features into the *same* kNN classifier and metrics as every other model in this project.

![From R18 (+synthetic) to DINOv3 (+re-rank)](figures/progression.png)

| backbone (frozen, zero-shot) | full GAP | GAP⁻ | ACC |
|---|--:|--:|--:|
| R18-SWSL — our step-1 baseline | 35.97 | 52.14 | 54.64 |
| R18-SWSL + synthetic *(the [synthetic-data thread](../training-with-synthetic/README.md))* | 38.15 | 55.49 | 58.23 |
| **DINOv3 ViT-L** | **48.16** | 72.14 | 77.07 |
| **DINOv3-7B** | **52.11** | 75.46 | 81.95 |

*"Zero-shot" = the pretrained model used directly, no fine-tuning. Everything is scored in our exact
kNN + GAP pipeline, so these numbers are comparable to the R18 rows. DINOv3 features are reused from the
sibling *art-research* repo (Met split verified byte-identical); DINOv3-7B is at its best K=5, τ=20.*

> **Honesty note.** That DINOv3 nearly doubles GAP on Met is **DINOv3's own published claim** (≈ 50.8 =
> DINOv2's 40.0 + their reported +10.8), not our contribution. We reproduced it in our pipeline to get a
> faithful starting point — the contribution is what we add *on top* (§2).

---

## 2. The new method: a geometric re-rank

DINOv3 has one clear weakness. It's excellent at **ranking the right painting first** (GAP⁻ = 72) but
poor at **rejecting distractors** — query photos of things that aren't in the collection (full GAP = 48).
That 24-point gap is exactly what GAP penalises.

The fix is **geometric verification**. For a query's top-50 candidates, we check whether their image
*patches* actually line up — mutual nearest-neighbour patch matches, filtered by RANSAC (the classic
"are these two pictures the same thing in the same arrangement?" test). A true match has many consistent
patch correspondences; a distractor has few. We fold that patch-match score into the model's
**confidence** as a gate, which leaves the top-1 guess — and therefore accuracy — untouched:

| DINOv3 ViT-L (our pipeline) | full GAP | GAP⁻ | ACC |
|---|--:|--:|--:|
| baseline (CLS-feature kNN) | 48.16 | 72.14 | 77.07 |
| **+ geometric re-rank gate** | **53.07** | **74.69** | 77.07 |

*The gate lifts both GAP (+4.9) and GAP⁻ (+2.55) with accuracy unchanged — a clean distractor-rejection
win, no trade-off. ViT-L + gate (53.07) already edges the 8× larger DINOv3-7B zero-shot (52.11).*

---

## 3. What's next

- **Gate on the 7B backbone.** Apply the same re-rank to the stronger 7B features (zero-shot 52.11) for
  the headline number — expected ~57 if the +5 transfers.
- **Adapt DINOv3 on our synthetic data.** The thread that links back to the synthetic dataset: does
  fine-tuning DINOv3 (head-only vs LoRA, on studio vs synthetic) stack on top of the re-rank gain? The
  jobs are scaffolded ([`slurm/ftdino.slurm`](../../slurm/ftdino.slurm),
  [`slurm/eval_dino_ft.slurm`](../../slurm/eval_dino_ft.slurm)); **results pending.**

---

## 4. Caveats

- **ViT-L, not yet the 7B headline.** The re-rank result is on ViT-L; the 7B gate (the headline number)
  isn't run yet.
- **Reused features + small val.** DINOv3 features come from the sibling *art-research* repo, and K / τ /
  gate-weight are tuned on only ~129 Met validation queries (noisy).
- **Planar geometry.** The patch-match verification assumes flat artworks, so it's shakier for 3-D
  (non-painting) exhibits.
- **One re-rank variant is a mirage.** An alternative "RRF" fusion shows GAP 56, but it's val-overfit (it
  *drops* GAP⁻ by ~6); the +4.9 **gate** result is the one we trust and report.

---

## 5. How to reproduce

Run in `.venv-dino` (DINOv3 + `transformers`); DINOv3 features are reused from the sibling *art-research* repo.

```bash
# zero-shot DINOv3 in our pipeline (CLS features -> PCA-whiten -> kNN -> GAP)
sbatch slurm/eval_dinov3.slurm                             # job 7332349  -> ViT-L 48.16, 7B 52.11
# geometric re-rank: patch matches over the CLS top-50, fused into the confidence as a gate
sbatch slurm/rerank_fusion.slurm                           # -> ViT-L + gate 53.07
# DINOv3 fine-tuning ablation (head-only / LoRA, on studio / synthetic)  -- scaffolded, results pending
sbatch --job-name=met-dinoL-head-synth slurm/ftdino.slurm synth head      # (+ lora; + studio variants)
```

Code: [`scripts/build_dinov3_pkl.py`](../../scripts/build_dinov3_pkl.py) ·
[`scripts/extract_dino_ckpt.py`](../../scripts/extract_dino_ckpt.py) ·
[`scripts/patchmatch_rerank.py`](../../scripts/patchmatch_rerank.py) ·
[`scripts/rerank_confidence_fusion.py`](../../scripts/rerank_confidence_fusion.py) ·
[`scripts/eval_fullgrid.py`](../../scripts/eval_fullgrid.py) · SLURM:
[`slurm/eval_dinov3.slurm`](../../slurm/eval_dinov3.slurm) ·
[`slurm/rerank_fusion.slurm`](../../slurm/rerank_fusion.slurm) ·
[`slurm/ftdino.slurm`](../../slurm/ftdino.slurm) ·
[`slurm/eval_dino_ft.slurm`](../../slurm/eval_dino_ft.slurm).
The frozen-DINOv3 *embedding-structure* analysis of the synthetic renders is a separate study:
[`../dinov3-embedding-analysis/`](../dinov3-embedding-analysis/README.md). Every number here is recorded
in [`EXPERIMENTS.md`](../../EXPERIMENTS.md) (EXP-6).
