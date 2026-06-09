#!/usr/bin/env python
"""Figures for the 'training with synthetic data' write-up (core plan, steps 1-4).

All numbers are the val-tuned test results recorded in EXPERIMENTS.md (EXP-1 / EXP-4):
  - Step 1 baseline (no synthetic) ........ eval job 7313742-derived (slurm/extract_eval.slurm)
  - Combined FT (studio + synthetic) ...... eval job 7332888
  - Synth-only FT (synthetic only) ........ eval job 7330036
  - From-scratch +synth (clean A/B) ....... eval job 7342026
Full GAP / GAP-/ ACC are at each model's own best (K, tau); the paint columns are at K=7, tau=50.

Run (NOT on a login node):
  srun --account=pl0896-03 --partition=standard --time=0:10:00 --mem=4G \
      .venv-dino/bin/python scripts/plot_synthetic_training.py
Writes -> experiments/training-with-synthetic/figures/{gap_by_config.png, baseline_vs_synth.png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "experiments/training-with-synthetic/figures"
os.makedirs(OUT, exist_ok=True)

# ----- palette -----
GREY   = "#b3b3b3"   # baselines (paper, step 1)
CONF   = "#9ecae1"   # synthetic, but confounded (LR re-warm + extra epochs)
CLEAN  = "#2a9d8f"   # synthetic, clean A/B  -> the headline
PAPER  = "#6b6b6b"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

def annotate(ax, bars, fmt="{:.2f}", dy=0.15, fs=10, weight="normal"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=fs, fontweight=weight)

# =====================================================================
# Figure 1 — full-benchmark GAP across the four training configurations
# =====================================================================
labels = ["Paper\nbest", "Step 1\n(no synth)", "Combined\nFT", "Synth-only\nFT", "From-scratch\n+synth"]
gap    = [36.1,           35.97,               37.38,          38.61,           38.15]
colors = [GREY,           GREY,                CONF,           CONF,            CLEAN]
hatch  = [None,           None,                "//",           "//",            None]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
bars = ax.bar(labels, gap, color=colors, edgecolor="white", width=0.66, zorder=3)
for b, h in zip(bars, hatch):
    if h:
        b.set_hatch(h)
        b.set_edgecolor("#5a8bb0")
annotate(ax, bars, dy=0.08, weight="bold")

ax.axhline(36.1, ls="--", lw=1.2, color=PAPER, zorder=2)
ax.text(4.45, 36.1 + 0.06, "paper best 36.1", color=PAPER, fontsize=9, ha="right", va="bottom")

ax.set_ylim(34, 39.6)
ax.set_ylabel("Full-benchmark GAP")
ax.set_title("Adding synthetic gallery data raises GAP — clean A/B beats the paper")
ax.grid(axis="y", color="#e6e6e6", zorder=0)
ax.set_axisbelow(True)

legend = [Patch(fc=GREY, label="baseline (no synthetic)"),
          Patch(fc=CONF, hatch="//", ec="#5a8bb0", label="with synthetic — confounded*"),
          Patch(fc=CLEAN, label="with synthetic — clean A/B (headline)")]
ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=9)
fig.text(0.012, -0.02,
         "*confounded = fine-tunes that also re-warmed the LR and trained 5 extra epochs.",
         fontsize=8, color="#666666")
fig.savefig(f"{OUT}/gap_by_config.png")
plt.close(fig)

# =====================================================================
# Figure 2 — baseline vs the clean from-scratch +synth, every metric
# =====================================================================
metrics = ["Full GAP", "GAP⁻\n(no distr.)", "Accuracy", "Paint GAP⁻\n(n=148)", "Paint ACC\n(n=148)"]
base    = [35.97, 52.14, 54.64, 67.86, 69.59]
synth   = [38.15, 55.49, 58.23, 70.41, 72.30]
x = range(len(metrics))
w = 0.38

fig, ax = plt.subplots(figsize=(8.4, 4.6))
b1 = ax.bar([i - w/2 for i in x], base,  w, color=GREY,  label="Step 1 — no synthetic", zorder=3)
b2 = ax.bar([i + w/2 for i in x], synth, w, color=CLEAN, label="From-scratch +synth (clean A/B)", zorder=3)
# only the lift is annotated (exact values live in the doc's table) -> keeps the figure clean
for i, (lo, hi) in enumerate(zip(base, synth)):
    ax.text(i + w/2, hi + 1.0, f"+{hi-lo:.2f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=CLEAN)

ax.set_xticks(list(x))
ax.set_xticklabels(metrics)
ax.set_ylim(0, 86)
ax.set_ylabel("score")
ax.set_title("Same recipe + synthetic data only: every metric improves")
ax.grid(axis="y", color="#e6e6e6", zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=9.5,
          bbox_to_anchor=(0.0, 0.88))
fig.savefig(f"{OUT}/baseline_vs_synth.png")
plt.close(fig)

# =====================================================================
# Figure 3 — the full progression: R18(+synthetic)  ->  DINOv3(+re-rank)
# DINOv3 numbers are from EXPERIMENTS.md EXP-6 (frozen backbones eval'd in OUR
# kNN-softmax pipeline; DINOv3-7B at its best K=5, τ=20). grey = off-the-shelf /
# reproduced reference; teal = what we added on top (synthetic data; geometric re-rank).
# =====================================================================
labels3 = ["Paper\nbest", "R18\nbaseline", "R18\n+synth",
           "DINOv3-L\nzero-shot", "DINOv3-7B\nzero-shot", "DINOv3-L\n+re-rank"]
gap3 = [36.1, 35.97, 38.15, 48.16, 52.11, 53.07]
ours = [False, False, True, False, False, True]   # teal where we contributed

fig, ax = plt.subplots(figsize=(8.8, 4.7))
bars = ax.bar(labels3, gap3, color=[CLEAN if o else GREY for o in ours],
              edgecolor="white", width=0.70, zorder=3)
annotate(ax, bars, dy=0.5, fs=10, weight="bold")

ax.axvline(2.5, color="#d6d6d6", lw=1.0, zorder=1)               # R18 | DINOv3 divider
ax.text(1.0, 58.0, "ResNet-18", ha="center", fontsize=10, color="#777", fontweight="bold")
ax.text(4.0, 58.0, "DINOv3 (frozen)", ha="center", fontsize=10, color="#777", fontweight="bold")

ax.set_ylim(0, 62)
ax.set_ylabel("Full-benchmark GAP")
ax.set_title("The backbone is the bigger lever — our re-rank tops it off")
ax.grid(axis="y", color="#e6e6e6", zorder=0)
ax.set_axisbelow(True)
legend = [Patch(fc=GREY, label="off-the-shelf / reproduced reference"),
          Patch(fc=CLEAN, label="our contribution (synthetic data · geometric re-rank)")]
ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=9.5,
          bbox_to_anchor=(0.0, 0.80))
fig.savefig(f"{OUT}/progression.png")
plt.close(fig)

print("wrote:")
for f in ("gap_by_config.png", "baseline_vs_synth.png", "progression.png"):
    print(" ", os.path.join(OUT, f))
