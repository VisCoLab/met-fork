"""Figures for the phone-photo augmentation report (experiments/phone-photo-augmentation/, EXP-9).

Writes (each only when its inputs are present, so this can run mid-experiment):
  fig_examples.png -- illustrative montage: one render + each phone artifact applied at a
                      deliberately visible strength (no logs needed; always written).
  fig_arms.png     -- closed paint-DB GAP- & ACC per augmentation arm vs the synthall baseline,
                      with a shaded +/-2-pt "148-photo noise floor" band. Needs met-ev-aug-*.out
                      (+ the baseline met-ev-synthall-*.out).
  fig_confirm.png  -- full-benchmark confirmation (full-DB painting GAP- + whole-benchmark
                      GAP/GAP-/ACC) for the baseline + arms with a met-full-aug-*.out log.

Run: .venv-dino/bin/python scripts/plot_phone_aug.py
"""
import os, re, glob
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-" + os.environ.get("USER", "x"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
LOGS = os.path.join(REPO, "logs"); DOCS = os.path.join(REPO, "experiments/phone-photo-augmentation/figures")
ARMS = ["base", "jpeg", "blur", "sensor", "phoneall"]
LABELS = {"base": "base\n(no phone aug)", "jpeg": "+JPEG", "blur": "+blur",
          "sensor": "+sensor\n(noise+res)", "phoneall": "+all three"}
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#888888"
NOISE = 2.0   # the ~2-pt 148-photo noise floor the README warns about


def last(pat):
    fs = sorted(glob.glob(os.path.join(LOGS, pat))); return fs[-1] if fs else None


# ---- parse closed-world 2-fold-CV GAP- / ACC ------------------------------------------------
def closed(arm):
    f = last("met-ev-synthall-*.out" if arm == "base" else f"met-ev-aug-{arm}-*.out")
    if not f: return (None, None)
    m = re.search(r"2-fold mean:\s*GAP-\s*([\d.]+)\s*ACC\s*([\d.]+)", open(f).read())
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


# ---- parse full-benchmark eval (eval_fullgrid + eval_paintings) -----------------------------
def full(arm):
    f = last("met-full-synthall-*.out" if arm == "base" else f"met-full-aug-{arm}-*.out")
    if not f: return None
    t = open(f).read()
    g  = re.search(r"^GAP\s+([\d.]+)", t, re.M)
    gn = re.search(r"^GAP-\s+([\d.]+)", t, re.M)
    ac = re.search(r"^ACC\s+([\d.]+)", t, re.M)
    pn = re.search(r"PAINT148\s+GAP\(\+all distr\)\s+[\d.]+\s+GAP-\s+([\d.]+)\s+ACC\s+([\d.]+)", t)
    if not (g and gn and ac): return None
    return dict(gap=float(g.group(1)), gnd=float(gn.group(1)), acc=float(ac.group(1)),
                paint_gnd=float(pn.group(1)) if pn else None,
                paint_acc=float(pn.group(2)) if pn else None)


# ============================================================================================
# Fig: illustrative augmentation montage (no logs needed)
# ============================================================================================
def fig_examples():
    """Per-method strength SWEEP: rows = the 5 phone transforms, columns = clean reference then
    the sampled strength from the range's min (subtle) to its max (strongest). Every value shown
    is exactly what training draws (uniformly, each at p=0.5) -- nothing exaggerated."""
    import torch
    from PIL import Image, ImageFilter
    from torchvision import transforms as T
    from code.utils.augmentations import JPEGCompress, RandomDownscale, MotionBlur
    cands = sorted(glob.glob(os.path.join(REPO, "data/aug/images/SYNTH/0/*front*.png"))) \
        or sorted(glob.glob(os.path.join(REPO, "data/aug/images/SYNTH/0/*.png"))) \
        or sorted(glob.glob(os.path.join(REPO, "data/aug/images/SYNTH/*/*.png")))
    if not cands:
        print("fig_examples.png SKIPPED -- no SYNTH render found"); return
    # center-crop ~62% so the (wall-mounted) painting fills the frame -> artifacts are legible
    full = Image.open(cands[0]).convert("RGB"); w, h = full.size; f = 0.62
    cw, ch = int(w * f), int(h * f); l, t0 = (w - cw) // 2, (h - ch) // 2
    img = full.crop((l, t0, l + cw, t0 + ch)).resize((256, 256), Image.BICUBIC)

    def jpeg(q):  return JPEGCompress(q, q)(img)
    def gblur(s): return T.GaussianBlur(5, (s, s))(img)
    mb = MotionBlur()
    def mblur(i): return img.filter(ImageFilter.Kernel((5, 5), mb._KERNELS[i], scale=1.0))
    def down(fc): return RandomDownscale(fc, fc)(img)
    def noise(s):
        torch.manual_seed(0)                          # fixed draw -> a representative grain pattern
        x = T.ToTensor()(img); x = (x + torch.randn_like(x) * s).clamp_(0, 1)
        return T.ToPILImage()(x)

    REF = ("original", img)                            # clean reference, column 1 of every row
    rows = [                                           # (row label, [(cell label, image), ...x5])
        ("JPEG\nquality 30-90",    [REF] + [(f"q={q}",     jpeg(q))  for q in (90, 70, 50, 30)]),
        ("Gaussian blur\nσ 0.1-2.0", [REF] + [(f"σ={s:.2g}", gblur(s)) for s in (0.1, 0.73, 1.37, 2.0)]),
        ("Motion blur\n5×5, random angle", [REF] + [(f"{a}°", mblur(i)) for i, a in enumerate((0, 45, 90, 135))]),
        ("Downscale\nfactor 0.3-0.7", [REF] + [(f"{fc:.2g}×", down(fc)) for fc in (0.7, 0.57, 0.43, 0.3)]),
        ("Sensor noise\nσ 0.01-0.06", [REF] + [(f"σ={s:.3g}", noise(s)) for s in (0.01, 0.027, 0.043, 0.06)]),
    ]
    nr, nc = len(rows), 5
    fig, axes = plt.subplots(nr, nc, figsize=(2.25 * nc, 2.45 * nr))
    for r, (rowlab, cells) in enumerate(rows):
        for c, (cellab, im) in enumerate(cells):
            ax = axes[r][c]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(cellab, fontsize=8.5, pad=2)
            if c == 0:
                ax.set_ylabel(rowlab, fontsize=9.5)
                for s in ax.spines.values(): s.set_color("#1f77b4"); s.set_linewidth(1.6)
    fig.suptitle("Phone-photo augmentation — sampled strength range per method\n"
                 "col 1 = clean render;  cols 2-5 = range min → max", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(DOCS, "fig_examples.png"), dpi=150); plt.close(fig)
    print("wrote fig_examples.png  (render:", os.path.relpath(cands[0], REPO), ")")


# ============================================================================================
# Fig: closed-world GAP- & ACC per arm
# ============================================================================================
def fig_arms():
    cg = {a: closed(a) for a in ARMS}
    if cg["base"][0] is None or all(cg[a][0] is None for a in ARMS if a != "base"):
        print("fig_arms.png SKIPPED -- closed-world eval logs not ready "
              f"(have: {[a for a in ARMS if cg[a][0] is not None]})"); return
    base_g, base_a = cg["base"]
    xs = list(range(len(ARMS))); w = 0.38
    gnd = [cg[a][0] for a in ARMS]; acc = [cg[a][1] for a in ARMS]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.axhspan(base_g - NOISE, base_g + NOISE, color=GREY, alpha=.13, zorder=0,
               label=f"±{NOISE:.0f}-pt noise floor (148 photos)")
    ax.axhline(base_g, color=BLUE, ls=":", lw=1.2, alpha=.7)
    b1 = ax.bar([x - w/2 for x in xs], gnd, w, color=BLUE,  label="GAP- (closed paint DB)")
    b2 = ax.bar([x + w/2 for x in xs], acc, w, color=GREEN, label="ACC (closed paint DB)")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            if h is not None:
                ax.text(r.get_x() + r.get_width()/2, h + 0.15, f"{h:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels([LABELS[a] for a in ARMS])
    lo = min(v for v in gnd + acc if v is not None) - 2
    hi = max(v for v in gnd + acc if v is not None) + 2
    ax.set_ylim(lo, hi)
    ax.set_ylabel("score on 148 real painting photos (%)")
    ax.set_title("Phone-photo augmentation on the synth-only painting recognizer\n"
                 "(all 24,490 renders, 0% real; closed painting world)")
    ax.grid(axis="y", alpha=.3); ax.legend(loc="lower right", framealpha=.95, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "fig_arms.png"), dpi=150); plt.close(fig)
    print("wrote fig_arms.png | closed GAP-:", {a: cg[a][0] for a in ARMS})


# ============================================================================================
# Fig: full-benchmark confirmation
# ============================================================================================
def fig_confirm():
    fm = {a: full(a) for a in ARMS}
    have = [a for a in ARMS if fm[a] is not None]
    if "base" not in have or len(have) < 2:
        print(f"fig_confirm.png SKIPPED -- full-benchmark logs not ready (have: {have})"); return
    arms = [a for a in ARMS if a in have]
    metrics = [("paint_gnd", "full-DB painting GAP-", RED),
               ("gnd", "whole-benchmark GAP-", BLUE),
               ("gap", "whole-benchmark GAP", GREEN)]
    xs = list(range(len(arms))); n = len(metrics); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for j, (key, lab, col) in enumerate(metrics):
        vals = [fm[a].get(key) for a in arms]
        off = (j - (n - 1) / 2) * w
        bars = ax.bar([x + off for x in xs], [v if v is not None else 0 for v in vals], w, color=col, label=lab)
        for r, v in zip(bars, vals):
            if v is not None:
                ax.text(r.get_x() + r.get_width()/2, v + 0.2, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([LABELS[a] for a in arms])
    ax.set_ylabel("score (%)"); ax.set_ylim(0, max(fm[a]["paint_gnd"] or 0 for a in arms) + 8)
    ax.set_title("Full-benchmark confirmation (original 397k studio DB)\nbaseline vs phone-augmented")
    ax.grid(axis="y", alpha=.3); ax.legend(loc="upper right", framealpha=.95, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "fig_confirm.png"), dpi=150); plt.close(fig)
    print("wrote fig_confirm.png | full:", {a: fm[a] for a in arms})


if __name__ == "__main__":
    import sys; sys.path.insert(0, REPO)
    fig_examples()
    fig_arms()
    fig_confirm()
