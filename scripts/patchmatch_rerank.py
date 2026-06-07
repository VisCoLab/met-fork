"""Decisive geometric-rerank-for-GAP experiment.

Stage 1 (precomputed, art-research): DINOv3 CLS top-50 candidates per query
(raw cosine sims, cross-query comparable).
Stage 2 (here): C2 PatchMatch geometric score per (query, candidate) on DINOv3
patches (mutual-NN + RANSAC homography over the 14x14 patch grid; proven recipe
copied from art-research's c2_patchmatch_met.py).
Stage 3 (the novel bit): fuse the geometric score into OUR CANONICAL
softmax-temperature confidence (NOT the z-score+argmax the art-research repo used,
which crippled its global baseline to GAP 0.22). lambda=0 recovers the pure-CLS
baseline as a correctness check; lambda>0 tests whether geometry lifts GAP.

Hypothesis: geometry raises true-Met-query confidence above distractors (which
have no geometric match), closing DINOv3's 26-pt GAP- - GAP distractor gap.

Reuses art-research's already-extracted patches + top-50 (byte-identical Met split).
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import cv2
import numpy as np
from code.utils.utils import evaluate  # official GAP / GAP- / ACC


# ---- C2 PatchMatch primitives (verbatim from art-research c2_patchmatch_met.py) ----
def patch_coords(grid, input_size):
    cell = input_size / grid
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    return np.stack([(cell / 2 + cell * xx).flatten(),
                     (cell / 2 + cell * yy).flatten()], axis=1).astype(np.float32)


def mutual_nn(sim):
    q_to_r = sim.argmax(axis=1)
    r_to_q = sim.argmax(axis=0)
    return [(qi, q_to_r[qi]) for qi in range(sim.shape[0]) if r_to_q[q_to_r[qi]] == qi]


def ransac_score(pairs, sim, pc, threshold_px=24.0, max_iters=200):
    if len(pairs) < 4:
        return 0.0
    q_idx = np.array([p[0] for p in pairs]); r_idx = np.array([p[1] for p in pairs])
    sim_vals = sim[q_idx, r_idx]
    _, mask = cv2.findHomography(pc[q_idx], pc[r_idx], method=cv2.RANSAC,
                                 ransacReprojThreshold=threshold_px, maxIters=max_iters)
    if mask is None:
        return 0.0
    mask = mask.ravel().astype(bool); n_in = int(mask.sum())
    return n_in * (float(sim_vals[mask].mean()) if n_in > 0 else 0.0)


def load_query_patches(patches_dir, model, size, split):
    f = os.path.join(patches_dir, f"{model}_size{size}_patches_{split}.npz")
    d = np.load(f)
    return d["features"], d["indices"]  # (Nq,P,D) float16, indices align to split order


def compute_pm(splits, chunks, pc, thr):
    """Streaming PatchMatch: ONE pass over train patch chunks (one resident at a
    time, ~17 GB peak vs 66 GB load-all), inner over all query splits so the 66 GB
    of chunks are read once rather than per-split."""
    for s in splits:
        s["q_pos"] = {int(g): i for i, g in enumerate(s["q_index"])}
        s["pm"] = np.zeros(s["top_idx"].shape, dtype=np.float64)
    for ci, cpath in enumerate(chunks):
        t0 = time.time()
        d = np.load(cpath); cf = d["features"]; cidx = d["indices"]
        cmap = {int(cidx[j]): j for j in range(len(cidx))}
        for s in splits:
            top_idx, q_patches, q_pos, pm = s["top_idx"], s["q_patches"], s["q_pos"], s["pm"]
            Nq, K = top_idx.shape
            for qi in range(Nq):
                ks = [k for k in range(K) if int(top_idx[qi, k]) in cmap]
                if not ks:
                    continue
                qf = q_patches[q_pos.get(qi, qi)].astype(np.float32)
                qf /= np.linalg.norm(qf, axis=1, keepdims=True).clip(min=1e-9)
                for k in ks:
                    tf = cf[cmap[int(top_idx[qi, k])]].astype(np.float32)
                    tf /= np.linalg.norm(tf, axis=1, keepdims=True).clip(min=1e-9)
                    sim = qf @ tf.T
                    pm[qi, k] = ransac_score(mutual_nn(sim), sim, pc, thr)
        print(f"  chunk {ci+1}/{len(chunks)} done ({time.time()-t0:.0f}s)", flush=True)
    return {s["name"]: s["pm"] for s in splits}


# ---- canonical softmax-temperature confidence on fused similarities (mirrors KNN_Classifier) ----
def canonical_gap(top_idx, top_sim, pm_norm, train_labels, gt, n_classes, K, tau, lam):
    fused = top_sim[:, :K] + lam * pm_norm[:, :K]
    labs = train_labels[top_idx[:, :K]]
    Nq = fused.shape[0]
    preds = np.empty(Nq, np.int64); confs = np.empty(Nq, np.float64)
    for q in range(Nq):
        uniq = np.unique(labs[q])
        tot = np.array([fused[q][labs[q] == c].max() for c in uniq])
        e = np.exp(tau * tot)
        e /= (e.sum() + (n_classes - uniq.shape[0]) * 1.0)
        a = e.argmax(); preds[q] = uniq[a]; confs[q] = e[a]
    return evaluate(preds, confs, gt, verbose=False)  # (gap, gap_minus, acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-root", default="/mnt/storage_6/project_data/pl0896-03/art-research")
    ap.add_argument("--model", default="dinov3_vitl16")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--info-dir", default="data/ground_truth")
    ap.add_argument("--out-dir", default="data/rerank")
    ap.add_argument("--ransac-threshold", type=float, default=24.0)
    ap.add_argument("--reuse-pm", action="store_true", help="load saved pm_scores.npz, skip matching")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    top_dir = os.path.join(args.ar_root, f"experiments/met_protocol/top50_{args.model}")
    pat_dir = os.path.join(args.ar_root, f"experiments/met_protocol/patches/{args.model}_size{args.size}")
    grid_n = args.size // 16
    pc = patch_coords(grid_n, args.size)

    print(f"top50 dir: {top_dir}\npatches dir: {pat_dir}", flush=True)
    test_idx = np.load(os.path.join(top_dir, "top50_test_indices.npy"))
    test_sim = np.load(os.path.join(top_dir, "top50_test_sims.npy"))
    val_idx = np.load(os.path.join(top_dir, "top50_val_indices.npy"))
    val_sim = np.load(os.path.join(top_dir, "top50_val_sims.npy"))

    train_labels = np.array([e["id"] for e in json.load(open(os.path.join(args.info_dir, "MET_database.json")))])
    n_classes = int(np.unique(train_labels).shape[0])
    def gt(fn):
        return np.array([int(e["MET_id"]) if "MET_id" in e else -1
                         for e in json.load(open(os.path.join(args.info_dir, fn)))])
    test_gt, val_gt = gt("testset.json"), gt("valset.json")
    print(f"n_classes={n_classes}  test={len(test_gt)} val={len(val_gt)}", flush=True)

    pm_path = os.path.join(args.out_dir, f"pm_scores_{args.model}.npz")
    if args.reuse_pm and os.path.exists(pm_path):
        z = np.load(pm_path); test_pm, val_pm = z["test_pm"], z["val_pm"]
        print(f"reused PM scores from {pm_path}", flush=True)
    else:
        chunks = sorted(glob.glob(os.path.join(pat_dir, f"{args.model}_size{args.size}_patches_train*.npz")))
        assert chunks, f"no train patch chunks in {pat_dir}"
        vqf, vqi = load_query_patches(pat_dir, args.model, args.size, "val")
        tqf, tqi = load_query_patches(pat_dir, args.model, args.size, "test")
        print(f"\n=== PatchMatch (val {len(val_gt)} + test {len(test_gt)} q) over {len(chunks)} train chunks ===", flush=True)
        pms = compute_pm(
            [{"name": "val", "top_idx": val_idx, "q_patches": vqf, "q_index": vqi},
             {"name": "test", "top_idx": test_idx, "q_patches": tqf, "q_index": tqi}],
            chunks, pc, args.ransac_threshold)
        val_pm, test_pm = pms["val"], pms["test"]
        del vqf, tqf
        np.savez(pm_path, test_pm=test_pm, val_pm=val_pm)
        print(f"saved PM scores -> {pm_path}", flush=True)

    # global PM scale from val positives (cross-query comparable normalizer)
    nz = val_pm[val_pm > 0]
    g0 = float(np.percentile(nz, 95)) if nz.size else 1.0
    test_pn, val_pn = test_pm / g0, val_pm / g0
    frac = float((test_pm[:, 0] > 0).mean())
    print(f"\nPM scale g0(val p95)={g0:.2f}; test queries with any top1 geom match: {frac*100:.1f}%", flush=True)

    Ks = [1, 3, 5, 10, 20, 50]; taus = [5, 10, 15, 30, 50, 100]; lams = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    print("\n=== tuning (K, tau, lambda) on val GAP ===", flush=True)
    best = {"gap": -1}
    base = {"gap": -1}  # best lambda=0 config (the pure-CLS baseline in this harness)
    for K in Ks:
        for tau in taus:
            for lam in lams:
                g, gm, a = canonical_gap(val_idx, val_sim, val_pn, train_labels, val_gt, n_classes, K, tau, lam)
                if g > best["gap"]:
                    best = {"gap": g, "K": K, "tau": tau, "lam": lam}
                if lam == 0.0 and g > base["gap"]:
                    base = {"gap": g, "K": K, "tau": tau, "lam": 0.0}
    print(f"best val: {best}\nbaseline(lam=0) val: {base}", flush=True)

    def test_eval(cfg):
        return canonical_gap(test_idx, test_sim, test_pn, train_labels, test_gt, n_classes,
                             cfg["K"], cfg["tau"], cfg["lam"])
    bg, bgm, ba = test_eval(base)
    rg, rgm, ra = test_eval(best)

    print("\n================  TEST (Met protocol, full 19,319 queries)  ================", flush=True)
    print(f"  baseline CLS (lam=0)   K={base['K']:<2} tau={base['tau']:<5} : "
          f"GAP {bg*100:6.2f}  GAP- {bgm*100:6.2f}  ACC {ba*100:6.2f}", flush=True)
    print(f"  + geom rerank          K={best['K']:<2} tau={best['tau']:<5} lam={best['lam']:<4}: "
          f"GAP {rg*100:6.2f}  GAP- {rgm*100:6.2f}  ACC {ra*100:6.2f}", flush=True)
    print(f"  delta                  : GAP {(rg-bg)*100:+6.2f}  GAP- {(rgm-bgm)*100:+6.2f}  ACC {(ra-ba)*100:+6.2f}", flush=True)

    json.dump({"model": args.model, "baseline": {**base, "test_gap": bg, "test_gap_minus": bgm, "test_acc": ba},
               "rerank": {**best, "test_gap": rg, "test_gap_minus": rgm, "test_acc": ra}, "g0": g0},
              open(os.path.join(args.out_dir, f"rerank_eval_{args.model}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
