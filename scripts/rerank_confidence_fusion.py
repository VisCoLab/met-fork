"""Geometric re-rank, take 2: fuse the C2 PatchMatch score into the CONFIDENCE
(GAP's global ranking signal), NOT into the pre-softmax similarity.

Diagnosis from take 1: DINOv3's softmax confidence SATURATES (GAP- 72 >> GAP 48),
so true Met queries and distractors are interleaved at conf~=1. The geometric
maxPM separates them (true mean 34 vs distractor 21). So we keep the CLS
prediction (ACC unchanged) and use geometry to RE-ORDER the confidence:

  per query: c* = CLS argmax (per-class-max sim), conf_cls = softmax-temp conf,
             g* = max PatchMatch score among top-K candidates of class c*.
  fused confidence -> evaluate GAP/GAP-/ACC (ACC fixed; GAP/GAP- depend on order).

Schemes (tuned on val GAP): none(=baseline), add, gate, rrf (reciprocal-rank
fusion), pureg. Operates on the SAVED PM scores -> no re-matching, seconds to run.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from code.utils.utils import evaluate


def rank_desc(x):
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r  # 0 == highest


def cls_predict(top_idx, top_sim, pm, labels, K, tau, n_classes):
    """Canonical per-class-max softmax-temp confidence (mirrors KNN_Classifier),
    plus geometric support g* of the predicted class."""
    fv = top_sim[:, :K]; lb = labels[top_idx[:, :K]]; pmk = pm[:, :K]
    Nq = fv.shape[0]
    pred = np.empty(Nq, np.int64); conf = np.empty(Nq); gstar = np.empty(Nq)
    for q in range(Nq):
        row, sims, pms = lb[q], fv[q], pmk[q]
        uniq = np.unique(row)
        cmax = np.array([sims[row == c].max() for c in uniq])
        e = np.exp(tau * cmax); e /= (e.sum() + (n_classes - uniq.shape[0]))
        a = int(e.argmax()); c_star = uniq[a]
        pred[q] = c_star; conf[q] = e[a]
        gstar[q] = pms[row == c_star].max()
    return pred, conf, gstar


def fuse(conf, gstar, scheme, w, g0):
    if scheme == "none":
        return conf
    gn = gstar / g0
    if scheme == "add":
        return conf + w * gn
    if scheme == "gate":
        return conf * (1.0 + w * gn)
    if scheme == "rrf":
        k0 = 60.0
        return 1.0 / (k0 + rank_desc(conf)) + w / (k0 + rank_desc(gstar))
    if scheme == "pureg":
        return gn
    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-root", default="/mnt/storage_6/project_data/pl0896-03/art-research")
    ap.add_argument("--model", default="dinov3_vitl16")
    ap.add_argument("--pm", default="data/rerank/pm_scores_dinov3_vitl16.npz")
    ap.add_argument("--info-dir", default="data/ground_truth")
    ap.add_argument("--out", default="data/rerank/conffusion_dinov3_vitl16.json")
    args = ap.parse_args()

    top_dir = os.path.join(args.ar_root, f"experiments/met_protocol/top50_{args.model}")
    test_idx = np.load(os.path.join(top_dir, "top50_test_indices.npy"))
    test_sim = np.load(os.path.join(top_dir, "top50_test_sims.npy"))
    val_idx = np.load(os.path.join(top_dir, "top50_val_indices.npy"))
    val_sim = np.load(os.path.join(top_dir, "top50_val_sims.npy"))
    z = np.load(args.pm); test_pm, val_pm = z["test_pm"], z["val_pm"]

    labels = np.array([e["id"] for e in json.load(open(os.path.join(args.info_dir, "MET_database.json")))])
    n_classes = int(np.unique(labels).shape[0])
    def gt(fn):
        return np.array([int(e["MET_id"]) if "MET_id" in e else -1
                         for e in json.load(open(os.path.join(args.info_dir, fn)))])
    test_gt, val_gt = gt("testset.json"), gt("valset.json")

    nz = val_pm[val_pm > 0]; g0 = float(np.percentile(nz, 95)) if nz.size else 1.0
    Ks = [1, 3, 5, 10, 20, 50]; taus = [10, 15, 20, 30, 50]
    schemes = [("none", 0.0)] + [(s, w) for w in (0.25, 0.5, 1, 2, 4, 8) for s in ("add", "gate", "rrf")] + [("pureg", 1.0)]

    print(f"g0={g0:.2f}  n_classes={n_classes}  grid: {len(Ks)}K x {len(taus)}tau x {len(schemes)} schemes", flush=True)

    # cache per-(K,tau) val predictions; tune all schemes on val GAP
    best = {"val_gap": -1}; base = {"val_gap": -1}
    val_cache = {}
    for K in Ks:
        for tau in taus:
            pv, cv, gv = cls_predict(val_idx, val_sim, val_pm, labels, K, tau, n_classes)
            val_cache[(K, tau)] = (pv, cv, gv)
            for scheme, w in schemes:
                g, _, _ = evaluate(pv, fuse(cv, gv, scheme, w, g0), val_gt, verbose=False)
                rec = {"K": K, "tau": tau, "scheme": scheme, "w": w, "val_gap": g}
                if g > best["val_gap"]:
                    best = rec
                if scheme == "none" and g > base["val_gap"]:
                    base = rec
    print(f"best val: {best}", flush=True)
    print(f"baseline(none) val: {base}", flush=True)

    def test_eval(cfg):
        pt, ct, gtt = cls_predict(test_idx, test_sim, test_pm, labels, cfg["K"], cfg["tau"], n_classes)
        return evaluate(pt, fuse(ct, gtt, cfg["scheme"], cfg["w"], g0), test_gt, verbose=False)

    bg, bgm, ba = test_eval(base)
    rg, rgm, ra = test_eval(best)
    print("\n================  TEST (Met protocol, 19,319 queries)  ================", flush=True)
    print(f"  baseline CLS         K={base['K']:<2} tau={base['tau']:<3}                    : "
          f"GAP {bg*100:6.2f}  GAP- {bgm*100:6.2f}  ACC {ba*100:6.2f}", flush=True)
    print(f"  + geom conf-fusion   K={best['K']:<2} tau={best['tau']:<3} {best['scheme']}(w={best['w']}) : "
          f"GAP {rg*100:6.2f}  GAP- {rgm*100:6.2f}  ACC {ra*100:6.2f}", flush=True)
    print(f"  delta                                              : "
          f"GAP {(rg-bg)*100:+6.2f}  GAP- {(rgm-bgm)*100:+6.2f}  ACC {(ra-ba)*100:+6.2f}", flush=True)

    json.dump({"model": args.model, "g0": g0,
               "baseline": {**base, "test_gap": bg, "test_gap_minus": bgm, "test_acc": ba},
               "rerank": {**best, "test_gap": rg, "test_gap_minus": rgm, "test_acc": ra}},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
