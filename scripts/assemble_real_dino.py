"""Assemble the 'real' DINOv3 ViT-L reference clouds for the synthetic-vs-real
analysis, REUSING art-research's already-extracted dinov3_vitl16_aspect512 features
(identical model + aspect512 preprocessing as scripts/extract_synth_dino.py).

Two real clouds:
  studio  : the catalog source image (MET/<id>/0.jpg) for each synthetic class
            -> the exact studio photos the renders were generated from (paired).
  query   : real visitor photos of PAINTINGS from the Met test set (the true target
            domain). CANONICAL painting set = `Classification == "Paintings"` = the 148
            committed query paths in data/gt_paint/testset.json -- the single painting
            definition used project-wide (see scripts/eval_paintings.py).

Output (--out-dir): real_dino_vitl16_aspect512.npz
  studio_feats(C,1024 f16), studio_met_id(C,)
  pq_feats(Q,1024 f16), pq_met_id(Q,)                       # 148 Classification==Paintings queries
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/
from synth_meta import SYNTH_ROOT, parse_metadata

AR_FEAT = ("/mnt/storage_6/project_data/pl0896-03/art-research/"
           "experiments/met_protocol/features/dinov3_vitl16_aspect512")
MODEL, VARIANT = "dinov3_vitl16", "aspect512"
N = {"train": 397121, "val": 2165, "test": 19319}


def chunk_files(split):
    stem = f"{MODEL}_{VARIANT}_{split}"
    single = os.path.join(AR_FEAT, f"{stem}.npz")
    if os.path.isfile(single):
        return [single]
    return sorted(glob.glob(os.path.join(AR_FEAT, f"{stem}_chunk*of*.npz")))


def load_split_full(split):
    """Reassemble a full (N[split], 1024) feature matrix indexed by split position."""
    out = None
    seen = np.zeros(N[split], dtype=bool)
    for f in chunk_files(split):
        z = np.load(f)
        feats, idxs = z["features"], z["indices"]
        if out is None:
            out = np.empty((N[split], feats.shape[1]), dtype=feats.dtype)
        out[idxs] = feats
        seen[idxs] = True
    assert out is not None and seen.all(), f"{split}: missing rows"
    return out


def synth_class_met_ids(synth_root):
    """Per-class met_id in folder order (one row per synthetic class)."""
    folders = sorted((d for d in os.listdir(synth_root)
                      if d.isdigit() and os.path.isdir(os.path.join(synth_root, d))), key=int)
    return [parse_metadata(os.path.join(synth_root, fl))["met_id"] for fl in folders]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info-dir", default="data/ground_truth")
    ap.add_argument("--paint-testset", default="data/gt_paint/testset.json",
                    help="committed Classification==Paintings query set (148)")
    ap.add_argument("--synth-root", default=SYNTH_ROOT)
    ap.add_argument("--out-dir", default="data/synth_dino")
    args = ap.parse_args()

    db = json.load(open(os.path.join(args.info_dir, "MET_database.json")))
    path_to_idx = {e["path"]: i for i, e in enumerate(db)}

    # ---- studio sources (paired with synthetic classes) ----
    met_ids = synth_class_met_ids(args.synth_root)
    studio_idx, studio_met_id, missing = [], [], 0
    for mid in met_ids:
        p = f"MET/{mid}/0.jpg"
        if p in path_to_idx:
            studio_idx.append(path_to_idx[p])
            studio_met_id.append(mid)
        else:
            missing += 1
    print(f"studio classes: {len(studio_idx)} (missing 0.jpg in DB: {missing})", flush=True)

    print("loading train features (for studio rows)...", flush=True)
    train_feats = load_split_full("train")
    studio_feats = train_feats[np.array(studio_idx)]
    del train_feats

    # ---- real painting test queries: committed Classification=="Paintings" (148) ----
    ts = json.load(open(os.path.join(args.info_dir, "testset.json")))
    test_met_id = np.array([int(e["MET_id"]) if "MET_id" in e else -1 for e in ts])
    paint_paths = {e["path"] for e in json.load(open(args.paint_testset))}   # 148 committed
    is_paint = np.array([e["path"] in paint_paths for e in ts])
    print(f"painting test queries (Classification==Paintings): {int(is_paint.sum())}", flush=True)

    print("loading test features...", flush=True)
    test_feats = load_split_full("test")
    qidx = np.where(is_paint)[0]
    pq_feats = test_feats[qidx]
    pq_met_id = test_met_id[qidx]

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "real_dino_vitl16_aspect512.npz")
    np.savez(out,
             studio_feats=studio_feats.astype(np.float16),
             studio_met_id=np.array(studio_met_id, dtype=np.int64),
             pq_feats=pq_feats.astype(np.float16),
             pq_met_id=pq_met_id.astype(np.int64))
    print(f"wrote {out}\n  studio {studio_feats.shape} | painting-queries {pq_feats.shape}", flush=True)


if __name__ == "__main__":
    main()
