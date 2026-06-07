"""Assemble raw DINOv3 CLS features (extracted by the art-research repo over the
IDENTICAL Met split — JSONs verified byte-identical) into a descriptors.pkl in
THIS repo's format, so our own eval_fullgrid.py (our PCAw + kNN + GAP) can
reproduce the DINOv3 Met number. Reuses the already-extracted 7B features rather
than re-running the 7B model.

Output: <out_dir>/descriptors.pkl  {train_descriptors, val_descriptors,
test_descriptors}  (float16; eval_fullgrid upcasts to float32 + applies PCAw).
"""
import argparse
import os
import pickle
import sys

import numpy as np

# expected split sizes for the full Met benchmark (must match MET_database/testset/valset)
N = {"train": 397121, "val": 2165, "test": 19319}


def load_split(fdir, model, variant, split):
    stem = f"{model}_{variant}_{split}"
    single = os.path.join(fdir, f"{stem}.npz")
    if os.path.isfile(single):
        files = [single]
    else:
        import glob
        files = sorted(glob.glob(os.path.join(fdir, f"{stem}_chunk*of*.npz")))
    assert files, f"no npz for {stem} in {fdir}"

    out = None
    seen = np.zeros(N[split], dtype=bool)
    for f in files:
        z = np.load(f)
        feats, idxs = z["features"], z["indices"]
        if out is None:
            out = np.empty((N[split], feats.shape[1]), dtype=feats.dtype)
        out[idxs] = feats
        seen[idxs] = True
    assert seen.all(), f"{split}: {(~seen).sum()} of {N[split]} rows missing"
    print(f"  {split}: {out.shape} {out.dtype}  ({len(files)} file(s))", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True, help="art-research met_protocol/features/<variant> dir")
    ap.add_argument("--model", default="dinov3_vit7b16")
    ap.add_argument("--variant", default="aspect512")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    print(f"assembling {args.model}_{args.variant} from {args.features_dir}", flush=True)
    d = {}
    for split in ("train", "val", "test"):
        d[f"{split}_descriptors"] = load_split(args.features_dir, args.model, args.variant, split)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "descriptors.pkl")
    with open(out, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {out}  "
          f"train {d['train_descriptors'].shape} val {d['val_descriptors'].shape} "
          f"test {d['test_descriptors'].shape}", flush=True)


if __name__ == "__main__":
    main()
