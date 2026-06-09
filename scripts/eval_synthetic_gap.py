"""GAP / GAP- / ACC for the SYNTHETIC painting renders used as queries.

Mirrors the real painting slice (scripts/eval_paintings_cls.py): kNN classifier K=7/tau=50 over the
full 397k studio DB, GAP scored against all 18,316 real test distractors, GAP-/ACC over the renders
only. Painting subset = committed def (Classification=="Paintings" -> data/gt_paint/testset.json,
122 classes); the synthetic renders of those classes = 610 imgs (5 views each). Distractor preds are
identical to eval_paintings_cls.py (same descriptors + classifier), so the synthetic GAP is directly
comparable to the real-painting-query GAP. Reports all-views and per camera view. Reuses
already-extracted descriptors (no re-extraction).

CPU-only; run via SLURM, not the login node.
"""
import os, sys, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO)
import numpy as np
from code.utils.utils import *                      # gap, gap_non_distr, classif_accuracy, PCAw helpers
from code.classifiers.knn_classifier import KNN_Classifier
K, TAU, DIM = 7, 50.0, 512

INFO = os.path.join(REPO, "data/ground_truth")
DB   = os.path.join(REPO, "data/descriptors/r18_contr_loss_gem_fc_swsl_ms/descriptors.pkl")
SYN  = os.path.join(REPO, "data/descriptors/synthetic/synth_descriptors.pkl")

# DB train + real distractors (label -1) from the step-1 descriptors
d = pickle.load(open(DB, "rb"))
train = np.ascontiguousarray(d["train_descriptors"], dtype="float32")
test  = np.ascontiguousarray(d["test_descriptors"],  dtype="float32")
train_labels = np.array([e["id"] for e in json.load(open(os.path.join(INFO, "MET_database.json")))])
test_labels  = np.array([int(e["MET_id"]) if "MET_id" in e else -1 for e in json.load(open(os.path.join(INFO, "testset.json")))])
distr = test[test_labels == -1]; n_distr = len(distr)

# synthetic painting renders (committed Classification=="Paintings" def)
paint_cls = {int(e["MET_id"]) for e in json.load(open(os.path.join(REPO, "data/gt_paint/testset.json"))) if "MET_id" in e}
s = pickle.load(open(SYN, "rb"))
synd = np.ascontiguousarray(s["descriptors"], dtype="float32")
mids = np.asarray(s["met_ids"]); angles = np.asarray(s["angles"])
pmask = np.isin(mids, list(paint_cls))
synp, synp_lab, synp_ang = synd[pmask], mids[pmask].astype(np.int64), angles[pmask]
print(f"synthetic painting renders: {len(synp)} ({len(set(synp_lab.tolist()))} classes) | "
      f"real distractors: {n_distr} | K={K} tau={TAU}", flush=True)

# PCAw learned on train, applied to train + queries (same recipe as the eval)
m, P = estimate_pca_whiten_with_shrinkage(train, shrinkage=1.0, dimensions=DIM)
train = apply_pca_whiten_and_normalize(train, m, P).astype("float32")
synp  = apply_pca_whiten_and_normalize(synp,  m, P).astype("float32")
distr = apply_pca_whiten_and_normalize(distr, m, P).astype("float32")

clf = KNN_Classifier(K=K, t=TAU); clf.fit(train, train_labels)

# predict once over [renders + distractors]; slice per view afterwards (queries are independent)
Q = np.ascontiguousarray(np.vstack([synp, distr]), dtype="float32")
preds, confs = clf.predict(Q); preds = np.array(preds); confs = np.array(confs)
nR = len(synp_lab)
rp, rc = preds[:nR], confs[:nR]                       # render preds / confs
dp, dc = preds[nR:], confs[nR:]                       # distractor preds / confs
dlab = -np.ones(n_distr, dtype=np.int64)

def score(mask, label):
    p = np.concatenate([rp[mask], dp]); c = np.concatenate([rc[mask], dc])
    lab = np.concatenate([synp_lab[mask], dlab])      # renders labeled by source class, distractors -1
    g_all = gap(p, c, lab) * 100                       # GAP: renders + all distractors
    g_no  = gap_non_distr(rp[mask], rc[mask], synp_lab[mask]) * 100   # GAP- : renders only
    acc   = classif_accuracy(rp[mask], synp_lab[mask]) * 100          # ACC  : renders only
    print(f"SYNGAP {label:<12} N={int(mask.sum()):>4} GAP {g_all:6.2f}  GAP- {g_no:6.2f}  ACC {acc:6.2f}", flush=True)
    return {"N": int(mask.sum()), "GAP": round(float(g_all), 2), "GAP-": round(float(g_no), 2), "ACC": round(float(acc), 2)}

out = {"K": K, "tau": TAU, "n_distractors": int(n_distr), "all_views": score(np.ones(nR, bool), "ALL views")}
for a in sorted(set(synp_ang.tolist())):
    out[a] = score(synp_ang == a, a)
json.dump(out, open(os.path.join(REPO, "data/descriptors/synthetic/gap_summary.json"), "w"), indent=2)
print("saved data/descriptors/synthetic/gap_summary.json", flush=True)
