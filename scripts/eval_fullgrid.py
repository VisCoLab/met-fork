"""Faithful knn_eval reproduction but tuning the FULL K grid (the released
knn_eval.py defaults --k 1, so it only tunes tau). Usage:
  .venv/bin/python data/eval_fullgrid.py [descr_dir] [info_dir] [pcaw_dim]
"""
import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
from code.utils.utils import *                       # PCAw helpers, evaluate, np
from code.classifiers.knn_classifier import *        # KNN_Classifier, tune_KNN

DDIR = sys.argv[1] if len(sys.argv) > 1 else "data/authors/descriptors"
INFO = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth"
DIM  = int(sys.argv[3]) if len(sys.argv) > 3 else 512

with open(os.path.join(DDIR, "descriptors.pkl"), "rb") as f:
    d = pickle.load(f)
train = np.ascontiguousarray(d["train_descriptors"], dtype="float32")
test  = np.ascontiguousarray(d["test_descriptors"],  dtype="float32")
val   = np.ascontiguousarray(d["val_descriptors"],   dtype="float32")
print(f"descriptors: train {train.shape} test {test.shape} val {val.shape}", flush=True)

train_labels = np.array([e["id"] for e in json.load(open(os.path.join(INFO, "MET_database.json")))])
def qlab(fn):
    return np.array([int(e["MET_id"]) if "MET_id" in e else -1
                     for e in json.load(open(os.path.join(INFO, fn)))])
test_labels, val_labels = qlab("testset.json"), qlab("valset.json")

print(f"PCAw -> {DIM} ...", flush=True)
m, P = estimate_pca_whiten_with_shrinkage(train, shrinkage=1.0, dimensions=DIM)
train = apply_pca_whiten_and_normalize(train, m, P).astype("float32")
val   = apply_pca_whiten_and_normalize(val,   m, P).astype("float32")
test  = apply_pca_whiten_and_normalize(test,  m, P).astype("float32")

grid = {'K': np.array([1, 2, 3, 5, 7, 10, 15, 20, 50]),
        't': np.array([0.01, 0.1, 1., 5., 10., 15., 20., 25., 30., 50., 100., 500.])}
print("tuning FULL K x tau grid on val ...", flush=True)
best_score, best = tune_KNN(grid, train, train_labels, val, val_labels, verbose=False)
print("BEST by val GAP:", best, "| val GAP =", round(float(best_score) * 100, 2), flush=True)

clf = KNN_Classifier(K=int(best['K']), t=float(best['t']))
clf.fit(train, train_labels)
preds, confs = clf.predict(test)
gap, gapnd, acc = evaluate(np.array(preds), np.array(confs), test_labels, verbose=False)
print(f"\n=== TEST — {DDIR} ===", flush=True)
print(f"GAP   {gap*100:6.2f}   (paper 36.1)", flush=True)
print(f"GAP-  {gapnd*100:6.2f}   (paper 52.4)", flush=True)
print(f"ACC   {acc*100:6.2f}   (paper 55.0)", flush=True)
