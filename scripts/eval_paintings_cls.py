"""Full-benchmark painting slice under the COMMITTED definition
(Classification=="Paintings" = the 148 query paths in data/gt_paint/testset.json).

Re-scores already-extracted FULL descriptors (no re-extraction): full 397k studio DB index,
K=7 / tau=50 (the EXP-2 painting protocol), then GAP(+all distractors) / GAP- / ACC on the
148 painting queries. Loops over the 6 mix models + the step-1 full-data reference, all under
the same query set -> directly comparable to the closed-world 148-query numbers.

Compute (full-DB kNN) -> run via SLURM, not the login node:  sbatch slurm/eval_paint_cls.slurm
"""
import os, sys, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO)
import numpy as np
from code.utils.utils import *                       # PCAw helpers, gap, gap_non_distr, classif_accuracy
from code.classifiers.knn_classifier import KNN_Classifier
K, TAU, DIM = 7, 50.0, 512

INFO = os.path.join(REPO, "data/ground_truth")
paint_paths = {e["path"] for e in json.load(open(os.path.join(REPO, "data/gt_paint/testset.json")))}  # 148
test_meta = json.load(open(os.path.join(INFO, "testset.json")))
test_labels = np.array([int(e["MET_id"]) if "MET_id" in e else -1 for e in test_meta])
paint_mask = np.array([e["path"] in paint_paths for e in test_meta])
distr_mask = test_labels == -1
train_labels = np.array([e["id"] for e in json.load(open(os.path.join(INFO, "MET_database.json")))])

RUNS = [(0, "data/descriptors_full_0s"), (20, "data/descriptors_full_20s"), (40, "data/descriptors_full_40s"),
        (60, "data/descriptors_full_60s"), (80, "data/descriptors_full_80s"), (100, "data/descriptors_full_100s"),
        ("ref", "data/descriptors"),                 # step-1 full-data model (EXP-1)
        ("synth125", "data/descriptors_full_synth125"),   # synth-only data-scaling runs (>100% budget)
        ("synth150", "data/descriptors_full_synth150"),
        ("synthall", "data/descriptors_full_synthall")]

print(f"painting queries (Classification==Paintings): {int(paint_mask.sum())} | "
      f"distractors: {int(distr_mask.sum())} | K={K} tau={TAU}", flush=True)
for lab, ddir in RUNS:
    pkl = os.path.join(REPO, ddir, "r18_contr_loss_gem_fc_swsl_ms/descriptors.pkl")
    if not os.path.isfile(pkl):
        print(f"CLSPAINT synth={lab} MISSING {pkl}", flush=True); continue
    d = pickle.load(open(pkl, "rb"))
    train = np.ascontiguousarray(d["train_descriptors"], dtype="float32")
    test  = np.ascontiguousarray(d["test_descriptors"],  dtype="float32")
    m, P = estimate_pca_whiten_with_shrinkage(train, shrinkage=1.0, dimensions=DIM)
    train = apply_pca_whiten_and_normalize(train, m, P).astype("float32")
    test  = apply_pca_whiten_and_normalize(test,  m, P).astype("float32")
    clf = KNN_Classifier(K=K, t=TAU); clf.fit(train, train_labels)
    preds, confs = clf.predict(test); preds = np.array(preds); confs = np.array(confs)
    allidx = paint_mask | distr_mask
    g_all = gap(preds[allidx], confs[allidx], test_labels[allidx]) * 100
    g_no  = gap_non_distr(preds[paint_mask], confs[paint_mask], test_labels[paint_mask]) * 100
    acc   = classif_accuracy(preds[paint_mask], test_labels[paint_mask]) * 100
    print(f"CLSPAINT synth={lab} GAP {g_all:.2f} GAP- {g_no:.2f} ACC {acc:.2f}", flush=True)
