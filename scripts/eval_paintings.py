"""Step 2: evaluate a model ONLY on the painting Met queries.

CANONICAL painting test set = `Classification == "Paintings"` = the **148** query paths committed in
`data/gt_paint/testset.json` (the single definition used project-wide: docs/real-synth-mixing and
`scripts/eval_paintings_cls.py`). Reuses step-1's tuned K=7, tau=50 (val has too few paintings to
retune) and the full 397k studio DB index. Reports GAP (+all distractors), and GAP- / ACC (no
distractors) over the 148 painting queries.

Usage: scripts/eval_paintings.py [descr_dir] [info_dir] [K] [tau] [pcaw_dim] [paint_testset_json]
Compute (full-DB faiss-CPU kNN) -> run via SLURM, not the login node.
"""
import os, sys, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import numpy as np
from code.utils.utils import *                         # PCAw helpers, gap, gap_non_distr, classif_accuracy
from code.classifiers.knn_classifier import KNN_Classifier

DDIR  = sys.argv[1] if len(sys.argv) > 1 else "data/descriptors/r18_contr_loss_gem_fc_swsl_ms"
INFO  = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth"
K     = int(sys.argv[3]) if len(sys.argv) > 3 else 7
TAU   = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0
DIM   = int(sys.argv[5]) if len(sys.argv) > 5 else 512
PAINT = sys.argv[6] if len(sys.argv) > 6 else "data/gt_paint/testset.json"   # committed 148 (Classification=="Paintings")

d = pickle.load(open(os.path.join(DDIR, "descriptors.pkl"), "rb"))
train = np.ascontiguousarray(d["train_descriptors"], dtype="float32")
test  = np.ascontiguousarray(d["test_descriptors"], dtype="float32")
train_labels = np.array([e["id"] for e in json.load(open(os.path.join(INFO, "MET_database.json")))])
test_meta    = json.load(open(os.path.join(INFO, "testset.json")))
test_labels  = np.array([int(e["MET_id"]) if "MET_id" in e else -1 for e in test_meta])

paint_paths = {e["path"] for e in json.load(open(PAINT))}              # 148 committed painting queries
paint_mask  = np.array([e["path"] in paint_paths for e in test_meta])
distr_mask  = test_labels == -1

m, P = estimate_pca_whiten_with_shrinkage(train, shrinkage=1.0, dimensions=DIM)   # same PCAw as full eval
train = apply_pca_whiten_and_normalize(train, m, P).astype("float32")
test  = apply_pca_whiten_and_normalize(test, m, P).astype("float32")
clf = KNN_Classifier(K=K, t=TAU); clf.fit(train, train_labels)
preds, confs = clf.predict(test); preds, confs = np.array(preds), np.array(confs)

allidx = paint_mask | distr_mask                                       # paintings + all distractors
g_all = gap(preds[allidx], confs[allidx], test_labels[allidx]) * 100
g_no  = gap_non_distr(preds[paint_mask], confs[paint_mask], test_labels[paint_mask]) * 100
acc   = classif_accuracy(preds[paint_mask], test_labels[paint_mask]) * 100
print(f"model: {DDIR} | K={K} tau={TAU} | paintings (Classification==Paintings) = "
      f"{int(paint_mask.sum())} | distractors {int(distr_mask.sum())}")
print(f"FULL test (sanity vs step 1): GAP {gap(preds,confs,test_labels)*100:.2f}  "
      f"GAP- {gap_non_distr(preds,confs,test_labels)*100:.2f}  ACC {classif_accuracy(preds,test_labels)*100:.2f}")
print(f"PAINT148  GAP(+all distr) {g_all:.2f}  GAP- {g_no:.2f}  ACC {acc:.2f}")
