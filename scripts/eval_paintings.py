"""Step 2: evaluate a model ONLY on painting Met queries, reusing step-1's tuned
k,tau (val has too few paintings to retune). Reports GAP with ALL distractors and
GAP-/ACC with NO distractors, for strict & broad painting definitions.

Usage: scripts/eval_paintings.py [descr_dir] [info_dir] [metobjects_csv] [K] [tau] [pcaw_dim]
"""
import os, sys, json, pickle, csv
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import numpy as np
from code.utils.utils import *                      # gap, gap_non_distr, classif_accuracy, PCAw
from code.classifiers.knn_classifier import KNN_Classifier

DDIR = sys.argv[1] if len(sys.argv) > 1 else "data/descriptors/r18_contr_loss_gem_fc_swsl_ms"
INFO = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth"
CSVF = sys.argv[3] if len(sys.argv) > 3 else "data/MetObjects.csv"
K    = int(sys.argv[4]) if len(sys.argv) > 4 else 7
TAU  = float(sys.argv[5]) if len(sys.argv) > 5 else 50.0
DIM  = int(sys.argv[6]) if len(sys.argv) > 6 else 512

d = pickle.load(open(os.path.join(DDIR, "descriptors.pkl"), "rb"))
train = np.ascontiguousarray(d["train_descriptors"], dtype="float32")
test  = np.ascontiguousarray(d["test_descriptors"], dtype="float32")
train_labels = np.array([e["id"] for e in json.load(open(os.path.join(INFO, "MET_database.json")))])
test_labels  = np.array([int(e["MET_id"]) if "MET_id" in e else -1
                         for e in json.load(open(os.path.join(INFO, "testset.json")))])

m, P = estimate_pca_whiten_with_shrinkage(train, shrinkage=1.0, dimensions=DIM)   # same PCAw as full eval
train = apply_pca_whiten_and_normalize(train, m, P).astype("float32")
test  = apply_pca_whiten_and_normalize(test, m, P).astype("float32")
clf = KNN_Classifier(K=K, t=TAU); clf.fit(train, train_labels)
preds, confs = clf.predict(test); preds, confs = np.array(preds), np.array(confs)

# painting Met-class sets from the Met Open Access metadata (same defs as scripts/count_paintings.py)
MED = ("oil on", "tempera on", "oil and tempera", "tempera and oil", "acrylic on",
       "distemper on", "encaustic", "oil colors on", "oil paint")
strict, broad = set(), set()
for row in csv.DictReader(open(CSVF, encoding="utf-8-sig", newline="")):
    oid = (row.get("Object ID") or "").strip()
    if not oid:
        continue
    cls = (row.get("Classification") or "").lower(); name = (row.get("Object Name") or "").lower()
    med = (row.get("Medium") or "").lower()
    s = "painting" in cls
    if s: strict.add(oid)
    if s or "painting" in name or any(p in med for p in MED): broad.add(oid)

distr = test_labels == -1
lstr = test_labels.astype(str)
print(f"model: {DDIR} | K={K} tau={TAU} | test {len(test_labels)} = {(~distr).sum()} Met + {distr.sum()} distractors")
print(f"FULL test (sanity vs step 1): GAP {gap(preds,confs,test_labels)*100:.2f}  "
      f"GAP- {gap_non_distr(preds,confs,test_labels)*100:.2f}  ACC {classif_accuracy(preds,test_labels)*100:.2f}")
print(f"{'definition':<18}{'#paint q':>9}{'GAP(+all distr)':>18}{'GAP- (no distr)':>18}{'ACC':>8}")
for name, pset in (("strict", strict), ("broad", broad)):
    paint = np.array([(l != -1) and (s in pset) for l, s in zip(test_labels, lstr)])
    allidx = paint | distr                                    # paintings + all distractors
    g_all = gap(preds[allidx], confs[allidx], test_labels[allidx]) * 100
    g_no  = gap_non_distr(preds[paint], confs[paint], test_labels[paint]) * 100
    acc   = classif_accuracy(preds[paint], test_labels[paint]) * 100
    print(f"{name:<18}{int(paint.sum()):>9}{g_all:>18.2f}{g_no:>18.2f}{acc:>8.2f}")
