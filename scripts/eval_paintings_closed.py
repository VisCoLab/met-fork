"""Closed-world painting eval (descriptors extracted with --info_dir data/gt_paint).

The closed world has only 1 val query, so K/tau can't be tuned on val. Instead we
tune on the 148 painting TEST queries via 2-fold cross-validation: tune (K,tau) on
one half, report GAP-/ACC on the held-out half, and vice-versa. Every query is
scored exactly once, on a fold it did NOT help tune -> no leakage. The closed world
has no distractors, so GAP == GAP- (we report GAP- and ACC). An "oracle" line (tune
== report on all 148) is printed only as an optimistic upper bound.

Input: descriptors.pkl from extract_descriptors run with --info_dir data/gt_paint
(train = 12,403-image painting DB, test = 148 painting queries).

Usage: scripts/eval_paintings_closed.py <descr_dir> [info_dir=data/gt_paint] [pcaw_dim=512] [seed=0]
"""
import os, sys, json, pickle, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import numpy as np
from code.utils.utils import *                       # PCAw helpers, evaluate
from code.classifiers.knn_classifier import KNN_Classifier, tune_KNN

DDIR = sys.argv[1] if len(sys.argv) > 1 else "data/descriptors_paint/r18_contr_loss_gem_fc_swsl_ms"
INFO = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "data/gt_paint")
DIM  = int(sys.argv[3]) if len(sys.argv) > 3 else 512
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

d = pickle.load(open(os.path.join(DDIR, "descriptors.pkl"), "rb"))
train = np.ascontiguousarray(d["train_descriptors"], dtype="float32")
test  = np.ascontiguousarray(d["test_descriptors"],  dtype="float32")
train_labels = np.array([e["id"] for e in json.load(open(os.path.join(INFO, "MET_database.json")))])
test_labels  = np.array([int(e["MET_id"]) if "MET_id" in e else -1
                         for e in json.load(open(os.path.join(INFO, "testset.json")))])
assert (test_labels != -1).all(), "closed world expects no distractors in testset.json"

m, P = estimate_pca_whiten_with_shrinkage(train, shrinkage=1.0, dimensions=DIM)
train = apply_pca_whiten_and_normalize(train, m, P).astype("float32")
test  = apply_pca_whiten_and_normalize(test,  m, P).astype("float32")

grid = {'K': np.array([1, 2, 3, 5, 7, 10, 15, 20, 50]),
        't': np.array([0.01, 0.1, 1., 5., 10., 15., 20., 25., 30., 50., 100., 500.])}

def tune(dsc, lab):                       # -> (K, t); tune_KNN prints every combo, so silence it
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        _, best = tune_KNN(grid, train, train_labels, np.ascontiguousarray(dsc), lab, verbose=False)
    return int(best['K']), float(best['t'])

def report(K, t, idx):                    # GAP- (== GAP here) and ACC on test[idx]
    clf = KNN_Classifier(K=K, t=t); clf.fit(train, train_labels)
    preds, confs = clf.predict(np.ascontiguousarray(test[idx]))
    _, gnd, acc = evaluate(np.array(preds), np.array(confs), test_labels[idx], verbose=False)
    return gnd, acc

N = len(test_labels)
perm = np.random.RandomState(SEED).permutation(N)
A, B = perm[:N // 2], perm[N // 2:]

print(f"closed-world paintings | DB {train.shape[0]:,} imgs / {len(np.unique(train_labels)):,} classes "
      f"| test {N} queries (no distractors -> GAP==GAP-)")
print(f"2-fold CV, seed {SEED}\n")
gaps, accs = [], []
for f, (tune_idx, rep_idx) in enumerate([(A, B), (B, A)], 1):
    K, t = tune(test[tune_idx], test_labels[tune_idx])
    gnd, acc = report(K, t, rep_idx)
    gaps.append(gnd); accs.append(acc)
    print(f"  fold {f}: tune {len(tune_idx)}q -> K={K:<3} tau={t:<5g} | report {len(rep_idx)}q:  "
          f"GAP- {gnd*100:6.2f}   ACC {acc*100:6.2f}")
print(f"\n  >> 2-fold mean:  GAP- {np.mean(gaps)*100:6.2f}   ACC {np.mean(accs)*100:6.2f}   (headline; no leakage)")

Ko, to = tune(test, test_labels)          # oracle: tune & report on all N
gnd, acc = report(Ko, to, np.arange(N))
print(f"  oracle (tune=report=all {N}): K={Ko} tau={to:g}  GAP- {gnd*100:6.2f}  ACC {acc*100:6.2f}  (upper bound)")
