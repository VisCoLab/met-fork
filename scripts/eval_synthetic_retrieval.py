"""Step 3b: retrieval of synthetic gallery images against the Met train DB.
Correct = the source Met class (whose studio image is in the DB). Reports recall@1/5/10
pooled over all 5 camera angles, then per angle. Same PCAw (learned on train) as the eval.
"""
import os, sys, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import numpy as np, faiss
from code.utils.utils import estimate_pca_whiten_with_shrinkage, apply_pca_whiten_and_normalize

DB  = os.path.join(REPO, "data/descriptors/r18_contr_loss_gem_fc_swsl_ms/descriptors.pkl")
SYN = os.path.join(REPO, "data/descriptors/synthetic/synth_descriptors.pkl")
GT  = os.path.join(REPO, "data/ground_truth"); DIM = 512

train = np.ascontiguousarray(pickle.load(open(DB, "rb"))["train_descriptors"], dtype="float32")
train_labels = np.array([e["id"] for e in json.load(open(GT + "/MET_database.json"))])
s = pickle.load(open(SYN, "rb"))
synd = np.ascontiguousarray(s["descriptors"], dtype="float32"); mids = np.asarray(s["met_ids"]); angles = np.asarray(s["angles"])

m, P = estimate_pca_whiten_with_shrinkage(train, shrinkage=1.0, dimensions=DIM)   # PCAw learned on train, applied to both
train = apply_pca_whiten_and_normalize(train, m, P).astype("float32")
synd  = apply_pca_whiten_and_normalize(synd,  m, P).astype("float32")

index = faiss.IndexFlatIP(DIM); index.add(train)
_, I = index.search(synd, 10)                 # top-10 train neighbours per synthetic query
nbl = train_labels[I]                          # neighbour class labels (N, 10)

def recalls(mask):
    sub, gt = nbl[mask], mids[mask]; n = int(mask.sum())
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    r1 = np.mean(sub[:, 0] == gt) * 100
    r5 = np.mean([gt[i] in sub[i, :5] for i in range(n)]) * 100
    r10 = np.mean([gt[i] in sub[i, :10] for i in range(n)]) * 100
    return n, r1, r5, r10

# query groups: all synthetic, then the step-2 painting test-class subsets (if the lists exist)
groups = [("ALL synthetic (4,952 paintings)", np.ones(len(mids), bool))]
for nm, f in (("step-2 STRICT (138 painting test-classes)", "data/synth_gen/step2_painting_train_images_strict.json"),
              ("step-2 BROAD  (176 painting test-classes)", "data/synth_gen/step2_painting_train_images_broad.json")):
    if os.path.exists(f):
        ids = {int(p.split("/MET/")[1].split("/")[0]) for p in json.load(open(f))}
        groups.append((nm, np.isin(mids, list(ids))))

print(f"DB: {len(train_labels):,} train images")
for gname, gmask in groups:
    print(f"\n=== {gname} ===")
    print(f"{'angle':<14}{'N':>8}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
    n, r1, r5, r10 = recalls(gmask); print(f"{'ALL angles':<14}{n:>8}{r1:>8.2f}{r5:>8.2f}{r10:>8.2f}")
    for a in sorted(set(angles.tolist())):
        n, r1, r5, r10 = recalls(gmask & (angles == a)); print(f"{a:<14}{n:>8}{r1:>8.2f}{r5:>8.2f}{r10:>8.2f}")
