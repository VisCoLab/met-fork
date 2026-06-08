"""Build the CLOSED-WORLD painting-only training/eval manifests.

Painting definition (committed): Met Open Access `Classification == "Paintings"`
-> 4,898 dataset classes. This is the clean single-field core: it excludes broad's
snuffbox/miniature/painted-object noise AND the substring-strict variants
(Fans|Paintings, Ceramics-Paintings, ...), while keeping all genuine flat paintings
incl. Asian-format scrolls/fans. It equals the synthetic dataset's exact universe
(every one of these classes has gallery renders). Strictly nested:
Western-easel (2,093) subset of THIS (4,898) subset of broad (7,310).

Closed world => train DB, val, and test are ALL restricted to these classes and
distractors are DROPPED (so GAP == GAP-, no label -1). This is a smaller, easier,
self-contained painting benchmark -- NOT comparable to the full-DB GAP 36.1 -- and
is aligned with the synthetic-paintings experiment.

Writes data/gt_paint/{MET_database.json, testset.json, valset.json}.
NO image symlinks needed: paintings are a subset of the existing MET/ and test_*/
trees, so paths are unchanged and resolve via data/images. Train/eval with the same
--im_root ./data/ as the full pipeline, only swapping --info_dir to ./data/gt_paint.

stdlib only; login-node safe.  Run:  .venv/bin/python scripts/build_paintings_data.py
"""
import os, csv, json

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
INFO = os.path.join(REPO, "data/ground_truth")          # symlink -> met-dataset
CSVF = os.path.join(REPO, "data/MetObjects.csv")
OUT  = os.path.join(REPO, "data/gt_paint")

# painting class-id set: Met Open Access Classification == "Paintings" (exact)
paint = set()
for r in csv.DictReader(open(CSVF, encoding="utf-8-sig", newline="")):
    oid = (r.get("Object ID") or "").strip()
    if oid and (r.get("Classification") or "").strip() == "Paintings":
        paint.add(int(oid))

db   = json.load(open(os.path.join(INFO, "MET_database.json")))
test = json.load(open(os.path.join(INFO, "testset.json")))
val  = json.load(open(os.path.join(INFO, "valset.json")))

db_p   = [e for e in db   if int(e["id"]) in paint]
test_p = [e for e in test if "MET_id" in e and int(e["MET_id"]) in paint]   # drops distractors
val_p  = [e for e in val  if "MET_id" in e and int(e["MET_id"]) in paint]

os.makedirs(OUT, exist_ok=True)
for fn, entries in (("MET_database.json", db_p), ("testset.json", test_p), ("valset.json", val_p)):
    json.dump(entries, open(os.path.join(OUT, fn), "w"))

n_cls = len({e["id"] for e in db_p})
print(f'painting classes (Classification=="Paintings"): {len(paint):,}')
print(f"data/gt_paint: DB {len(db_p):,} imgs / {n_cls:,} classes | test {len(test_p)} q | val {len(val_p)} q")
print("train/eval with --im_root ./data/ --info_dir ./data/gt_paint")
