"""How many of the Met dataset classes (and scored test/val queries) are paintings?

Uses the committed, project-wide painting definition: Met Open Access
`Classification == "Paintings"` (exact, single field). Joins the dataset's class
ids (= Met object IDs) against The Met Open Access metadata CSV. stdlib only;
login-node safe.  Run with the repo .venv:  .venv/bin/python scripts/count_paintings.py
"""
import os, csv, json
from collections import Counter

REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVF   = REPO + "/data/MetObjects.csv"
DBJSON = REPO + "/data/ground_truth/MET_database.json"
TEST   = REPO + "/data/ground_truth/testset.json"
VAL    = REPO + "/data/ground_truth/valset.json"

# dataset classes (one id per exhibit) + images per class, from the local ground truth
db = json.load(open(DBJSON))
img_per_class = Counter(str(e["id"]) for e in db)
class_ids = set(img_per_class)
n_classes, n_images = len(class_ids), len(db)

# committed painting class set: Met Open Access Classification == "Paintings" (exact)
paint, dept_of = set(), {}
with open(CSVF, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        oid = (row.get("Object ID") or "").strip()
        if oid in class_ids and (row.get("Classification") or "").strip() == "Paintings":
            paint.add(oid)
            dept_of[oid] = (row.get("Department") or "").strip()

paint_imgs = sum(img_per_class[i] for i in paint)
print("=" * 70)
print(f"Met dataset classes: {n_classes:,}    training images: {n_images:,}")
print(f'PAINTINGS (Classification == "Paintings"): {len(paint):,} classes  '
      f"/ {paint_imgs:,} images  ({100 * len(paint) / n_classes:.1f}% of classes)")
print("=" * 70)
print("Painting classes by department (top 12):")
for d, c in Counter(dept_of.values()).most_common(12):
    print(f"  {d or '[blank]':<36} {c:>6,}")

# query side: how many scored test/val queries depict a painting?
def qstats(path, label):
    q = json.load(open(path))
    met   = [str(e["MET_id"]) for e in q if "MET_id" in e]
    other = sum(1 for e in q if (e.get("path") or "").startswith("test_other"))
    noart = sum(1 for e in q if (e.get("path") or "").startswith("test_noart"))
    p = sum(1 for m in met if m in paint)
    print(f"\n{label}: {len(q):,} queries = {len(met):,} Met + {other:,} other-art + {noart:,} non-art distractors")
    print(f'   Met queries depicting a PAINTING (Classification=="Paintings"): {p:,}')

print("\n" + "=" * 70)
print("QUERY SIDE (the actually-scored targets)")
print("=" * 70)
qstats(TEST, "TEST")
qstats(VAL,  "VAL")
