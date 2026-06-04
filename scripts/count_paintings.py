"""One-off: how many of the 224,408 Met dataset classes are paintings?

Joins the dataset's class ids (= Met object IDs) against The Met Open Access
metadata CSV and counts paintings under several definitions. stdlib only.
Run with the repo .venv:  .venv/bin/python data/count_paintings.py
"""
import os, csv, json
from collections import Counter

REPO  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS   = REPO + "/data/met_class_ids.txt"
CSVF  = REPO + "/data/MetObjects.csv"
DBJSON= REPO + "/data/ground_truth/MET_database.json"

# class ids (one per exhibit/class)
with open(IDS) as f:
    class_ids = {ln.strip() for ln in f if ln.strip()}
n_classes = len(class_ids)

# images per class (from local ground truth)
with open(DBJSON) as f:
    db = json.load(f)
img_per_class = Counter(str(e["id"]) for e in db)
n_images = len(db)

# join with Met metadata (stream the big CSV)
meta = {}
with open(CSVF, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        oid = (row.get("Object ID") or "").strip()
        if oid in class_ids and oid not in meta:
            meta[oid] = ((row.get("Department") or "").strip(),
                         (row.get("Classification") or "").strip(),
                         (row.get("Object Name") or "").strip(),
                         (row.get("Medium") or "").strip())
matched = len(meta)

MED_PAT = ("oil on", "tempera on", "oil and tempera", "tempera and oil",
           "acrylic on", "distemper on", "encaustic", "oil colors on", "oil paint")
def med_is_paint(m):
    ml = m.lower(); return any(p in ml for p in MED_PAT)

strict, broad = set(), set()
for oid, (dept, cls, name, med) in meta.items():
    s = "painting" in cls.lower()
    if s: strict.add(oid)
    if s or ("painting" in name.lower()) or med_is_paint(med): broad.add(oid)

euro = {oid for oid, (d, _, _, _) in meta.items() if d == "European Paintings"}
imgs = lambda ids: sum(img_per_class[i] for i in ids)
dept_classes  = Counter(d for (d, _, _, _) in meta.values())
strict_by_dept = Counter(meta[oid][0] for oid in strict)

def line(label, ids):
    print(f"  {label:<36} {len(ids):>7,} classes   {imgs(ids):>8,} images")

print("=" * 74)
print(f"Met dataset classes (exhibits): {n_classes:,}    training images: {n_images:,}")
print(f"Matched in Met Open Access CSV: {matched:,}    (absent from CSV: {n_classes-matched:,})")
print("=" * 74)
print("PAINTINGS by definition:")
line("strict  (Classification ~ painting)", strict)
line("broad   (classification/name/medium)", broad)
line("dept = 'European Paintings'", euro)
print(f"\n  broad share: {100*len(broad)/n_classes:.1f}% of all classes, "
      f"{100*len(broad)/matched:.1f}% of matched")
print("\nStrict paintings by department (top 12):")
for d, c in strict_by_dept.most_common(12):
    print(f"  {d or '[blank]':<44} {c:>6,}")
print("\nALL matched classes by department (top 20):")
for d, c in dept_classes.most_common(20):
    print(f"  {d or '[blank]':<44} {c:>6,}")
print(f"\n  distinct departments: {len(dept_classes)}")

# ---- query side: how many of the scored Met queries depict a painting? ----
def qstats(path, label):
    with open(path) as f:
        q = json.load(f)
    met   = [str(e["MET_id"]) for e in q if "MET_id" in e]
    other = sum(1 for e in q if (e.get("path") or "").startswith("test_other"))
    noart = sum(1 for e in q if (e.get("path") or "").startswith("test_noart"))
    s = sum(1 for m in met if m in strict)
    b = sum(1 for m in met if m in broad)
    print(f"\n{label}: {len(q):,} queries = {len(met):,} Met + {other:,} other-art + {noart:,} non-art distractors")
    print(f"   Met queries depicting a PAINTING:  strict {s:,}  /  broad {b:,}   "
          f"(of {len(met):,} Met queries = {100*b/len(met):.0f}% broad)")

print("\n" + "=" * 74)
print("QUERY SIDE (the actually-scored targets)")
print("=" * 74)
qstats(REPO + "/data/ground_truth/testset.json", "TEST")
qstats(REPO + "/data/ground_truth/valset.json",  "VAL")
