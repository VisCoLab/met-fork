"""Parse the per-folder metadata.json of the synthetic gallery dataset into the
procedural factors we want to test for clustering structure (floor material,
canvas aspect, placard position, per-camera pose) plus the definitive source
Met-id (from the CanvasMaterial texture path).

Reusable: `parse_metadata(folder)` -> dict. Run as a script for a dataset-wide
survey of factor cardinalities (which hyperparameters are recoverable at all).
"""
import json
import os
import re
import sys
from collections import Counter

SYNTH_ROOT = "/mnt/storage_6/project_data/pl0896-03/visart-dataset"
ANGLES = ["front", "left upper", "right upper", "left bottom", "right bottom"]
_MET_ID_RE = re.compile(r"/MET/(\d+)/")


def met_id_from_path(p):
    if not p:
        return None
    m = _MET_ID_RE.search(p)
    return int(m.group(1)) if m else None


def parse_metadata(folder):
    """folder: int or path. Returns dict of procedural factors + source met_id."""
    fdir = folder if isinstance(folder, str) else os.path.join(SYNTH_ROOT, str(folder))
    with open(os.path.join(fdir, "metadata.json")) as f:
        d = json.load(f)
    objs = {o["name"]: o for o in d.get("objects", [])}
    mats = {m["name"]: m for m in d.get("materials", [])}

    # definitive source image (canvas texture) -> met id
    src = None
    cm = mats.get("CanvasMaterial")
    if cm:
        for t in cm.get("textures", []):
            if t.get("image"):
                src = t["image"]
                break

    # floor material: name like "floor_5"
    floor = next((n for n in mats if n.lower().startswith("floor")), None)

    # canvas aspect ratio (scale_x / scale_y); also raw scales
    canvas = objs.get("canvas")
    aspect = None
    cscale = None
    if canvas and "scale" in canvas:
        cscale = canvas["scale"]
        if cscale[1]:
            aspect = cscale[0] / cscale[1]

    # placard ("plakietka") x-position
    plk = objs.get("plakietka")
    placard_x = plk["location"][0] if plk else None

    # cameras: name -> {loc, rot}
    cams = {n: {"loc": o.get("location"), "rot": o.get("rotation_euler")}
            for n, o in objs.items() if o.get("type") == "CAMERA"}

    # light object: capture whatever fields exist (to learn if shape/spread/energy recoverable)
    light = objs.get("light")
    light_keys = sorted(light.keys()) if light else []

    return dict(
        folder=os.path.basename(fdir.rstrip("/")),
        src=src,
        met_id=met_id_from_path(src),
        floor=floor,
        aspect=aspect,
        canvas_scale=cscale,
        placard_x=placard_x,
        cams=cams,
        light_keys=light_keys,
    )


def _survey():
    folders = sorted(
        (d for d in os.listdir(SYNTH_ROOT)
         if d.isdigit() and os.path.isdir(os.path.join(SYNTH_ROOT, d))),
        key=int,
    )
    print(f"folders: {len(folders)}")
    floors = Counter()
    aspects = []
    placards = []
    cam_name_sets = Counter()
    light_keysets = Counter()
    met_ids = []
    n_bad = 0
    for fl in folders:
        try:
            r = parse_metadata(os.path.join(SYNTH_ROOT, fl))
        except Exception as e:
            n_bad += 1
            if n_bad <= 5:
                print(f"  [bad] {fl}: {e}")
            continue
        floors[r["floor"]] += 1
        if r["aspect"] is not None:
            aspects.append(r["aspect"])
        if r["placard_x"] is not None:
            placards.append(r["placard_x"])
        cam_name_sets[tuple(sorted(r["cams"].keys()))] += 1
        light_keysets[tuple(r["light_keys"])] += 1
        if r["met_id"] is not None:
            met_ids.append(r["met_id"])

    print(f"\nbad/unparseable: {n_bad}")
    print(f"\nfloor materials ({len(floors)} distinct):")
    for k, v in floors.most_common():
        print(f"  {k!r}: {v}")
    print(f"\ncamera name-sets ({len(cam_name_sets)} distinct):")
    for k, v in cam_name_sets.most_common():
        print(f"  {v:5d} x {k}")
    print(f"\nlight object key-sets (what fields exist on the LIGHT object):")
    for k, v in light_keysets.most_common():
        print(f"  {v:5d} x {k}")
    if aspects:
        import statistics as st
        print(f"\ncanvas aspect: n={len(aspects)} min={min(aspects):.3f} "
              f"max={max(aspects):.3f} mean={st.mean(aspects):.3f} "
              f"distinct(2dp)={len({round(a,2) for a in aspects})}")
    if placards:
        import statistics as st
        print(f"placard_x:     n={len(placards)} min={min(placards):.3f} "
              f"max={max(placards):.3f} mean={st.mean(placards):.3f} "
              f"distinct(2dp)={len({round(p,2) for p in placards})}")
    print(f"\nmet_ids: n={len(met_ids)} distinct={len(set(met_ids))} "
          f"(dupes={len(met_ids)-len(set(met_ids))})")


if __name__ == "__main__":
    _survey()
