"""
prep_openimages.py
Build the Open Images V7 validation subset used for the cross-dataset test.

Three steps, run in order (see the step-by-step guide for the downloads
in between):

  python prep_openimages.py select   # pick images, write the download list
  python prep_openimages.py masks    # pull only the needed masks out of the zips
  python prep_openimages.py build    # rasterise, filter, write annotations.json

The selection mirrors the COCO protocol so the two datasets are comparable:
instance masks covering 2%-60% of the image, one instance drawn per sample at
evaluation time, images resized to 512. Rotated images are skipped -- their
stored pixels and their annotations disagree on orientation, and fixing that
is not worth it for a 1000-image test set.

Everything is seeded, so the same subset is produced on every machine.
"""
import argparse
import csv
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image

_DATA = os.environ.get("ECSNET_DATA", "./data")
ROOT = os.path.join(_DATA, "openimages_val")   # the prepared subset
RAW = os.path.join(_DATA, "openimages_raw")    # the downloaded csv/zip files


def cmd_select(a):
    os.makedirs(a.root, exist_ok=True)

    rotated = set()
    if a.rot_csv and os.path.isfile(a.rot_csv):
        with open(a.rot_csv) as f:
            for r in csv.DictReader(f):
                rot = (r.get("Rotation") or "").strip()
                if rot not in ("", "0", "0.0", "nan"):
                    rotated.add(r["ImageID"])
        print(f"  {len(rotated)} rotated images will be skipped")

    # Box area is a cheap upper bound on mask area, known before anything is
    # downloaded. It is a loose pre-filter; the exact mask-area filter runs in
    # `build`, so select a margin of extra images to survive it.
    cands = defaultdict(list)
    n_rows = 0
    with open(a.seg_csv) as f:
        for r in csv.DictReader(f):
            n_rows += 1
            iid = r["ImageID"]
            if iid in rotated:
                continue
            w = float(r["BoxXMax"]) - float(r["BoxXMin"])
            h = float(r["BoxYMax"]) - float(r["BoxYMin"])
            if 0.03 <= w * h <= 0.80:
                cands[iid].append(r["MaskPath"])
    ids = sorted(cands)
    print(f"  {n_rows} mask rows, {len(ids)} images with a usable instance")

    random.Random(a.seed).shuffle(ids)
    ids = ids[:a.n]
    with open(os.path.join(a.root, "image_list.txt"), "w") as f:
        f.writelines(f"validation/{i}\n" for i in ids)
    with open(os.path.join(a.root, "selected_masks.json"), "w") as f:
        json.dump({i: cands[i] for i in ids}, f)
    print(f"  selected {len(ids)} images -> {a.root}/image_list.txt")


def cmd_masks(a):
    sel = json.load(open(os.path.join(a.root, "selected_masks.json")))
    want = {m for ms in sel.values() for m in ms}
    out = os.path.join(a.raw, "masks_raw")
    os.makedirs(out, exist_ok=True)
    got = 0
    for z in sorted(os.listdir(a.zip_dir)):
        if not z.endswith(".zip"):
            continue
        with zipfile.ZipFile(os.path.join(a.zip_dir, z)) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base in want:
                    with zf.open(name) as src, open(os.path.join(out, base), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    got += 1
        print(f"  {z}: {got}/{len(want)} masks so far")
    print(f"  extracted {got} of {len(want)} wanted masks -> {out}")
    if got < len(want):
        print("  [!] some masks were not found; check that all 16 zips are present")


def cmd_build(a):
    sel = json.load(open(os.path.join(a.root, "selected_masks.json")))
    img_raw = os.path.join(a.raw, "images_raw")
    mask_raw = os.path.join(a.raw, "masks_raw")
    img_out = os.path.join(a.root, "images")
    mask_out = os.path.join(a.root, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    images, n_missing, n_nomask = [], 0, 0
    for iid in sel:          # json keeps insertion order, i.e. the select order
        src = os.path.join(img_raw, f"{iid}.jpg")
        if not os.path.isfile(src):
            n_missing += 1
            continue
        im = Image.open(src)
        W, H = im.size
        keep = []
        for mp in sel[iid]:
            mpath = os.path.join(mask_raw, mp)
            if not os.path.isfile(mpath):
                continue
            # Mask PNGs are not guaranteed to be stored at image resolution, so
            # bring them to it before measuring area.
            m = Image.open(mpath).convert("L").resize((W, H), Image.NEAREST)
            frac = (np.asarray(m) > 127).mean()
            if 0.02 <= frac <= 0.60:          # same band as the COCO loader
                m.save(os.path.join(mask_out, mp))
                keep.append(mp)
        if not keep:
            n_nomask += 1
            continue
        shutil.copy(src, os.path.join(img_out, f"{iid}.jpg"))
        images.append({"id": iid, "file_name": f"{iid}.jpg",
                       "width": W, "height": H, "masks": keep})
        if len(images) >= a.n_final:
            break

    with open(os.path.join(a.root, "annotations.json"), "w") as f:
        json.dump({"images": images}, f)
    print(f"  kept {len(images)} images "
          f"({n_missing} not downloaded, {n_nomask} without a mask in the area band)")
    if len(images) < a.n_final:
        print(f"  [!] fewer than {a.n_final}: rerun `select` with a larger --n, "
              f"download the extra images, and build again")
    print(f"  -> {a.root}/annotations.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--seg_csv", default=None, help="default: <raw>/validation-annotations-object-segmentation.csv")
    s.add_argument("--rot_csv", default=None, help="default: <raw>/validation-images-with-rotation.csv")
    s.add_argument("--n", type=int, default=1400,
                   help="images to download; more than n_final because some "
                        "fail the exact mask-area filter in `build`")
    s.add_argument("--seed", type=int, default=1234)

    m = sub.add_parser("masks")
    m.add_argument("--zip_dir", default=None, help="default: <raw>/mask_zips")

    b = sub.add_parser("build")
    b.add_argument("--n_final", type=int, default=1000)

    for p in (s, m, b):
        p.add_argument("--root", default=ROOT)
        p.add_argument("--raw", default=RAW)
    a = ap.parse_args()
    # Derive file defaults from --raw, so moving the raw directory moves them too.
    if getattr(a, "seg_csv", "x") is None:
        a.seg_csv = os.path.join(a.raw, "validation-annotations-object-segmentation.csv")
    if getattr(a, "rot_csv", "x") is None:
        a.rot_csv = os.path.join(a.raw, "validation-images-with-rotation.csv")
    if getattr(a, "zip_dir", "x") is None:
        a.zip_dir = os.path.join(a.raw, "mask_zips")
    {"select": cmd_select, "masks": cmd_masks, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
