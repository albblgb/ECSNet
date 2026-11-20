"""
report_tables.py
Turn the metrics.json files written by evaluate.py into the two tables of the
paper that concern ECSNet, and check them against the published numbers.

  python report_tables.py --root results
  python report_tables.py --root results --latex      # LaTeX instead of text

Only the ECSNet rows are produced. The competing methods of Table 2 are not
part of this release, so their rows are left to the reader's own runs.
"""
import argparse
import json
import os

# experiment subdirectory -> row label, for Table 1
MASKS = [("instance", "Instance Segmentation"),
         ("box", "Random Rectangular"),
         ("brush", "Free-form")]
DATASETS = [("coco_val2017", "COCO"), ("openimages_val", "Open Images")]

T1_COLS = [("psnr", 2), ("ssim", 4), ("lpips", 4), ("clip_score", 2)]
T2_COLS = [("psnr", 2), ("ssim", 4), ("lpips", 4), ("clip_score", 2),
           ("f1", 4), ("iou", 4), ("cer", 4), ("white_wrong", 4),
           ("white_cover", 4)]

# What this release should reproduce: the ECSNet rows of the paper, n=1000.
# A rerun on the same checkpoint should land within a few 0.01 of these; the
# evaluation is seeded, but GPU kernels differ between cards and drivers.
REFERENCE = {
    ("t1", "coco_val2017", "instance"): [23.10, 0.8358, 0.1207, 24.84],
    ("t1", "coco_val2017", "box"):      [19.67, 0.7747, 0.1749, 24.65],
    ("t1", "coco_val2017", "brush"):    [25.49, 0.8439, 0.1004, 24.68],
    ("t1", "openimages_val", "instance"): [21.48, 0.8217, 0.1531, 24.11],
    ("t1", "openimages_val", "box"):      [20.59, 0.8154, 0.1511, 23.81],
    ("t1", "openimages_val", "brush"):    [27.11, 0.8806, 0.0807, 23.75],
    ("t2", "coco_val2017", ""):   [28.52, 0.8507, 0.0947, 24.68,
                                   0.9987, 0.9975, 0.0109, 0.0000, 0.0000],
    ("t2", "openimages_val", ""): [30.92, 0.8903, 0.0736, 23.73,
                                   0.9986, 0.9973, 0.0110, 0.0000, 0.0000],
}


def load(root, dataset, subdir):
    """(summary dict, n) for one result directory, or (None, 0) if absent."""
    p = os.path.join(root, "ecsnet", dataset, subdir, "metrics.json")
    if not os.path.isfile(p):
        return None, 0
    j = json.load(open(p))
    s = j["summary"]
    n = max((v.get("n", 0) for v in s.values() if isinstance(v, dict)), default=0)
    return s, n


def val(summary, key):
    """The mean of one metric, or None when the run did not produce it."""
    if summary is None or key not in summary:
        return None
    v = summary[key]
    v = v.get("mean") if isinstance(v, dict) else v
    # white_wrong is NaN for a keyless method; json gives it back as a float nan
    return None if v is None or v != v else v


def fmt(v, dec):
    return "-" if v is None else f"{v:.{dec}f}"


def row_cells(summary, cols):
    return [fmt(val(summary, k), d) for k, d in cols]


def drift(summary, cols, ref):
    """Largest absolute deviation from the published row, or None."""
    if ref is None or summary is None:
        return None
    worst = 0.0
    for (k, _), r in zip(cols, ref):
        v = val(summary, k)
        if v is not None:
            worst = max(worst, abs(v - r))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ECSNET_RESULTS", "results"),
                    help="the --out directory the evaluation wrote to")
    ap.add_argument("--strength", default="0.5", help="Table 2: stego strength")
    ap.add_argument("--cell", default="64", help="Table 2: secret cell height")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    missing = []

    # ---------------- Table 1: normal inpainting ----------------
    print("\n=== Table 1 (tab:inpainting), ECSNet rows ===")
    hdr = f"{'Mask':<24}" + "".join(f"{k:>12}" for k, _ in T1_COLS)
    for ds, dslabel in DATASETS:
        print(f"\n-- {dslabel}")
        print(hdr + f"{'n':>8}{'max drift':>11}")
        for mask, label in MASKS:
            sub = f"exp1_inpainting_{mask}"
            s, n = load(a.root, ds, sub)
            if s is None:
                missing.append(f"{ds}/{sub}")
            cells = row_cells(s, T1_COLS)
            d = drift(s, T1_COLS, REFERENCE.get(("t1", ds, mask)))
            print(f"{label:<24}" + "".join(f"{c:>12}" for c in cells)
                  + f"{n:>8}" + (f"{d:>11.4f}" if d is not None else f"{'-':>11}"))

    # ---------------- Table 2: stego ----------------
    print("\n\n=== Table 2 (tab:steganographic_performance), ECSNet row ===")
    sub = f"exp2_stego_cell{a.cell}_s{a.strength}"
    print(f"{'Dataset':<24}" + "".join(f"{k:>12}" for k, _ in T2_COLS)
          + f"{'n':>8}{'max drift':>11}")
    for ds, dslabel in DATASETS:
        s, n = load(a.root, ds, sub)
        if s is None:
            missing.append(f"{ds}/{sub}")
        cells = row_cells(s, T2_COLS)
        d = drift(s, T2_COLS, REFERENCE.get(("t2", ds, "")))
        print(f"{dslabel:<24}" + "".join(f"{c:>12}" for c in cells)
              + f"{n:>8}" + (f"{d:>11.4f}" if d is not None else f"{'-':>11}"))

    if a.latex:
        print("\n\n% ---- LaTeX, ECSNet rows only ----")
        for ds, dslabel in DATASETS:
            for mask, label in MASKS:
                s, _ = load(a.root, ds, f"exp1_inpainting_{mask}")
                print(f"% T1 {dslabel} {label}: "
                      + " & ".join(row_cells(s, T1_COLS)) + r" \\")
        for ds, dslabel in DATASETS:
            s, _ = load(a.root, ds, sub)
            print(f"% T2 {dslabel}: ECSNet & "
                  + " & ".join(row_cells(s, T2_COLS)) + r" \\")

    print()
    if missing:
        print("MISSING -- these runs have not been done yet:")
        for m in missing:
            print("   ", m)
    else:
        print("All ECSNet cells present.")
    print("\n'max drift' is the largest absolute difference from the published "
          "row. PSNR is in dB, so a drift of 0.05 there is 0.2%; the other "
          "columns are 0-1 quantities.")


if __name__ == "__main__":
    main()
