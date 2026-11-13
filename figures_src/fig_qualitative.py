"""Qualitative figures (R3.3, R4.7). All tiles are recomposited outputs (Eq. 2) at the native 256x256 resolution.

fig_visual_fi_p1/p2.jpg : masked input, GT, all FI methods ordered by hole PSNR (Table 9), 2 images per hole-ratio bin
                          (seeded random choice, seed 0; the >50% bin uses 2 of its 4 images)
fig_visual_fu.jpg       : masked input, GT, all FU methods ordered by hole PSNR, 8 images (seed 0)
fig_visual_hard.jpg     : challenging FI cases: the 3 largest holes, 2 masks spanning several components (30-40% bin,
                          seed 1) and 3 strong expressions (broad smile/laugh), chosen by visual inspection of a seeded
                          contact sheet (25770, 27634, 28453; see figures_src/out/_expr_candidates.png); rows = one
                          representative per category/family (best hole PSNR within the group)
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV / "benchmark_v2"))
from common import DATA, METHODS, composite, gt_files, load_mask, load_rgb, output_path  # noqa: E402

OUT = REV / "FI_FU_Survey_AIR_revised"
T = 256
try:
    FONT = ImageFont.truetype("arial.ttf", 22); FONT_S = ImageFont.truetype("arial.ttf", 18)
except OSError:
    FONT = FONT_S = ImageFont.load_default()


def order(task):
    S = {r["method"]: float(r["comp_psnr_hole"]) for r in csv.DictReader(open(REV / "benchmark_v2" / "results" / task / "summary.csv"))
         if r.get("status") not in ("excluded", "reference")}
    return sorted([m for m in METHODS[task] if m.name in S], key=lambda m: -S[m.name])


def tile(task, m, f, files):
    ic = load_rgb(DATA[task] / "masked" / f)
    if isinstance(m, str) and m == "masked":
        return ic
    if isinstance(m, str) and m == "gt":
        return load_rgb(DATA[task] / "gt" / f)
    return composite(ic, load_rgb(output_path(task, m, f, files)), load_mask(task, f))


def grid(task, rows, cols, col_labels, path, label_w=210):
    files = gt_files(task)
    W = label_w + T * len(cols); H = 40 + T * len(rows)
    canvas = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(canvas)
    for j, lab in enumerate(col_labels):
        d.text((label_w + j * T + T // 2, 20), lab, fill="black", font=FONT_S, anchor="mm")
    for i, m in enumerate(rows):
        name = {"masked": "Masked input", "gt": "Ground truth"}[m] if isinstance(m, str) else m.name
        d.text((label_w - 8, 40 + i * T + T // 2), name, fill="black", font=FONT, anchor="rm")
        for j, f in enumerate(cols):
            im = (np.clip(tile(task, m, f, files), 0, 1) * 255 + 0.5).astype(np.uint8)
            canvas.paste(Image.fromarray(im), (label_w + j * T, 40 + i * T))
    canvas.save(path, quality=92)
    print("wrote", path, canvas.size)


def main():
    rng = np.random.default_rng(0)
    ts = list(csv.DictReader(open(REV / "benchmark_v2" / "testsets" / "fi_testset.csv")))
    bins = ["0-10", "10-20", "20-30", "30-40", "40-50", ">50"]
    cols, labs = [], []
    for b in bins:
        pool = sorted(r["file"] for r in ts if r["bin"] == b)
        for f in rng.choice(pool, 2, replace=False):
            cols.append(f); labs.append(b + "%")
    fi = order("fi"); half = (len(fi) + 1) // 2
    grid("fi", ["masked", "gt"] + fi[:half], cols, labs, OUT / "fig_visual_fi_p1.jpg")
    grid("fi", ["masked", "gt"] + fi[half:], cols, labs, OUT / "fig_visual_fi_p2.jpg")
    fu_files = sorted(rng.choice(gt_files("fu"), 8, replace=False))
    grid("fu", ["masked", "gt"] + order("fu"), fu_files, ["" for _ in fu_files], OUT / "fig_visual_fu.jpg")
    largest = [r["file"] for r in sorted(ts, key=lambda r: -float(r["mask_ratio"]))[:3]]
    span = list(np.random.default_rng(1).choice(sorted(r["file"] for r in ts if r["bin"] == "30-40"), 2, replace=False))
    expr = ["25770.png", "27634.png", "28453.png"]
    hard = largest + span + expr
    labs = [f"{100 * float(next(r['mask_ratio'] for r in ts if r['file'] == f)):.0f}% hole" for f in largest] + ["multi-component"] * 2 + ["expression"] * 3
    reps, seen = [], set()
    for m in fi:
        key = (m.category, m.family)
        if key not in seen:
            seen.add(key); reps.append(m)
    grid("fi", ["masked", "gt"] + reps, hard, labs, OUT / "fig_visual_hard.jpg")


if __name__ == "__main__":
    main()
