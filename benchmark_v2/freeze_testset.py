"""Freeze the exact test sets: filename, mask ratio, bin, and SHA-256 of GT/mask/masked files.

Output: benchmark_v2/testsets/{fi,fu}_testset.csv  (+ a check of the FI bin folders).
"""
import csv
import hashlib
import os

from common import DATA, ORIG, HERE, gt_files, load_mask, bin_of


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    out = HERE / "testsets"
    out.mkdir(exist_ok=True)
    folder_bin = {}
    gb = ORIG["fi"] / "gt_binned"
    for b in os.listdir(gb):
        if (gb / b).is_dir():
            for f in os.listdir(gb / b):
                folder_bin[f] = b
    for task in ("fi", "fu"):
        rows = []
        for f in gt_files(task):
            hole = load_mask(task, f)
            r = float(hole.mean())
            rows.append({
                "file": f, "mask_ratio": round(r, 6), "bin": bin_of(r),
                "gt_binned_folder": folder_bin.get(f, "") if task == "fi" else "",
                "sha256_gt": sha(DATA[task] / "gt" / f),
                "sha256_mask": sha(DATA[task] / "mask" / f),
                "sha256_masked": sha(DATA[task] / "masked" / f),
            })
        with open(out / f"{task}_testset.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        from collections import Counter
        print(task, len(rows), dict(sorted(Counter(r["bin"] for r in rows).items())))
        if task == "fi":
            mism = [(r["file"], r["bin"], r["gt_binned_folder"]) for r in rows
                    if r["gt_binned_folder"].replace("_", "-").replace("over50", ">50") != r["bin"]]
            print("  bin folder vs computed ratio mismatches:", len(mism), mism[:5])


if __name__ == "__main__":
    main()
