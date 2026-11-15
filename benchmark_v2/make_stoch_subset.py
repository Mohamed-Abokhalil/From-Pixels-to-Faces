"""Stratified subsets for repeated stochastic sampling. Seed 0.

FI: 100 images = all 4 images with >50% hole + 96 drawn proportionally from the other bins.
FU: 100 images drawn uniformly. Written to testsets/<task>_stoch_subset.txt.
"""
import csv

import numpy as np

from common import HERE

rng = np.random.default_rng(0)
for task, n in (("fi", 100), ("fu", 100)):
    rows = list(csv.DictReader(open(HERE / "testsets" / f"{task}_testset.csv")))
    if task == "fi":
        quota = {"0-10": 7, "10-20": 13, "20-30": 58, "30-40": 15, "40-50": 3, ">50": 4}
        pick = []
        for b, q in quota.items():
            pool = sorted(r["file"] for r in rows if r["bin"] == b)
            pick += list(rng.choice(pool, q, replace=False))
    else:
        pick = list(rng.choice(sorted(r["file"] for r in rows), n, replace=False))
    pick = sorted(pick)
    assert len(pick) == n
    open(HERE / "testsets" / f"{task}_stoch_subset.txt", "w").write("\n".join(pick) + "\n")
    print(task, len(pick))
    # 20-image subset for the expensive diffusion samplers (RePaint, CoPaint), nested in the 100-image subset
    rng20 = np.random.default_rng(1)
    if task == "fi":
        quota20 = {"0-10": 1, "10-20": 2, "20-30": 10, "30-40": 2, "40-50": 1, ">50": 4}
        bins = {r["file"]: r["bin"] for r in rows}
        p20 = []
        for b, q in quota20.items():
            pool = sorted(f for f in pick if bins[f] == b)
            p20 += list(rng20.choice(pool, q, replace=False))
    else:
        p20 = list(rng20.choice(pick, 20, replace=False))
    open(HERE / "testsets" / f"{task}_stoch20_subset.txt", "w").write("\n".join(sorted(p20)) + "\n")
