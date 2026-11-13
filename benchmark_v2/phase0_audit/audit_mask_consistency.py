"""Mask-consistency audit: fraction of benchmark-hole pixels that equal the GT exactly (all 3 channels,
|diff| <= 1/255). A method that really filled the benchmark hole has ~0%; a method run with a different mask
(and composited with that mask) copies GT inside our hole. First 200 images of each test set."""
import numpy as np
from common import DATA, METHODS, gt_files, load_mask, load_rgb, output_path
for task in ("fi", "fu"):
    files = gt_files(task)[:200]
    for m in METHODS[task]:
        fr = []
        for f in files:
            g = load_rgb(DATA[task] / "gt" / f); h = load_mask(task, f); o = load_rgb(output_path(task, m, f, files if False else gt_files(task)))
            same = (np.abs(o - g).max(axis=2) <= 1.0 / 255 + 1e-6)
            fr.append(same[h].mean())
        print(f"{task} {m.name:14s} exact-GT pixels inside benchmark hole: {100*np.mean(fr):6.2f}%  (images >20%: {sum(x>0.2 for x in fr)}/{len(fr)})", flush=True)
