"""Per-image distortion/perceptual metrics, raw and recomposited, over three regions.

For every method and image we compute, on the raw output ("raw") and on the
recomposited output I_c*(1-M) + out*M ("comp"):
  full image : PSNR, SSIM, L1, LPIPS(alex)
  hole (M=1) : PSNR, SSIM (mean of the SSIM map over hole pixels), L1
  band       : PSNR, SSIM, L1 over dilate(M,4)&~erode(M,4) (8-px band straddling the boundary)
PSNR is computed per image from the region MSE (data range 1); SSIM uses
skimage.structural_similarity (Gaussian window sigma=1.5, as in Wang et al. 2004) on RGB.

Also evaluates two non-learned reference rows:
  "Visible-only (mean fill)": hole filled with the mean colour of the visible pixels
  "Masked input": the benchmark's masked input as given (white hole for FI)

Output: results/<task>/per_image/<method>.csv  (one row per image)
"""
import argparse
import csv

import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity
from tqdm import tqdm

from common import (DATA, METHODS, RESULTS, Method, band, composite, gt_files, load_mask,
                    load_rgb, output_path)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def region_scores(pred, gt, ssim_map, region):
    if region is None:
        diff = pred - gt
        mse = float((diff ** 2).mean()); l1 = float(np.abs(diff).mean()); s = float(ssim_map.mean())
    else:
        if not region.any():
            return np.nan, np.nan, np.nan
        d = (pred - gt)[region]
        mse = float((d ** 2).mean()); l1 = float(np.abs(d).mean()); s = float(ssim_map[region].mean())
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    return psnr, s, l1


def score(pred, gt, hole, bnd, lp):
    _, smap = structural_similarity(gt, pred, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                    sigma=1.5, use_sample_covariance=False, full=True)
    smap = smap.mean(axis=2)
    r = {}
    for tag, reg in (("full", None), ("hole", hole), ("band", bnd)):
        r[f"psnr_{tag}"], r[f"ssim_{tag}"], r[f"l1_{tag}"] = region_scores(pred, gt, smap, reg)
    with torch.no_grad():
        t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV) * 2 - 1
        r["lpips_full"] = float(lp(t(pred), t(gt)).item())
    return r


def run(task, only=None):
    files = gt_files(task)
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    outdir = RESULTS / task / "per_image"
    outdir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for f in files:
        cache[f] = (load_rgb(DATA[task] / "gt" / f), load_mask(task, f), load_rgb(DATA[task] / "masked" / f))
    refs = [Method("Visible-only (mean fill)", "_meanfill", 0, "", "reference", "none"),
            Method("Masked input", "_masked", 0, "", "reference", "none")]
    for m in METHODS[task] + refs:
        if only and m.folder not in only and m.name not in only:
            continue
        rows = []
        for f in tqdm(files, desc=f"{task}/{m.name}", leave=False):
            gt, hole, ic = cache[f]
            bnd = band(hole)
            if m.folder == "_meanfill":
                vis_mean = ic[~hole].mean(axis=0)
                out = ic.copy(); out[hole] = vis_mean
            elif m.folder == "_masked":
                out = ic
            else:
                out = load_rgb(output_path(task, m, f, files))
            row = {"file": f, "mask_ratio": float(hole.mean()),
                   "vis_abs_err_raw": float(np.abs(out - gt)[~hole].mean())}
            for kind, img in (("raw", out), ("comp", composite(ic, out, hole))):
                if kind == "comp" and m.folder.startswith("_"):
                    continue
                for k, v in score(img, gt, hole, bnd, lp).items():
                    row[f"{kind}_{k}"] = v
            rows.append(row)
        with open(outdir / f"{m.folder}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"{task}/{m.name}: comp PSNR full {np.mean([r.get('comp_psnr_full', r['raw_psnr_full']) for r in rows]):.2f}"
              f" hole {np.mean([r.get('comp_psnr_hole', r['raw_psnr_hole']) for r in rows]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fi", "fu", "all"], default="all")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for t in (["fi", "fu"] if a.task == "all" else [a.task]):
        run(t, a.only)
