"""FID and KID with sampling-variability estimates.

Implementation: pytorch-fid InceptionV3 (FID weights pt_inception-2015-12-05, pool3, 2048-d),
images at 256x256 fed as [0,1] tensors (pytorch-fid resizes to 299 internally, bilinear).
Real set = the benchmark GT images; generated set = method outputs (raw and recomposited).

FID variability (recomposited outputs only): 100 bootstrap resamples (images resampled with replacement, independently
in both sets, seed 0). Resampling with replacement duplicates images, and FID is biased upward for small effective
samples, so percentile intervals of the bootstrap distribution do NOT bracket the full-set FID (observed: FID 6.9,
bootstrap percentiles ~20-23). We therefore report only the bootstrap standard deviation (fid_boot_sd) as a measure of
sampling variability, plus KID, which is unbiased, as mean +- std over subsets.
KID: unbiased MMD^2 with cubic polynomial kernel (Binkowski et al. 2018), 100 random subsets
of size 250 (seed 0), reported as mean +- std (x1e3).

Features are cached in results/<task>/features/*.npy so the script is cheap to re-run.
"""
import argparse
import csv
import json

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from tqdm import tqdm

from common import DATA, METHODS, RESULTS, composite, gt_files, load_mask, load_rgb, output_path

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT, KID_SUBSETS, KID_SIZE = 100, 100, 250


def fid_from_feats(a, b):
    """Frechet distance on GPU (float64). tr sqrt(S1 S2) = tr sqrt(S1^1/2 S2 S1^1/2); the inner matrix is symmetric
    PSD, so only symmetric eigendecompositions (eigh/eigvalsh) are needed.
    Checked against the scipy formula for every method (assert |diff| < 0.05; value stored as fid_gpu_check)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = torch.from_numpy(np.cov(a, rowvar=False)).to(DEV)
    s2 = torch.from_numpy(np.cov(b, rowvar=False)).to(DEV)
    w, V = torch.linalg.eigh(s1)
    r1 = (V * w.clamp(min=0).sqrt()) @ V.T
    ev = torch.linalg.eigvalsh(r1 @ s2 @ r1).clamp(min=0)
    tr = float(torch.sqrt(ev).sum())
    return float(((mu1 - mu2) ** 2).sum() + float(torch.trace(s1)) + float(torch.trace(s2)) - 2 * tr)


def fid_scipy(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1.dot(s2), disp=False)[0].real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kid(a, b, rng):
    d = a.shape[1]
    vals = []
    for _ in range(KID_SUBSETS):
        x = a[rng.choice(len(a), KID_SIZE, replace=False)]
        y = b[rng.choice(len(b), KID_SIZE, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        m = KID_SIZE
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


class Feat:
    def __init__(self):
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEV).eval()

    @torch.no_grad()
    def __call__(self, imgs):
        out = []
        for i in range(0, len(imgs), 50):
            x = torch.from_numpy(np.stack(imgs[i:i + 50])).permute(0, 3, 1, 2).to(DEV)
            out.append(self.net(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def run(task, only=None):
    files = gt_files(task)
    fdir = RESULTS / task / "features"; fdir.mkdir(parents=True, exist_ok=True)
    F = Feat()
    gt_p = fdir / "gt.npy"
    if not gt_p.exists():
        np.save(gt_p, F([load_rgb(DATA[task] / "gt" / f) for f in files]))
    g = np.load(gt_p)
    rows = []
    for m in METHODS[task]:
        if only and m.folder not in only:
            continue
        for kind in ("raw", "comp"):
            p = fdir / f"{m.folder}__{kind}.npy"
            if not p.exists():
                imgs = []
                for f in tqdm(files, desc=f"{task}/{m.name}/{kind}", leave=False):
                    o = load_rgb(output_path(task, m, f, files))
                    if kind == "comp":
                        o = composite(load_rgb(DATA[task] / "masked" / f), o, load_mask(task, f))
                    imgs.append(o.astype(np.float32))
                np.save(p, F(imgs))
            x = np.load(p)
            rng = np.random.default_rng(0)
            fid = fid_scipy(x, g)                      # point estimate: exact pytorch-fid formula
            fid_gpu = fid_from_feats(x, g)             # bootstrap engine; consistency logged
            assert abs(fid - fid_gpu) < 0.05, (fid, fid_gpu)
            boots = ([fid_from_feats(x[rng.integers(0, len(x), len(x))], g[rng.integers(0, len(g), len(g))])
                      for _ in range(N_BOOT)] if kind == "comp" else [np.nan])
            k_mean, k_std = kid(x, g, np.random.default_rng(0))
            rows.append({"method": m.name, "folder": m.folder, "kind": kind, "n": len(x), "fid": fid,
                         "fid_boot_sd": float(np.std(boots, ddof=1)) if kind == "comp" else "", "fid_boot_mean": float(np.mean(boots)) if kind == "comp" else "",
                         "fid_gpu_check": fid_gpu, "kid_x1e3": k_mean * 1e3, "kid_std_x1e3": k_std * 1e3})
            print(f"{task} {m.name:14s} {kind}: FID {fid:6.2f} (boot sd {rows[-1]['fid_boot_sd']})"
                  f" KID {k_mean*1e3:.2f}+-{k_std*1e3:.2f}", flush=True)
    out = RESULTS / task / "fid_kid.csv"
    if only and out.exists():  # merge
        old = [r for r in csv.DictReader(open(out)) if r["folder"] not in only]
        rows = old + rows
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump({"implementation": "pytorch-fid InceptionV3 pool3 (pt_inception-2015-12-05)",
               "n_boot": N_BOOT, "kid_subsets": KID_SUBSETS, "kid_subset_size": KID_SIZE, "seed": 0},
              open(RESULTS / task / "fid_kid_settings.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fi", "fu", "all"], default="all")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for t in (["fi", "fu"] if a.task == "all" else [a.task]):
        run(t, a.only)
