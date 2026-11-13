"""Repeated-sampling analysis for stochastic methods (R1.6b).

For each method in reruns/stoch/<task>/<Method>/<stem>_k<j>.png (all samples recomposited with Eq. 2):
  * per-sample hole PSNR, full LPIPS, aligned ArcFace ID-Sim (same protocol as eval_arcface.py)
  * across-sample variability: for each image the std over its K samples; reported as the mean per-image std
  * run-to-run variability of the method mean: the K "runs" (sample j of every image) give K method means;
    reported as mean +- std over runs and a 95% t-interval
  * diversity: mean pairwise LPIPS between the K samples of an image (full image), averaged over images
Output: results/stoch/<task>_<Method>.json and results/stoch/summary.csv
"""
import csv
import itertools
import json
import os

import cv2
import lpips
import numpy as np
import torch
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from scipy import stats

from common import DATA, RERUNS, RESULTS, composite, load_mask, load_rgb

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def psnr(a, b, region):
    mse = float(((a - b)[region] ** 2).mean())
    return 10 * np.log10(1 / max(mse, 1e-12))


def main():
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(256, 256)); rec = app.models["recognition"]
    t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().to(DEV) * 2 - 1
    bgr = lambda a: cv2.cvtColor((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    emb = lambda im, k: (lambda f: f / np.linalg.norm(f))(rec.get_feat(face_align.norm_crop(bgr(im), landmark=np.array(k), image_size=112)).ravel())
    out_dir = RESULTS / "stoch"; out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for task in ("fi", "fu"):
        lm = json.load(open(RESULTS / task / "arcface_gt_landmarks.json"))
        base = RERUNS / "stoch" / task
        if not base.exists():
            continue
        for meth in sorted(os.listdir(base)):
            files = {}
            for f in os.listdir(base / meth):
                stem, k = f.rsplit("_k", 1)
                files.setdefault(stem, []).append(int(k.split(".")[0]))
            K = min(len(v) for v in files.values())
            stems = sorted(s for s, v in files.items() if len(v) >= K)
            P = np.full((len(stems), K), np.nan); L = P.copy(); I = P.copy(); D = []
            for i, st in enumerate(stems):
                fn = st + ".png"
                gt = load_rgb(DATA[task] / "gt" / fn); h = load_mask(task, fn); ic = load_rgb(DATA[task] / "masked" / fn)
                kp = lm[fn]["kps"]; ge = emb(gt, kp) if kp else None
                samples = []
                for j in range(K):
                    o = composite(ic, load_rgb(base / meth / f"{st}_k{j}.png"), h); samples.append(o)
                    P[i, j] = psnr(o, gt, h)
                    with torch.no_grad():
                        L[i, j] = lp(t(o), t(gt)).item()
                    if ge is not None:
                        I[i, j] = float(emb(o, kp) @ ge)
                with torch.no_grad():
                    D.append(np.mean([lp(t(a), t(b)).item() for a, b in itertools.combinations(samples, 2)]))
            res = {"task": task, "method": meth, "n_images": len(stems), "K": K}
            for nm, M in (("psnr_hole", P), ("lpips", L), ("id_aligned", I)):
                runs = np.nanmean(M, axis=0)                     # K method means
                ci = stats.t.interval(0.95, K - 1, loc=runs.mean(), scale=stats.sem(runs)) if K > 1 else (np.nan, np.nan)
                res[nm] = {"mean": float(np.nanmean(M)), "run_mean_std": float(runs.std(ddof=1)) if K > 1 else 0.0,
                           "run_ci95": [float(ci[0]), float(ci[1])], "per_image_std_mean": float(np.nanmean(np.nanstd(M, axis=1, ddof=1)))}
            res["diversity_pairwise_lpips"] = float(np.mean(D))
            json.dump(res, open(out_dir / f"{task}_{meth}.json", "w"), indent=1)
            summary.append({"task": task, "method": meth, "n_images": len(stems), "K": K,
                            **{f"{nm}_{k}": (json.dumps(v) if isinstance(v, list) else v) for nm in ("psnr_hole", "lpips", "id_aligned") for k, v in res[nm].items()},
                            "diversity": res["diversity_pairwise_lpips"]})
            print(task, meth, len(stems), K, f"PSNRhole {res['psnr_hole']['mean']:.2f} (run std {res['psnr_hole']['run_mean_std']:.3f}, per-img std {res['psnr_hole']['per_image_std_mean']:.2f})",
                  f"ID {res['id_aligned']['mean']:.3f} (per-img std {res['id_aligned']['per_image_std_mean']:.3f})", f"div {res['diversity_pairwise_lpips']:.4f}", flush=True)
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)


if __name__ == "__main__":
    main()
