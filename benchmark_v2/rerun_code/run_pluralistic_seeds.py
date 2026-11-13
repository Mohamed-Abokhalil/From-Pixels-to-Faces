"""Benchmark re-run (--subset all --k 1; original outputs failed the mask-consistency audit: 49.8%% of
hole pixels were exact GT copies, i.e. generated with other masks) and repeated sampling of Pluralistic (Zheng et al., CVPR 2019) on the 100-image stochastic subsets (R1.6b).

Official code (FI/000_Related work/Pluralistic) read in place; checkpoint checkpoints/celeba_random
(latest_net_E.pth, latest_net_G.pth). Network definitions and the sampling step reproduce
PluralisticModel.test(): z ~ q(z | masked image) from net_E, one decoder pass per sample, output
composited with the masked input. 256x256, benchmark mask per image.
Seeds: torch.manual_seed(100000 + index of the image in the sorted full test list); K samples drawn
sequentially after seeding. Output: reruns/stoch/<task>/Pluralistic/<stem>_k<j>.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
REPO = ROOT / "FI" / "000_Related work" / "Pluralistic"
sys.path.insert(0, str(REPO)); os.chdir(REPO)
from model import network  # noqa: E402
from util import task as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["fi", "fu"], required=True)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--subset", default="stoch_subset")
a = ap.parse_args()
dev = "cuda"
E = network.define_e(ngf=32, z_nc=128, img_f=128, layers=5, norm='none', activation='LeakyReLU', init_type='orthogonal', gpu_ids=[0])
G = network.define_g(ngf=32, z_nc=128, img_f=128, L=0, layers=5, output_scale=4, init_type='orthogonal', gpu_ids=[0])
for net, f in ((E, "latest_net_E.pth"), (G, "latest_net_G.pth")):
    sd = torch.load(REPO / "checkpoints" / "celeba_random" / f, map_location=dev)
    tgt = net.module if hasattr(net, "module") else net
    tgt.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
E.eval(); G.eval()

data = ROOT / "data" / ("CelebA_FI" if a.task == "fi" else "CelebA_FU")
allfiles = sorted(os.listdir(data / "gt"))
if a.subset == "all":   # benchmark mode: one seeded sample per image, saved as <file>.png
    subset, out = allfiles, HERE / "reruns" / a.task / "Pluralistic"
else:
    subset = [l.strip() for l in open(HERE / "testsets" / f"{a.task}_{a.subset}.txt") if l.strip()]
    out = HERE / "reruns" / "stoch" / a.task / "Pluralistic"
out.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    for f in subset:
        img = torch.from_numpy(np.asarray(Image.open(data / "gt" / f).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].to(dev)
        hole = torch.from_numpy((np.asarray(Image.open(data / "mask" / f).convert("L")) > 0).astype(np.float32))[None, None].to(dev)
        keep = (1 - hole).repeat(1, 3, 1, 1)
        im = keep * (img * 2 - 1)
        torch.manual_seed(100000 + allfiles.index(f))
        dist, feat = E(im)
        q = torch.distributions.Normal(dist[-1][0], dist[-1][1])
        sm = T.scale_img(keep, size=[feat[2].size(2), feat[2].size(3)])
        for j in range(a.k):
            g, _ = G(q.sample(), f_m=feat[-1], f_e=feat[2], mask=sm.chunk(3, dim=1)[0])
            y = (((1 - keep) * g[-1] + keep * im) + 1) / 2
            Image.fromarray((y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(out / (f if a.subset == "all" else f"{Path(f).stem}_k{j}.png"))
print("done", a.task, len(subset))
