"""Repeated sampling of RePaint (Lugmayr et al., CVPR 2022) on a stratified benchmark subset.

Official code (RePaint, commit 8e2a055) read in place; nothing written into it.
Config: the repo's face settings (confs/face_example.yml) with model_path forced to the official
CelebA-HQ checkpoint data/pretrained/celeba256_250000.pt: 1000 training steps respaced to 250,
resampling schedule t_T=250, jump_length=10, jump_n_sample=10, no classifier guidance.
Input: benchmark GT with the benchmark mask (keep-mask = 1 - hole), 256x256.
Seeds: for image index i (position in the sorted full test list) all K samples are drawn in one
batch after torch.manual_seed(100000 + i). Output: reruns/stoch/<task>/RePaint/<stem>_k<j>.png
(raw sampler output; recompositing is done by the evaluator).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
REPO = ROOT / "NII" / "RePaint"
sys.path.insert(0, str(REPO)); os.chdir(REPO)
import conf_mgt  # noqa: E402
from utils import yamlread  # noqa: E402
from guided_diffusion.script_util import (create_model_and_diffusion, model_and_diffusion_defaults,  # noqa: E402
                                          select_args)

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["fi", "fu"], required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--subset", default="stoch20_subset")
a = ap.parse_args()

data = ROOT / "data" / ("CelebA_FI" if a.task == "fi" else "CelebA_FU")
allfiles = sorted(os.listdir(data / "gt"))
subset = [l.strip() for l in open(HERE / "testsets" / f"{a.task}_{a.subset}.txt") if l.strip()]
if a.limit:
    subset = subset[:a.limit]
out = HERE / "reruns" / "stoch" / a.task / "RePaint"; out.mkdir(parents=True, exist_ok=True)

conf = conf_mgt.conf_base.Default_Conf(); conf.update(yamlread(str(REPO / "confs" / "face_example.yml")))
conf["model_path"] = "./data/pretrained/celeba256_250000.pt"
conf["show_progress"] = False
assert conf["schedule_jump_params"] == {"t_T": 250, "n_sample": 1, "jump_length": 10, "jump_n_sample": 10}
assert str(conf["timestep_respacing"]) == "250"
dev = "cuda"
model, diffusion = create_model_and_diffusion(**select_args(conf, model_and_diffusion_defaults().keys()), conf=conf)
model.load_state_dict(th.load(conf["model_path"], map_location="cpu")); model.to(dev).eval()


def model_fn(x, t, y=None, gt=None, **kw):
    return model(x, t, None, gt=gt)


for f in subset:
    stem = Path(f).stem
    outs = [out / f"{stem}_k{j}.png" for j in range(a.k)]
    if all(o.exists() for o in outs):
        continue
    img = np.asarray(Image.open(data / "gt" / f).convert("RGB"), np.float32) / 127.5 - 1
    hole = (np.asarray(Image.open(data / "mask" / f).convert("L")) > 0).astype(np.float32)
    keep = np.repeat((1 - hole)[None], 3, 0)
    gt = th.from_numpy(img).permute(2, 0, 1)[None].to(dev) * th.from_numpy(keep)[None].to(dev)
    km = th.from_numpy(keep)[None].to(dev)
    th.manual_seed(100000 + allfiles.index(f)); np.random.seed(allfiles.index(f))
    t0 = time.time()
    kw = {"gt": gt.repeat(a.k, 1, 1, 1), "gt_keep_mask": km.repeat(a.k, 1, 1, 1),
          "y": th.zeros(a.k, dtype=th.long, device=dev)}
    with th.no_grad():
        r = diffusion.p_sample_loop(model_fn, (a.k, 3, 256, 256), clip_denoised=conf.clip_denoised, model_kwargs=kw,
                                    cond_fn=None, device=dev, progress=False, return_all=True, conf=conf)
    x = ((r["sample"] + 1) * 127.5).clamp(0, 255).round().to(th.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for j, o in enumerate(outs):
        Image.fromarray(x[j]).save(o)
    print(f"{f}: {time.time() - t0:.1f}s for {a.k} samples", flush=True)
