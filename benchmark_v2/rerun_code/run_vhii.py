"""Re-run VHII (Campana et al., Computers & Graphics 2023; repo jose13579/variable-hyperparameter-image-impainting)
on the FI benchmark with each image's OWN benchmark mask.

Why: the official test_dataset.py draws a random mask per image (random.randint over the mask
folder), so the original outputs do not correspond to the benchmark masks.
Settings reproduced from test_dataset.py: 256x256, input in [-1,1], masked = x*(1-m),
model(masked, m), output composited with the input (inpainted*m + image*(1-m)).
Checkpoint: trained_models/celeba/celeba_VHII_efficient/best_model_celeba.pth (official CelebA
model), module model.VHII_efficient. Deterministic (no sampling).
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
REPO = ROOT / "FI" / "000_Related work" / "variable-hyperparameter-image-impainting"
CKPT = REPO / "trained_models" / "celeba" / "celeba_VHII_efficient" / "best_model_celeba.pth"
DATA = ROOT / "data" / "CelebA_FI"
OUT = HERE / "reruns" / "fi" / "VHII"

sys.path.insert(0, str(REPO)); os.chdir(REPO)
dev = "cuda"
g = importlib.import_module("model.VHII_efficient").InpaintGenerator().to(dev)
g.load_state_dict(torch.load(str(CKPT), map_location=dev)["netG"]); g.eval()
OUT.mkdir(parents=True, exist_ok=True)
files = sorted(f for f in os.listdir(DATA / "gt") if f.endswith(".png"))
with torch.no_grad():
    for f in files:
        img = np.asarray(Image.open(DATA / "gt" / f).convert("RGB").resize((256, 256)), np.float32) / 255
        m = (np.asarray(Image.open(DATA / "mask" / f).convert("L").resize((256, 256), Image.NEAREST)) > 0).astype(np.float32)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(dev) * 2 - 1
        mt = torch.from_numpy(m)[None, None].to(dev)
        y = (g(x * (1 - mt), mt) + 1) / 2
        y = y[0].permute(1, 2, 0).cpu().numpy() * 255
        out = (y.astype(np.uint8) * m[..., None] + np.round(img * 255).astype(np.uint8) * (1 - m[..., None])).astype(np.uint8)
        Image.fromarray(out).save(OUT / f)
print("done", len(files), "sha256(ckpt) see logs")
