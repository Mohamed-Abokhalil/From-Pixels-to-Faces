"""Memorisation probe (train/test overlap). For a checkpoint that never saw an image, the content of a fully hidden lower half
cannot be reproduced. We hide the entire lower half (rows 128-255) of N test images and measure hole PSNR / ID-Sim against
the ground truth, for Spa-former (CelebA-HQ checkpoint) and, as a reference, Pluralistic (CelebA checkpoint) and VHII.
Compared on FI images (CelebA-HQ-indexed, 25001-29993) and FU images (CelebA ids 030000+, ground truths unmasked).
Output: phase0_audit/memorization_probe.json and a montage."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import DATA, ROOT  # noqa: E402

dev = "cuda"
FIR = ROOT / "FI" / "000_Related work"


def spaformer():
    sys.path.insert(0, str(FIR / "spa-former")); cwd = os.getcwd(); os.chdir(FIR / "spa-former")
    from model import netU_spa_former
    g = netU_spa_former.define_g(gpu_ids=[0])
    sd = torch.load(HERE / "reruns" / "_ckpt" / "spa-former" / "celeba_hq" / "latest_net_G.pth", map_location=dev)
    (g.module if hasattr(g, "module") else g).load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
    g.eval(); os.chdir(cwd); sys.path.pop(0)
    for k in [k for k in sys.modules if k == "model" or k.startswith("model.") or k == "util" or k.startswith("util.")]:
        del sys.modules[k]

    def run(img01, hole):  # PluralisticModel.test(): mask = 1 keep
        keep = 1 - hole; t = img01 * 2 - 1
        y = g(keep * t, keep)
        return ((y * (1 - keep) + t * keep) + 1) / 2
    return run


def vhii():
    import importlib
    sys.path.insert(0, str(FIR / "variable-hyperparameter-image-impainting")); cwd = os.getcwd(); os.chdir(FIR / "variable-hyperparameter-image-impainting")
    g = importlib.import_module("model.VHII_efficient").InpaintGenerator().to(dev)
    g.load_state_dict(torch.load("trained_models/celeba/celeba_VHII_efficient/best_model_celeba.pth", map_location=dev)["netG"]); g.eval()
    os.chdir(cwd); sys.path.pop(0)

    def run(img01, hole):
        x = img01 * 2 - 1; y = (g(x * (1 - hole), hole) + 1) / 2
        return y * hole + img01 * (1 - hole)
    return run


def main():
    rng = np.random.default_rng(0)
    sets = {"fi": sorted(rng.choice(sorted(os.listdir(DATA["fi"] / "gt")), 40, replace=False)),
            "fu": sorted(rng.choice(sorted(os.listdir(DATA["fu"] / "gt")), 40, replace=False))}
    hole = torch.zeros(1, 1, 256, 256, device=dev); hole[..., 128:, :] = 1
    res = {}; tiles = []
    for name, ctor in (("Spa-former", spaformer), ("VHII", vhii)):
        run = ctor()
        for task, files in sets.items():
            ps = []
            for k, f in enumerate(files):
                img = torch.from_numpy(np.asarray(Image.open(DATA[task] / "gt" / f).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].to(dev)
                with torch.no_grad():
                    o = run(img, hole).clamp(0, 1)
                mse = float(((o - img)[..., 128:, :] ** 2).mean()); ps.append(10 * np.log10(1 / mse))
                if k < 4:
                    tiles.append((name, task, (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), (o[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)))
            res[f"{name}_{task}"] = {"n": len(ps), "lower_half_psnr_mean": float(np.mean(ps)), "std": float(np.std(ps))}
            print(name, task, res[f"{name}_{task}"], flush=True)
    json.dump(res, open(HERE / "phase0_audit" / "memorization_probe.json", "w"), indent=1)
    W = Image.new("RGB", (256 * 8, 256 * 2 * 2), "white")
    for i, (name, task, gt, out) in enumerate(tiles):
        r, c = divmod(i, 4)
        W.paste(Image.fromarray(gt), (c * 512, r * 256)); W.paste(Image.fromarray(out), (c * 512 + 256, r * 256))
    W.save(HERE / "phase0_audit" / "memorization_probe.jpg", quality=90)


if __name__ == "__main__":
    main()
