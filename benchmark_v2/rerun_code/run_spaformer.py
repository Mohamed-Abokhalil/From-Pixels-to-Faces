"""Re-run Spa-former (Huang et al., PR 2024) on the FI benchmark with the benchmark masks.

Why: the original outputs were produced with test option mask_type=[2,4] (masks generated at
random per image), so they do not correspond to the benchmark masks.
Fix: use the official code and the CelebA-HQ checkpoint unchanged; force mask_type=[3]
(external masks, paired by sorted index = same filenames), no shuffling, no visualizer.
Checkpoint: copy of FI/000_Related work/spa-former/checkpoints/celeba_hq (latest_net_G.pth
sha256 d12a94a0...b580d). Code is read in place; nothing is written into the repo.

Usage (from benchmark_v2):
  set PYTHONDONTWRITEBYTECODE=1
  python rerun_code/run_spaformer.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
REPO = ROOT / "FI" / "000_Related work" / "spa-former"
sys.path.insert(0, str(REPO))
os.chdir(REPO)                                   # repo uses relative imports of its packages

import torch  # noqa: E402

torch.manual_seed(0)
sys.argv = [sys.argv[0],
            "--name", "celeba_hq",
            "--checkpoints_dir", str(HERE / "reruns" / "_ckpt" / "spa-former"),
            "--img_file", str(ROOT / "data" / "CelebA_FI" / "gt"),
            "--mask_file", str(ROOT / "data" / "CelebA_FI" / "mask"),   # white = hole; loader uses (m == 0) = valid
            "--results_dir", str(HERE / "reruns" / "fi"),
            "--no_shuffle", "--nThreads", "0", "--gpu_ids", "0", "--display_id", "0"]

from options import test_options  # noqa: E402
from dataloader import data_loader  # noqa: E402
from model import create_model  # noqa: E402

opt = test_options.TestOptions().parse()
opt.mask_type = [3]
dataset = data_loader.dataloader(opt)
model = create_model(opt)
model.eval()
with torch.no_grad():
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
print("done", i + 1)
