"""Shared paths, method registry and image helpers for the FPFS benchmark v2.

All source data are read in place (read-only) from the original study folders.
Everything derived is written under FPFS_Revision/benchmark_v2/.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REV = HERE.parent                         # FPFS_Revision
ROOT = REV.parent                         # main directory (read-only)
DATA = {"fi": ROOT / "data" / "CelebA_FI", "fu": ROOT / "data" / "CelebA_FU"}
ORIG = {"fi": ROOT / "FI", "fu": ROOT / "FU"}      # MD5-identical originals
RERUNS = HERE / "reruns"                  # outputs we regenerated (Phase 2)
RESULTS = HERE / "results"
SIZE = 256
MASK_THRESHOLD = 0                         # mask > 0 -> hole. FU masks are anti-aliased; any non-zero pixel carries mask texture in I_c
BAND_RADIUS = 4                           # boundary band = dilate(M, r) & ~erode(M, r): 2r = 8 px wide


@dataclass
class Method:
    name: str                 # display name used in tables
    folder: str               # sub-folder of SOTA/ (or of reruns/<task>/)
    year: int
    cite: str                 # bibtex key
    category: str             # FI-specific | FU-specific | BFR (related) | General inpainting
    family: str               # GAN/CNN | Transformer | Latent/VQ | StyleGAN prior | Diffusion
    stochastic: bool = False
    team: bool = False        # developed by members of the author team
    status: str = "ok"        # ok | rerun | excluded
    note: str = ""
    name_map: Optional[Callable[[List[str]], Dict[str, str]]] = field(default=None, repr=False)


def _tsformer_map(gt_files: List[str]) -> Dict[str, str]:
    """TSFormer wrote outputs as NNN_im.png in the order of the sorted GT list.

    Verified by image matching (benchmark_v2/phase0_audit): 1000/1000 outputs match
    GT index NNN-1 with median abs. difference 0. The renamed copy in
    data/CelebA_FI/SOTA/TSFormer is mis-paired and must not be used.
    """
    return {g: f"{i + 1:03d}_im.png" for i, g in enumerate(gt_files)}


FI_METHODS: List[Method] = [
    Method("EdgeConnect", "EdgeConnect", 2019, "nazeri2019edgeconnect", "General inpainting", "GAN/CNN"),
    Method("StructureFlow", "StructureFlow", 2019, "ren2019structureflow", "General inpainting", "GAN/CNN"),
    Method("Pluralistic", "Pluralistic", 2019, "zheng2019pluralistic", "General inpainting", "GAN/CNN", stochastic=True,
           status="rerun", note="original outputs failed the mask-consistency audit (49.8% of hole pixels = GT); re-run with benchmark masks, one seeded sample (reruns/fi/Pluralistic)"),
    Method("DeepFillv2", "DeepFillv2", 2019, "yu2019free", "General inpainting", "GAN/CNN"),
    Method("PEN-Net", "PEN-Net", 2019, "zeng2019learning", "General inpainting", "GAN/CNN"),
    Method("LaFIn", "LaFIn", 2019, "yang2019lafin", "FI-specific", "GAN/CNN"),
    Method("E2I", "E2I", 2020, "xu2020e2i", "General inpainting", "GAN/CNN"),
    Method("MADF", "MADF", 2021, "zhu2021image", "General inpainting", "GAN/CNN"),
    Method("VHII", "VHII", 2023, "campana2023variable", "General inpainting", "Transformer",
           status="rerun", note="original outputs used randomly drawn masks (test_dataset.py); re-run with benchmark masks and the official CelebA checkpoint (reruns/fi/VHII)"),
    Method("WaveFill", "WaveFill", 2021, "yu2021wavefill", "General inpainting", "GAN/CNN"),
    Method("BAT-Fill", "BAT-Fill", 2021, "yu2021diverse", "General inpainting", "Transformer", stochastic=True),
    Method("latent-code", "latent-code", 2021, "liu2021pd", "General inpainting", "Latent/VQ"),
    Method("CoModGAN", "CoModGAN", 2021, "zhao2021large", "General inpainting", "StyleGAN prior", stochastic=True),
    Method("CTSDG", "CTSDG", 2021, "guo2021image", "General inpainting", "GAN/CNN"),
    Method("AttrFaceNet", "AttrFaceNet", 2021, "xiao2021face", "FI-specific", "GAN/CNN",
           status="excluded", note="outputs come from a model trained by us (RelatedWorks/AttrFaceNet/checkpoints: config MODE 1, training log from iteration 1, 2025-09-25/26), not from the released checkpoint; excluded under the official-checkpoint rule"),
    Method("GPEN", "GPEN", 2021, "yang2021gan", "BFR (related)", "StyleGAN prior"),
    Method("MISF", "MISF", 2022, "li2022misf", "General inpainting", "GAN/CNN"),
    Method("LaMa", "LaMa", 2022, "suvorov2022resolution", "General inpainting", "GAN/CNN"),
    Method("DGTS", "DGTS", 2022, "liu2022delving", "General inpainting", "GAN/CNN"),
    Method("RePaint", "RePaint", 2022, "lugmayr2022repaint", "General inpainting", "Diffusion", stochastic=True),
    Method("CodeFormer", "CodeFormer", 2022, "zhou2022towards", "BFR (related)", "Latent/VQ"),
    Method("MFCL", "MFCL", 2022, "lin2022multi", "General inpainting", "GAN/CNN",
           status="excluded", note="no official pretrained model was released (README test section has empty checkpoint paths); the outputs are identical to MFCL_30, written by a model we trained for 30 epochs (checkpoints dated 2025-09/10); excluded under the official-checkpoint rule"),
    Method("AOT-GAN", "AOT-GAN", 2022, "zeng2022aggregated", "General inpainting", "GAN/CNN"),
    Method("SCAT", "SCAT", 2023, "zuo2023generative", "General inpainting", "GAN/CNN"),
    Method("CMT", "CMT", 2023, "ko2023continuously", "General inpainting", "Transformer"),
    Method("CoPaint", "CoPaint", 2023, "zhang2023towards", "General inpainting", "Diffusion", stochastic=True),
    Method("NDMA", "NDMA", 2023, "phutke2023nested", "FI-specific", "Transformer"),
    Method("MuFIN", "MuFIN", 2024, "zhan2024learning", "FI-specific", "GAN/CNN",
           status="excluded", note="outputs identical to RelatedWorks/MuFIN/checkpoints/results, written by a model we trained (config MODE 1, 40 epochs, 161,500 iterations, log of 2025-09-16), not the released BaiduNetdisk checkpoint; excluded under the official-checkpoint rule"),
    Method("Spa-former", "spa-former", 2024, "huang2024sparse", "General inpainting", "Transformer",
           status="rerun", note="original outputs used generated random masks (mask_type [2,4]); re-run with benchmark masks and the official CelebA-HQ checkpoint (reruns/fi/spa-former)"),
    Method("TSFormer", "TSFormer", 2024, "lin2024tsformer", "General inpainting", "Transformer",
           note="read from FI/SOTA/TSFormer with recovered NNN_im.png pairing", name_map=_tsformer_map),
    Method("MADIN", "MADIN", 2026, "oh2026learning", "General inpainting", "Transformer"),
]

FU_METHODS: List[Method] = [
    Method("GLCIC", "GLCIC", 2017, "iizuka2017globally", "General inpainting", "GAN/CNN",
           status="excluded", note="outputs come from the comparison set of our earlier unmasking papers (Face_Unmasking/00_SOTA/GLCIC); they cannot be traced to the official Torch7 release (no run of it exists; local .t7 files are empty placeholders); excluded under the official-checkpoint rule"),
    Method("EdgeConnect", "EdgeConnect", 2019, "nazeri2019edgeconnect", "General inpainting", "GAN/CNN"),
    Method("DeepFillv2", "DeepFillv2", 2019, "yu2019free", "General inpainting", "GAN/CNN",
           status="excluded", note="outputs (Face_Unmasking/00_SOTA/GatedConv) are not reproduced by the official CelebA-HQ DeepFillv2 checkpoint (mean hole difference 25-34 grey levels on two test images); model of unknown provenance; excluded under the official-checkpoint rule"),
    Method("Din et al.", "Din2020", 2020, "din2020novel", "FU-specific", "GAN/CNN",
           status="excluded", note="no official code or checkpoint was released; the outputs (Face_Unmasking/00_SOTA/GUMF) come from a re-implementation trained by us (kaylode/facemask-removal, local weights stage2/model_99_99000.pth); excluded under the official-checkpoint rule"),
    Method("CoModGAN", "CoModGAN", 2021, "zhao2021large", "General inpainting", "StyleGAN prior", stochastic=True),
    Method("RePaint", "RePaint", 2022, "lugmayr2022repaint", "General inpainting", "Diffusion", stochastic=True),
    Method("GANMasker", "GANMasker", 2023, "mahmoud2023ganmasker", "FU-specific", "GAN/CNN", team=True),
    Method("SCAT", "SCAT", 2023, "zuo2023generative", "General inpainting", "GAN/CNN"),
    Method("CoPaint", "CoPaint", 2023, "zhang2023towards", "General inpainting", "Diffusion", stochastic=True),
    Method("MuFIN", "MuFIN", 2024, "zhan2024learning", "FI-specific", "GAN/CNN",
           status="excluded", note="the only MuFIN model in the workspace is the one we trained (see FI); the FU outputs (Face_Unmasking/00_SOTA/MuFIN) cannot come from the released checkpoint; excluded under the official-checkpoint rule"),
    Method("M2UNet", "M2UNet", 2026, "mahmoud2026m2unet", "FU-specific", "GAN/CNN", team=True),
]

METHODS = {"fi": FI_METHODS, "fu": FU_METHODS}


def gt_files(task: str) -> List[str]:
    return sorted(f for f in os.listdir(DATA[task] / "gt") if f.lower().endswith(".png"))


def output_path(task: str, m: Method, fname: str, files: List[str]) -> Path:
    rerun = RERUNS / task / m.folder
    if rerun.is_dir():
        return rerun / fname
    if m.name_map is not None:
        return ORIG[task] / "SOTA" / m.folder / m.name_map(files)[fname]
    return DATA[task] / "SOTA" / m.folder / fname


def has_rerun(task: str, m: Method) -> bool:
    return (RERUNS / task / m.folder).is_dir()


def load_rgb(p: Path) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_mask(task: str, fname: str) -> np.ndarray:
    m = np.asarray(Image.open(DATA[task] / "mask" / fname).convert("L"))
    if m.shape != (SIZE, SIZE):
        m = np.asarray(Image.open(DATA[task] / "mask" / fname).convert("L").resize((SIZE, SIZE), Image.NEAREST))
    return (m > MASK_THRESHOLD)          # True = hole


def composite(masked_input: np.ndarray, out: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Eq. (2): I_hat = I_c * (1 - M) + f(I_c, M) * M, with I_c = the benchmark's masked input."""
    h = hole[..., None].astype(np.float32)
    return masked_input * (1.0 - h) + out * h


def band(hole: np.ndarray, r: int = BAND_RADIUS) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion
    st = np.ones((2 * r + 1, 2 * r + 1), bool)
    return binary_dilation(hole, st) & ~binary_erosion(hole, st, border_value=0)


def bin_of(ratio: float) -> str:
    edges = [(0.1, "0-10"), (0.2, "10-20"), (0.3, "20-30"), (0.4, "30-40"), (0.5, "40-50")]
    for e, lab in edges:
        if ratio < e:
            return lab
    return ">50"
