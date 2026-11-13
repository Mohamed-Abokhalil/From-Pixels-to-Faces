"""Redrawn Figure 2 (R3.1, R4.1, R4.2, R5.1a-b): (A) roadmap of architectural families as parallel
lanes on a publication-year axis, with arrows for mechanisms that one family borrowed from another
(no arrow implies that a later family superseded an earlier one); (B) the five-dimensional taxonomy
with every label listed exactly once. Every method/year below is taken from the verified bibliography.
Output: FI_FU_Survey_AIR_revised/fig_evolution_taxonomy.pdf (vector) + figures_src/out/fig_roadmap.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REV = Path(__file__).resolve().parents[1]
LANES = [  # (lane label, colour, [(year, method, x-offset)])
    ("CNN/GAN\nencoder-decoder", "#4C78A8", [(2016, "Context Enc.", 0), (2017, "GFC", 0), (2018, "MCGAN", 0), (2020, "Symmetry-CNN", 0), (2022, "DE-GAN", 0), (2025, "PKDFIN", 0)]),
    ("Structure / geometry\nguided GAN", "#72B7B2", [(2018, "Sun et al.", -0.15), (2019, "LaFIn", 0), (2019.55, "GeoFace", 0), (2022, "3DFaceFill", 0), (2024, "SFI-Swin", 0)]),
    ("Reference / attribute /\nsketch conditioning", "#54A24B", [(2018, "Eye in-painting", 0), (2019, "SC-FEGAN", 0.15), (2021, "AttrFaceNet", 0), (2023, "RFTB", 0), (2025, "EXE-GAN", 0)]),
    ("Pre-trained GAN prior\n(StyleGAN)", "#EECA3B", [(2021, "GPEN / GFP-GAN", 0), (2022, "He et al.", 0.1), (2024, "E2F-Net", 0)]),
    ("Codebook / Transformer", "#F58518", [(2021, "ICT", -0.1), (2022, "CodeFormer", 0), (2023, "PatMAT / NDMA", 0), (2024, "MuFIN", 0), (2025, "IN$^2$", -0.1), (2026, "MADIN*", 0)]),
    ("Diffusion", "#E45756", [(2022, "RePaint*", 0), (2023, "PGDiff", 0), (2024, "PVA", 0), (2025, "RAD*", 0)]),
    ("Multimodal (text / audio)\nconditioning", "#B279A2", [(2024, "Audio-Face, MuFIN", 0), (2025, "InpaintFormer", 0), (2026, "LLM-Inpainter", 0)]),
    ("Face unmasking\n(protective masks)", "#9D755D", [(2020, "Din et al.", 0), (2022, "MRA", 0), (2023, "GANMasker$^\\dagger$, SAM C-GAN", 0), (2024, "RAFU (preprint)", 0), (2026, "M2UNet$^\\dagger$", 0)]),
]
LINKS = [  # (from lane, from year, to lane, to year, label) -- only mechanisms stated in the cited papers
    (0, 2017, 1, 2019, "parsing loss -> explicit landmark prior"),
    (3, 2021, 4, 2022, "pre-trained face prior"),
    (2, 2023, 5, 2024, "reference-image identity"),
    (4, 2024, 6, 2024, "cross-attention conditioning"),
]


def panel_a(ax):
    n = len(LANES)
    ax.set_xlim(2012.6, 2026.8); ax.set_ylim(-0.9, n - 0.4); ax.invert_yaxis()
    for i, (lab, col, pts) in enumerate(LANES):
        ax.axhspan(i - 0.45, i + 0.45, color=col, alpha=0.10, lw=0)
        ax.text(2012.7, i, lab, va="center", ha="left", fontsize=6.6, color="#222", fontweight="bold")
        for k, (y, m, dx) in enumerate(pts):
            ax.plot(y, i, "o", ms=4, color=col, mec="#333", mew=0.4, zorder=3)
            above = (k % 2 == 0)   # alternate labels above/below the lane centre to avoid collisions
            ax.text(y + dx, i - 0.12 if above else i + 0.12, m, ha="center", va="bottom" if above else "top",
                    fontsize=5.6, zorder=4)
    for a, ya, b, yb, lab in LINKS:
        ax.annotate("", xy=(yb, b - 0.05), xytext=(ya, a + 0.05),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555", connectionstyle="arc3,rad=-0.25"))
        ax.text(max(ya, yb) + 0.12, (a + b) / 2, lab, fontsize=5, color="#444", style="italic", ha="left", va="center")
    ax.set_xticks(range(2016, 2027)); ax.tick_params(axis="x", labelsize=6.5); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Year of publication (lanes are concurrent families, not successive generations)", fontsize=6.8)
    ax.set_title("(A) Architectural families of deep face inpainting and unmasking, and mechanisms they share", fontsize=8, loc="left", fontweight="bold")
    ax.text(2026.8, -0.75, "* general inpainting method evaluated on faces;  $\\dagger$ developed by members of this author team",
            fontsize=5.3, ha="right", color="#444")


TAX = [
    ("1. Inpainting scenario", ["Irregular free-form holes", "Component-shaped masks", "Lower-face (protective mask)", "Large / hemi-face occlusion", "Real occluders (glasses, HMD)"]),
    ("2. Guidance modality", ["None (image + mask only)", "Landmarks", "Parsing / segmentation", "Attributes", "Reference image(s)", "Sketch / colour strokes", "Text", "Audio", "3D / UV prior"]),
    ("3. Generative paradigm", ["Adversarial CNN", "Pre-trained GAN prior", "Codebook / Transformer", "Diffusion"]),
    ("4. Technical component", ["Mask-aware convolution", "Attention / cross-attention", "Multi-stage coarse-to-fine", "Identity / perceptual losses", "Personalization / fine-tuning"]),
    ("5. Evaluation aspect", ["Distortion (PSNR, SSIM, L1)", "Perceptual (LPIPS, FID, KID)", "Identity (ArcFace similarity)", "Diversity / stochasticity", "User studies"]),
]


def panel_b(ax):
    ax.axis("off"); ax.set_xlim(0, 5); ax.set_ylim(0, 1)
    cols = ["#4C78A8", "#54A24B", "#F58518", "#B279A2", "#E45756"]
    ax.set_title("(B) Five-dimensional taxonomy (each method in the complete taxonomy table is coded on every dimension)", fontsize=8, loc="left", fontweight="bold")
    for j, (head, items) in enumerate(TAX):
        x = j + 0.04
        ax.add_patch(FancyBboxPatch((x, 0.86), 0.92, 0.11, boxstyle="round,pad=0.005", fc=cols[j], ec="none"))
        ax.text(x + 0.46, 0.915, head, ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
        for k, it in enumerate(items):
            y = 0.76 - k * 0.085
            ax.add_patch(FancyBboxPatch((x + 0.04, y - 0.03), 0.84, 0.065, boxstyle="round,pad=0.004", fc="white", ec=cols[j], lw=0.6))
            ax.text(x + 0.46, y + 0.0025, it, ha="center", va="center", fontsize=5.7)


def main():
    fig = plt.figure(figsize=(7.2, 8.2))
    ax1 = fig.add_axes([0.02, 0.47, 0.96, 0.50]); panel_a(ax1)
    ax2 = fig.add_axes([0.02, 0.0, 0.96, 0.38]); panel_b(ax2)
    out = REV / "FI_FU_Survey_AIR_revised" / "fig_evolution_taxonomy.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(REV / "figures_src" / "out" / "fig_roadmap.png", dpi=220, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
