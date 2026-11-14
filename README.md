# From Pixels to Faces — review artefacts and re-executable benchmark

Companion repository of the review

> M. Mahmoud, M. Abdalla, M. S. Kasem, A. Salem, H.-S. Kang. **From Pixels to Faces: A Comprehensive Survey of Deep Learning-Based Face Inpainting and Face Unmasking.** *Artificial Intelligence Review* (under review, revised version).

Repository: https://github.com/Mohamed-Abokhalil/From-Pixels-to-Faces · Archived code and artefacts (Zenodo concept DOI, all versions): https://doi.org/10.5281/zenodo.24190797 · Generated outputs (Zenodo concept DOI, all versions): https://doi.org/10.5281/zenodo.24193552 (CC BY-NC 4.0)

It contains everything needed to check the literature search and to regenerate every benchmark number, table and figure of the article from per-image results. The generated outputs of the benchmarked methods are archived in a separate Zenodo record (see below), not in git.

## Benchmarks

| | Face inpainting (FI) | Face unmasking (FU) |
|---|---|---|
| Images | 1,000 CelebA-HQ images (`25001.png`…, file `n` = CelebA-HQ image `n-1`), 256×256 | 500 CelebA images (renumbered `030000.png`…`030518.png`), 256×256 |
| Masks | QD-IMD irregular masks, six hole-ratio bins | MaskTheFace protective masks (surgical, cloth, N95, KN95) |
| Methods | 28 | 7 |

All test images were randomly selected from the CelebA test split (no seed was recorded); no training data were used, and every method was run on the same images with the same masks and protocol. Because the files were renumbered, membership of the official test partition (CelebA images 182638–202599) cannot be checked by filename; the exact filename lists are in `benchmark_v2/testsets/`.

Rule: only methods evaluated with checkpoints released by, or obtained from, the original authors; no model was trained or fine-tuned for the benchmark. Entries of the submitted version whose outputs came from models we had trained, or could not be traced to the official checkpoints, were removed (FI: MFCL, AttrFaceNet, MuFIN; FU: Din et al., MuFIN, DeepFillv2, GLCIC); the evidence is in `logs/PROVENANCE.md` and `benchmark_v2/configs/`.

GANMasker and M2UNet (FU) were developed by members of the author team (see the Competing interests statement of the article). Their code and checkpoints are not public; their outputs were produced by their original authors with the models of the published papers, and the checkpoints are available from the authors on request. The released outputs, masks and filename lists allow every metric to be recomputed.

## Contents

| Path | What |
|---|---|
| `literature/` | Scopus query and yearly counts (export as downloaded from Scopus in `exports/`, search run on 9 July 2026, parsed by `scopus_from_csv.py` into `scopus_counts.json`; Scopus gives no record-level export for this search), supplementary OpenAlex/arXiv search scripts and exports, eligibility criteria, de-duplication, the full screening log with a decision and reason per record (`screening/screening_log.csv`, single screener), the coded taxonomy (`taxonomy_methods.csv`) and its table generator |
| `benchmark_v2/testsets/` | Exact FI and FU test lists with mask ratio, bin and SHA-256 of every ground-truth, mask and masked image; repeated-sampling subsets |
| `data/{fi,fu}/masks/` | The benchmark masks (white = missing; every non-zero pixel is treated as missing) |
| `benchmark_v2/configs/` | One record per method: repository, commit, checkpoint files with SHA-256, the folder in which the stored outputs were produced (checked byte-for-byte), run settings, status after the audit; `online_resource_2.csv` summarises them |
| `benchmark_v2/rerun_code/` | Re-runs of VHII, Spa-former and Pluralistic with the benchmark masks, and the repeated-sampling runs (RePaint, Pluralistic) |
| `benchmark_v2/phase0_audit/` | Audit scripts (compositing, pairing, mask consistency, memorisation probe) |
| `benchmark_v2/*.py` | Evaluation (`eval_pixel.py`, `eval_fid.py`, `eval_arcface.py`, `eval_stoch.py`) and table generation (`make_tables.py`) |
| `results/` | Per-image scores of every method, ArcFace per-image scores, FID/KID with bootstrap s.d., summaries, old-vs-new comparison, statistics, repeated-sampling results |
| `figures_src/` | PRISMA diagram, roadmap, trend and qualitative figures |
| `logs/` | Provenance audit of every stored output (`PROVENANCE.md`) and the revision's decision log |
| `MANIFEST_outputs.csv` | SHA-256 of every generated output image |
| `zenodo_record.json`, `download_outputs.py` | Location and checksums of the archives with the generated outputs (not in git) |

## Getting the generated outputs

The outputs of all benchmarked methods (as scored, before recompositing; named by benchmark file) and the repeated-sampling samples are archived on Zenodo, one zip per method and task, at https://doi.org/10.5281/zenodo.24193552 (licence CC BY-NC 4.0; `SHA256SUMS.txt` lists the checksums). Download them from the record page or with

```bash
python download_outputs.py                     # all archives -> outputs/<task>/<method>/, samples_stoch/<task>/<method>/
python download_outputs.py --only outputs
python download_outputs.py --files outputs_fi_LaMa.zip outputs_fu_M2UNet.zip
```

## Reproducing the tables

1. Obtain CelebA and CelebA-HQ from their authors (non-commercial research terms) and place the ground-truth and masked images as listed in `benchmark_v2/testsets/*.csv`; verify them with the SHA-256 column. FI file `n.png` is CelebA-HQ image `n-1`.
2. Get the method outputs with `download_outputs.py` (or re-run each method with the repository and checkpoint in `benchmark_v2/configs/`).
3. `python benchmark_v2/eval_pixel.py --task all`, `python benchmark_v2/eval_fid.py --task all`, `python benchmark_v2/eval_arcface.py --task all`, `python benchmark_v2/eval_stoch.py`
4. `python benchmark_v2/make_tables.py` regenerates every benchmark table and number of the article (`bench_numbers.tex`, `tab_*_rows.tex`).

Environment: Python 3.10, PyTorch 2.5 (CUDA 12.1), see `requirements.txt`. Identity similarity uses the InsightFace `buffalo_l` models (non-commercial research licence). The scripts expect the original folder layout described in `benchmark_v2/common.py` (`DATA`, `ORIG`, `RERUNS`).

## Protocol in one paragraph

All outputs are recomposited with the observed pixels (Eq. 2 of the article) before scoring. PSNR/SSIM/L1 are reported on the full image, the hole and an 8-px boundary band; LPIPS (AlexNet), FID and KID (pytorch-fid Inception weights; bootstrap s.d.; KID over 100 subsets of 250) on full images. ArcFace identity similarity uses five landmarks detected on the ground truth with SCRFD and the same alignment for ground truth and output (`norm_crop`, 112×112), with the `w600k_r50` recognition model; a visible-region-only reference separates the identity carried by the visible pixels.

## Data licences

Code: MIT licence (`LICENSE`). Masks, test lists, scores and metadata: CC BY 4.0. The test images come from CelebA and CelebA-HQ (non-commercial research terms of their authors) and are not redistributed. The generated outputs are derived from these images and are archived separately under CC BY-NC 4.0 (https://doi.org/10.5281/zenodo.24193552); the method outputs were produced with checkpoints released by, or obtained from, the original authors, which have their own licences (see each repository in `benchmark_v2/configs/`).

## Citation

Please cite the article. To cite the released artefacts, see `CITATION.cff`, or

```bibtex
@software{mahmoud2026fpf_code,
  author    = {Mahmoud, Mohamed and Abdalla, Mahmoud and Kasem, Mahmoud SalahEldin and Salem, Ahmed and Kang, Hyun-Soo},
  title     = {From Pixels to Faces: review artefacts and re-executable face inpainting/unmasking benchmark},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24190797},
  url       = {https://github.com/Mohamed-Abokhalil/From-Pixels-to-Faces}
}

@dataset{mahmoud2026fpf_outputs,
  author    = {Mahmoud, Mohamed and Abdalla, Mahmoud and Kasem, Mahmoud SalahEldin and Salem, Ahmed and Kang, Hyun-Soo},
  title     = {Generated outputs of the face inpainting and face unmasking benchmark of "From Pixels to Faces"},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24193552}
}
```
