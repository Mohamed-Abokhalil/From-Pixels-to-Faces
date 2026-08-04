# From Pixels to Faces: A Comprehensive Survey of Deep Learning-Based Face Inpainting and Face Unmasking

[![Paper](https://img.shields.io/badge/Paper-ACM%20CSUR-blue)]() [![Maintenance](https://img.shields.io/badge/Maintained-yes-green.svg)]() [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Companion repository for the survey **"From Pixels to Faces: A Comprehensive Survey of Deep Learning-Based Face Inpainting and Face Unmasking"** (submitted to ACM Computing Surveys, 2026).

> Mohamed Mahmoud, Mahmoud Abdalla, Mahmoud SalahEldin Kasem, Hyun-Soo Kang
> Chungbuk National University · Assiut University

## 📌 Highlights
- **70+ methods** (2017–2026) organized by architectural generation: GAN → Latent/Transformer → Diffusion
- **Dedicated face unmasking treatment**: 10 dedicated methods, unified benchmark
- **Unified benchmark**: 31 FI + 11 FU methods on CelebA, 256×256, with PSNR / SSIM / FID / LPIPS / L1 / **ArcFace ID-Sim**
- Identity-aware evaluation protocol and reproducible mask sets
- **Electronic supplement** with the full reported-metrics table and the complete chronological method taxonomy
- **Latest additions (2025–2026):** TransRef, Pyramid Fusion, SqSFill, NLMSA, MATdiff, FreeInpaint

## 🗂 Repository Structure
```
├── papers/          # Curated paper lists by generation and category
├── benchmark/       # Unified benchmark protocol, mask sets, ArcFace script
├── taxonomy/        # Live version of the survey taxonomy
└── assets/          # Figures and qualitative results
```

## 📚 Paper Lists
| Category | File |
|---|---|
| GAN-Based Face Completion (2017–2022) | [papers/01_gan_based.md](papers/01_gan_based.md) |
| Latent Space & Transformer (2020–2024) | [papers/02_latent_transformer.md](papers/02_latent_transformer.md) |
| Diffusion-Based (2022–present) | [papers/03_diffusion.md](papers/03_diffusion.md) |
| Face Unmasking (dedicated subtask) | [papers/04_face_unmasking.md](papers/04_face_unmasking.md) |
| Datasets & Tools | [papers/05_datasets.md](papers/05_datasets.md) |

> 📄 **Paper & supplement.** The article is split into a main manuscript (≤35 pp.) and an electronic supplement containing (i) the table of metrics originally reported by each method and (ii) the complete chronological taxonomy of all reviewed methods. This repository mirrors both.

## 🏆 Unified Benchmark (top-5 excerpt)
**Face Inpainting** — CelebA, 1K images, QD-IMD masks, 256×256:
| Method | PSNR↑ | SSIM↑ | FID↓ | ID-Sim↑ |
|---|---|---|---|---|
| MADIN (WACV'26) | **33.80** | 0.9525 | **5.59** | 0.9384 |
| SCAT (AAAI'23) | 33.57 | **0.9536** | 6.47 | 0.9177 |
| MISF (CVPR'22) | 33.45 | 0.9513 | 5.93 | 0.9165 |
| MADF (TIP'21) | 33.05 | 0.9508 | 6.48 | 0.9174 |
| CMT (ICCV'23) | 32.88 | 0.9472 | 6.45 | 0.9141 |

**Face Unmasking** — CelebA, 500 images, MaskTheFace, 256×256:
| Method | PSNR↑ | SSIM↑ | FID↓ | ID-Sim↑ |
|---|---|---|---|---|
| M2UNet (Mathematics'26) | **31.33** | **0.9577** | **7.64** | **0.7040** |
| GANMasker (Sensors'23) | 30.78 | 0.9518 | 10.10 | 0.6718 |
| MuFIN (TCSVT'24) | 29.88 | 0.9481 | 9.07 | 0.6488 |

Full tables: [benchmark/](benchmark/)

## 🔬 Reproduce the ArcFace ID-Sim Metric
```bash
pip install insightface onnxruntime-gpu opencv-python-headless tqdm
python benchmark/compute_arcface_similarity.py \
    --root /path/to/method_outputs --gt /path/to/ground_truth \
    --out results/arcface.json
```

## 📖 Citation
```bibtex
@article{mahmoud2026pixels,
  title   = {From Pixels to Faces: A Comprehensive Survey of Deep
             Learning-Based Face Inpainting and Face Unmasking},
  author  = {Mahmoud, Mohamed and
             Abdalla, Mahmoud and Kasem, Mahmoud SalahEldin and
             Kang, Hyun-Soo},
  journal = {ACM Computing Surveys},
  year    = {2026}
}
```

## 🤝 Contributing
New method? Missing paper? Open an issue or PR — see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License
MIT — see [LICENSE](LICENSE).
