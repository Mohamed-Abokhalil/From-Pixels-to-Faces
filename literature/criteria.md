# Eligibility criteria and screening procedure

## Inclusion
- **I1 Publication type.** Peer-reviewed journal article or peer-reviewed conference/workshop paper; arXiv-only works are recorded separately and labelled *preprint* (they are never the sole support for a claim).
- **I2 Task.** Proposes a deep-learning method whose input is a face image with a known or estimated missing/occluded region and whose output fills that region (face inpainting, face completion, face de-occlusion, face unmasking), **or** a general image-inpainting method that reports results on a face benchmark (CelebA, CelebA-HQ, FFHQ).
- **I3 Period/language.** 1 Jan 2016 – 12 Nov 2026; English.

## Exclusion (one code per excluded record, first applicable)
| Code | Meaning |
|---|---|
| E-TYPE | Not a primary research article (thesis, dataset/software record, editorial, erratum, abstract-only, report) |
| E-REVIEW | Survey/review (kept as a separate list for Table 1) |
| E-DOMAIN | Not about face images |
| E-TASK | Face task without mask-conditioned completion as the contribution (recognition, detection, FER, alignment, swapping/reenactment, editing, unconditional generation, 3D/texture reconstruction) |
| E-NONDL | No learned model (PCA, sparse coding, patch/PDE only) — background only |
| E-NOTMETHOD | Evaluation/comparison/application study proposing no new completion method (may be cited in discussion) |
| E-BFR | Blind face restoration without a mask (cited as related work, not counted) |
| E-ACCESS | Neither abstract nor full text available to judge |

## Classification of included studies
- **FI-specific**: designed for faces; occlusions other than protective masks (irregular holes, components, glasses, HMDs, hands).
- **FU-specific**: designed for removing protective/face masks (surgical, cloth, N95, synthetic MaskTheFace).
- **Blind face restoration (related)**: restoration priors (GFP-GAN, CodeFormer, DifFace, …) used as related methods/baselines; not counted as FI.
- **General inpainting (baseline)**: designed for general scenes; included only as benchmark baselines or when evaluated on faces.

## Hybrid assignment rule
The chapter is decided by the **primary generative mechanism that produces the filled pixels** (adversarial CNN decoder → GAN family; pre-trained StyleGAN/VQ codebook or Transformer token prediction → latent/Transformer family; iterative denoising → diffusion family). Secondary mechanisms (attention, landmarks, parsing, references, text, audio) are recorded as attributes in the taxonomy table, not as chapters.

## Procedure
1. Records from all databases are merged; duplicates removed by DOI, then normalised title (`dedup.py`); peer-reviewed versions preferred over preprints.
2. First-pass title/abstract screen by the assistant with a logged reason (`screen_first_pass.py`, manual overrides in `overrides.json`). **This is not a substitute for human screening.**
3. Two authors (screeners A and B) independently screen the blind sheets `screening/sheet_A.csv`, `sheet_B.csv` (all records, no first-pass decision shown). Agreement: Cohen's κ (`kappa.py`). Disagreements resolved by discussion with a third author; the resolution is logged.
4. Full-text eligibility for records retained at stage 3; reasons for full-text exclusion logged.
5. Included studies → taxonomy table (Online Resource 1, complete) → studies discussed in detail in the main text: every included study that is (a) peer-reviewed, and (b) either introduces a mechanism not already represented, is evaluated in our benchmark, has ≥ 1 citation per year since publication (OpenAlex), or was published 2025–2026 (recency).
