# Decisions log

| ID | Date | Decision | Rationale | Status |
|---|---|---|---|---|
| D-01 | 2026-11-12 | Evaluated data are `data\CelebA_FI` / `data\CelebA_FU` (MD5-identical to `FI\`/`FU\`), read in place | These are what the original scripts used | adopted |
| D-02 | 2026-11-12 | E2 defended (CV proceedings are archival venues) + venue-type column and distribution | Revision prompt Phase 1.5 | adopted |
| D-03 | 2026-11-12 | Tyagi et al. (R2.7): none meets the inclusion criteria (detection/recognition, no completion). **Authors decide** whether to cite any for a specific recognition statement; otherwise the response explains scope | Integrity rule; avoid citations that support no statement | **resolved 2026-11-13: cite all four in the Introduction (authors)** |
| D-04 | 2026-11-12 | Fig. 2 redrawn from scratch as scripted vector figure (`figures_src/fig_roadmap.py`); arrows only for mechanisms stated in cited papers | No editable source; PNG had many errors | adopted |
| D-05 | 2026-11-12 | ArcFace protocol: SCRFD landmarks on GT only, identical `norm_crop` for GT and output; legacy (unaligned) scores kept in Online Resource 3 | R1.6d/R5.4b | adopted |
| D-06 | 2026-11-12 / 2026-11-13 | Release mirrors the NII revision (authors): GitHub = code, configs, lists, masks, scores, logs (MIT for code); generated outputs in a separate Zenodo dataset record, one zip per method + SHA256SUMS + archives.json, CC BY-NC 4.0 (CelebA non-commercial terms), linked to the software record; `memorization_probe.jpg` (shows CelebA faces) kept out of GitHub | CelebA licence; NII precedent | adopted |
| D-07 | 2026-11-12 | FU masks binarised at `mask > 0` (anti-aliased border pixels carry mask texture in the masked input) | Measured: visible-pixel error of masked input 0.016/255 at >0 vs 0.246/255 at >127 | adopted |
| D-08 | 2026-11-12 | TSFormer pairing recovered by image matching (1000/1000 exact); VHII, Spa-former, Pluralistic re-run with benchmark masks; MFCL excluded | Phase-2 audit (mask-consistency 49.8%/74.3% exact-GT hole pixels for Pluralistic/MFCL; random-mask test scripts) | adopted |
| D-09 | 2026-11-12 | Repeated sampling: Pluralistic 10 samples × 100 images per task; RePaint 5 × 10 images per task; CoPaint single sample | RePaint measured at ~4 min/sample on the shared RTX 3090 | adopted |
| D-10 | 2026-11-12 | FID uncertainty reported as bootstrap s.d. (not percentile CI); KID mean ± std as unbiased alternative | Observed bootstrap bias (FID 6.9 vs bootstrap percentiles 20–23) | adopted |
| D-11 | 2026-11-12 / 2026-11-13 | Database search = Scopus only (authors; query and yearly counts supplied); OpenAlex + arXiv scripted search kept as supplementary arm ("other methods" in PRISMA) | Authors' answer N-01 | adopted |
| D-12 | 2026-11-12 | Methods discussed in detail = peer-reviewed + (new mechanism or benchmarked or 2025–26); full set in taxonomy table | 272 records retained after first pass cannot all be discussed individually | adopted |
| D-13 | 2026-11-12 / 2026-11-13 | latexdiff: authors permitted installing Algorithm::Diff | Tooling permission | see CHANGES |
| D-14 | 2026-11-13 | Benchmark rule: only methods evaluated with the models released by their authors; no model trained or fine-tuned by us. Excluded after the provenance audit: FI MFCL, AttrFaceNet, MuFIN; FU Din et al., MuFIN, DeepFillv2, GLCIC (`logs/PROVENANCE.md`) | Authors' rule N-04 | adopted |
| D-15 | 2026-11-13 | Screening by a single screener; no κ; 211 retained records without full-text decision are reported as such, not given invented reasons | Authors' answer N-06; integrity rule | adopted |
| D-16 | 2026-11-13 | No AI-use declaration in the manuscript (authors' decision N-09) | Authors | adopted |
| D-17 | 2026-11-13 | Zenodo: software = new version of concept 10.5281/zenodo.24190797 created by a GitHub release (integration, as NII); outputs = separate dataset record (API draft, published after approval); manuscript cites concept DOIs (authors) | Authors / NII precedent | adopted |
| D-18 | 2026-11-14 | Scopus: run date 9 July 2026; the yearly-count export is the source of the counts; no record-level export, so the database arm of PRISMA is counts-only (limitation) | Authors' answer N-01 | adopted |
| D-19 | 2026-11-14 | Test split: stated as randomly selected from the CelebA test split, not verified by filename (files renumbered, no mapping); training-overlap limitation added | Authors' answer N-07; integrity rule | adopted |
| D-20 | 2026-11-14 | Benchmark rule worded 'checkpoints released by, or obtained from, the original authors'; GANMasker/M2UNet checkpoints available on request | Authors' answer N-15 | adopted |
