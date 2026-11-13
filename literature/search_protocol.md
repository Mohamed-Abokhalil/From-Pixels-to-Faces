# Systematic search protocol (E1, R1.1, R2.2, R5.3b)

**Search window:** 1 Jan 2016 – 12 Nov 2026 (search date). Language: English.
**Unit of inclusion:** a publication proposing a method (one row in the taxonomy table).

## Concept blocks
- **A (task):** face inpainting · face completion · facial inpainting · facial image completion · face image inpainting · face de-occlusion · face unmasking · mask removal · masked face (reconstruction|restoration|completion|recovery) · occluded face (completion|restoration|reconstruction)
- **B (method):** deep learning · neural network · GAN / generative adversarial · transformer · diffusion · convolutional · autoencoder (used only where the database does not support A alone)

Final string = A AND (B) for Scopus/WoS/IEEE; A alone for OpenAlex/arXiv (title+abstract).

## 1. Scopus (run by the authors; export CSV+BibTeX with abstracts → `literature/exports/scopus_YYYYMMDD.csv`)
```
TITLE-ABS-KEY ( ( "face inpainting" OR "facial inpainting" OR "face completion" OR "facial image completion"
  OR "face image completion" OR "face image inpainting" OR "face de-occlusion" OR "face deocclusion"
  OR "face unmasking" OR "facial unmasking" OR ( "mask removal" AND face ) OR ( "masked face" AND ( reconstruct* OR restor* OR complet* OR recover* OR inpaint* ) )
  OR ( "occluded face" AND ( complet* OR restor* OR reconstruct* OR inpaint* ) ) )
  AND ( "deep learning" OR "neural network*" OR "generative adversarial" OR GAN OR transformer* OR diffusion OR convolutional OR autoencoder ) )
AND PUBYEAR > 2015 AND PUBYEAR < 2027 AND ( LIMIT-TO ( LANGUAGE , "English" ) )
```
## 2. IEEE Xplore (Command Search, "All Metadata"; export CSV → `exports/ieee_YYYYMMDD.csv`)
```
("face inpainting" OR "facial inpainting" OR "face completion" OR "facial image completion" OR "face de-occlusion"
 OR "face unmasking" OR "masked face reconstruction" OR "masked face restoration" OR "occluded face completion")
AND ("deep learning" OR "neural network" OR "generative adversarial" OR GAN OR transformer OR diffusion)
```
Filters: Publication year 2016–2026; Conferences + Journals + Early Access.
## 3. Web of Science Core Collection (Advanced; export "Full record" txt → `exports/wos_YYYYMMDD.txt`)
```
TS=("face inpainting" OR "facial inpainting" OR "face completion" OR "facial image completion" OR "face de-occlusion"
 OR "face unmasking" OR ("mask removal" AND face) OR ("masked face" AND (reconstruct* OR restor* OR complet* OR inpaint*))
 OR ("occluded face" AND (complet* OR restor* OR reconstruct* OR inpaint*)))
AND TS=("deep learning" OR "neural network*" OR "generative adversarial" OR GAN OR transformer* OR diffusion OR convolutional)
```
Timespan 2016-01-01 to 2026-11-12. (ACM DL may replace WoS if WoS is unavailable: same string on Title+Abstract, export BibTeX.)
## 4. OpenAlex (run by script `search_openalex.py`; free, reproducible) — title+abstract
Same block A, run as one boolean query on `title_and_abstract.search`, `from_publication_date:2016-01-01`, types article / proceedings-article / preprint.
## 5. arXiv (script `search_arxiv.py`) — cs.CV, title+abstract, block A.

## Supplementary identification (reported separately in PRISMA "other methods")
- Backward/forward citation chasing from included FU papers and from the 10 papers named by Reviewer 1.
- Papers named by reviewers (R1.3, R2.7, R4.3).
- Benchmark baselines (general inpainting methods run in our benchmark) are entered through this route and labelled "General inpainting (baseline)".
