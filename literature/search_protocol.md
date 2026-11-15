# Search protocol as carried out

**Unit of inclusion:** a publication proposing a method (one row in the taxonomy table). Language: English.

## 1. Database search: Scopus (run by the authors on 9 July 2026)
```
(TITLE-ABS-KEY( "face inpainting" OR "face completion" OR "facial inpainting" OR "face image inpainting"
  OR "face image completion" OR "face unmasking" OR "face mask removal" OR "masked face restoration"
  OR "face occlusion removal" OR "face mask inpainting" OR "face demasking" OR "unmasking face" )
 AND PUBYEAR > 2015 AND PUBYEAR < 2027 AND (SRCTYPE(j) OR SRCTYPE(p)))
```
Source types: journals (`j`) and conference proceedings (`p`). Result: 290 records (2016–2025: 284; 2026, partial up to
the run date: 6). Only the yearly counts were exported from Scopus (`exports/Scopus-10-Analyze-Year (1).csv`, parsed by
`scopus_from_csv.py` into `scopus_counts.json`). **No record-level Scopus export exists**, so the Scopus records were not
screened record by record and the database arm of the PRISMA diagram reports counts only. IEEE Xplore and Web of Science
were not searched. Both points are stated as limitations of the review.

## 2. Supplementary scripted search (12 November 2026; reported under "other methods" in PRISMA)
Task phrases (title and abstract): face inpainting · face completion · facial inpainting · facial image completion ·
face image inpainting · face de-occlusion · face unmasking · mask removal · masked face (reconstruction|restoration|completion|recovery) ·
occluded face (completion|restoration|reconstruction) · eye in-painting · head inpainting.
- OpenAlex: `search_openalex.py` (`title_and_abstract.search`, from 2016-01-01; article / proceedings-article / preprint).
- arXiv: `search_arxiv.py` (cs.CV, title and abstract).
De-duplication: `dedup.py` (DOI, then normalised title). Title/abstract screening by one author with a logged reason per record
(`screening/screening_log.csv`, built by `make_screening_log.py`).

## 3. Other sources (reported under "other methods" in PRISMA)
- Backward/forward citation chasing from included FU papers and from the 10 papers suggested during peer review; curated method lists.
- Papers named by the reviewers.
- Benchmark baselines (general inpainting methods run in our benchmark), labelled "General inpainting (baseline)".
