"""First-pass title/abstract screening with a logged reason per record.

This is a transparent, rule-based PRE-screen produced by the assistant; it is NOT the human
screening. Two authors screen independently from screening_sheet_*.csv; Cohen's kappa is
computed by kappa.py from their sheets. Manual overrides (checked by reading title+abstract)
are listed in OVERRIDES with a reason.

Criteria (see criteria.md):
 I1 peer-reviewed journal/conference paper or arXiv preprint (preprints labelled)
 I2 proposes a deep-learning method for mask-conditioned FACE completion / de-occlusion / unmasking,
    or a general inpainting method evaluated on a face benchmark
 E-TYPE  not a primary research article (thesis, dataset/software record, editorial, erratum, abstract-only, report)
 E-REVIEW survey/review (kept in a separate list for the related-surveys comparison, Table 1)
 E-DOMAIN not about faces
 E-TASK  face task without mask-conditioned completion (recognition/detection/verification/editing/generation only)
 E-NONDL no learned model (PCA/graph/patch/PDE only)
 E-ACCESS no abstract and no accessible full text to judge (flagged for human check)
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = HERE / "screening"

FACE = re.compile(r"\bfac(e|es|ial)\b|\bportrait|\bhead\b|\bperiocular|\bmouth|\blip", re.I)
COMPLETION = re.compile(r"inpaint|complet|de-?occlu|unmask|mask removal|remov\w* (the )?(face )?mask|restor|reconstruct|hallucinat|fill|recover", re.I)
DL = re.compile(r"network|deep|learn|gan\b|gans\b|adversarial|transformer|diffusion|autoencoder|\bcnn|convolution|lstm|vae|stylegan|neural", re.I)
REVIEW = re.compile(r"\bsurvey\b|\breview\b|state of the art and|: a review|overview of", re.I)
RECOG_ONLY = re.compile(r"recogni|detect|verification|identif|classif|attendance|authenticat", re.I)
BAD_TYPES = {"dissertation", "dataset", "software", "editorial", "erratum", "conference-abstract", "report", "other", "paratext", "letter"}

OVERRIDES = []   # [title regex, decision, code, reason]; manual decisions after reading title+abstract
if (HERE / "overrides.json").exists():
    OVERRIDES = json.load(open(HERE / "overrides.json", encoding="utf8"))


def fu_flag(t):
    """FU = removal of protective/face masks only. HMD, glasses, hands etc. are structured occlusions (FI)."""
    return bool(re.search(r"unmask|mask removal|remov\w* (the )?(face ?)?masks?|masked[- ]face|face ?masks?|surgical mask|covid|respirator|mask-occluded|removal of face masks", t, re.I))


def screen(r):
    t, a = r["title"], r["abstract"]
    ta = t + " " + a
    for pat, dec, code, why in OVERRIDES:
        if re.search(pat, t):
            return (dec, code, why + " [manual]")
    if r["type"] in BAD_TYPES:
        return ("exclude", "E-TYPE", f"record type '{r['type']}'")
    if REVIEW.search(t):
        return ("exclude", "E-REVIEW", "survey/review in title")
    if not FACE.search(ta):
        return ("exclude", "E-DOMAIN", "no face term in title/abstract")
    if not COMPLETION.search(ta):
        return ("exclude", "E-TASK", "no completion/inpainting/de-occlusion term")
    if RECOG_ONLY.search(t) and not COMPLETION.search(t):
        return ("exclude", "E-TASK", "recognition/detection task in title; completion only auxiliary")
    if not a.strip():
        return ("include?", "CHECK", "no abstract in export: title suggests eligibility; full text needed")
    if not DL.search(ta):
        return ("exclude", "E-NONDL", "no learned model mentioned")
    return ("include", "I", "face completion/de-occlusion method with learned model")


def main():
    rows = list(csv.DictReader(open(S / "records_dedup.csv", encoding="utf8")))
    out = []
    for r in rows:
        d, code, why = screen(r)
        cat = ""
        if code == "I-GEN":
            cat = "General inpainting (baseline)"
        elif d.startswith("include"):
            cat = "FU-specific (candidate)" if fu_flag(r["title"]) or (fu_flag(r["abstract"]) and re.search(r"remov|inpaint|reconstruct|restor|complet", r["title"], re.I) and re.search(r"mask", r["title"], re.I)) else "FI-specific (candidate)"
        out.append({"rec_id": r["rec_id"], "year": r["year"], "type": r["type"], "venue": r["venue"], "doi": r["doi"],
                    "title": r["title"], "decision": d, "code": code, "reason": why, "category_guess": cat})
    with open(S / "first_pass_screen.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    from collections import Counter
    print(Counter(o["code"] for o in out)); print(Counter(o["category_guess"] for o in out if o["category_guess"]))


if __name__ == "__main__":
    main()
