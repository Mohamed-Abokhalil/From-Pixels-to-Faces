"""PRISMA 2020 flow diagram (R1.1c), two-column layout of PRISMA 2020 for new reviews that include searches of
databases and other sources.

Left column  : the database search (Scopus; counts in literature/scopus_counts.json, supplied by the authors).
               The record-level Scopus export is not available, so its screening counts are shown as not available.
Right column : other sources -- the scripted supplementary search of OpenAlex and arXiv run for this revision
               (literature/screening/counts_identification.json, screening_log.csv / screening_summary.json from
               make_screening_log.py), citation chasing and curated lists, and reviewer suggestions
               (route column of literature/taxonomy_methods.csv).
Screening was done by one author (no inter-rater agreement).
Output: FI_FU_Survey_AIR_revised/fig_prisma.pdf (vector) and figures_src/out/fig_prisma.png
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REV = Path(__file__).resolve().parents[1]
L = REV / "literature"
S = L / "screening"
OUT = REV / "FI_FU_Survey_AIR_revised"; (REV / "figures_src" / "out").mkdir(exist_ok=True)
LAB = {"E-DOMAIN": "not face images", "E-TASK": "no mask-conditioned completion", "E-TYPE": "not a research article",
       "E-REVIEW": "survey/review", "E-NONDL": "no learned model", "E-NOTMETHOD": "evaluation study only",
       "E-BFR": "blind restoration, no mask", "E-ACCESS": "not assessable"}


def box(ax, x, y, w, h, text, fc="#ffffff", fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.01", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.22)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))


def main():
    sc = json.load(open(L / "scopus_counts.json"))
    ident = json.load(open(S / "counts_identification.json"))
    summ = json.load(open(S / "screening_summary.json"))
    log = list(csv.DictReader(open(S / "screening_log.csv", encoding="utf8")))
    excl = Counter(r["exclusion_code"] for r in log if r["title_abstract_decision"] == "exclude")
    routes = summ["taxonomy_route_counts"]
    per = ident["per_export"]
    n_oa = sum(v for k, v in per.items() if k.startswith("openalex")); n_ax = sum(v for k, v in per.items() if k.startswith("arxiv"))

    fig, ax = plt.subplots(figsize=(7.4, 7.0)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.27, 0.985, "Identification of studies via databases", ha="center", fontsize=7.6, fontweight="bold")
    ax.text(0.75, 0.985, "Identification of studies via other methods", ha="center", fontsize=7.6, fontweight="bold")
    for y, t in ((0.86, "Identification"), (0.52, "Screening"), (0.08, "Included")):
        ax.text(0.012, y, t, rotation=90, ha="center", va="center", fontsize=7.6, fontweight="bold", color="#555")
    # database column (Scopus)
    box(ax, 0.04, 0.84, 0.24, 0.12, f"Records identified from Scopus\n(n = {sc['total']}; 2016-2025: {sc['total_2016_2025']},\n2026 partial: {sc['by_year']['2026']})\njournals and\nconference proceedings")
    box(ax, 0.30, 0.84, 0.20, 0.12, "Record-level export\nnot available;\nrecords not screened\nindividually")
    arrow(ax, 0.28, 0.90, 0.30, 0.90)
    box(ax, 0.04, 0.66, 0.24, 0.10, "Records screened\n(n = not available)", fc="#f4f4f4")
    arrow(ax, 0.16, 0.84, 0.16, 0.76)
    # other-methods column
    box(ax, 0.54, 0.84, 0.44, 0.12, f"Supplementary scripted search (12 Nov 2026):\nOpenAlex (n = {n_oa}), arXiv (n = {n_ax})\nduplicates removed (n = {ident['duplicates_removed']})\n"
                                    f"Citation chasing and curated lists (n = {routes.get('other', 0)} methods)\nReviewer suggestions (n = {routes.get('reviewer', 0)} methods)")
    box(ax, 0.54, 0.66, 0.20, 0.12, f"Search records screened\n(title/abstract, one author)\n(n = {summ['records_screened']})")
    arrow(ax, 0.64, 0.84, 0.64, 0.78)
    ex_txt = f"Records excluded\n(n = {summ['excluded_title_abstract']})\n" + "\n".join(
        f"{LAB.get(k, k)}: {v}" for k, v in sorted(excl.items(), key=lambda kv: -kv[1]))
    box(ax, 0.765, 0.48, 0.225, 0.30, ex_txt.replace("no mask-conditioned completion", "no mask-conditioned\ncompletion"), fs=6.0)
    arrow(ax, 0.74, 0.72, 0.77, 0.66)
    box(ax, 0.54, 0.44, 0.20, 0.13, f"Records retained\n(n = {summ['retained_title_abstract']})")
    arrow(ax, 0.64, 0.66, 0.64, 0.57)
    box(ax, 0.77, 0.26, 0.21, 0.17, f"Retained but not\nreviewed individually;\nno full-text decision\nrecorded\n(n = {summ['retained_not_reviewed_individually']})", fs=6.2)
    arrow(ax, 0.74, 0.50, 0.77, 0.36)
    box(ax, 0.54, 0.26, 0.20, 0.13, f"Retained records that are\nmethods of the taxonomy\n(n = {summ['retained_and_in_taxonomy']})")
    arrow(ax, 0.64, 0.44, 0.64, 0.39)
    # included
    n_only_other = summ["taxonomy_total"] - summ["retained_and_in_taxonomy"]
    box(ax, 0.30, 0.02, 0.44, 0.13, f"Methods included in the review (taxonomy table)\n(n = {summ['taxonomy_total']}: {summ['retained_and_in_taxonomy']} found by the scripted search,\n"
                                    f"{n_only_other} only through other sources)", fc="#eef3f8")
    arrow(ax, 0.64, 0.26, 0.60, 0.15)
    ax.annotate("", xy=(0.70, 0.15), xytext=(0.99, 0.84),
                                                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333", connectionstyle="angle,angleA=-90,angleB=0"))
    for p in (OUT / "fig_prisma.pdf", REV / "figures_src" / "out" / "fig_prisma.png"):
        fig.savefig(p, dpi=300, bbox_inches="tight")
    print("PRISMA written")


if __name__ == "__main__":
    main()
