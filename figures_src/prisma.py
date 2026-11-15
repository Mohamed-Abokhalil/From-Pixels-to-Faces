"""PRISMA 2020 flow diagram, laid out as the PRISMA 2020 template for new reviews that include searches of databases
and other sources (Page et al. 2021): grey column headers, grey phase bars, square boxes with thin black borders.

Database arm : Scopus (literature/scopus_counts.json; yearly counts only, no record-level export).
Other arm    : supplementary OpenAlex/arXiv search (literature/screening/*), citation searching and curated lists,
               and suggested studies (route column of literature/taxonomy_methods.csv). Single screener.
Output: manuscript/fig_prisma.pdf (vector) and figures_src/out/fig_prisma.png
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REV = Path(__file__).resolve().parents[1]
L = REV / "literature"
S = L / "screening"
OUT = REV / "manuscript"; (REV / "figures_src" / "out").mkdir(exist_ok=True)
LAB = {"E-DOMAIN": "Not face images", "E-TASK": "No mask-conditioned completion", "E-TYPE": "Not a research article",
       "E-REVIEW": "Survey or review", "E-NONDL": "No learned model", "E-NOTMETHOD": "Evaluation study only",
       "E-BFR": "Blind restoration without mask", "E-ACCESS": "Not assessable"}
GREY = "#d9d9d9"
FS = 6.2
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def box(ax, x, y, w, h, text, fc="white", fs=FS, ha="center", bold=False):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="black", lw=0.6))
    tx = x + w / 2 if ha == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, linespacing=1.3, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black", shrinkA=0, shrinkB=0,
                                                                  mutation_scale=7))


def main():
    sc = json.load(open(L / "scopus_counts.json"))
    ident = json.load(open(S / "counts_identification.json"))
    summ = json.load(open(S / "screening_summary.json"))
    log = list(csv.DictReader(open(S / "screening_log.csv", encoding="utf8")))
    excl = Counter(r["exclusion_code"] for r in log if r["title_abstract_decision"] == "exclude")
    routes = summ["taxonomy_route_counts"]
    per = ident["per_export"]
    n_oa = sum(v for k, v in per.items() if k.startswith("openalex")); n_ax = sum(v for k, v in per.items() if k.startswith("arxiv"))
    n_only_other = summ["taxonomy_total"] - summ["retained_and_in_taxonomy"]
    d = __import__("datetime").date.fromisoformat(sc["run_date"])

    fig, ax = plt.subplots(figsize=(7.4, 6.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # column headers and phase bars
    box(ax, 0.06, 0.935, 0.42, 0.055, "Identification of studies via databases", fc=GREY, bold=True)
    box(ax, 0.52, 0.935, 0.47, 0.055, "Identification of studies via other methods", fc=GREY, bold=True)
    for y0, y1, t in ((0.715, 0.92, "Identification"), (0.18, 0.70, "Screening"), (0.01, 0.165, "Included")):
        ax.add_patch(Rectangle((0.0, y0), 0.045, y1 - y0, fc=GREY, ec="black", lw=0.6))
        ax.text(0.0225, (y0 + y1) / 2, t, rotation=90, ha="center", va="center", fontsize=FS + 0.4, fontweight="bold")

    # database arm
    box(ax, 0.06, 0.73, 0.22, 0.19, f"Records identified from\nScopus (n = {sc['total']})\n\n2016–2025: {sc['total_2016_2025']}\n"
                                   f"2026 (to {d.day} {d:%b} {d.year}): {sc['by_year']['2026']}")
    box(ax, 0.30, 0.73, 0.18, 0.19, f"Records not screened\nindividually (n = {sc['total']}):\nyearly counts only,\nno record-level export")
    arrow(ax, 0.28, 0.825, 0.30, 0.825)

    # other-methods arm
    box(ax, 0.52, 0.73, 0.24, 0.19, f"Records identified from:\nOpenAlex (n = {n_oa})\narXiv (n = {n_ax})\n\nMethods identified from:\n"
                                   f"citation searching and\ncurated lists (n = {routes.get('other', 0)})\nsuggested studies (n = {routes.get('suggested', 0)})", fs=FS - 0.3)
    box(ax, 0.79, 0.77, 0.20, 0.11, f"Duplicate records\nremoved (n = {ident['duplicates_removed']})")
    arrow(ax, 0.76, 0.825, 0.79, 0.825)

    box(ax, 0.52, 0.55, 0.24, 0.10, f"Records screened\n(title and abstract)\n(n = {summ['records_screened']})")
    arrow(ax, 0.64, 0.73, 0.64, 0.65)
    reasons = "\n".join(f"{LAB.get(k, k)}: {v}".replace("No mask-conditioned ", "No mask-conditioned\n").replace("Blind restoration ", "Blind restoration\n")
                        for k, v in sorted(excl.items(), key=lambda kv: -kv[1]))
    box(ax, 0.79, 0.45, 0.20, 0.28, f"Records excluded\n(n = {summ['excluded_title_abstract']})\n\n{reasons}", fs=FS - 0.5)
    arrow(ax, 0.76, 0.60, 0.79, 0.60)

    box(ax, 0.52, 0.37, 0.24, 0.10, f"Records retained\n(n = {summ['retained_title_abstract']})")
    arrow(ax, 0.64, 0.55, 0.64, 0.47)
    box(ax, 0.79, 0.20, 0.20, 0.13, f"Records not assessed\nindividually\n(n = {summ['retained_not_reviewed_individually']})")

    box(ax, 0.52, 0.20, 0.24, 0.13, f"Records describing\nmethods included\nin the review\n(n = {summ['retained_and_in_taxonomy']})")
    arrow(ax, 0.64, 0.37, 0.64, 0.33)
    ax.plot([0.76, 0.89], [0.42, 0.42], color="black", lw=0.6); arrow(ax, 0.89, 0.42, 0.89, 0.33)

    # included
    box(ax, 0.30, 0.025, 0.46, 0.12, f"Methods included in the review (n = {summ['taxonomy_total']})\n\n"
                                    f"{summ['retained_and_in_taxonomy']} from the screened records;\n"
                                    f"{n_only_other} from citation searching, curated\nlists and suggested studies")
    arrow(ax, 0.64, 0.20, 0.64, 0.145)
    # methods identified outside the scripted search go straight to inclusion
    ax.plot([0.52, 0.50, 0.50], [0.76, 0.76, 0.20], color="black", lw=0.6)
    arrow(ax, 0.50, 0.20, 0.50, 0.145)
    for p in (OUT / "fig_prisma.pdf", REV / "figures_src" / "out" / "fig_prisma.png"):
        fig.savefig(p, dpi=300, bbox_inches="tight")
    print("PRISMA written")


if __name__ == "__main__":
    main()
