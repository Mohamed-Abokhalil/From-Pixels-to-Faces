"""Figure 1: Scopus records per publication year for the query in literature/scopus_counts.json (journals and conference
proceedings, 2016-2026; 2026 is a partial year). Counts supplied by the authors from Scopus "Analyze search results".
Output: manuscript/fig_fi_trend.pdf"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = Path(__file__).resolve().parents[1]
sc = json.load(open(REV / "literature" / "scopus_counts.json"))
years = sorted(int(y) for y in sc["by_year"])
n = [sc["by_year"][str(y)] for y in years]
assert sum(n) == sc["total"]
fig, ax = plt.subplots(figsize=(6.6, 2.5))
ax.bar(years[:-1], n[:-1], color="#4C78A8", label="Scopus records (journal and conference papers)")
ax.bar(years[-1:], n[-1:], color="#4C78A8", alpha=0.45, hatch="///", edgecolor="#4C78A8", label="2026 (partial year)")
for y, v in zip(years, n):
    ax.text(y, v + 0.8, str(v), ha="center", fontsize=7)
ax.set_xticks(years); ax.set_xticklabels([str(y) for y in years], fontsize=7)
ax.set_ylabel("Records", fontsize=8); ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(REV / "manuscript" / "fig_fi_trend.pdf", bbox_inches="tight")
fig.savefig(REV / "figures_src" / "out" / "fig_fi_trend.png", dpi=200, bbox_inches="tight")
print("total", sum(n), dict(zip(years, n)))
