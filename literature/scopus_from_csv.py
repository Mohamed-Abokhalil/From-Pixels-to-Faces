"""Build literature/scopus_counts.json from the official Scopus yearly-count export (N-01).

Source: literature/exports/Scopus-10-Analyze-Year (1).csv  (Scopus > Analyze search results > Documents by year),
query run by the authors on 9 July 2026. Scopus provides only yearly counts here; no record-level export exists.
The script parses the query and the per-year counts from the CSV, checks them against the counts used in the
manuscript (previous scopus_counts.json, if present) and rewrites scopus_counts.json.
"""
import csv
import json
import re
from pathlib import Path

L = Path(__file__).resolve().parent
SRC = L / "exports" / "Scopus-10-Analyze-Year (1).csv"
OUT = L / "scopus_counts.json"
RUN_DATE = "2026-07-09"

rows = list(csv.reader(open(SRC, encoding="utf-8-sig")))
query = None
by_year = {}
for r in rows:
    if r and r[0].startswith("Your query"):
        query = r[0].split(":", 1)[1].strip()
    elif len(r) >= 2 and re.fullmatch(r"\d{4}", r[0].strip()) and r[1].strip().isdigit():
        by_year[r[0].strip()] = int(r[1])
assert query and by_year, "could not parse the Scopus export"
by_year = dict(sorted(by_year.items()))
total = sum(by_year.values())
ten = sum(v for y, v in by_year.items() if int(y) <= 2025)

if OUT.exists():
    old = json.load(open(OUT, encoding="utf8"))
    assert old["by_year"] == by_year, ("yearly counts differ from those used in the manuscript", old["by_year"], by_year)
    assert old["total"] == total and old["total_2016_2025"] == ten
    assert re.sub(r"\s+", " ", old["query"]) == re.sub(r"\s+", " ", query), "query differs"

json.dump({
    "source": "literature/exports/Scopus-10-Analyze-Year (1).csv (Scopus, Analyze search results > Documents by year), "
              "exported by the authors. Only yearly counts are available; there is no record-level Scopus export.",
    "database": "Scopus",
    "query": query,
    "source_types": "journals (j) and conference proceedings (p)",
    "run_date": RUN_DATE,
    "by_year": by_year,
    "total": total,
    "total_2016_2025": ten,
    "note_2026": f"partial year: records up to the run date ({RUN_DATE})",
}, open(OUT, "w", encoding="utf8"), indent=1)
print(f"Scopus export parsed: total {total}, 2016-2025 {ten}, 2026 {by_year.get('2026')}; matches manuscript counts")
