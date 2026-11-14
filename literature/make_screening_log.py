"""Full screening log for Online Resource 1 (R1.1, N-06: single screener).

One row per de-duplicated record of the scripted OpenAlex/arXiv search with its title/abstract decision and reason
(first_pass_screen.csv, applied by one author with the rules in criteria.md and overrides.json), and, for every record
retained at title/abstract, whether it is one of the methods of the taxonomy table (matched by normalised title against
the bibliography entry of each taxonomy key). Retained records that are not in the taxonomy have no individual full-text
decision; they are labelled as such rather than given an invented reason.
Output: screening/screening_log.csv and screening/screening_summary.json
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

L = Path(__file__).resolve().parent
REV = L.parent
norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())

bib = open(REV / "FI_FU_Survey_AIR_revised" / "sn-bibliography.bib", encoding="utf8").read()
key_title = {}
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S):
    t = re.search(r"\btitle\s*=\s*[{\"](.+?)[}\"],?\s*\n", m.group(2), re.S)
    if t:
        key_title[m.group(1).strip()] = norm(re.sub(r"[{}]", "", t.group(1)))
tax = list(csv.DictReader(open(L / "taxonomy_methods.csv", encoding="utf8")))
title_key = {key_title[r["key"]]: r for r in tax if r["key"] in key_title}

rows = list(csv.DictReader(open(L / "screening" / "first_pass_screen.csv", encoding="utf8")))
out = []
for r in rows:
    t = title_key.get(norm(r["title"]))
    retained = r["decision"].startswith("include")
    if not retained:
        stage2 = ""
    elif t:
        stage2 = f"included: taxonomy table ({t['key']}, {t['category']})"
    else:
        stage2 = "retained at title/abstract; not reviewed individually (no full-text decision recorded)"
    out.append({"rec_id": r["rec_id"], "year": r["year"], "doi": r["doi"], "title": r["title"], "venue": r["venue"],
                "title_abstract_decision": "retain" if retained else "exclude", "exclusion_code": "" if retained else r["code"],
                "exclusion_reason": "" if retained else r["reason"], "stage2": stage2})
with open(L / "screening" / "screening_log.csv", "w", newline="", encoding="utf8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
c = Counter(o["stage2"].split(":")[0] if o["stage2"] else "excluded at title/abstract" for o in out)
summ = {"records_screened": len(out), "excluded_title_abstract": sum(o["title_abstract_decision"] == "exclude" for o in out),
        "retained_title_abstract": sum(o["title_abstract_decision"] == "retain" for o in out),
        "retained_and_in_taxonomy": c.get("included", 0),
        "retained_not_reviewed_individually": sum(o["stage2"].startswith("retained") for o in out),
        "taxonomy_route_counts": dict(Counter(r["route"] for r in tax)), "taxonomy_total": len(tax)}
json.dump(summ, open(L / "screening" / "screening_summary.json", "w"), indent=1)
sc = json.load(open(L / "scopus_counts.json"))
mac = {"SLscreened": summ["records_screened"], "SLexcluded": summ["excluded_title_abstract"], "SLretained": summ["retained_title_abstract"],
       "SLinTax": summ["retained_and_in_taxonomy"], "SLnotReviewed": summ["retained_not_reviewed_individually"],
       "SLonlyOther": summ["taxonomy_total"] - summ["retained_and_in_taxonomy"],
       "ScopusTotal": sc["total"], "ScopusTenYears": sc["total_2016_2025"], "ScopusPartial": sc["by_year"]["2026"],
       "ScopusRunDate": (lambda d: f"{d.day} {d:%B %Y}")(__import__("datetime").date.fromisoformat(sc["run_date"]))}
open(REV / "FI_FU_Survey_AIR_revised" / "screening_numbers.tex", "w", encoding="utf8").write(
    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print(json.dumps(summ, indent=1))
