"""Recall check: which methods in the submitted taxonomy (Table 10) were retrieved by the search?
Matching: normalised title of the bib entry vs. normalised titles of de-duplicated records (exact or
one contained in the other). Output: screening/recall_vs_submitted_table10.csv"""
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "FI_FU_Survey_AIR" / "sn-article.tex"
BIB = HERE.parent / "FI_FU_Survey_AIR" / "sn-bibliography.bib"


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def bib_titles():
    s = open(BIB, encoding="utf8").read()
    out = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", s, re.S):
        t = re.search(r"title\s*=\s*\{(.*?)\},?\s*\n", m.group(2), re.S)
        if t:
            out[m.group(1).strip()] = re.sub(r"[{}]", "", t.group(1))
    return out


def main():
    L = open(TEX, encoding="utf8").read().split("\n")
    keys = [k.strip() for g in re.findall(r"\\cite\{([^}]*)\}", "\n".join(L[1203:1262])) for k in g.split(",")]
    bt = bib_titles()
    recs = list(csv.DictReader(open(HERE / "screening" / "records_dedup.csv", encoding="utf8")))
    rn = [(r["rec_id"], norm(r["title"])) for r in recs]
    rows, hit = [], 0
    for k in keys:
        t = norm(bt.get(k, ""))
        m = [rid for rid, x in rn if t and (x == t or (len(t) > 25 and (t in x or x in t)))]
        hit += bool(m)
        rows.append({"bibkey": k, "title": bt.get(k, ""), "retrieved": bool(m), "rec_ids": ";".join(m)})
    with open(HERE / "screening" / "recall_vs_submitted_table10.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"retrieved {hit}/{len(keys)}")
    for r in rows:
        if not r["retrieved"]:
            print("  missed:", r["bibkey"], "|", r["title"][:80])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
